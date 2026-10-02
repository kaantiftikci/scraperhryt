"""Hürriyet Gündem kaynağı.

Keşif: ``/rss/gundem`` (100 öğe; ``<text>`` içinde tam HTML gövde, ``<abstract>`` spot) + ``/gundem/`` liste sayfası
(30 bağlantı). ``parse_rss`` beslemedeki her Hürriyet haberini döndürür (besleme ara sıra ``/dunya/``, ``/kelebek/``
gibi eski haberleri de listeler); ``discover`` ise yalnızca ``/gundem/<slug>-<id>`` biçimindeki URL'leri geçirir.
Haber sayfası JSON-LD ``NewsArticle`` (dict) ve ``div.news-content`` paragraflarından ayrıştırılır; sayfa
alınamazsa RSS gövdesinden kayıt üretilir. ``robots.txt`` ``/api/`` ve ``/arama/`` yollarını yasaklar; bu yüzden
arşiv/arama taraması yapılmaz (``backfill_days`` yok sayılır).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlsplit

import requests

from ..config import Settings, get_settings
from ..models import NewsRecord
from ..textutil import html_to_text
from .base import (
    DiscoveredLink,
    HttpClient,
    HttpError,
    clean_text,
    dedupe_links,
    extract_meta,
    extract_paragraphs,
    first_nonempty,
    jsonld_keywords,
    jsonld_name,
    jsonld_url,
    make_soup,
    paragraphize_flat_text,
    parse_jsonld_newsarticle,
    parse_tr_date,
    rss_items,
    split_keywords,
    text_of,
    xml_child_attr,
    xml_child_text,
    xml_root,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.hurriyet.com.tr"
_HOSTS = frozenset({"www.hurriyet.com.tr", "hurriyet.com.tr"})
_ARTICLE_PATH_RE = re.compile(r"^(?:/[a-z0-9-]+)+/[a-z0-9-]+-\d{4,}$")  # herhangi bir Hürriyet haberi
_GUNDEM_PATH_RE = re.compile(r"^/gundem/[a-z0-9-]+-\d{4,}$")

# articleBody ve sayfa gövdesine sızan Google "tercih edilen kaynak" kutusu ile reklam kalıpları
_BOILERPLATE_PHRASES = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"haberlerimizi google['’`]?da takip edin\.?",
        r"gelişmelerden anında haberdar olun\.?",
        r"google['’`]?da tercih edilen\s+kaynak olarak ekleyin\.?",
    )
)
_DROP_PARAGRAPH_PATTERNS = _BOILERPLATE_PHRASES + tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"^haberin devamı$",
        r"^haberle ilgili daha fazlası",
        r"^bakmadan geçme!?$",
    )
)
_DROP_SELECTORS = (
    ".google-preferred-source",
    ".medyanet-inline-adv",
    ".adRenderer",
    ".inline-ads",
    ".read-more-detail",
    ".news-more-tags",
    ".news-tb-adv",
    ".taboolaAd",
    ".promo",
    "script",
    "style",
    "noscript",
    "iframe",
    "figure",
    "ins",
    "form",
)


class HurriyetSource:
    """Hürriyet Gündem kazıyıcısı (``Source`` protokolü)."""

    name = "hurriyet"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.rss_url = self.settings.hurriyet_gundem_rss
        self.listing_url = self.settings.hurriyet_gundem_listing
        parts = urlsplit(self.listing_url)
        self.base_url = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else DEFAULT_BASE_URL

    # --- URL ---
    def normalize_url(self, href: str | None, *, gundem_only: bool = False) -> str | None:
        """Göreli/mutlak bağlantıyı kanonik haber URL'sine çevirir; Hürriyet haberi değilse None.

        ``gundem_only=True`` ile yalnızca ``/gundem/<slug>-<id>`` yolları kabul edilir.
        """
        if not href:
            return None
        parts = urlsplit(urljoin(self.base_url + "/", href.strip()))
        if parts.netloc.lower() not in _HOSTS:
            return None
        path = re.sub(r"/{2,}", "/", parts.path).rstrip("/").lower()
        pattern = _GUNDEM_PATH_RE if gundem_only else _ARTICLE_PATH_RE
        if not pattern.match(path):
            return None
        return f"{DEFAULT_BASE_URL}{path}"

    @staticmethod
    def is_gundem_url(url: str) -> bool:
        return bool(_GUNDEM_PATH_RE.match(urlsplit(url).path.rstrip("/").lower()))

    # --- keşif ---
    def discover(
        self, client: HttpClient, *, backfill_days: int = 0, limit: int | None = None
    ) -> list[DiscoveredLink]:
        if backfill_days > 0:
            log.debug("Hürriyet için arşiv taraması yok (robots.txt /arama/ yasak); backfill_days yok sayıldı")
        links: list[DiscoveredLink] = []
        steps = (("RSS", self.rss_url, self.parse_rss), ("liste", self.listing_url, self.parse_listing))
        for label, url, parser in steps:
            try:
                text = client.get_text(url)
            except (HttpError, requests.RequestException) as exc:
                log.warning("Hürriyet %s alınamadı (%s): %s", label, url, exc)
                continue
            found = parser(text)
            gundem = [link for link in found if self.is_gundem_url(link.url)]
            log.info("Hürriyet %s: %d haber bağlantısı (%d gündem)", label, len(found), len(gundem))
            links.extend(gundem)
            if limit is not None and len(dedupe_links(links)) >= limit:
                break
        unique = dedupe_links(links)
        return unique[:limit] if limit is not None else unique

    def parse_rss(self, xml_text: str | bytes) -> list[DiscoveredLink]:
        """Beslemedeki tüm Hürriyet haber bağlantıları (gündem dışı yollar dahil; süzme ``discover``'da yapılır)."""
        root = xml_root(xml_text)
        if root is None:
            log.warning("Hürriyet RSS ayrıştırılamadı (XML değil)")
            return []
        links: list[DiscoveredLink] = []
        for item in rss_items(root):
            url = self.normalize_url(xml_child_text(item, "link") or xml_child_text(item, "guid"))
            if not url:
                continue
            published = parse_tr_date(xml_child_text(item, "pubDate")) or parse_tr_date(
                xml_child_text(item, "dateTimeWritten")
            )
            links.append(
                DiscoveredLink(
                    url=url,
                    title_hint=clean_text(xml_child_text(item, "title")),
                    published_hint=published,
                    category_hint=clean_text(xml_child_text(item, "category")),
                    rss_html=xml_child_text(item, "text"),
                    rss_summary=clean_text(xml_child_text(item, "abstract") or xml_child_text(item, "description")),
                    updated_hint=parse_tr_date(xml_child_text(item, "modified")),
                    image_hint=xml_child_attr(item, "content", "url")
                    or xml_child_attr(item, "enclosure", "url")
                    or xml_child_attr(item, "thumbnail", "url"),
                    origin="rss",
                )
            )
        return dedupe_links(links)

    def parse_listing(self, html: str) -> list[DiscoveredLink]:
        """``/gundem/`` liste sayfasındaki gündem haberi bağlantıları (sayfadaki diğer bölümlerin haberleri elenir)."""
        soup = make_soup(html)
        links: list[DiscoveredLink] = []
        for anchor in soup.select("a[href]"):
            url = self.normalize_url(anchor.get("href"), gundem_only=True)
            if not url:
                continue
            links.append(
                DiscoveredLink(
                    url=url,
                    title_hint=clean_text(anchor.get("title") or anchor.get_text(" ")),
                    category_hint="Gündem",
                    origin="listing",
                )
            )
        return dedupe_links(links)

    # --- haber ---
    def fetch_article(self, client: HttpClient, link: DiscoveredLink) -> NewsRecord | None:
        try:
            html = client.get_text(link.url)
        except (HttpError, requests.RequestException) as exc:
            if link.rss_html:
                log.warning("Hürriyet sayfası alınamadı, RSS gövdesiyle kayıt üretiliyor (%s): %s", link.url, exc)
                return self.record_from_rss(link)
            raise
        record = self.parse_article(html, link)
        if record is None and link.rss_html:
            log.warning("Hürriyet sayfası ayrıştırılamadı, RSS gövdesi kullanılıyor: %s", link.url)
            return self.record_from_rss(link)
        return record

    def parse_article(self, html: str, link: DiscoveredLink | None = None, url: str = "") -> NewsRecord | None:
        """Haber sayfasını ``NewsRecord``'a çevirir. ``link`` verilirse RSS ipuçları yedek olarak kullanılır."""
        link = link or DiscoveredLink(url=url)
        if not link.url:
            raise ValueError("parse_article için link veya url gerekli")
        soup = make_soup(html)
        ld = parse_jsonld_newsarticle(soup) or {}
        meta = extract_meta(soup)

        title = first_nonempty(
            ld.get("headline"), text_of(soup.select_one("h1.news-detail-title")), meta.get("og:title"), link.title_hint
        )
        subtitle = first_nonempty(
            ld.get("description"), text_of(soup.select_one("div.news-content__inf h2")), link.rss_summary
        )
        time_tag = soup.select_one("time[datetime]")
        time_attr = str(time_tag.get("datetime") or "") if time_tag is not None else ""
        published_at = (
            parse_tr_date(ld.get("datePublished"))
            or parse_tr_date(meta.get("datepublished"))
            or link.published_hint
            or parse_tr_date(text_of(soup.select_one("span.news-date")))
            or parse_tr_date(time_attr)
        )
        updated_at = (
            parse_tr_date(ld.get("dateModified"))
            or parse_tr_date(meta.get("datemodified"))
            or parse_tr_date(time_attr)
            or link.updated_hint
        )
        content = (
            self._content_from_html(soup)
            or self.clean_article_body(ld.get("articleBody"))
            or clean_text(html_to_text(link.rss_html))
        )
        if not title or not content:
            log.warning("Hürriyet haberi eksik (başlık=%s, içerik=%d karakter): %s", bool(title), len(content), link.url)
            return None

        category = first_nonempty(ld.get("articleSection"), link.category_hint, meta.get("article:section"), "Gündem")
        author = first_nonempty(
            jsonld_name(ld.get("author")),
            meta.get("article:author"),
            text_of(soup.select_one(".news-profile__account--editorname")),
        )
        tags = jsonld_keywords(ld.get("keywords")) or split_keywords(meta.get("keywords"))
        image_url = first_nonempty(meta.get("og:image"), jsonld_url(ld.get("image")), link.image_hint)
        return NewsRecord.new(
            source=self.name,
            content_url=link.url,
            title=title,
            subtitle=subtitle,
            content=content,
            published_at=published_at,
            updated_at=updated_at,
            category=category,
            author=author,
            image_url=image_url,
            tags=tags,
        )

    def record_from_rss(self, link: DiscoveredLink) -> NewsRecord | None:
        """Sayfa alınamadığında RSS ``<text>`` gövdesi ve ``<abstract>`` spotundan kayıt üretir."""
        content = clean_text(html_to_text(link.rss_html))
        if not link.title_hint or not content:
            log.warning("RSS verisi kayıt üretmeye yetmiyor: %s", link.url)
            return None
        return NewsRecord.new(
            source=self.name,
            content_url=link.url,
            title=link.title_hint,
            subtitle=link.rss_summary,
            content=content,
            published_at=link.published_hint,
            updated_at=link.updated_hint,
            category=link.category_hint or "Gündem",
            image_url=link.image_hint,
        )

    @staticmethod
    def _content_from_html(soup) -> str:
        body = soup.select_one("div.news-content")
        if body is None:
            return ""
        paragraphs = extract_paragraphs(body, drop_selectors=_DROP_SELECTORS, drop_patterns=_DROP_PARAGRAPH_PATTERNS)
        return "\n\n".join(paragraphs)

    @staticmethod
    def clean_article_body(body: str | None) -> str:
        """JSON-LD ``articleBody``'den Google kalıp metnini atar ve kaybolan paragraf sınırlarını geri getirir."""
        text = clean_text(body)
        for pattern in _BOILERPLATE_PHRASES:
            text = pattern.sub("", text)
        return paragraphize_flat_text(text)
