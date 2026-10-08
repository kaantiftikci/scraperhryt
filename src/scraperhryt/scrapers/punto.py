"""12punto kaynağı (gerçek site: https://12punto.com.tr — 12punto.com park edilmiş alan adıdır).

Keşif: ``/rss`` (20 karışık öğe) + ``/rss/<kategori>`` (ayarlardaki her kategori) + ``/<kategori>`` liste sayfaları +
``backfill_days > 0`` ise gün gün ``/Arama/Ara?search=&StartDate=GG/AA/YYYY&EndDate=GG/AA/YYYY&sayfa=N`` arşiv araması
(site tarihleri yalnızca ``GG/AA/YYYY`` biçiminde kabul eder; ISO ya da noktalı biçim "Sonuç bulunamadı" döndürür;
sayfa başına 20 sonuç, sayfalar ``div.pagination`` içinde listelenir). Sonuçlar yalnızca ``section.category``
kapsayıcısından okunur; kenar çubuğundaki "Çok Okunanlar" bağlantıları arşiv sonucu sayılmaz. Sitenin arşiv dizini
haftalarca gecikebilir (bugün ve yakın günler "Sonuç bulunamadı" döndürür); bu yüzden sonuçsuz bir gün taramayı
DURDURMAZ, istenen pencerenin her günü sorgulanır ve sonunda sonuçsuz gün sayısı loglanır (tümü boşsa uyarı).
RSS bağlantıları ``http://`` gelir → https'e çevrilir, ``www.`` atılır. Yalnızca ``/<kategori>/<slug>-<id>`` biçimi
(``^/[a-z0-9-]+/[a-z0-9-]+-\\d{3,}$``) haber kabul edilir; yazar köşeleri (``/yazarlar/<ad>/<slug>-<id>``) elenir.
Haber sayfası JSON-LD LİSTESİNİN ilk ``NewsArticle`` öğesinden ve ``section.details`` paragraflarından ayrıştırılır.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from urllib.parse import quote, urljoin, urlsplit

import requests
from bs4 import Tag

from ..config import Settings, get_settings
from ..models import NewsRecord
from ..textutil import html_to_text, tr_lower
from .base import (
    ISTANBUL,
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
    looks_like_xml,
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

ARTICLE_PATH_RE = re.compile(r"^/[a-z0-9-]+/[a-z0-9-]+-\d{3,}$")
_NO_RESULTS_RE = re.compile(r"sonuç bulunamadı")
_PAGE_PARAM_RE = re.compile(r"[?&]sayfa=(\d+)")
# 2026 yeniden tasarımı: sonuçlar section.punto-listing, sayfalama nav.punto-pagination; eski düzen de desteklenir.
ARCHIVE_RESULTS_SELECTOR = "section.punto-listing, section.category"
ARCHIVE_PAGINATION_SELECTOR = "nav.punto-pagination a[href], div.pagination a[href]"
ARCHIVE_MAX_PAGES = 50  # gün başına güvenlik tavanı (20 sonuç/sayfa → 1000 haber)
_PUBLISHED_RE = re.compile(r"yayınlanma\s*:\s*(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)")
_UPDATED_RE = re.compile(r"güncelle(?:n)?me\s*:\s*(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)")
_DROP_PARAGRAPH_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"haberlerini algoritmaya bırakma",
        r"^haber kaynağı\s*:",
        r"^abone ol$",
        r"^bağlantı kopyalandı$",
    )
)
_DROP_SELECTORS = (
    ".punto-article-body__source",  # "Haber Kaynağı : 12punto"
    ".punto-ad",
    "img",
    "script",
    "style",
    "noscript",
    "iframe",
    "figure",
    "video",
    "audio",
    "ins",
    "form",
    ".news-source",
    ".add-source",
    ".add-source-link",
    ".share",
    ".date",
    ".gnews",
    ".tooltip",
)


@dataclass
class ArchivePage:
    """Bir arşiv arama sayfasının çözümü: sonuç kapsayıcısındaki haber bağlantıları ve toplam sayfa sayısı."""

    links: list[DiscoveredLink] = field(default_factory=list)
    page_count: int = 1

    @property
    def empty(self) -> bool:
        return not self.links


class PuntoSource:
    """12punto kazıyıcısı (``Source`` protokolü)."""

    name = "12punto"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.punto_base_url.rstrip("/")
        self.host = urlsplit(self.base_url).netloc.lower().removeprefix("www.")
        self.excluded = set(self.settings.punto_excluded_category_list)
        self.categories = [c for c in self.settings.punto_category_list if c.lower() not in self.excluded]

    # --- URL ---
    def normalize_url(self, href: str | None) -> str | None:
        """Bağlantıyı ``https://12punto.com.tr/<kategori>/<slug>-<id>`` biçimine çevirir; haber değilse None."""
        if not href:
            return None
        parts = urlsplit(urljoin(self.base_url + "/", href.strip()))
        if parts.netloc.lower().removeprefix("www.") != self.host:
            return None
        path = re.sub(r"/{2,}", "/", parts.path).rstrip("/").lower()
        if not ARTICLE_PATH_RE.match(path):
            return None
        return f"https://{self.host}{path}"

    @staticmethod
    def _category_from_path(url: str) -> str:
        return urlsplit(url).path.strip("/").split("/", 1)[0]

    def is_excluded(self, url: str) -> bool:
        """Haber dışlanan bir bölümde mi (ör. ``/spor/...``)?"""
        return self._category_from_path(url).lower() in self.excluded

    # --- keşif ---
    def discover(
        self, client: HttpClient, *, backfill_days: int = 0, limit: int | None = None
    ) -> list[DiscoveredLink]:
        collected: list[DiscoveredLink] = []
        seen_keys: set[str] = set()

        def add(found: list[DiscoveredLink]) -> bool:
            """Bağlantıları ekler; limit dolduysa True döndürür (daha fazla istek yapılmaz)."""
            for link in found:
                key = link.canonical
                if key not in seen_keys:
                    seen_keys.add(key)
                collected.append(link)
            return limit is not None and len(seen_keys) >= limit

        feeds = [("", f"{self.base_url}/rss")] + [(cat, f"{self.base_url}/rss/{cat}") for cat in self.categories]
        for category, url in feeds:
            if add(self._fetch_feed(client, url, category)):
                return self._finalize(collected, limit)
        for category in self.categories:
            if add(self._fetch_listing(client, category)):
                return self._finalize(collected, limit)
        if backfill_days > 0:
            today = datetime.now(ISTANBUL).date()
            empty_days: list[date] = []
            for offset in range(backfill_days + 1):
                day = today - timedelta(days=offset)
                found = self._fetch_archive_day(client, day)
                if found is None:
                    continue  # geçici HTTP hatası: gün atlanır, tarama sürer
                if not found:
                    empty_days.append(day)  # arşiv dizini gecikmiş olabilir: sonuçsuz gün taramayı durdurmaz
                    continue
                if add(found):
                    return self._finalize(collected, limit)
            self._log_backfill_summary(backfill_days + 1, empty_days)
        return self._finalize(collected, limit)

    @staticmethod
    def _log_backfill_summary(scanned_days: int, empty_days: list[date]) -> None:
        """Geriye dönük taramanın özeti: tüm günler sonuçsuzsa uyarı (dizin gecikmesi), aksi halde bilgi."""
        if not empty_days:
            log.info("12punto arşiv taraması tamamlandı: %d günün tamamında sonuç var", scanned_days)
            return
        level = logging.WARNING if len(empty_days) >= scanned_days else logging.INFO
        log.log(
            level,
            "12punto arşiv taraması tamamlandı: %d günün %d'i sonuçsuz (%s – %s arası); sitenin arşiv dizini "
            "haftalarca gecikebilir, yakın günler için sonuç olmaması normaldir",
            scanned_days,
            len(empty_days),
            min(empty_days).isoformat(),
            max(empty_days).isoformat(),
        )

    def _finalize(self, links: list[DiscoveredLink], limit: int | None) -> list[DiscoveredLink]:
        all_links = dedupe_links(links)
        unique = [link for link in all_links if not self.is_excluded(link.url)]
        if len(unique) < len(all_links):
            log.info("12punto: %s bölümünden %d bağlantı atlandı", ", ".join(sorted(self.excluded)), len(all_links) - len(unique))
        return unique[:limit] if limit is not None else unique

    def _fetch_feed(self, client: HttpClient, url: str, category: str) -> list[DiscoveredLink]:
        try:
            text = client.get_text(url)
        except HttpError as exc:
            level = logging.INFO if exc.status == 404 else logging.WARNING
            log.log(level, "12punto RSS atlandı (%s): %s", url, exc)
            return []
        except requests.RequestException as exc:
            log.warning("12punto RSS alınamadı (%s): %s", url, exc)
            return []
        if not looks_like_xml(text):
            log.info("12punto RSS XML değil, atlandı: %s", url)
            return []
        found = self.parse_rss(text, category_hint=category)
        log.info("12punto RSS %s: %d haber bağlantısı", category or "genel", len(found))
        return found

    def _fetch_listing(self, client: HttpClient, category: str) -> list[DiscoveredLink]:
        url = f"{self.base_url}/{category}"
        try:
            html = client.get_text(url)
        except HttpError as exc:
            level = logging.INFO if exc.status == 404 else logging.WARNING
            log.log(level, "12punto liste sayfası atlandı (%s): %s", url, exc)
            return []
        except requests.RequestException as exc:
            log.warning("12punto liste sayfası alınamadı (%s): %s", url, exc)
            return []
        found = self.parse_listing(html, category_hint=category)
        log.info("12punto liste %s: %d haber bağlantısı", category, len(found))
        return found

    def archive_url(self, day: date | str, page: int = 1) -> str:
        """Bir günün arşiv arama URL'si. ``day`` ISO metin (``2026-10-02``) ya da ``date`` olabilir; site
        yalnızca ``GG/AA/YYYY`` biçimini tanıdığından tarih bu biçime çevrilip URL içinde kodlanır."""
        if isinstance(day, str):
            day = date.fromisoformat(day)
        stamp = quote(day.strftime("%d/%m/%Y"), safe="")
        return f"{self.base_url}/Arama/Ara?search=&StartDate={stamp}&EndDate={stamp}&sayfa={max(1, int(page))}"

    def _fetch_archive_day(self, client: HttpClient, day: date) -> list[DiscoveredLink] | None:
        """Bir günün tüm arşiv sayfalarını çeker. HTTP hatasında (ilk sayfada) None, sonuç yoksa boş liste."""
        links: list[DiscoveredLink] = []
        page_count = 1
        page = 1
        while page <= min(page_count, ARCHIVE_MAX_PAGES):
            url = self.archive_url(day, page)
            try:
                html = client.get_text(url)
            except (HttpError, requests.RequestException) as exc:
                log.warning("12punto arşiv araması alınamadı (%s, sayfa %d): %s", day.isoformat(), page, exc)
                if page == 1:
                    return None
                break
            parsed = self.parse_search_page(html)
            if parsed.empty:
                break
            links.extend(parsed.links)
            if page == 1:
                page_count = parsed.page_count
            page += 1
        found = dedupe_links(links)
        log.info("12punto arşiv %s: %d haber bağlantısı (%d sayfa)", day.isoformat(), len(found), page - 1)
        return found

    def parse_rss(self, xml_text: str | bytes, category_hint: str = "") -> list[DiscoveredLink]:
        root = xml_root(xml_text)
        if root is None:
            log.warning("12punto RSS ayrıştırılamadı (XML değil)")
            return []
        links: list[DiscoveredLink] = []
        for item in rss_items(root):
            url = self.normalize_url(xml_child_text(item, "link") or xml_child_text(item, "guid"))
            if not url:
                continue
            summary = xml_child_text(item, "description")
            links.append(
                DiscoveredLink(
                    url=url,
                    title_hint=clean_text(xml_child_text(item, "title")),
                    published_hint=parse_tr_date(xml_child_text(item, "pubDate")),
                    category_hint=category_hint or self._category_from_path(url),
                    rss_summary=clean_text(html_to_text(summary)) if "<" in summary else clean_text(summary),
                    image_hint=xml_child_attr(item, "enclosure", "url") or xml_child_attr(item, "content", "url"),
                    origin="rss",
                )
            )
        return dedupe_links(links)

    def _links_from_anchors(self, root: str | Tag, *, category_hint: str, origin: str) -> list[DiscoveredLink]:
        """``root`` (HTML metni ya da bir kapsayıcı etiket) altındaki haber bağlantılarını toplar."""
        container = root if isinstance(root, Tag) else make_soup(root)
        links: list[DiscoveredLink] = []
        for anchor in container.select("a[href]"):
            url = self.normalize_url(anchor.get("href"))
            if not url:
                continue
            links.append(
                DiscoveredLink(
                    url=url,
                    title_hint=clean_text(anchor.get("title") or anchor.get_text(" ")),
                    published_hint=parse_tr_date(anchor.get("data-yayintarihi2") or anchor.get("data-yayintarihi")),
                    category_hint=category_hint or self._category_from_path(url),
                    origin=origin,
                )
            )
        return dedupe_links(links)

    def parse_listing(self, html: str, category_hint: str = "") -> list[DiscoveredLink]:
        """Kategori sayfasındaki haber bağlantıları (``a.category-item`` dahil sayfadaki tüm haber URL'leri)."""
        return self._links_from_anchors(html, category_hint=category_hint, origin="listing")

    def parse_search_page(self, html: str) -> ArchivePage:
        """Arşiv arama sayfasını çözer: yalnızca sonuç kapsayıcısındaki haber bağlantıları (kenar çubuğu/menü
        bağlantıları hariç) ve sayfalama bağlantılarından okunan toplam sayfa sayısı.

        Kapsayıcı yoksa ya da çözülmüş metni "Sonuç bulunamadı" içeriyorsa (sayfa ``ç``'yi ``&#231;`` olarak
        kodlar; bu yüzden ham HTML değil metin denetlenir) boş sonuç döner.
        """
        soup = make_soup(html)
        container = soup.select_one(ARCHIVE_RESULTS_SELECTOR)
        if container is None:
            log.info("12punto arşiv sayfasında sonuç kapsayıcısı (%s) yok", ARCHIVE_RESULTS_SELECTOR)
            return ArchivePage()
        if _NO_RESULTS_RE.search(tr_lower(container.get_text(" "))):
            return ArchivePage()
        links = self._links_from_anchors(container, category_hint="", origin="archive")
        page_count = 1
        for anchor in soup.select(ARCHIVE_PAGINATION_SELECTOR):
            match = _PAGE_PARAM_RE.search(str(anchor.get("href") or ""))
            if match:
                page_count = max(page_count, int(match.group(1)))
        return ArchivePage(links=links, page_count=page_count)

    def parse_search(self, html: str) -> list[DiscoveredLink]:
        """Arşiv arama sonucu sayfasındaki haber bağlantıları (yalnızca sonuç kapsayıcısından)."""
        return self.parse_search_page(html).links

    # --- haber ---
    def fetch_article(self, client: HttpClient, link: DiscoveredLink) -> NewsRecord | None:
        if self.is_excluded(link.url):
            return None  # önceki turlardan bekleyen listede kalmış dışlanan bölüm haberi: çekilmez, listeden düşer
        html = client.get_text(link.url)
        return self.parse_article(html, link)

    def parse_article(self, html: str, link: DiscoveredLink | None = None, url: str = "") -> NewsRecord | None:
        """Haber sayfasını ``NewsRecord``'a çevirir (JSON-LD listesi + ``section.details``)."""
        link = link or DiscoveredLink(url=url)
        if not link.url:
            raise ValueError("parse_article için link veya url gerekli")
        soup = make_soup(html)
        ld = parse_jsonld_newsarticle(soup) or {}
        meta = extract_meta(soup)

        title = first_nonempty(ld.get("headline"), text_of(soup.select_one("h1")), meta.get("og:title"), link.title_hint)
        subtitle = first_nonempty(
            ld.get("description"), meta.get("description"), meta.get("og:description"), link.rss_summary
        )
        date_spans = soup.select("div.date span")
        date_text = tr_lower(" | ".join(text_of(span) for span in date_spans) if date_spans else soup.get_text(" "))
        published_match = _PUBLISHED_RE.search(date_text)
        updated_match = _UPDATED_RE.search(date_text)
        published_at = (
            parse_tr_date(ld.get("datePublished"))
            or parse_tr_date(published_match.group(1) if published_match else None)
            or link.published_hint
            or parse_tr_date(meta.get("datepublished"))
        )
        updated_at = (
            parse_tr_date(ld.get("dateModified"))
            or parse_tr_date(updated_match.group(1) if updated_match else None)
            or parse_tr_date(meta.get("datemodified"))
        )
        content = self._choose_content(self._content_from_html(soup), paragraphize_flat_text(ld.get("articleBody")))
        if not content and title and subtitle:
            # Galeri/video sayfalarında gövde metni yok; açıklama haberin tek metnidir (anahtar kelime de orada olabilir).
            log.info("12punto haberinde gövde yok, açıklama içerik olarak kullanılıyor: %s", link.url)
            content = subtitle
        elif subtitle and content.startswith(subtitle) and len(content) > len(subtitle) + 50:
            content = content[len(subtitle) :].lstrip()  # gövde spotu tekrar ediyorsa ikinci kopyayı at
        if not title or not content:
            log.warning("12punto haberi eksik (başlık=%s, içerik=%d karakter): %s", bool(title), len(content), link.url)
            return None

        category = first_nonempty(ld.get("articleSection"), link.category_hint, self._category_from_path(link.url))
        author = first_nonempty(jsonld_name(ld.get("author")), meta.get("author"), meta.get("article:author"))
        tags = jsonld_keywords(ld.get("keywords")) or split_keywords(meta.get("keywords"))
        image_url = first_nonempty(
            meta.get("og:image"), meta.get("og:image:url"), jsonld_url(ld.get("image")), link.image_hint
        )
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

    @staticmethod
    def _content_from_html(soup) -> str:
        # 2026 yeniden tasarımı: div.punto-article-body__content; eski düzen: section.details
        body = soup.select_one("div.punto-article-body__content") or soup.select_one("section.details")
        if body is None:
            return ""
        paragraphs = extract_paragraphs(body, drop_selectors=_DROP_SELECTORS, drop_patterns=_DROP_PARAGRAPH_PATTERNS)
        return "\n\n".join(paragraphs)

    @staticmethod
    def _choose_content(html_text: str, body_text: str) -> str:
        """HTML paragrafları (apostrof/başlık/paragraf yapısı korunmuş) tercih edilir; HTML çıkarımı
        JSON-LD gövdesinden belirgin biçimde kısa kaldıysa (eksik ayrıştırma) JSON-LD gövdesi kullanılır."""
        if html_text and len(html_text) >= 0.7 * len(body_text):
            return html_text
        return body_text or html_text
