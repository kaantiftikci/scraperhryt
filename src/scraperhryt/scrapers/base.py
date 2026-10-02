"""Kaynak kazıyıcılarının ortak altyapısı.

- ``HttpClient``: requests tabanlı, kibar (host başına bekleme), yeniden deneyen HTTP istemcisi.
- ``DiscoveredLink``: keşif aşamasında bulunan haber bağlantısı ve RSS'ten gelen ipuçları.
- ``Source``: her kaynağın uyması gereken protokol (``discover`` + ``fetch_article``).
- Ayrıştırma yardımcıları: JSON-LD, meta etiketleri, Türkçe/RFC/ISO tarih çözümleme, metin temizliği.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, Tag
from lxml import etree
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings
from ..models import NewsRecord, article_id_for, canonical_url
from ..textutil import normalize_ws, tr_lower

log = logging.getLogger(__name__)

ISTANBUL = ZoneInfo("Europe/Istanbul")

# ---------------------------------------------------------------------------------------------------------
# Metin temizliği
# ---------------------------------------------------------------------------------------------------------

_INVISIBLE_RE = re.compile("[�​‌‍﻿­]")


def clean_text(text: str | None) -> str:
    """Görünmez/bozuk karakterleri (U+FFFD, sıfır genişlikli boşluklar) atar ve boşlukları normalize eder."""
    if not text:
        return ""
    return normalize_ws(_INVISIBLE_RE.sub("", str(text)))


def first_nonempty(*values: str | None) -> str:
    """Verilen adaylardan ilk boş olmayan (temizlenmiş) metni döndürür."""
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?:…»”\"'’])(?=[A-ZÇĞİÖŞÜ0-9“\"«(])")


def paragraphize_flat_text(text: str | None) -> str:
    """Paragraf sınırları kaybolmuş düz metni (JSON-LD articleBody gibi) cümle sonlarından böler."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return normalize_ws(_SENTENCE_BOUNDARY_RE.sub("\n", cleaned))


# ---------------------------------------------------------------------------------------------------------
# Karakter kodlaması
# ---------------------------------------------------------------------------------------------------------

_CHARSET_RE = re.compile(r"charset=['\"]?([\w.:-]+)", re.I)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset=['\"]?([\w.:-]+)", re.I)
_XML_ENCODING_RE = re.compile(rb"^\s*<\?xml[^>]*encoding=['\"]([\w.:-]+)", re.I)


def decode_bytes(raw: bytes, declared: str | None = None) -> str:
    """Ham gövdeyi metne çevirir.

    Sıra: HTTP başlığındaki charset → belge içi (xml/meta) bildirim → UTF-8. Hiçbiri hatasız çözülemezse
    UTF-8 ile toleranslı çözülür ve bozuk baytlar (U+FFFD) atılır; sayfalar ara sıra kesik baytlar içerir.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    candidates: list[str] = []
    if declared:
        candidates.append(declared)
    match = _XML_ENCODING_RE.match(raw[:256]) or _META_CHARSET_RE.search(raw[:4096])
    if match:
        candidates.append(match.group(1).decode("ascii", "ignore"))
    candidates.append("utf-8")
    for encoding in candidates:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace").replace("�", "")


# ---------------------------------------------------------------------------------------------------------
# HTTP istemcisi
# ---------------------------------------------------------------------------------------------------------


class HttpError(RuntimeError):
    """HTTP 4xx/5xx yanıtı. 5xx ve 429 yeniden denenebilir; diğer 4xx'ler kalıcı hatadır."""

    def __init__(self, url: str, status: int, reason: str = "") -> None:
        super().__init__(f"HTTP {status} {reason}".rstrip() + f" ← {url}")
        self.url = url
        self.status = status
        self.reason = reason

    @property
    def retryable(self) -> bool:
        return self.status >= 500 or self.status == 429


_RETRYABLE_REQUEST_ERRORS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        return exc.retryable
    return isinstance(exc, _RETRYABLE_REQUEST_ERRORS)


class HttpClient:
    """Tek bir ``requests.Session`` üzerinden kibar ve dayanıklı GET istekleri.

    - Host başına ``settings.request_delay_seconds`` bekleme (thread güvenli).
    - 5xx/429 ve bağlantı/zaman aşımı hatalarında üstel geri çekilmeli yeniden deneme (tenacity).
    - ``get_text`` gövdeyi doğru karakter kodlamasıyla çözer.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session: requests.Session | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.5",
            }
        )
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    # --- kibarlık ---
    def _throttle(self, host: str) -> None:
        delay = float(self.settings.request_delay_seconds)
        if delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last = self._last_request.get(host)
            wait = 0.0 if last is None else max(0.0, last + delay - now)
            self._last_request[host] = now + wait
        if wait > 0:
            time.sleep(wait)

    # --- istek ---
    def _request(self, url: str, *, params: dict[str, Any] | None, timeout: float | None) -> requests.Response:
        host = requests.utils.urlparse(url).netloc.lower()
        self._throttle(host)
        read_timeout = float(timeout or self.settings.request_timeout)
        response = self.session.get(
            url, params=params, timeout=(min(10.0, read_timeout), read_timeout), allow_redirects=True
        )
        if response.status_code >= 400:
            raise HttpError(response.url or url, response.status_code, response.reason or "")
        return response

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome is not None else None
        sleep = state.next_action.sleep if state.next_action is not None else 0.0
        log.warning(
            "HTTP isteği başarısız (deneme %d/%d): %s; %.1fs sonra yeniden denenecek",
            state.attempt_number,
            self.max_attempts,
            exc,
            sleep,
        )

    def get(self, url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> requests.Response:
        """GET isteği; 4xx'te ``HttpError``, tükenen denemelerde son istisna fırlatılır."""
        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.backoff_seconds, min=self.backoff_seconds, max=20.0),
            retry=retry_if_exception(_is_retryable),
            before_sleep=self._log_retry,
            reraise=True,
        )
        return retrying(self._request, url, params=params, timeout=timeout)

    def get_text(self, url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        response = self.get(url, params=params, timeout=timeout)
        match = _CHARSET_RE.search(response.headers.get("content-type", ""))
        return decode_bytes(response.content, match.group(1) if match else None)

    def close(self) -> None:
        self.session.close()


# ---------------------------------------------------------------------------------------------------------
# Keşif bağlantısı ve kaynak protokolü
# ---------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class DiscoveredLink:
    """Keşif aşamasında bulunan haber bağlantısı. RSS'ten gelen alanlar sayfa alınamazsa yedek olarak kullanılır."""

    url: str
    title_hint: str = ""
    published_hint: datetime | None = None
    category_hint: str = ""
    rss_html: str = ""  # RSS'teki tam HTML gövde (Hürriyet <text>)
    rss_summary: str = ""  # RSS özeti / <abstract>
    updated_hint: datetime | None = None
    image_hint: str = ""
    origin: str = ""  # "rss" | "listing" | "archive"

    @property
    def canonical(self) -> str:
        return canonical_url(self.url)

    @property
    def id(self) -> str:
        return article_id_for(self.url)

    def merge(self, other: DiscoveredLink) -> DiscoveredLink:
        """Aynı haberin ikinci kez bulunmasında boş alanları diğer kayıttan doldurur (ilk URL korunur)."""
        return DiscoveredLink(
            url=self.url,
            title_hint=self.title_hint or other.title_hint,
            published_hint=self.published_hint or other.published_hint,
            category_hint=self.category_hint or other.category_hint,
            rss_html=self.rss_html or other.rss_html,
            rss_summary=self.rss_summary or other.rss_summary,
            updated_hint=self.updated_hint or other.updated_hint,
            image_hint=self.image_hint or other.image_hint,
            origin=self.origin or other.origin,
        )


def dedupe_links(links: Iterable[DiscoveredLink]) -> list[DiscoveredLink]:
    """Kanonik URL'ye göre tekilleştirir; tarihi bilinenler en yeniden eskiye, tarihsizler keşif sırasıyla sona."""
    merged: dict[str, DiscoveredLink] = {}
    order: list[str] = []
    for link in links:
        key = link.canonical
        if not key:
            continue
        if key in merged:
            merged[key] = merged[key].merge(link)
        else:
            merged[key] = link
            order.append(key)
    unique = [merged[key] for key in order]
    dated = [link for link in unique if link.published_hint is not None]
    dated.sort(key=lambda link: link.published_hint, reverse=True)  # type: ignore[arg-type, return-value]
    undated = [link for link in unique if link.published_hint is None]
    return dated + undated


@runtime_checkable
class Source(Protocol):
    """Bir haber kaynağı: bağlantı keşfi ve tek haber çekme."""

    name: str

    def discover(
        self, client: HttpClient, *, backfill_days: int = 0, limit: int | None = None
    ) -> list[DiscoveredLink]: ...

    def fetch_article(self, client: HttpClient, link: DiscoveredLink) -> NewsRecord | None: ...


# ---------------------------------------------------------------------------------------------------------
# HTML / XML yardımcıları
# ---------------------------------------------------------------------------------------------------------


def make_soup(html: str | BeautifulSoup) -> BeautifulSoup:
    return html if isinstance(html, BeautifulSoup) else BeautifulSoup(html or "", "lxml")


def text_of(element: Tag | None) -> str:
    return clean_text(element.get_text(" ")) if element is not None else ""


def extract_meta(html: str | BeautifulSoup) -> dict[str, str]:
    """``<meta property|name|itemprop=... content=...>`` etiketlerini sözlüğe çevirir (anahtarlar küçük harf,
    ilk görülen değer korunur). ``<link rel=canonical>`` varsa ``canonical`` anahtarıyla eklenir."""
    soup = make_soup(html)
    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        content = tag.get("content")
        if content is None:
            continue
        for attr in ("property", "name", "itemprop"):
            key = tag.get(attr)
            if isinstance(key, str) and key.strip():
                meta.setdefault(key.strip().lower(), clean_text(content))
    canonical = soup.find("link", rel="canonical")
    if canonical is not None and canonical.get("href"):
        meta.setdefault("canonical", str(canonical["href"]).strip())
    return meta


_ARTICLE_TYPES = ("NewsArticle", "ReportageNewsArticle", "Article", "BlogPosting")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _load_jsonld(text: str) -> Any | None:
    text = text.strip()
    if text.startswith("<!--"):
        text = text[4:]
    if text.endswith("-->"):
        text = text[:-3]
    text = text.replace("<![CDATA[", "").replace("]]>", "").strip()
    if not text:
        return None
    for candidate in (text, _TRAILING_COMMA_RE.sub(r"\1", text)):
        try:
            return json.loads(candidate, strict=False)
        except ValueError:
            continue
    return None


def _iter_jsonld_nodes(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_nodes(item)
    elif isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict):
                    yield item


def parse_jsonld_blocks(html: str | BeautifulSoup) -> list[Any]:
    """Sayfadaki tüm ``application/ld+json`` bloklarını çözümler (bozuk bloklar atlanır)."""
    soup = make_soup(html)
    blocks: list[Any] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        data = _load_jsonld(script.string or script.get_text() or "")
        if data is not None:
            blocks.append(data)
    return blocks


def parse_jsonld_newsarticle(html: str | BeautifulSoup) -> dict[str, Any] | None:
    """Sayfadaki ilk ``NewsArticle`` JSON-LD düğümünü döndürür (dict, liste ve ``@graph`` biçimlerini destekler).

    ``NewsArticle`` bulunamazsa ``Article``/``BlogPosting`` türleri yedek olarak kabul edilir.
    """
    fallback: dict[str, Any] | None = None
    for block in parse_jsonld_blocks(html):
        for node in _iter_jsonld_nodes(block):
            raw_type = node.get("@type")
            types = [str(t) for t in raw_type] if isinstance(raw_type, list) else [str(raw_type or "")]
            if "NewsArticle" in types or "ReportageNewsArticle" in types:
                return node
            if fallback is None and any(t in _ARTICLE_TYPES for t in types):
                fallback = node
    return fallback


def jsonld_name(value: Any) -> str:
    """``author`` gibi alanlardan isim çıkarır: str, {"name": ...} veya bunların listesi."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return clean_text(value.get("name") or value.get("alternateName") or "")
    if isinstance(value, list):
        names = [jsonld_name(item) for item in value]
        return ", ".join(name for name in names if name)
    return ""


def jsonld_url(value: Any) -> str:
    """``image`` gibi alanlardan URL çıkarır: str, {"url": ...}, {"contentUrl": ...} veya liste."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("url") or value.get("contentUrl") or value.get("@id") or "").strip()
    if isinstance(value, list):
        for item in value:
            url = jsonld_url(item)
            if url:
                return url
    return ""


def jsonld_keywords(value: Any) -> list[str]:
    """``keywords`` alanını listeye çevirir (liste ya da virgülle ayrılmış metin)."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        return []
    seen: list[str] = []
    for item in items:
        cleaned = clean_text(item)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def split_keywords(value: str | None) -> list[str]:
    return jsonld_keywords(value or "")


_BLOCK_TAGS = ("p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote")


def extract_paragraphs(
    container: Tag,
    *,
    drop_selectors: Sequence[str] = (),
    drop_patterns: Sequence[re.Pattern[str]] = (),
) -> list[str]:
    """Bir kapsayıcıdaki paragraf/başlık/liste metinlerini sırayla toplar.

    ``drop_selectors`` ile eşleşen alt ağaçlar (reklam, promosyon) atılır; ``drop_patterns`` ile eşleşen
    paragraflar (kalıp metinler) elenir; ardışık/aynı paragraflar tekrar eklenmez. Blok etiketi yoksa
    kapsayıcının satır satır düz metnine düşülür.
    """
    for selector in drop_selectors:
        for element in container.select(selector):
            element.decompose()
    paragraphs: list[str] = []
    for element in container.find_all(_BLOCK_TAGS):
        if element.find(_BLOCK_TAGS) is not None:
            continue  # iç içe blok: alt öğeler ayrıca gezilecek
        text = clean_text(element.get_text(" "))
        if not text or any(pattern.search(text) for pattern in drop_patterns):
            continue
        if text not in paragraphs:
            paragraphs.append(text)
    if not paragraphs:
        for line in clean_text(container.get_text("\n")).split("\n"):
            line = line.strip()
            if line and not any(pattern.search(line) for pattern in drop_patterns) and line not in paragraphs:
                paragraphs.append(line)
    return paragraphs


def looks_like_xml(text: str | bytes) -> bool:
    head = (text[:512].decode("utf-8", "ignore") if isinstance(text, bytes) else text[:512]).lstrip().lower()
    return head.startswith("<?xml") or "<rss" in head or "<feed" in head or "<rdf:rdf" in head


def xml_root(text: str | bytes) -> etree._Element | None:
    """RSS/Atom metnini (toleranslı) lxml ağacına çevirir; bozuksa None."""
    raw = text.encode("utf-8") if isinstance(text, str) else text
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    if isinstance(text, str):
        # Metin zaten çözülmüş: XML bildirimindeki farklı bir encoding lxml'i yanıltmasın.
        raw = _XML_ENCODING_RE.sub(lambda m: m.group(0).replace(m.group(1), b"utf-8"), raw, count=1)
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(raw, parser=parser)
    except etree.XMLSyntaxError:
        return None
    return root


def _localname(element: Any) -> str:
    tag = getattr(element, "tag", None)
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname


def rss_items(root: etree._Element) -> list[etree._Element]:
    """RSS ``<item>`` ve Atom ``<entry>`` öğelerini belge sırasıyla döndürür."""
    return [element for element in root.iter() if _localname(element) in ("item", "entry")]


def xml_child_text(item: etree._Element, localname: str) -> str:
    """Ad alanından bağımsız olarak ilk ``<localname>`` alt öğesinin metnini döndürür (CDATA dahil)."""
    for child in item:
        if _localname(child) == localname:
            return (child.text or "").strip()
    return ""


def xml_child_attr(item: etree._Element, localname: str, attr: str) -> str:
    """İlk ``<localname attr=...>`` alt öğesinin (ya da onun altındakilerin) özniteliğini döndürür."""
    for element in item.iter():
        if _localname(element) == localname and element.get(attr):
            return str(element.get(attr)).strip()
    return ""


# ---------------------------------------------------------------------------------------------------------
# Tarih çözümleme
# ---------------------------------------------------------------------------------------------------------

_TR_MONTHS = {
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "mayis": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9, "ekim": 10, "kasım": 11, "kasim": 11,
    "aralık": 12, "aralik": 12,
}
_EN_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
    "nov": 11, "dec": 12,
}
_TZ_PART = r"(?:\s*(?P<tz>Z|UTC|GMT|[+-]\d{2}:?\d{2}))?"
_TIME_PART = r"(?:[T\s,]+(?P<H>\d{1,2})[:.](?P<M>\d{2})(?::(?P<S>\d{2})(?:[.,]\d+)?)?)?"
_ISO_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})" + _TIME_PART + _TZ_PART)
_TR_NUMERIC_RE = re.compile(r"(?P<d>\d{1,2})[./](?P<m>\d{1,2})[./](?P<y>\d{4})" + _TIME_PART + _TZ_PART)
_RFC_RE = re.compile(
    r"(?:[a-z]{3},?\s+)?(?P<d>\d{1,2})\s+(?P<mon>[a-z]{3})[a-z]*\.?\s+(?P<y>\d{4})" + _TIME_PART + _TZ_PART
)
_TR_MONTH_FIRST_RE = re.compile(r"(?P<mon>[a-zçğıöşü]{3,8})\s+(?P<d>\d{1,2}),?\s+(?P<y>\d{4})" + _TIME_PART)
_TR_DAY_FIRST_RE = re.compile(r"(?P<d>\d{1,2})\s+(?P<mon>[a-zçğıöşü]{3,8})\s+(?P<y>\d{4})" + _TIME_PART)


def _tzinfo(token: str | None) -> timezone | ZoneInfo:
    if not token:
        return ISTANBUL
    token = token.upper()
    if token in ("Z", "UTC", "GMT"):
        return timezone.utc
    sign = -1 if token[0] == "-" else 1
    digits = token[1:].replace(":", "")
    offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:4]))
    return timezone(sign * offset)


def _build(groups: dict[str, str | None], month: int) -> datetime | None:
    try:
        return datetime(
            int(groups["y"] or 0),
            month,
            int(groups["d"] or 0),
            int(groups.get("H") or 0),
            int(groups.get("M") or 0),
            int(groups.get("S") or 0),
            tzinfo=_tzinfo(groups.get("tz")),
        )
    except ValueError:
        return None


def parse_tr_date(text: str | None) -> datetime | None:
    """Haber sitelerinde görülen tarih biçimlerini saat dilimli ``datetime``'a çevirir.

    Desteklenenler: ISO 8601 (``2026-10-02T22:20:00+03:00``, ``...Z``, yalnız tarih), RFC 822
    (``Fri, 02 Oct 2026 19:24:54 Z`` / ``+0300``), ``2.10.2026 19:24:54 +00:00``, ``02.10.2026 22:41``,
    ``Ekim 02, 2026 22:34``, ``2 Ekim 2026 22:34``. Saat dilimi yoksa Europe/Istanbul varsayılır.
    """
    if not text:
        return None
    cleaned = clean_text(text)
    if not cleaned:
        return None
    lowered = tr_lower(cleaned)

    for pattern in (_ISO_RE, _TR_NUMERIC_RE):
        match = pattern.search(lowered)
        if match:
            result = _build(match.groupdict(), int(match.group("m")))
            if result is not None:
                return result

    match = _RFC_RE.search(lowered)
    if match and match.group("mon") in _EN_MONTHS:
        result = _build(match.groupdict(), _EN_MONTHS[match.group("mon")])
        if result is not None:
            return result

    for pattern in (_TR_MONTH_FIRST_RE, _TR_DAY_FIRST_RE):
        for match in pattern.finditer(lowered):
            month = _TR_MONTHS.get(match.group("mon"))
            if month is None:
                continue
            result = _build(match.groupdict(), month)
            if result is not None:
                return result
    return None


def ensure_aware(value: datetime | None) -> datetime | None:
    """Saat dilimsiz bir datetime'ı Europe/Istanbul kabul eder; aware olanı aynen döndürür."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=ISTANBUL)
