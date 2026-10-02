"""Kazıyıcı katmanının çevrimdışı testleri (tests/fixtures altındaki gerçek sayfa kopyalarıyla)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import requests
from lxml import etree

from scraperhryt.broker import InMemoryBroker, Queue, RoutingKey
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord, Stage
from scraperhryt.scrapers import (
    DiscoveredLink,
    HttpClient,
    HttpError,
    HurriyetSource,
    PuntoSource,
    ScrapeRunner,
    ScrapeStats,
    SeenStore,
    Source,
    build_sources,
    extract_meta,
    parse_jsonld_newsarticle,
    parse_tr_date,
)
from scraperhryt.scrapers.base import ISTANBUL, decode_bytes, dedupe_links, paragraphize_flat_text

FIXTURES = Path(__file__).parent / "fixtures"
HURRIYET_ARTICLE_URL = (
    "https://www.hurriyet.com.tr/gundem/son-dakika-fon-sorusturmasinda-20-kisiye-tutuklama-talebi-43327782"
)
PUNTO_ARTICLE_URL = (
    "https://12punto.com.tr/kulis/tip-milletvekili-ahmet-siktan-fon-vurgunu-sorusturmasina-iliskin-"
    "yeni-iddialar-akp-icinden-kulis-bilgileri-paylasti-154231"
)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_text(name: str) -> str:
    return decode_bytes(fixture_bytes(name))


def trimmed_rss(name: str, keep: Callable[[str], bool], count: int) -> bytes:
    """RSS fixture'ından ``keep`` koşulunu sağlayan ilk ``count`` öğeyi tutar (çalıştırıcı testleri için küçük besleme)."""
    root = etree.fromstring(fixture_bytes(name), parser=etree.XMLParser(recover=True, huge_tree=True))
    kept = 0
    for item in list(root.iter("item")):
        link = (item.findtext("link") or "").strip()
        if kept < count and keep(link):
            kept += 1
            continue
        item.getparent().remove(item)
    return etree.tostring(root, xml_declaration=True, encoding="utf-8")


class FakeHttpClient:
    """URL → fixture içeriği sunan çevrimdışı istemci; bilinmeyen URL'ler için HTTP 404 fırlatır."""

    def __init__(
        self,
        routes: dict[str, bytes | str | Exception] | None = None,
        fallback: Callable[[str], bytes | str | Exception | None] | None = None,
    ) -> None:
        self.routes = dict(routes or {})
        self.fallback = fallback
        self.calls: list[str] = []

    def get_text(self, url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None) -> str:
        self.calls.append(url)
        value = self.routes.get(url)
        if value is None and self.fallback is not None:
            value = self.fallback(url)
        if value is None:
            raise HttpError(url, 404, "Not Found")
        if isinstance(value, Exception):
            raise value
        return decode_bytes(value) if isinstance(value, bytes) else value

    def close(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        state_db_path=str(tmp_path / "state" / "state.sqlite3"),
        request_delay_seconds=0,
        sources="hurriyet,12punto",
        backfill_days=0,
    )


@pytest.fixture
def hurriyet(settings: Settings) -> HurriyetSource:
    return HurriyetSource(settings)


@pytest.fixture
def punto(settings: Settings) -> PuntoSource:
    return PuntoSource(settings)


def hurriyet_routes(settings: Settings) -> dict[str, bytes | str | Exception]:
    return {
        settings.hurriyet_gundem_rss: fixture_bytes("hurriyet_gundem_rss.xml"),
        settings.hurriyet_gundem_listing: fixture_bytes("hurriyet_gundem_listing.html"),
        HURRIYET_ARTICLE_URL: fixture_bytes("hurriyet_article.html"),
    }


def punto_routes(settings: Settings) -> dict[str, bytes | str | Exception]:
    base = settings.punto_base_url
    return {
        f"{base}/rss": fixture_bytes("punto_rss.xml"),
        f"{base}/rss/gundem": fixture_bytes("punto_rss.xml"),
        f"{base}/rss/siyaset": b"<html><body>404</body></html>",
        f"{base}/gundem": fixture_bytes("punto_gundem_listing.html"),
        PUNTO_ARTICLE_URL: fixture_bytes("punto_article.html"),
    }


# ---------------------------------------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-02T22:20:00+03:00", "2026-10-02T22:20:00+03:00"),
        ("2026-10-02T19:20:00Z", "2026-10-02T19:20:00+00:00"),
        ("2026-10-02T22:41:28.123+0300", "2026-10-02T22:41:28+03:00"),
        ("Fri, 02 Oct 2026 19:24:54 Z", "2026-10-02T19:24:54+00:00"),
        ("Fri, 02 Oct 2026 22:41:28 +0300", "2026-10-02T22:41:28+03:00"),
        ("2.10.2026 19:24:54 +00:00", "2026-10-02T19:24:54+00:00"),
        ("02.10.2026 22:41", "2026-10-02T22:41:00+03:00"),
        ("Ekim 02, 2026 22:34", "2026-10-02T22:34:00+03:00"),
        ("Oluşturulma Tarihi: Ekim 02, 2026 22:20", "2026-10-02T22:20:00+03:00"),
        ("Yayınlanma:  02.10.2026 22:41", "2026-10-02T22:41:00+03:00"),
        ("2 Ekim 2026 22:34", "2026-10-02T22:34:00+03:00"),
        ("15 Ağustos 2026", "2026-08-15T00:00:00+03:00"),
        ("2026-10-02", "2026-10-02T00:00:00+03:00"),
    ],
)
def test_parse_tr_date(text: str, expected: str) -> None:
    parsed = parse_tr_date(text)
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.isoformat() == expected


@pytest.mark.parametrize("text", ["", None, "garbage", "32.13.2026 10:00", "Ekim 2026"])
def test_parse_tr_date_rejects_invalid(text: str | None) -> None:
    assert parse_tr_date(text) is None


def test_parse_tr_date_same_instant_across_formats() -> None:
    rss = parse_tr_date("Fri, 02 Oct 2026 19:20:00 Z")
    page = parse_tr_date("2026-10-02T22:20:00+03:00")
    assert rss == page
    assert page is not None and page.astimezone(ISTANBUL).hour == 22


def test_parse_jsonld_handles_dict_list_and_graph() -> None:
    as_dict = '<script type="application/ld+json">{"@type": "NewsArticle", "headline": "A"}</script>'
    as_list = (
        '<script type="application/ld+json">[{"@type": "Organization"}, '
        '{"@type": "NewsArticle", "headline": "B"}]</script>'
    )
    as_graph = (
        '<script type="application/ld+json">{"@graph": [{"@type": "WebPage"}, '
        '{"@type": ["NewsArticle", "Article"], "headline": "C"}]}</script>'
    )
    broken_then_ok = (
        '<script type="application/ld+json">{not json</script>'
        '<script type="application/ld+json">{"@type": "Article", "headline": "D",}</script>'
    )
    assert parse_jsonld_newsarticle(as_dict)["headline"] == "A"
    assert parse_jsonld_newsarticle(as_list)["headline"] == "B"
    assert parse_jsonld_newsarticle(as_graph)["headline"] == "C"
    assert parse_jsonld_newsarticle(broken_then_ok)["headline"] == "D"
    assert parse_jsonld_newsarticle("<html><body>yok</body></html>") is None


def test_extract_meta_lowercases_keys_and_keeps_first() -> None:
    meta = extract_meta(fixture_text("hurriyet_article.html"))
    assert meta["og:image"].startswith("https://image.hurimg.com/")
    assert meta["datepublished"] == "2026-10-02T22:20:52+03:00"
    assert meta["article:section"] == "gundem"
    assert "canonical" in meta or "og:url" in meta


def test_decode_bytes_drops_broken_bytes_and_honours_charset() -> None:
    text = decode_bytes(fixture_bytes("hurriyet_article.html"))
    assert "�" not in text
    assert "SORUŞTURMANIN DETAYLARI" in text
    assert decode_bytes("şğü".encode("iso-8859-9"), "iso-8859-9") == "şğü"
    assert decode_bytes(b"\xef\xbb\xbf<p>ok</p>") == "<p>ok</p>"


def test_paragraphize_flat_text_restores_boundaries() -> None:
    text = paragraphize_flat_text("İlk cümle bitti.İkinci cümle başladı. Üçüncü cümle.")
    assert text == "İlk cümle bitti.\nİkinci cümle başladı. Üçüncü cümle."


def test_dedupe_links_merges_hints_and_sorts_newest_first() -> None:
    older = datetime(2026, 10, 1, 10, 0, tzinfo=ISTANBUL)
    newer = datetime(2026, 10, 2, 10, 0, tzinfo=ISTANBUL)
    links = [
        DiscoveredLink(url="https://12punto.com.tr/gundem/a-1001", title_hint="A"),
        DiscoveredLink(url="https://12punto.com.tr/gundem/b-1002", published_hint=older),
        DiscoveredLink(url="http://www.12punto.com.tr/gundem/a-1001", published_hint=newer, rss_html="<p>x</p>"),
        DiscoveredLink(url="https://12punto.com.tr/gundem/c-1003"),
    ]
    unique = dedupe_links(links)
    assert [link.url for link in unique] == [
        "https://12punto.com.tr/gundem/a-1001",
        "https://12punto.com.tr/gundem/b-1002",
        "https://12punto.com.tr/gundem/c-1003",
    ]
    assert unique[0].title_hint == "A" and unique[0].rss_html == "<p>x</p>" and unique[0].published_hint == newer


# ---------------------------------------------------------------------------------------------------------
# HttpClient
# ---------------------------------------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"", content_type: str = "text/html; charset=utf-8") -> None:
        self.status_code = status
        self.content = body
        self.headers = {"content-type": content_type}
        self.url = "https://example.test/x"
        self.reason = "Err" if status >= 400 else "OK"


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        return None


def test_http_client_retries_transient_errors(settings: Settings) -> None:
    session = _FakeSession(
        [
            _FakeResponse(503),
            requests.ConnectionError("kopuk"),
            _FakeResponse(200, b"merhaba", "text/html; charset=UTF-8"),
        ]
    )
    client = HttpClient(settings, session=session, max_attempts=4, backoff_seconds=0.0)
    assert client.get_text("https://example.test/x") == "merhaba"
    assert session.calls == 3


def test_http_client_does_not_retry_client_errors(settings: Settings) -> None:
    session = _FakeSession([_FakeResponse(404), _FakeResponse(200, b"asla")])
    client = HttpClient(settings, session=session, max_attempts=3, backoff_seconds=0.0)
    with pytest.raises(HttpError) as excinfo:
        client.get_text("https://example.test/x")
    assert excinfo.value.status == 404 and not excinfo.value.retryable
    assert session.calls == 1


def test_http_client_gives_up_after_max_attempts(settings: Settings) -> None:
    session = _FakeSession([_FakeResponse(500), _FakeResponse(502)])
    client = HttpClient(settings, session=session, max_attempts=2, backoff_seconds=0.0)
    with pytest.raises(HttpError) as excinfo:
        client.get("https://example.test/x")
    assert excinfo.value.status == 502
    assert session.calls == 2


def test_http_client_politeness_delay_per_host(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "s.db"), request_delay_seconds=0.2)
    session = _FakeSession([_FakeResponse(200, b"a"), _FakeResponse(200, b"b"), _FakeResponse(200, b"c")])
    client = HttpClient(settings, session=session)
    started = time.monotonic()
    client.get_text("https://a.test/1")
    client.get_text("https://b.test/1")  # farklı host: beklemez
    client.get_text("https://a.test/2")  # aynı host: en az 0.2s bekler
    assert time.monotonic() - started >= 0.15


# ---------------------------------------------------------------------------------------------------------
# Hürriyet
# ---------------------------------------------------------------------------------------------------------


def test_sources_satisfy_protocol(hurriyet: HurriyetSource, punto: PuntoSource) -> None:
    assert isinstance(hurriyet, Source) and isinstance(punto, Source)
    assert hurriyet.name == "hurriyet" and punto.name == "12punto"


def test_hurriyet_rss_yields_100_links_with_full_text(hurriyet: HurriyetSource) -> None:
    links = hurriyet.parse_rss(fixture_text("hurriyet_gundem_rss.xml"))
    assert len(links) == 100
    assert all(link.rss_html for link in links)
    assert all(link.rss_summary for link in links)
    assert all(link.published_hint is not None and link.published_hint.tzinfo is not None for link in links)
    assert all(link.url.startswith("https://www.hurriyet.com.tr/") for link in links)
    dates = [link.published_hint for link in links]
    assert dates == sorted(dates, reverse=True)
    fon = next(link for link in links if link.url == HURRIYET_ARTICLE_URL)
    assert fon.title_hint == "Son dakika : Fon soruşturmasında 20 şüpheliye tutuklama talebi"
    assert fon.published_hint == datetime(2026, 10, 2, 22, 20, tzinfo=ISTANBUL)
    assert fon.updated_hint is not None and fon.image_hint.startswith("https://image.hurimg.com/")
    assert "İstanbul Cumhuriyet Başsavcılığı" in fon.rss_html


def test_hurriyet_listing_yields_30_links(hurriyet: HurriyetSource) -> None:
    links = hurriyet.parse_listing(fixture_text("hurriyet_gundem_listing.html"))
    assert len(links) == 30
    assert all(hurriyet.is_gundem_url(link.url) for link in links)
    assert all(link.title_hint for link in links)
    assert all(link.origin == "listing" for link in links)


def test_hurriyet_url_filter(hurriyet: HurriyetSource) -> None:
    assert hurriyet.normalize_url("/gundem/abc-def-43327782", gundem_only=True) == (
        "https://www.hurriyet.com.tr/gundem/abc-def-43327782"
    )
    assert hurriyet.normalize_url("https://hurriyet.com.tr/gundem/abc-43327782/?utm_source=x", gundem_only=True) == (
        "https://www.hurriyet.com.tr/gundem/abc-43327782"
    )
    assert hurriyet.normalize_url("/gundem/", gundem_only=True) is None
    assert hurriyet.normalize_url("/ekonomi/abc-43327782", gundem_only=True) is None
    assert hurriyet.normalize_url("/ekonomi/abc-43327782") == "https://www.hurriyet.com.tr/ekonomi/abc-43327782"
    assert hurriyet.normalize_url("https://bigpara.hurriyet.com.tr/gundem/abc-43327782") is None
    assert hurriyet.normalize_url("/haberleri/son-dakika") is None


def test_hurriyet_discover_only_gundem_deduplicated(settings: Settings, hurriyet: HurriyetSource) -> None:
    client = FakeHttpClient(hurriyet_routes(settings))
    links = hurriyet.discover(client)
    urls = [link.url for link in links]
    assert len(urls) == len(set(urls))
    assert all(hurriyet.is_gundem_url(url) for url in urls)
    assert 80 <= len(links) <= 110
    assert links[0].published_hint is not None  # tarihli (RSS) bağlantılar önce gelir
    fon = next(link for link in links if link.url == HURRIYET_ARTICLE_URL)
    assert fon.rss_html and fon.title_hint
    assert client.calls == [settings.hurriyet_gundem_rss, settings.hurriyet_gundem_listing]


def test_hurriyet_discover_limit_stops_early(settings: Settings, hurriyet: HurriyetSource) -> None:
    client = FakeHttpClient(hurriyet_routes(settings))
    links = hurriyet.discover(client, limit=5)
    assert len(links) == 5
    assert client.calls == [settings.hurriyet_gundem_rss]


def test_hurriyet_discover_survives_feed_failure(settings: Settings, hurriyet: HurriyetSource) -> None:
    routes = hurriyet_routes(settings)
    routes[settings.hurriyet_gundem_rss] = requests.Timeout("zaman aşımı")
    links = hurriyet.discover(FakeHttpClient(routes))
    assert len(links) == 30


def test_hurriyet_article_parse(hurriyet: HurriyetSource) -> None:
    record = hurriyet.parse_article(fixture_text("hurriyet_article.html"), url=HURRIYET_ARTICLE_URL)
    assert record is not None
    assert record.source == "hurriyet"
    assert record.content_url == HURRIYET_ARTICLE_URL
    assert record.title == "Son dakika : Fon soruşturmasında 20 şüpheliye tutuklama talebi"
    assert record.subtitle.startswith("İstanbul'da sermaye piyasasında")
    assert record.published_at == datetime.fromisoformat("2026-10-02T22:20:00+03:00")
    assert record.updated_at == datetime.fromisoformat("2026-10-02T22:34:53+03:00")
    assert "Google’da Takip Edin" not in record.content
    assert "Gelişmelerden anında haberdar olun" not in record.content
    assert "kaynak olarak ekleyin" not in record.content
    assert "Haberin Devamı" not in record.content
    assert "İstanbul Cumhuriyet Başsavcılığı" in record.content
    assert "SORUŞTURMANIN DETAYLARI" in record.content
    assert "�" not in record.content
    assert record.content.startswith("İstanbul Cumhuriyet Başsavcılığı")
    assert record.content.endswith("65'e ulaşmıştı.")
    assert record.content.count("\n\n") >= 10  # paragraf yapısı korunuyor
    assert record.category == "Gündem"
    assert record.author == "Gülden Kılıç"
    assert record.tags == ["son dakika", "fon soruşturması", "tutuklama talebi"]
    assert record.image_url == "https://image.hurimg.com/i/hurriyet/90/0x0/6ac0038083386a49177b17ea.jpg"
    assert record.stage == Stage.RAW and record.content_hash and record.id


def test_hurriyet_article_body_fallback_strips_boilerplate() -> None:
    body = parse_jsonld_newsarticle(fixture_text("hurriyet_article.html"))["articleBody"]
    cleaned = HurriyetSource.clean_article_body(body)
    assert "Google" not in cleaned
    assert cleaned.startswith("İstanbul Cumhuriyet Başsavcılığı")
    assert "\n" in cleaned


def test_hurriyet_article_without_html_body_uses_articlebody(hurriyet: HurriyetSource) -> None:
    html = fixture_text("hurriyet_article.html").replace('class="news-content readingTime"', 'class="gone"')
    record = hurriyet.parse_article(html, url=HURRIYET_ARTICLE_URL)
    assert record is not None
    assert "Google" not in record.content
    assert "İstanbul Cumhuriyet Başsavcılığı" in record.content


def test_hurriyet_fetch_falls_back_to_rss_on_http_error(settings: Settings, hurriyet: HurriyetSource) -> None:
    rss_links = hurriyet.parse_rss(fixture_text("hurriyet_gundem_rss.xml"))
    link = next(item for item in rss_links if item.url == HURRIYET_ARTICLE_URL)
    client = FakeHttpClient({HURRIYET_ARTICLE_URL: HttpError(HURRIYET_ARTICLE_URL, 503, "Service Unavailable")})
    record = hurriyet.fetch_article(client, link)
    assert record is not None
    assert record.title == "Son dakika : Fon soruşturmasında 20 şüpheliye tutuklama talebi"
    assert record.subtitle.startswith("İstanbul'da sermaye piyasasında")
    assert record.published_at == datetime(2026, 10, 2, 22, 20, tzinfo=ISTANBUL)
    assert record.content.startswith("İstanbul Cumhuriyet Başsavcılığı")
    assert "<p>" not in record.content and "&uuml;" not in record.content
    assert record.category == "Gündem"


def test_hurriyet_fetch_falls_back_to_rss_on_timeout(hurriyet: HurriyetSource) -> None:
    link = DiscoveredLink(
        url=HURRIYET_ARTICLE_URL, title_hint="Başlık", rss_html="<p>Gövde metni.</p>", rss_summary="Spot"
    )
    record = hurriyet.fetch_article(FakeHttpClient({HURRIYET_ARTICLE_URL: requests.Timeout("yavaş")}), link)
    assert record is not None and record.content == "Gövde metni." and record.subtitle == "Spot"


def test_hurriyet_fetch_raises_without_rss_fallback(hurriyet: HurriyetSource) -> None:
    link = DiscoveredLink(url=HURRIYET_ARTICLE_URL)
    with pytest.raises(HttpError):
        hurriyet.fetch_article(FakeHttpClient(), link)


def test_hurriyet_parse_returns_none_for_empty_page(hurriyet: HurriyetSource) -> None:
    assert hurriyet.parse_article("<html><body><h1>x</h1></body></html>", url=HURRIYET_ARTICLE_URL) is None


# ---------------------------------------------------------------------------------------------------------
# 12punto
# ---------------------------------------------------------------------------------------------------------


def test_punto_rss_yields_20_https_links(punto: PuntoSource) -> None:
    links = punto.parse_rss(fixture_text("punto_rss.xml"))
    assert len(links) == 20
    assert all(link.url.startswith("https://12punto.com.tr/") for link in links)
    assert all(link.published_hint is not None for link in links)
    first = links[0]
    assert first.url == PUNTO_ARTICLE_URL
    assert first.published_hint == datetime.fromisoformat("2026-10-02T22:41:28+03:00")
    assert first.category_hint == "kulis"
    assert first.rss_summary.startswith("TİP Milletvekili Ahmet Şık")
    assert first.image_hint.startswith("https://i.12punto.com.tr/")


def test_punto_listing_yields_25_links(punto: PuntoSource) -> None:
    links = punto.parse_listing(fixture_text("punto_gundem_listing.html"), category_hint="gundem")
    assert len(links) == 25
    assert all(link.url.startswith("https://12punto.com.tr/") for link in links)
    assert all(link.title_hint for link in links)


def test_punto_archive_search_yields_4_links(punto: PuntoSource) -> None:
    links = punto.parse_search(fixture_text("punto_arama.html"))
    assert len(links) == 4
    assert all(link.origin == "archive" for link in links)
    assert not any("/yazarlar/" in link.url for link in links)


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("http://12punto.com.tr/kulis/abc-154231", "https://12punto.com.tr/kulis/abc-154231"),
        ("https://www.12punto.com.tr/gundem/abc-def-154231/", "https://12punto.com.tr/gundem/abc-def-154231"),
        ("/gundem/abc-154231?utm_source=rss#top", "https://12punto.com.tr/gundem/abc-154231"),
        ("/yazarlar/muyesser-yildiz/trumpla-hayirli-isler-154145", None),
        ("/gundem/", None),
        ("/gundem/abc-12", None),
        ("https://other.example/gundem/abc-154231", None),
        ("", None),
    ],
)
def test_punto_url_filter(punto: PuntoSource, href: str, expected: str | None) -> None:
    assert punto.normalize_url(href) == expected


def test_punto_article_parse(punto: PuntoSource) -> None:
    record = punto.parse_article(fixture_text("punto_article.html"), url=PUNTO_ARTICLE_URL)
    assert record is not None
    assert record.source == "12punto"
    assert record.title.startswith("TİP milletvekili Ahmet Şık")
    assert record.subtitle.startswith("TİP Milletvekili Ahmet Şık, “fon vurgunu”")
    assert record.published_at == datetime.fromisoformat("2026-10-02T22:41:28+03:00")
    assert record.updated_at == datetime.fromisoformat("2026-10-02T22:41:28+03:00")
    assert record.category == "Kulis"
    assert "ŞIK'IN PAYLAŞIMINDA ÖNE ÇIKAN İDDİALAR" in record.content
    assert "Adalet Bakanlığı" in record.content
    assert "Haberlerini algoritmaya bırakma" not in record.content
    assert "Haber Kaynağı" not in record.content
    assert not record.content.startswith("TİP milletvekili Ahmet Şık’tan")  # <img alt> içeriğe sızmıyor
    assert record.content.startswith("Türkiye İşçi Partisi (TİP)")
    assert record.author == "12punto"
    assert record.image_url.startswith("https://i.12punto.com.tr/Archive/2026/10/2/154231/")


def test_punto_article_falls_back_to_jsonld_body_and_text_date(punto: PuntoSource) -> None:
    html = fixture_text("punto_article.html").replace('<section class="details">', '<section class="gone">')
    html = html.replace('"datePublished": "2026-10-02T22:41:28+03:00"', '"datePublished": ""')
    record = punto.parse_article(html, url=PUNTO_ARTICLE_URL)
    assert record is not None
    assert "Adalet Bakanlığı" in record.content
    assert record.published_at == datetime(2026, 10, 2, 22, 41, tzinfo=ISTANBUL)  # "Yayınlanma: 02.10.2026 22:41"


def test_punto_discover_collects_feeds_listings_and_skips_missing(settings: Settings, punto: PuntoSource) -> None:
    client = FakeHttpClient(punto_routes(settings))
    links = punto.discover(client)
    urls = [link.url for link in links]
    assert len(urls) == len(set(urls))
    assert all(url.startswith("https://12punto.com.tr/") for url in urls)
    assert 20 < len(links) <= 45
    assert links[0].published_hint is not None
    base = settings.punto_base_url
    assert client.calls[0] == f"{base}/rss"
    assert f"{base}/rss/siyaset" in client.calls  # XML olmayan yanıt sessizce atlandı
    assert f"{base}/gundem" in client.calls
    assert not any("/Arama/Ara" in call for call in client.calls)


def test_punto_discover_backfill_requests_each_day(settings: Settings, punto: PuntoSource) -> None:
    routes = punto_routes(settings)
    archive_hits: list[str] = []

    def fallback(url: str) -> bytes | None:
        if "/Arama/Ara?search=&StartDate=" in url:
            archive_hits.append(url)
            return fixture_bytes("punto_arama.html")
        return None

    client = FakeHttpClient(routes, fallback=fallback)
    links = punto.discover(client, backfill_days=2)
    assert len(archive_hits) == 3  # bugün + 2 gün
    today = datetime.now(ISTANBUL).date()
    for offset, url in enumerate(archive_hits):
        day = (today - timedelta(days=offset)).isoformat()
        assert url == f"{settings.punto_base_url}/Arama/Ara?search=&StartDate={day}&EndDate={day}"
    archive_urls = {link.url for link in punto.parse_search(fixture_text("punto_arama.html"))}
    assert archive_urls <= {link.url for link in links}  # arşiv bağlantıları (RSS/listeyle çakışanlar birleşik) sonuçta


def test_punto_discover_limit_stops_after_first_feed(settings: Settings, punto: PuntoSource) -> None:
    client = FakeHttpClient(punto_routes(settings))
    links = punto.discover(client, limit=3)
    assert len(links) == 3
    assert client.calls == [f"{settings.punto_base_url}/rss"]


def test_punto_fetch_article_propagates_http_errors(punto: PuntoSource) -> None:
    with pytest.raises(HttpError):
        punto.fetch_article(FakeHttpClient(), DiscoveredLink(url=PUNTO_ARTICLE_URL))


# ---------------------------------------------------------------------------------------------------------
# SeenStore
# ---------------------------------------------------------------------------------------------------------


def make_record(url: str = PUNTO_ARTICLE_URL, content: str = "içerik", published: datetime | None = None) -> NewsRecord:
    return NewsRecord.new(
        source="12punto",
        content_url=url,
        title="Başlık",
        content=content,
        published_at=published or datetime(2026, 10, 2, 22, 41, 28, tzinfo=ISTANBUL),
    )


def test_seen_store_status_transitions(tmp_path: Path) -> None:
    store = SeenStore(tmp_path / "nested" / "dir" / "state.sqlite3")
    record = make_record()
    assert store.status(record.id, record.content_hash) == "new"
    store.mark(record)
    assert store.status(record.id, record.content_hash) == "unchanged"
    changed = make_record(content="güncellenmiş içerik")
    assert changed.id == record.id and changed.content_hash != record.content_hash
    assert store.status(changed.id, changed.content_hash) == "updated"
    store.mark(changed)
    assert store.status(changed.id, changed.content_hash) == "unchanged"
    row = store.get(record.id)
    assert row is not None and row["times_seen"] == 2 and row["url"] == PUNTO_ARTICLE_URL
    assert row["first_seen"] <= row["last_seen"]
    assert store.count() == 1
    store.close()


def test_seen_store_seen_recently_and_pubdate_change(tmp_path: Path) -> None:
    store = SeenStore(tmp_path / "state.sqlite3", recent_hours=6)
    record = make_record()
    assert store.seen_recently(record.id) is False
    store.mark(record)
    assert store.seen_recently(record.id) is True
    assert store.seen_recently(record.id, published_at=record.published_at) is True
    assert store.seen_recently(record.id, published_at=record.published_at + timedelta(seconds=30)) is True
    assert store.seen_recently(record.id, published_at=record.published_at + timedelta(minutes=10)) is False
    assert store.seen_recently(record.id, within_hours=0) is False
    feed_stamp = record.published_at + timedelta(minutes=14)  # beslemedeki "modified" damgası sayfadan farklı olabilir
    store.mark(record, published_at=feed_stamp)
    assert store.get(record.id)["published_at"] == feed_stamp.isoformat()
    assert store.seen_recently(record.id, published_at=feed_stamp) is True
    assert store.seen_recently(record.id, published_at=record.published_at) is False
    store.close()
    disabled = SeenStore(tmp_path / "state.sqlite3", recent_hours=0)
    assert disabled.seen_recently(record.id) is False
    disabled.close()


def test_seen_store_seen_urls_and_thread_safety(tmp_path: Path) -> None:
    store = SeenStore(tmp_path / "state.sqlite3")
    records = [make_record(url=f"https://12punto.com.tr/gundem/haber-{n}-1000{n}") for n in range(8)]

    def worker(chunk: list[NewsRecord]) -> None:
        for item in chunk:
            store.mark(item)
            store.status(item.id, item.content_hash)

    threads = [threading.Thread(target=worker, args=(records[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.count() == 8
    urls = store.seen_urls(limit=3)
    assert len(urls) == 3 and all(url.startswith("https://12punto.com.tr/gundem/haber-") for url in urls)
    assert len(store.seen_urls(limit=100)) == 8
    store.close()


# ---------------------------------------------------------------------------------------------------------
# ScrapeRunner
# ---------------------------------------------------------------------------------------------------------


def test_build_sources(settings: Settings) -> None:
    names = [source.name for source in build_sources(settings)]
    assert names == ["hurriyet", "12punto"]
    assert [s.name for s in build_sources(settings.model_copy(update={"sources": "punto, bilinmeyen"}))] == ["12punto"]
    with pytest.raises(ValueError):
        build_sources(settings.model_copy(update={"sources": "yok"}))


def runner_with_small_feed(settings: Settings, *, article_html: bytes | None = None, **kwargs: Any) -> tuple[ScrapeRunner, InMemoryBroker, FakeHttpClient]:
    article = article_html or fixture_bytes("hurriyet_article.html")
    routes: dict[str, bytes | str | Exception] = {
        settings.hurriyet_gundem_rss: trimmed_rss("hurriyet_gundem_rss.xml", lambda link: "/gundem/" in link, 3),
    }
    client = FakeHttpClient(routes, fallback=lambda url: article if "/gundem/" in url else None)
    broker = InMemoryBroker(settings)
    runner = ScrapeRunner(settings, broker, sources=[HurriyetSource(settings)], client=client, **kwargs)
    return runner, broker, client


def test_runner_publishes_raw_records_then_nothing_on_second_run(settings: Settings) -> None:
    runner, broker, client = runner_with_small_feed(settings)
    stats = runner.run_once()
    assert isinstance(stats, ScrapeStats)
    assert stats.finished_at is not None and stats.duration_seconds >= 0
    source_stats = stats.per_source["hurriyet"]
    assert source_stats.discovered == 3
    assert source_stats.fetched == 3 and source_stats.published == 3 and source_stats.errors == 0
    assert stats.published == 3 and stats.unchanged == 0
    assert broker.size(Queue.ARTICLES_RAW) == 3
    messages = broker.drain(Queue.ARTICLES_RAW)
    assert all(msg.routing_key == RoutingKey.ARTICLE_RAW for msg in messages)
    assert all(msg.headers["x-source"] == "hurriyet" and msg.headers["x-change"] == "new" for msg in messages)
    records = [NewsRecord.from_message(msg.body) for msg in messages]
    assert {record.content_url for record in records} == {
        link.url for link in HurriyetSource(settings).parse_rss(client.get_text(settings.hurriyet_gundem_rss))
    }
    assert all(record.stage == Stage.RAW and record.content and record.published_at for record in records)
    assert all(record.alarm_score == 0 and not record.is_alarm and record.alarm_reason == "" for record in records)
    assert runner.seen.count() == 3
    assert "yayınlanan=3" in stats.summary()

    second = runner.run_once()
    assert second.published == 0
    assert second.per_source["hurriyet"].unchanged == 3
    assert second.per_source["hurriyet"].fetched == 0  # son 6 saatte görüldü, sayfa hiç çekilmedi
    assert broker.size(Queue.ARTICLES_RAW) == 0
    assert client.calls.count(HURRIYET_ARTICLE_URL) == 1
    runner.close()


def test_runner_republishes_updated_content(settings: Settings, tmp_path: Path) -> None:
    store = SeenStore(tmp_path / "state.sqlite3", recent_hours=0)  # "yakında görüldü" kısa devresi kapalı
    runner, broker, client = runner_with_small_feed(settings, seen_store=store)
    first = runner.run_once()
    assert first.published == 3
    broker.drain(Queue.ARTICLES_RAW)

    unchanged = runner.run_once()
    assert unchanged.published == 0 and unchanged.per_source["hurriyet"].unchanged == 3
    assert unchanged.per_source["hurriyet"].fetched == 3  # yeniden çekildi ama içerik özeti aynı

    updated_html = fixture_bytes("hurriyet_article.html").replace(
        "Böylece soruşturmada tutuklananların sayısı 65'e ulaşmıştı.".encode(),
        "Böylece soruşturmada tutuklananların sayısı 70'e ulaştı.".encode(),
    )
    client.fallback = lambda url: updated_html if "/gundem/" in url else None
    third = runner.run_once()
    assert third.published == 3
    messages = broker.drain(Queue.ARTICLES_RAW)
    assert all(msg.headers["x-change"] == "updated" for msg in messages)
    assert all("70'e ulaştı" in msg.body["content"] for msg in messages)
    runner.close()


def test_runner_counts_errors_and_continues(settings: Settings) -> None:
    runner, broker, client = runner_with_small_feed(settings)
    rss_links = HurriyetSource(settings).parse_rss(client.get_text(settings.hurriyet_gundem_rss))
    failing = rss_links[1].url
    client.routes[failing] = requests.ConnectionError("bağlantı koptu")
    # RSS <text> gövdesi var → Hürriyet bu hatada RSS'ten kayıt üretir, hata sayılmaz
    stats = runner.run_once()
    assert stats.errors == 0 and stats.published == 3
    broker.drain(Queue.ARTICLES_RAW)

    broken_source = HurriyetSource(settings)
    links_without_rss = [DiscoveredLink(url=link.url) for link in rss_links]
    broken_source.discover = lambda client, backfill_days=0, limit=None: links_without_rss  # type: ignore[method-assign]
    store = SeenStore(settings.state_db_path + ".2", recent_hours=0)
    runner2 = ScrapeRunner(settings, broker, seen_store=store, sources=[broken_source], client=client)
    stats2 = runner2.run_once()
    assert stats2.errors == 1 and stats2.published == 2
    assert broker.size(Queue.ARTICLES_RAW) == 2
    runner2.close()
    runner.close()


def test_runner_respects_total_article_budget(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_articles_per_run": 2})
    runner, broker, _ = runner_with_small_feed(limited)
    stats = runner.run_once()
    assert stats.published == 2 and stats.budget_exhausted is True
    assert broker.size(Queue.ARTICLES_RAW) == 2
    assert "bütçesi doldu" in stats.summary()
    runner.close()


def test_runner_with_both_sources_and_graceful_404s(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_articles_per_run": 12})
    routes = {**hurriyet_routes(settings), **punto_routes(settings)}

    def fallback(url: str) -> bytes | None:
        if url.startswith("https://www.hurriyet.com.tr/gundem/"):
            return fixture_bytes("hurriyet_article.html")
        if url.startswith("https://12punto.com.tr/") and "/rss" not in url and url.count("/") >= 4:
            return fixture_bytes("punto_article.html")
        return None

    client = FakeHttpClient(routes, fallback=fallback)
    broker = InMemoryBroker(limited)
    runner = ScrapeRunner(limited, broker, client=client)
    assert [source.name for source in runner.sources] == ["hurriyet", "12punto"]
    stats = runner.run_once()
    assert stats.published == 12 and stats.budget_exhausted is True
    # bütçe kaynaklar arasında adil paylaşılır: 12 / 2 kaynak → her biri 6
    assert stats.per_source["hurriyet"].published == 6
    assert stats.per_source["12punto"].discovered > 6 and stats.per_source["12punto"].published == 6
    records = [NewsRecord.from_message(msg.body) for msg in broker.drain(Queue.ARTICLES_RAW)]
    assert {record.source for record in records} == {"hurriyet", "12punto"}
    runner.close()


def test_runner_budget_rolls_over_unused_share(settings: Settings) -> None:
    """İlk kaynağın kullanmadığı pay ikinci kaynağa devreder (10 bütçe: hurriyet 2 → 12punto 8)."""
    limited = settings.model_copy(update={"max_articles_per_run": 10})
    routes = {**hurriyet_routes(settings), **punto_routes(settings)}
    hurriyet_rss = fixture_bytes("hurriyet_gundem_rss.xml").decode("utf-8", "replace")
    routes[settings.hurriyet_gundem_listing] = b"<html><body></body></html>"
    head, *items = hurriyet_rss.split("<item ")
    routes[settings.hurriyet_gundem_rss] = (head + "<item " + "<item ".join(items[:2]) + "</channel></rss>").encode(
        "utf-8"
    )

    def fallback(url: str) -> bytes | None:
        if url.startswith("https://www.hurriyet.com.tr/gundem/"):
            return fixture_bytes("hurriyet_article.html")
        if url.startswith("https://12punto.com.tr/") and "/rss" not in url and url.count("/") >= 4:
            return fixture_bytes("punto_article.html")
        return None

    runner = ScrapeRunner(limited, InMemoryBroker(limited), client=FakeHttpClient(routes, fallback=fallback))
    stats = runner.run_once()
    assert stats.per_source["hurriyet"].fetched <= 2
    assert stats.per_source["12punto"].fetched == 10 - stats.per_source["hurriyet"].fetched
    runner.close()


def test_runner_requires_broker(settings: Settings) -> None:
    with pytest.raises(ValueError):
        ScrapeRunner(settings, None)


def test_run_forever_stops_on_event(settings: Settings) -> None:
    fast = settings.model_copy(update={"scrape_interval_seconds": 1})
    stop_event = threading.Event()
    runs: list[int] = []

    class StopAfterFirstRun:
        name = "sahte"

        def discover(self, client: Any, *, backfill_days: int = 0, limit: int | None = None) -> list[DiscoveredLink]:
            runs.append(1)
            stop_event.set()
            return []

        def fetch_article(self, client: Any, link: DiscoveredLink) -> NewsRecord | None:
            return None

    runner = ScrapeRunner(fast, InMemoryBroker(fast), sources=[StopAfterFirstRun()], client=FakeHttpClient())
    started = time.monotonic()
    runner.run_forever(stop_event)
    assert runs == [1]
    assert time.monotonic() - started < 1.0
    runner.close()


def test_scrape_stats_summary_is_turkish() -> None:
    stats = ScrapeStats()
    stats.source("hurriyet").published = 2
    stats.source("12punto").errors = 1
    stats.finished_at = datetime.now(UTC)
    text = stats.summary()
    assert "Kazıma turu" in text and "yayınlanan=2" in text and "hata=1" in text
    assert stats.published == 2 and stats.errors == 1 and stats.discovered == 0
