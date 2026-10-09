"""Çekilemeyen haberlerin bekleyen listesi, sonraki turda yeniden deneme, tur kaydı ve kazıyıcı durumu."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scraperhryt.broker import InMemoryBroker, Queue
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord
from scraperhryt.pipeline.llm import FakeOllama
from scraperhryt.reporting.api import create_app
from scraperhryt.scrapers.base import DiscoveredLink, HttpError
from scraperhryt.scrapers.runner import ScrapeRunner
from scraperhryt.scrapers.state import SeenStore
from scraperhryt.scrapers.status import scraper_status
from scraperhryt.store import InMemoryStore


class FlakySource:
    """İlk turda 2. haberi alamayan, sonraki turda başaran sahte kaynak."""

    name = "hurriyet"

    def __init__(self) -> None:
        self.fail_once = {"https://www.hurriyet.com.tr/gundem/ikinci-haber-2"}

    def discover(self, client, *, backfill_days=0, limit=None):
        return [DiscoveredLink(url=f"https://www.hurriyet.com.tr/gundem/{slug}", title_hint=slug) for slug in ("ilk-haber-1", "ikinci-haber-2", "ucuncu-haber-3")]

    def fetch_article(self, client, link):
        if link.url in self.fail_once:
            self.fail_once.discard(link.url)
            raise HttpError(link.url, 503, "geçici hata")
        return NewsRecord.new(source="hurriyet", content_url=link.url, title=link.title_hint or link.url, content="Bakan açıkladı")


def _runner(tmp_path: Path, source, **kw) -> tuple[ScrapeRunner, InMemoryBroker, SeenStore]:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "state.sqlite3"), **kw)
    broker = InMemoryBroker(settings)
    seen = SeenStore(settings.state_db_path)
    return ScrapeRunner(settings, broker, seen_store=seen, sources=[source]), broker, seen


def test_failed_fetch_is_retried_next_run(tmp_path: Path) -> None:
    runner, broker, seen = _runner(tmp_path, FlakySource())
    first = runner.run_once()
    assert first.published == 2 and first.errors == 1 and first.pending_after == 1
    pending = seen.list_pending()
    assert len(pending) == 1 and pending[0]["url"].endswith("ikinci-haber-2") and pending[0]["attempts"] == 1
    assert "bekleyen=1" in first.summary()
    second = runner.run_once()
    assert second.retried == 1 and second.published == 1 and second.pending_after == 0 and seen.pending_count() == 0
    assert broker.size(Queue.ARTICLES_RAW) == 3
    assert seen.last_run()["published"] == 1 and len(seen.recent_runs()) == 2


def test_budget_exhaustion_defers_remaining_links(tmp_path: Path) -> None:
    runner, broker, seen = _runner(tmp_path, FlakySource(), max_articles_per_run=1)
    stats = runner.run_once()
    assert stats.budget_exhausted and stats.published == 1 and stats.pending_after == 2
    reasons = {p["reason"] for p in seen.list_pending()}
    assert reasons == {"haber bütçesi doldu"} and all(p["attempts"] == 0 for p in seen.list_pending())


def test_pending_dropped_after_max_attempts(tmp_path: Path) -> None:
    from scraperhryt.scrapers.state import MAX_PENDING_ATTEMPTS

    seen = SeenStore(tmp_path / "s.sqlite3")
    for i in range(MAX_PENDING_ATTEMPTS - 1):
        assert seen.add_pending(id="x", url="https://e/x", source="12punto", reason="hata") == i + 1
    assert seen.pending_count() == 1
    seen.remove_pending("x")
    assert seen.pending_count() == 0 and seen.get_meta("yok") is None
    seen.set_meta("k", "v")
    assert seen.get_meta("k") == "v"


def test_scraper_status_and_api(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "state.sqlite3"))
    missing = scraper_status(settings)
    assert missing["available"] is False and "hiç çalışmadı" in missing["warning"]
    runner, _broker, seen = _runner(tmp_path, FlakySource())
    runner.run_once()
    status = scraper_status(settings)
    assert status["available"] and status["pending_count"] == 1 and "henüz çekilemedi" in status["warning"]
    assert status["last_run"]["published"] == 2 and status["stale"] is False and status["pending"][0]["attempts"] == 1
    client = TestClient(create_app(settings, InMemoryStore(), FakeOllama()))
    body = client.get("/scraper/status").json()
    assert body["pending_count"] == 1 and body["last_run"]["errors"] == 1
    page = client.get("/ara")
    assert page.status_code == 200 and "Haber Radarı" in page.text and "/scraper/status" in page.text


def test_runner_reports_live_progress_and_status_exposes_it(tmp_path: Path) -> None:
    import json as _json

    runner, _broker, seen = _runner(tmp_path, FlakySource())
    states: list[dict] = []
    original = seen.set_meta

    def spy(key: str, value: str) -> None:
        if key == "run_state":
            states.append(_json.loads(value))
        original(key, value)

    seen.set_meta = spy  # type: ignore[method-assign]
    runner.run_once()
    assert states[0]["running"] is True and states[0]["phase"] == "başlıyor"
    assert any(s["phase"] == "çekme" and s["source"] == "hurriyet" for s in states)
    assert states[-1]["running"] is False and states[-1]["fraction"] == 1.0
    fractions = [s["fraction"] for s in states]
    assert fractions == sorted(fractions)

    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "state.sqlite3"))
    assert scraper_status(settings)["running"] is False
    seen.set_meta("run_state", _json.dumps({**states[1], "updated_at": states[1]["updated_at"]}))
    status = scraper_status(settings)
    assert status["running"] is True and status["progress"]["source"] in ("", "hurriyet") and status["stale"] is False


def test_ui_has_no_emoji_or_icons_and_shows_news_as_cards() -> None:
    import re as _re

    root = Path(__file__).resolve().parents[1] / "src" / "scraperhryt" / "reporting" / "templates"
    for page in root.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert not _re.search("[\U0001F300-\U0001FAFF☀-➿←-⇿■-◿]", text), page.name
        assert "<svg" not in text and "<symbol" not in text and 'role="progressbar"' in text
        assert "repeat(3, minmax(0, 1fr))" in text and "-webkit-line-clamp: 2" in text and "-webkit-line-clamp: 3" in text


def test_ask_ui_shows_elapsed_time_and_aborts_after_server_limits() -> None:
    from fastapi.testclient import TestClient

    from scraperhryt.pipeline.llm import FakeOllama
    from scraperhryt.reporting.api import create_app
    from scraperhryt.store import InMemoryStore

    s = Settings(_env_file=None, rag_rewrite_timeout=20, rag_answer_timeout=90)
    page = TestClient(create_app(s, InMemoryStore(), FakeOllama())).get("/").text
    assert "const ASK_LIMIT_MS = 150000;" in page  # 90 + 60 sn pay (sorgu yeniden yazma kapalı)
    s = s.model_copy(update={"rag_query_rewrite": True})
    page = TestClient(create_app(s, InMemoryStore(), FakeOllama())).get("/").text
    assert "const ASK_LIMIT_MS = 170000;" in page  # 20 + 90 + 60 sn pay
    assert "signal: ctrl.signal" in page and "typewriter(" in page and "AbortError" in page


def test_runner_skips_articles_older_than_max_age(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    fetched: list[str] = []

    class AgedSource:
        name = "hurriyet"

        def discover(self, client, *, backfill_days=0, limit=None):
            return [
                DiscoveredLink(url="https://www.hurriyet.com.tr/gundem/eski-rss-1", title_hint="eski", published_hint=now - timedelta(days=30)),
                DiscoveredLink(url="https://www.hurriyet.com.tr/gundem/tarihsiz-2", title_hint="tarihsiz"),
                DiscoveredLink(url="https://www.hurriyet.com.tr/gundem/yeni-3", title_hint="yeni", published_hint=now - timedelta(hours=2)),
            ]

        def fetch_article(self, client, link):
            fetched.append(link.title_hint)
            published = now - timedelta(days=40) if link.title_hint == "tarihsiz" else now - timedelta(hours=2)
            return NewsRecord.new(source="hurriyet", content_url=link.url, title=link.title_hint, content="Bakan açıkladı", published_at=published)

    runner, broker, _seen = _runner(tmp_path, AgedSource(), max_article_age_days=7)
    stats = runner.run_once()
    assert fetched == ["tarihsiz", "yeni"]  # beslemede tarihi eski olan hiç çekilmez
    assert stats.published == 1 and stats.source("hurriyet").too_old == 2 and "eski=2" in stats.summary()
    assert broker.size(Queue.ARTICLES_RAW) == 1
    # sonraki turda tarihi öğrenilmiş eski haber yeniden çekilmez
    fetched.clear()
    runner.run_once()
    assert "tarihsiz" not in fetched
    # geriye dönük taramada pencere BACKFILL_DAYS kadar genişler
    runner2, broker2, _ = _runner(tmp_path / "b", AgedSource(), max_article_age_days=7)
    assert runner2.run_once(backfill_days=45).published == 3 and broker2.size(Queue.ARTICLES_RAW) == 3


def test_hurriyet_newsletter_box_is_not_article_text() -> None:
    from bs4 import BeautifulSoup

    from scraperhryt.scrapers.hurriyet import HurriyetSource

    html = (
        '<div class="news-content"><p>Bakan yeni düzenlemeyi açıkladı.</p><p>Ayrıntılar Geliyor...</p>'
        "<p>Haber Bültenleri ve E-Posta Tercihleri</p><p>Türkiye ve dünyadaki en güncel gelişmelerden haberdar olmak "
        "için, bültenlerin gönderileceği e-posta adresini girin.</p></div>"
    )
    assert HurriyetSource._content_from_html(BeautifulSoup(html, "lxml")) == "Bakan yeni düzenlemeyi açıkladı."
    flat = "Bakan açıkladı. Haber Bültenleri ve E-Posta Tercihleri Türkiye ve dünyadaki en güncel gelişmelerden haberdar olmak için, bültenlerin gönderileceği e-posta adresini girin."
    assert "Bülten" not in HurriyetSource.clean_article_body(flat)


def test_punto_gallery_page_uses_description_as_content() -> None:
    from scraperhryt.scrapers.punto import PuntoSource

    html = (
        "<html><head><meta name='description' content='Mustafa Sandal\'ın eşinin fondan kazandığı para gündem oldu.'>"
        "</head><body><h1>Fondan kazandığı para dudak uçuklattı</h1></body></html>"
    )
    rec = PuntoSource(Settings(_env_file=None)).parse_article(html, url="https://12punto.com.tr/yasam/galeri-fon-haberi-154290")
    assert rec is not None and rec.content.startswith("Mustafa Sandal") and rec.title.startswith("Fondan")


def test_punto_2026_redesign_archive_and_article_markup() -> None:
    from datetime import date

    from scraperhryt.scrapers.punto import PuntoSource

    src = PuntoSource(Settings(_env_file=None))
    archive = (
        '<section class="punto-listing"><div class="punto-listing__grid">'
        '<a class="punto-listing__card" title="Ukrayna’da başbakan değişikliği" data-yayintarihi2="12.07.2026 00:00:00" '
        'href="/dunya/ukraynada-basbakan-degisikligi-gundemde-144293"><p class="punto-listing__title">x</p></a>'
        '<a class="punto-listing__card" title="Haluk Levent" data-yayintarihi2="12.07.2026 00:00:00" '
        'href="/gundem/haluk-levent-gozalti-144292"></a></div>'
        '<nav class="punto-pagination"><a href="/Arama/Ara?key=&amp;StartDate=12%2f07%2f2026&amp;sayfa=1">1</a>'
        '<a href="/Arama/Ara?key=&amp;StartDate=12%2f07%2f2026&amp;sayfa=4">4</a></nav></section>'
        '<section class="punto-sidebar"><a href="/spor/kenar-cubugu-haberi-144000">yan</a></section>'
    )
    page = src.parse_search_page(archive)
    assert page.page_count == 4 and len(page.links) == 2
    assert all(link.published_hint.date() == date(2026, 7, 12) for link in page.links)

    article = (
        "<html><body><section class='punto-article-header'><h1 class='punto-article-header__title'>Başlık</h1>"
        "<p class='punto-article-header__spot'>Spot metni.</p></section>"
        "<section class='punto-article-body'><div class='punto-article-body__content'><p>Birinci paragraf yeterince uzun bir metin.</p>"
        "<p>İkinci paragraf.</p><p class='punto-article-body__source'>Haber Kaynağı : <a>12punto</a></p></div></section></body></html>"
    )
    rec = src.parse_article(article, url="https://12punto.com.tr/gundem/baslik-154871")
    assert rec is not None and rec.content == "Birinci paragraf yeterince uzun bir metin.\n\nİkinci paragraf."
