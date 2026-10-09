"""Canlı site testleri: her kaynaktan en fazla 2 haber çeker. ``-m "not live"`` ile dışlanır."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest

from scraperhryt.config import Settings
from scraperhryt.scrapers import HttpClient, HurriyetSource, PuntoSource

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client() -> Iterator[HttpClient]:
    settings = Settings(_env_file=None, request_delay_seconds=0.5, request_timeout=30.0)
    http = HttpClient(settings)
    yield http
    http.close()


@pytest.mark.parametrize("source_cls", [HurriyetSource, PuntoSource], ids=["hurriyet", "12punto"])
def test_live_discover_and_fetch_two_articles(client: HttpClient, source_cls: type) -> None:
    source = source_cls(client.settings)
    links = source.discover(client, limit=2)
    assert links, f"{source.name}: keşif boş döndü"
    assert len(links) <= 2
    for link in links:
        record = source.fetch_article(client, link)
        assert record is not None, f"{source.name}: kayıt üretilemedi ({link.url})"
        assert record.source == source.name
        assert record.title
        assert len(record.content) > 100
        assert record.published_at is not None and record.published_at.tzinfo is not None
        assert record.content_url.startswith("https://")
        if source.name == "hurriyet":
            assert "/gundem/" in record.content_url
            assert "Google’da Takip Edin" not in record.content
        else:
            assert "Haberlerini algoritmaya bırakma" not in record.content


def test_live_punto_archive_search_returns_results_for_indexed_day(client: HttpClient) -> None:
    """Site arşivi yalnızca GG/AA/YYYY tarih biçimini tanır; dizinde bulunduğu doğrulanmış bir gün (12.07.2026,
    4 sayfa) için sonuç kapsayıcısından haber bağlantıları ve sayfa sayısı okunabilmelidir."""
    source = PuntoSource(client.settings)
    page = source.parse_search_page(client.get_text(source.archive_url(date(2026, 7, 12))))
    assert page.page_count >= 2
    assert len(page.links) >= 10
    assert all(link.origin == "archive" and link.url.startswith("https://12punto.com.tr/") for link in page.links)
    assert all(link.published_hint is not None and link.published_hint.date() == date(2026, 7, 12) for link in page.links)
