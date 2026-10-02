"""Canlı site testleri: her kaynaktan en fazla 2 haber çeker. ``-m "not live"`` ile dışlanır."""

from __future__ import annotations

from collections.abc import Iterator

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
