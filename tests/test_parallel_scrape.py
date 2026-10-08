"""Paralel kaynak taraması ve adil bütçe paylaşımı."""

from __future__ import annotations

import threading

from scraperhryt.broker import InMemoryBroker
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord
from scraperhryt.scrapers.base import DiscoveredLink
from scraperhryt.scrapers.runner import ScrapeRunner, _Budget
from scraperhryt.scrapers.state import SeenStore


# --- paralel tarama ---
class _BarrierSource:
    """Her iki kaynağın ilk haberi aynı anda çekilmezse bariyer zaman aşımına düşer (sıralı taramada olur)."""

    def __init__(self, name: str, barrier: threading.Barrier, count: int = 3) -> None:
        self.name = name
        self.barrier = barrier
        self.count = count
        self.first = True

    def discover(self, client, *, backfill_days=0, limit=None):
        return [DiscoveredLink(url=f"https://{self.name}.example/gundem/haber-{i}", title_hint=f"h{i}") for i in range(self.count)]

    def fetch_article(self, client, link):
        if self.first:
            self.first = False
            self.barrier.wait(timeout=5)
        return NewsRecord.new(source=self.name, content_url=link.url, title=link.title_hint, content="içerik")


def test_sources_are_scraped_in_parallel(tmp_path) -> None:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "s.sqlite3"), max_articles_per_run=10)
    barrier = threading.Barrier(2)
    sources = [_BarrierSource("hurriyet", barrier), _BarrierSource("12punto", barrier)]
    runner = ScrapeRunner(settings, InMemoryBroker(settings), seen_store=SeenStore(settings.state_db_path), sources=sources)
    stats = runner.run_once()
    assert not barrier.broken
    assert stats.per_source["hurriyet"].published == 3 and stats.per_source["12punto"].published == 3


def test_budget_is_shared_fairly_and_unused_share_moves_to_other_source() -> None:
    budget = _Budget(4, ["a", "b"])
    assert [budget.take("a") for _ in range(3)] == [True, True, False]  # b'nin payı (2) ayrılmış kalır
    budget.take("b")
    budget.finish("b")  # b yalnızca 1 haber çekti; kalan payı a'ya geçer
    assert budget.take("a") is True and budget.take("a") is False
