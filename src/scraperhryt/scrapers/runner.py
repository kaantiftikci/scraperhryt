"""Kazıma çalıştırıcısı: kaynakları keşfeder, yeni/güncellenen haberleri çekip ``article.raw`` olarak yayınlar."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime

import requests

from ..broker import Broker, RoutingKey
from ..config import Settings, get_settings
from ..models import Stage, utcnow
from .base import DiscoveredLink, HttpClient, HttpError, Source
from .hurriyet import HurriyetSource
from .punto import PuntoSource
from .state import SeenStore

log = logging.getLogger(__name__)

_SOURCE_FACTORIES: dict[str, type[HurriyetSource] | type[PuntoSource]] = {
    "hurriyet": HurriyetSource,
    "12punto": PuntoSource,
    "punto": PuntoSource,
}


def build_sources(settings: Settings | None = None) -> list[Source]:
    """``settings.sources`` listesinden kaynak nesneleri üretir; bilinmeyen adlar uyarıyla atlanır."""
    settings = settings or get_settings()
    sources: list[Source] = []
    names_seen: set[str] = set()
    for raw_name in settings.source_list:
        key = raw_name.strip().lower()
        factory = _SOURCE_FACTORIES.get(key)
        if factory is None:
            log.warning("Bilinmeyen kaynak adı yok sayıldı: %r (geçerli: %s)", raw_name, ", ".join(_SOURCE_FACTORIES))
            continue
        source = factory(settings)
        if source.name in names_seen:
            continue
        names_seen.add(source.name)
        sources.append(source)
    if not sources:
        raise ValueError(f"Geçerli kaynak yok: SOURCES={settings.sources!r}")
    return sources


@dataclass
class SourceStats:
    discovered: int = 0  # keşfedilen tekil bağlantı
    fetched: int = 0  # sayfası çekilen (başarılı) haber
    published: int = 0  # kuyruğa yayınlanan (yeni + güncellenen)
    unchanged: int = 0  # değişmediği için atlanan (son görülme penceresi ya da aynı içerik özeti)
    skipped: int = 0  # çekildi ama kayıt üretilemedi (ayrıştırılamadı)
    errors: int = 0  # çekme/ayrıştırma hatası


@dataclass
class ScrapeStats:
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    per_source: dict[str, SourceStats] = field(default_factory=dict)
    budget_exhausted: bool = False

    def source(self, name: str) -> SourceStats:
        return self.per_source.setdefault(name, SourceStats())

    def _total(self, attr: str) -> int:
        return sum(getattr(stats, attr) for stats in self.per_source.values())

    @property
    def discovered(self) -> int:
        return self._total("discovered")

    @property
    def fetched(self) -> int:
        return self._total("fetched")

    @property
    def published(self) -> int:
        return self._total("published")

    @property
    def unchanged(self) -> int:
        return self._total("unchanged")

    @property
    def skipped(self) -> int:
        return self._total("skipped")

    @property
    def errors(self) -> int:
        return self._total("errors")

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or utcnow()
        return max(0.0, (end - self.started_at).total_seconds())

    def summary(self) -> str:
        parts = [
            f"{name}: keşif={s.discovered} çekilen={s.fetched} yayınlanan={s.published} "
            f"değişmeyen={s.unchanged} atlanan={s.skipped} hata={s.errors}"
            for name, s in self.per_source.items()
        ]
        detail = "; ".join(parts) if parts else "kaynak yok"
        suffix = " (haber bütçesi doldu)" if self.budget_exhausted else ""
        return (
            f"Kazıma turu {self.duration_seconds:.1f}s sürdü — toplam yayınlanan={self.published}, "
            f"değişmeyen={self.unchanged}, hata={self.errors}{suffix} | {detail}"
        )


class ScrapeRunner:
    """Tüm kaynakları sırayla tarar ve ``NewsRecord.to_message()`` mesajlarını ``article.raw`` ile yayınlar.

    - ``SeenStore`` sayesinde son 6 saatte görülüp besleme damgası (RSS modified/pubDate) değişmeyen bağlantılar
      hiç çekilmez;
      çekilenlerde içerik özeti aynıysa yayınlanmaz, değiştiyse (güncellenen haber) aynı id ile yeniden yayınlanır.
    - ``settings.max_articles_per_run`` tur başına toplam sayfa çekme bütçesidir. Bütçe kaynaklar arasında adil
      paylaştırılır: her kaynak en fazla ``ceil(kalan bütçe / kalan kaynak sayısı)`` sayfa çeker, kullanılmayan pay
      sonraki kaynağa devreder (böylece küçük bütçede ikinci kaynak aç kalmaz).
    - Haber başına hatalar sayılır ve tur devam eder; broker hataları turu keser (``run_forever`` yakalar).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        broker: Broker | None = None,
        seen_store: SeenStore | None = None,
        sources: list[Source] | None = None,
        client: HttpClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if broker is None:
            raise ValueError("ScrapeRunner için bir broker gerekli")
        self.broker = broker
        self.seen = seen_store or SeenStore(self.settings.state_db_path)
        self.sources: list[Source] = list(sources) if sources is not None else build_sources(self.settings)
        self.client = client or HttpClient(self.settings)
        self._topology_ready = False

    def _ensure_topology(self) -> None:
        if not self._topology_ready:
            self.broker.declare_topology()
            self._topology_ready = True

    def run_once(self, *, backfill_days: int | None = None) -> ScrapeStats:
        """Tek kazıma turu. ``backfill_days`` verilmezse ``settings.backfill_days`` kullanılır."""
        self._ensure_topology()
        stats = ScrapeStats()
        days = self.settings.backfill_days if backfill_days is None else max(0, int(backfill_days))
        budget = max(0, int(self.settings.max_articles_per_run))
        attempts = 0
        for index, source in enumerate(self.sources):
            source_stats = stats.source(source.name)
            remaining_sources = len(self.sources) - index
            quota = math.ceil((budget - attempts) / remaining_sources) if budget > attempts else 0
            if quota <= 0:
                log.warning("Haber bütçesi (%d) doldu; %s bu turda taranmadı", budget, source.name)
                stats.budget_exhausted = True
                continue
            source_attempts = 0
            try:
                links = source.discover(self.client, backfill_days=days)
            except Exception:
                log.exception("%s keşfi başarısız", source.name)
                source_stats.errors += 1
                continue
            source_stats.discovered = len(links)
            log.info("%s: %d bağlantı keşfedildi", source.name, len(links))
            for link in links:
                if source_attempts >= quota:
                    log.warning(
                        "Haber bütçesi (%d, %s payı %d) doldu; %s için kalan bağlantılar sonraki tura kaldı",
                        budget,
                        source.name,
                        quota,
                        source.name,
                    )
                    stats.budget_exhausted = True
                    break
                feed_stamp = link.updated_hint or link.published_hint
                if self.seen.seen_recently(link.id, published_at=feed_stamp):
                    source_stats.unchanged += 1
                    continue
                attempts += 1
                source_attempts += 1
                self._process_link(source, link, source_stats, feed_stamp)
        stats.finished_at = utcnow()
        return stats

    def _process_link(
        self, source: Source, link: DiscoveredLink, source_stats: SourceStats, feed_stamp: datetime | None
    ) -> None:
        try:
            record = source.fetch_article(self.client, link)
        except (HttpError, requests.RequestException) as exc:
            source_stats.errors += 1
            log.warning("%s haberi alınamadı (%s): %s", source.name, link.url, exc)
            return
        except Exception:
            source_stats.errors += 1
            log.exception("%s haberi işlenirken beklenmeyen hata (%s)", source.name, link.url)
            return
        if record is None:
            source_stats.skipped += 1
            return
        source_stats.fetched += 1
        status = self.seen.status(record.id, record.content_hash)
        if status == "unchanged":
            source_stats.unchanged += 1
            self.seen.mark(record, published_at=feed_stamp)
            return
        record.stage = Stage.RAW
        self.broker.publish(
            RoutingKey.ARTICLE_RAW,
            record.to_message(),
            headers={"x-source": source.name, "x-change": status},
        )
        self.seen.mark(record, published_at=feed_stamp)
        source_stats.published += 1
        log.info("%s %s yayınlandı: %s", source.name, "güncellendi" if status == "updated" else "yeni", record.title)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        """``stop_event`` set edilene kadar ``scrape_interval_seconds`` aralıklarla tur atar."""
        stop_event = stop_event or threading.Event()
        interval = max(1, int(self.settings.scrape_interval_seconds))
        log.info("Kazıyıcı başladı: kaynaklar=%s, aralık=%ds", [s.name for s in self.sources], interval)
        while not stop_event.is_set():
            try:
                stats = self.run_once()
                log.info(stats.summary())
            except Exception:
                log.exception("Kazıma turu başarısız; %ds sonra yeniden denenecek", interval)
            if stop_event.wait(interval):
                break
        log.info("Kazıyıcı durduruldu")

    def close(self) -> None:
        self.client.close()
        self.seen.close()
