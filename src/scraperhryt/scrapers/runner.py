"""Kazıma çalıştırıcısı: kaynakları keşfeder, yeni/güncellenen haberleri çekip ``article.raw`` olarak yayınlar."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import requests

from ..broker import Broker, RoutingKey
from ..config import Settings, get_settings
from ..models import Stage, utcnow
from .base import DiscoveredLink, HttpClient, HttpError, Source
from .hurriyet import HurriyetSource
from .punto import PuntoSource
from .state import MAX_PENDING_ATTEMPTS, SeenStore

MAX_PENDING_BACKLOG = 2000  # bekleyen (bütçe nedeniyle ertelenen) bağlantı listesinin üst sınırı

log = logging.getLogger(__name__)

# Besleme güncelleme damgası (RSS <modified>) taşımayan bağlantılarda sayfayı çekmeden atlama penceresi (saat).
# 12punto RSS/liste öğeleri yalnızca pubDate verir; haber güncellense de damga değişmez, değişiklik ancak sayfa
# çekilip içerik özeti karşılaştırılınca anlaşılır. Bu yüzden bu bağlantılar depo penceresinden (6 s) çok daha
# sık yeniden denetlenir.
RECHECK_HOURS_WITHOUT_UPDATE_STAMP = 1.0

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
    too_old: int = 0  # MAX_ARTICLE_AGE_DAYS'ten eski olduğu için atlanan
    errors: int = 0  # çekme/ayrıştırma hatası


@dataclass
class ScrapeStats:
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    per_source: dict[str, SourceStats] = field(default_factory=dict)
    budget_exhausted: bool = False
    interrupted: bool = False  # durdurma sinyali geldi; tur yarıda kesildi (kalan bağlantılar sonraki tura kaldı)
    retried: int = 0  # bekleyen listesinden bu turda yeniden denenen bağlantılar
    pending_after: int = 0  # tur sonunda hâlâ çekilememiş (bekleyen) haber sayısı

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
            f"değişmeyen={s.unchanged} atlanan={s.skipped} eski={s.too_old} hata={s.errors}"
            for name, s in self.per_source.items()
        ]
        detail = "; ".join(parts) if parts else "kaynak yok"
        suffix = " (haber bütçesi doldu)" if self.budget_exhausted else ""
        if self.pending_after:
            suffix += f" | bekleyen={self.pending_after} (sonraki turda yeniden denenecek)"
        if self.interrupted:
            suffix += " (durdurma sinyaliyle yarıda kesildi)"
        return (
            f"Kazıma turu {self.duration_seconds:.1f}s sürdü — toplam yayınlanan={self.published}, "
            f"değişmeyen={self.unchanged}, hata={self.errors}{suffix} | {detail}"
        )


class ScrapeRunner:
    """Tüm kaynakları sırayla tarar ve ``NewsRecord.to_message()`` mesajlarını ``article.raw`` ile yayınlar.

    - ``SeenStore`` sayesinde son 6 saatte görülüp besleme damgası (RSS modified) değişmeyen bağlantılar hiç
      çekilmez; güncelleme damgası olmayan bağlantılar (12punto: yalnızca pubDate) ise en geç
      ``RECHECK_HOURS_WITHOUT_UPDATE_STAMP`` saat sonra yeniden çekilir ki düzenlenen haberler yakalansın.
      Çekilenlerde içerik özeti aynıysa yayınlanmaz, değiştiyse (güncellenen haber) aynı id ile yeniden yayınlanır.
    - ``settings.backfill_days`` (12punto arşiv taraması) ``run_forever``'da yalnızca ilk tamamlanan turda
      uygulanır; sonraki turlar arşivi yeniden taramaz.
    - ``settings.max_articles_per_run`` tur başına toplam sayfa çekme bütçesidir. Bütçe kaynaklar arasında adil
      paylaştırılır: her kaynak en fazla ``ceil(kalan bütçe / kalan kaynak sayısı)`` sayfa çeker, kullanılmayan pay
      sonraki kaynağa devreder (böylece küçük bütçede ikinci kaynak aç kalmaz).
    - Haber başına hatalar sayılır ve tur devam eder; broker hataları turu keser (``run_forever`` yakalar).
    - ``stop_event`` set edilince tur en geç bir sonraki bağlantıda biter (SIGTERM/SIGINT'te kibar kapanış);
      yayınlama ``SeenStore.mark``'tan önce yapıldığından veri kaybı olmaz, kalanlar sonraki tura kalır.
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

    def run_once(
        self, *, backfill_days: int | None = None, stop_event: threading.Event | None = None
    ) -> ScrapeStats:
        """Tek kazıma turu. ``backfill_days`` verilmezse ``settings.backfill_days`` kullanılır.

        ``stop_event`` set edildiğinde kaynak ve bağlantı döngüleri hemen bırakılır; kısmi istatistik döner.
        """
        self._ensure_topology()
        stats = ScrapeStats()

        def stop_requested() -> bool:
            if stop_event is not None and stop_event.is_set():
                if not stats.interrupted:
                    stats.interrupted = True
                    log.warning("Durdurma sinyali alındı; kazıma turu yarıda kesiliyor")
                return True
            return False

        days = self.settings.backfill_days if backfill_days is None else max(0, int(backfill_days))
        budget = max(0, int(self.settings.max_articles_per_run))
        self._progress_started = stats.started_at
        self._last_progress_write = 0.0
        self._report_progress(stats, fraction=0.0, source="", phase="başlıyor", force=True)
        try:
            self._run_sources(stats, days=days, budget=budget, stop_requested=stop_requested)
        finally:
            self._report_progress(stats, fraction=1.0, source="", phase="bitti", running=False, force=True)
        stats.finished_at = utcnow()
        stats.pending_after = self.seen.pending_count()
        try:
            self.seen.record_run(stats)
        except Exception:  # pragma: no cover - yalnızca durum kaydı
            log.exception("Kazıma turu kaydedilemedi")
        if stats.pending_after:
            log.warning("%d haber bu turda çekilemedi; sonraki turda yeniden denenecek", stats.pending_after)
        return stats

    # --- canlı ilerleme (arayüzdeki ilerleme çubuğu için) ---
    def _report_progress(
        self,
        stats: ScrapeStats,
        *,
        fraction: float,
        source: str,
        phase: str,
        running: bool = True,
        force: bool = False,
    ) -> None:
        """Tur ilerlemesini durum veritabanına yazar (en çok saniyede bir; faz değişince hemen)."""
        now = time.monotonic()
        if not force and now - getattr(self, "_last_progress_write", 0.0) < 1.0:
            return
        self._last_progress_write = now
        state = {
            "running": running,
            "started_at": getattr(self, "_progress_started", stats.started_at).isoformat(),
            "updated_at": utcnow().isoformat(),
            "fraction": round(max(0.0, min(1.0, fraction)), 4),
            "source": source,
            "phase": phase,
            "published": stats.published,
            "fetched": stats.fetched,
            "errors": stats.errors,
            "sources": len(self.sources),
        }
        try:
            self.seen.set_meta("run_state", json.dumps(state, ensure_ascii=False))
        except Exception:  # pragma: no cover - ilerleme kaydı taramayı durdurmamalı
            log.debug("İlerleme kaydedilemedi", exc_info=True)

    def _run_sources(self, stats: ScrapeStats, *, days: int, budget: int, stop_requested) -> None:
        attempts = 0
        n_sources = max(1, len(self.sources))
        for index, source in enumerate(self.sources):
            if stop_requested():
                break
            source_stats = stats.source(source.name)
            remaining_sources = len(self.sources) - index
            quota = math.ceil((budget - attempts) / remaining_sources) if budget > attempts else 0
            if quota <= 0:
                log.warning("Haber bütçesi (%d) doldu; %s bu turda taranmadı", budget, source.name)
                stats.budget_exhausted = True
                continue
            source_attempts = 0
            self._report_progress(stats, fraction=index / n_sources, source=source.name, phase="keşif", force=True)
            try:
                links = source.discover(self.client, backfill_days=days)
            except Exception:
                log.exception("%s keşfi başarısız", source.name)
                source_stats.errors += 1
                links = []
            source_stats.discovered = len(links)
            pending_ids = {row["id"] for row in self.seen.list_pending(source=source.name)}
            pending_links = self._pending_links(source.name, exclude={link.id for link in links})
            if pending_ids:
                log.info("%s: %d bekleyen bağlantı yeniden denenecek", source.name, len(pending_ids))
            # Yeni keşfedilen haberler önce: birikmiş bekleyen liste güncel haberleri ve alarmları geciktirmesin.
            links = links + pending_links
            log.info("%s: %d bağlantı keşfedildi", source.name, source_stats.discovered)
            self._report_progress(stats, fraction=index / n_sources, source=source.name, phase="çekme", force=True)
            for index_link, link in enumerate(links):
                within = max(index_link / max(1, len(links)), source_attempts / max(1, quota))
                self._report_progress(stats, fraction=(index + min(1.0, within)) / n_sources, source=source.name, phase="çekme")
                if stop_requested():
                    self._defer(source.name, links[index_link:], "durdurma sinyali: tur yarıda kesildi")
                    break
                if source_attempts >= quota:
                    log.warning(
                        "Haber bütçesi (%d, %s payı %d) doldu; %s için kalan bağlantılar sonraki tura kaldı",
                        budget,
                        source.name,
                        quota,
                        source.name,
                    )
                    stats.budget_exhausted = True
                    self._defer(source.name, links[index_link:], "haber bütçesi doldu")
                    break
                feed_stamp = link.updated_hint or link.published_hint
                known_date = link.published_hint
                if known_date is None and (row := self.seen.get(link.id)) and row.get("published_at"):
                    known_date = datetime.fromisoformat(row["published_at"])  # daha önce çekilip tarihi öğrenilmiş
                if self._too_old(known_date, days):
                    source_stats.too_old += 1
                    if link.id in pending_ids:
                        self.seen.remove_pending(link.id)
                    continue
                if link.id not in pending_ids and self.seen.seen_recently(
                    link.id, published_at=feed_stamp, within_hours=self._skip_window_hours(link)
                ):
                    source_stats.unchanged += 1
                    continue
                attempts += 1
                source_attempts += 1
                if link.id in pending_ids:
                    stats.retried += 1
                outcome = self._process_link(source, link, source_stats, feed_stamp, days=days)
                if outcome == "error":
                    self._remember_failure(source.name, link, "çekme hatası")
                else:
                    self.seen.remove_pending(link.id)

    # --- bekleyen bağlantılar ---
    def _pending_links(self, source_name: str, *, exclude: set[str]) -> list[DiscoveredLink]:
        links: list[DiscoveredLink] = []
        for row in self.seen.list_pending(source=source_name):
            if row["id"] in exclude:
                continue
            published = datetime.fromisoformat(row["published_at"]) if row.get("published_at") else None
            links.append(
                DiscoveredLink(
                    url=row["url"], title_hint=row.get("title") or "", published_hint=published,
                    category_hint=row.get("category") or "", origin="pending",
                )
            )
        return links

    def _remember_failure(self, source_name: str, link: DiscoveredLink, reason: str) -> None:
        attempts = self.seen.add_pending(
            id=link.id, url=link.url, source=source_name, reason=reason, title=link.title_hint,
            category=link.category_hint, published_at=link.published_hint,
        )
        if attempts >= MAX_PENDING_ATTEMPTS:
            self.seen.remove_pending(link.id)
            log.error("%s haberi %d denemede alınamadı, listeden düşürüldü: %s", source_name, attempts, link.url)

    def _defer(self, source_name: str, links: list[DiscoveredLink], reason: str) -> None:
        """Bu turda sıra gelmeyen bağlantıları (daha önce görülmemişse) bekleyen listesine yazar."""
        deferred = 0
        room = MAX_PENDING_BACKLOG - self.seen.pending_count()
        for link in links:
            if self.seen.get(link.id) is not None:
                continue  # daha önce alınmış haber: yalnızca yeniden kontrol sırası geldi, "çekilemedi" değil
            if room <= 0:
                log.warning("Bekleyen liste dolu (%d); kalan bağlantılar bir sonraki keşifte yeniden bulunacak", MAX_PENDING_BACKLOG)
                break
            room -= 1
            self.seen.add_pending(
                id=link.id, url=link.url, source=source_name, reason=reason, title=link.title_hint,
                category=link.category_hint, published_at=link.published_hint, count_attempt=False,
            )
            deferred += 1
        if deferred:
            log.info("%s: %d bağlantı bekleyen listesine alındı (%s)", source_name, deferred, reason)

    def _skip_window_hours(self, link: DiscoveredLink) -> float:
        """Bağlantıyı sayfasını çekmeden atlama penceresi (saat).

        Besleme güncelleme damgası taşıyan bağlantılarda (Hürriyet RSS ``<modified>``) damga değişmediği sürece
        depo penceresi geçerlidir; damgasız bağlantılarda değişiklik yalnızca sayfadan anlaşılabildiğinden pencere
        ``RECHECK_HOURS_WITHOUT_UPDATE_STAMP`` ile sınırlanır (depo penceresi daha kısaysa o korunur).
        """
        if link.updated_hint is not None:
            return self.seen.recent_hours
        return min(self.seen.recent_hours, RECHECK_HOURS_WITHOUT_UPDATE_STAMP)

    def _too_old(self, published: datetime | None, backfill_days: int = 0) -> bool:
        """Haber ``MAX_ARTICLE_AGE_DAYS``'ten (geriye dönük taramada ``backfill_days``'ten) eski mi? Tarih yoksa hayır."""
        limit = int(self.settings.max_article_age_days)
        if limit <= 0 or published is None:
            return False
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        return published < utcnow() - timedelta(days=max(limit, int(backfill_days)))

    def _process_link(
        self, source: Source, link: DiscoveredLink, source_stats: SourceStats, feed_stamp: datetime | None, *, days: int = 0
    ) -> str:
        """Bağlantıyı çeker ve yayınlar; sonuç: "published" | "unchanged" | "skipped" | "error"."""
        try:
            record = source.fetch_article(self.client, link)
        except (HttpError, requests.RequestException) as exc:
            source_stats.errors += 1
            log.warning("%s haberi alınamadı (%s): %s", source.name, link.url, exc)
            return "error"
        except Exception:
            source_stats.errors += 1
            log.exception("%s haberi işlenirken beklenmeyen hata (%s)", source.name, link.url)
            return "error"
        if record is None:
            source_stats.skipped += 1
            return "skipped"
        source_stats.fetched += 1
        if self._too_old(record.published_at, days):
            # Beslemede tarih yoktu ya da yanlıştı; sayfadaki tarih eski. Tekrar çekilmesin diye görüldü sayılır.
            source_stats.too_old += 1
            self.seen.mark(record, published_at=feed_stamp)
            return "skipped"
        status = self.seen.status(record.id, record.content_hash)
        if status == "unchanged":
            source_stats.unchanged += 1
            self.seen.mark(record, published_at=feed_stamp)
            return "unchanged"
        record.stage = Stage.RAW
        self.broker.publish(
            RoutingKey.ARTICLE_RAW,
            record.to_message(),
            headers={"x-source": source.name, "x-change": status},
        )
        self.seen.mark(record, published_at=feed_stamp)
        source_stats.published += 1
        log.info("%s %s yayınlandı: %s", source.name, "güncellendi" if status == "updated" else "yeni", record.title)
        return "published"

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        """``stop_event`` set edilene kadar ``scrape_interval_seconds`` aralıklarla tur atar."""
        stop_event = stop_event or threading.Event()
        interval = max(1, int(self.settings.scrape_interval_seconds))
        backfill_days = max(0, int(self.settings.backfill_days))
        log.info(
            "Kazıyıcı başladı: kaynaklar=%s, aralık=%ds, geriye dönük tarama=%d gün (yalnızca ilk turda)",
            [s.name for s in self.sources],
            interval,
            backfill_days,
        )
        self.seen.set_meta("interval_seconds", str(interval))
        while not stop_event.is_set():
            try:
                stats = self.run_once(backfill_days=backfill_days, stop_event=stop_event)
                log.info(stats.summary())
                backfill_days = 0  # arşiv taraması tamamlandı; sonraki turlar yalnızca güncel beslemeleri tarar
            except Exception:
                log.exception("Kazıma turu başarısız; %ds sonra yeniden denenecek", interval)
            self.seen.set_meta("next_run_at", (utcnow() + timedelta(seconds=interval)).isoformat())
            if stop_event.wait(interval):
                break
        self.seen.set_meta("next_run_at", "")
        log.info("Kazıyıcı durduruldu")

    def close(self) -> None:
        self.client.close()
        self.seen.close()
