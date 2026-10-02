"""Raporlama servisleri: ``q.alarms`` tüketicisi (alarm özetleri) ve periyodik rapor üretici.

**ReportingConsumer** gelen ``AlarmEvent``'leri bellek içi tamponda biriktirir; tamponda
``report_digest_every`` alarm birikince **veya** son özetten bu yana ``report_digest_minutes`` geçmişse (ve
tampon boş değilse) ``alarm_digest`` raporu kurar, ``news-reports``'a yazar, ``report.alarm_digest`` ile yayınlar
ve tamponu boşaltır. Teslimat "en az bir kez" olduğundan ``alarm_id`` ile tekilleştirme yapılır.

Zaman koşulu her gelen mesajda ve ``run()`` çıkışında değerlendirilir; ayrı bir zamanlayıcı iş parçacığı
kullanılmaz çünkü pika bağlantısı iş parçacığı güvenli değildir (``broker.publish`` tüketici iş parçacığında
kalmalıdır). Tüketiciyi kendi döngüsünde süren bir sahip ``flush_if_due()``'yu periyodik olarak çağırabilir.

**PeriodicReporter** her ``report_interval_minutes`` dakikada son ``report_window_hours`` saat için ``periodic``
raporu üretir, yazar ve ``report.generated`` ile yayınlar.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, RoutingKey
from ..config import Settings
from ..models import AlarmEvent, Report, utcnow
from ..store import ArticleStore
from ..textutil import excerpt
from .builder import ReportBuilder
from .prompts import format_tr, to_aware

log = logging.getLogger(__name__)

#: Özetlenmiş alarm kimliklerinden hatırlanan son N tanesi (özet sonrası yeniden teslimleri elemek için).
RECENT_IDS_LIMIT = 2000
#: Periyodik raporda art arda hata olduğunda en fazla bu kadar bekle (dakika cinsinden aralık yine geçerlidir).
_PERIODIC_RETRY_SECONDS = 60.0


@dataclass
class ReportingStats:
    received: int = 0
    buffered: int = 0
    duplicates: int = 0
    rejected: int = 0
    digests: int = 0
    periodic_reports: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ReportingConsumer:
    """``Queue.ALARMS`` tüketicisi: alarmları tamponlar ve ``alarm_digest`` raporları üretir."""

    def __init__(self, settings: Settings, broker: Broker, store: ArticleStore, builder: ReportBuilder) -> None:
        self.settings = settings
        self.broker = broker
        self.store = store
        self.builder = builder
        self.buffer: list[AlarmEvent] = []
        self.last_digest_at: datetime = utcnow()
        self.stats = ReportingStats()
        self._lock = threading.RLock()
        self._recent_ids: OrderedDict[str, None] = OrderedDict()

    # --- durum ---
    @property
    def buffered(self) -> int:
        with self._lock:
            return len(self.buffer)

    def is_digest_due(self, now: datetime | None = None) -> bool:
        """Tampon sayı eşiğine ulaştı mı ya da (tampon doluyken) süre eşiği aşıldı mı?"""
        now = now or utcnow()
        with self._lock:
            count = len(self.buffer)
            if count == 0:
                return False
            if count >= max(1, int(self.settings.report_digest_every)):
                return True
            elapsed = now - self.last_digest_at
            return elapsed >= timedelta(minutes=max(0, int(self.settings.report_digest_minutes)))

    # --- mesaj işleme ---
    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            event = AlarmEvent.model_validate(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz alarm mesajı (alarm.raised): {excerpt(str(exc), 300)}") from exc

        with self._lock:
            if event.alarm_id in self._recent_ids or any(e.alarm_id == event.alarm_id for e in self.buffer):
                self.stats.duplicates += 1
                log.info("Alarm yeniden teslim edildi, tampona eklenmedi: %s", event.alarm_id)
            else:
                self.buffer.append(event)
                self.stats.buffered += 1
                log.info(
                    "Alarm tampona alındı (%d/%d) [%s] skor=%d %s",
                    len(self.buffer),
                    self.settings.report_digest_every,
                    event.source,
                    event.alarm_score,
                    excerpt(event.title, 80),
                )
        self.flush_if_due()

    def flush_if_due(self, now: datetime | None = None) -> Report | None:
        """Koşul sağlanıyorsa özet üretir; aksi halde ``None``."""
        return self.flush() if self.is_digest_due(now) else None

    def flush(self) -> Report | None:
        """Tampondaki alarmlardan ``alarm_digest`` raporu kurar, yazar, yayınlar ve tamponu boşaltır.

        Hata durumunda (depo/broker) istisna yükselir ve tampon korunur; çağıran (broker) mesajı yeniden dener.
        Tampon boşsa ``None`` döner.
        """
        with self._lock:
            if not self.buffer:
                return None
            events = list(self.buffer)
            raised = [to_aware(e.raised_at) for e in events]
            window_start, window_end = min(raised), max(raised)
            try:
                report = self.builder.build(
                    "alarm_digest",
                    window_start,
                    window_end,
                    top_alarms=[e.to_es_document() for e in events],
                )
                self.store.index_report(report)
                self.broker.publish(RoutingKey.REPORT_ALARM_DIGEST, report.to_message())
            except Exception:
                self.stats.failures += 1
                log.exception("Alarm özeti üretilemedi; %d alarm tamponda tutuluyor", len(events))
                raise
            for event in events:
                self._remember(event.alarm_id)
            self.buffer.clear()
            self.last_digest_at = utcnow()
            self.stats.digests += 1
        log.info(
            "Alarm özeti yayınlandı: %s (%d alarm, pencere %s – %s) → %s",
            report.report_id,
            len(events),
            format_tr(window_start),
            format_tr(window_end),
            Queue.REPORTS,
        )
        return report

    def _remember(self, alarm_id: str) -> None:
        self._recent_ids[alarm_id] = None
        self._recent_ids.move_to_end(alarm_id)
        while len(self._recent_ids) > RECENT_IDS_LIMIT:
            self._recent_ids.popitem(last=False)

    # --- servis döngüsü ---
    def run(self, stop_event: threading.Event | None = None, max_messages: int | None = None) -> int:
        """``q.alarms`` kuyruğunu tüketir; çıkışta tamponu boşaltır. İşlenen mesaj sayısını döndürür."""
        stop_event = stop_event or threading.Event()
        self.broker.declare_topology()
        log.info(
            "Raporlama tüketicisi başlıyor: kuyruk=%s, özet eşiği=%d alarm / %d dakika",
            Queue.ALARMS,
            self.settings.report_digest_every,
            self.settings.report_digest_minutes,
        )
        processed = 0
        try:
            processed = self.broker.consume(
                Queue.ALARMS,
                self.handle,
                prefetch=self.settings.rabbitmq_prefetch,
                stop_event=stop_event,
                max_messages=max_messages,
            )
        finally:
            try:
                if self.flush() is not None:
                    log.info("Çıkışta bekleyen alarmlar özetlendi")
            except Exception as exc:
                log.error("Çıkışta alarm özeti üretilemedi (%d alarm kaybedildi): %s", self.buffered, exc)
        log.info("Raporlama tüketicisi durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed


class PeriodicReporter:
    """Her ``report_interval_minutes`` dakikada son ``report_window_hours`` saat için ``periodic`` raporu üretir."""

    def __init__(self, settings: Settings, store: ArticleStore, builder: ReportBuilder, broker: Broker) -> None:
        self.settings = settings
        self.store = store
        self.builder = builder
        self.broker = broker
        self.stats = ReportingStats()
        self.last_report: Report | None = None

    def run_once(self, *, now: datetime | None = None) -> Report:
        """Tek rapor: kur, ``news-reports``'a yaz, ``report.generated`` ile yayınla ve döndür."""
        window_end = now or utcnow()
        window_start = window_end - timedelta(hours=max(1, int(self.settings.report_window_hours)))
        report = self.builder.build("periodic", window_start, window_end)
        self.store.index_report(report)
        self.broker.publish(RoutingKey.REPORT_GENERATED, report.to_message())
        self.stats.periodic_reports += 1
        self.last_report = report
        log.info(
            "Periyodik rapor yayınlandı: %s (pencere %s – %s, haber=%s, alarm=%s) → %s",
            report.report_id,
            format_tr(window_start),
            format_tr(window_end),
            report.stats.get("total", 0),
            report.stats.get("alarms", 0),
            Queue.REPORTS,
        )
        return report

    def run(self, stop_event: threading.Event | None = None) -> int:
        """İlk raporu hemen, sonrakileri her aralıkta üretir; ``stop_event`` set edilince bekleme hemen biter.

        Geçici hatalar (depo/broker/LLM) loglanır ve bir sonraki denemeye kadar beklenir. Üretilen rapor sayısını
        döndürür.
        """
        stop_event = stop_event or threading.Event()
        self.broker.declare_topology()
        interval = max(1, int(self.settings.report_interval_minutes)) * 60.0
        log.info(
            "Periyodik raporlayıcı başlıyor: her %d dakika, pencere %d saat",
            self.settings.report_interval_minutes,
            self.settings.report_window_hours,
        )
        produced = 0
        while not stop_event.is_set():
            try:
                self.run_once()
                produced += 1
                wait_seconds = interval
            except Exception as exc:
                self.stats.failures += 1
                wait_seconds = min(interval, _PERIODIC_RETRY_SECONDS)
                log.error("Periyodik rapor üretilemedi: %s; %.0fs sonra yeniden denenecek", exc, wait_seconds)
            if stop_event.wait(wait_seconds):
                break
        log.info("Periyodik raporlayıcı durdu: %d rapor üretildi", produced)
        return produced
