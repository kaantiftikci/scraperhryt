"""Raporlama servisleri: ``q.alarms`` tüketicisi (alarm özetleri) ve periyodik rapor üretici.

**ReportingConsumer** gelen ``AlarmEvent``'leri bellek içi tamponda biriktirir; tamponda
``report_digest_every`` alarm birikince **veya** tampondaki en eski alarm ``report_digest_minutes`` dakikadır
bekliyorsa ``alarm_digest`` raporu kurar, ``news-reports``'a yazar, ``report.alarm_digest`` ile yayınlar ve
tamponu boşaltır. Teslimat "en az bir kez" olduğundan ``alarm_id`` ile tekilleştirme yapılır.

Sayı koşulu her gelen mesajda değerlendirilir. Süre koşulu için ``run()`` tüketim döngüsünü kendi uyandırma
olayıyla sürer: küçük bir gözcü iş parçacığı süre dolunca ``broker.consume``'u durdurur, özet tüketici iş
parçacığında üretilip yayınlanır (pika bağlantısı iş parçacığı güvenli olmadığından ``broker.publish``
tüketici iş parçacığında kalır) ve tüketim kaldığı yerden sürer. Çıkışta tampon yine boşaltılır.

Tampon bellek içi olduğundan ``run()`` başlarken ``news-alarms``'taki, henüz hiçbir özete girmemiş yakın
tarihli alarmlar tampona geri alınır (``recover_pending``); böylece çökme/yeniden başlatma sonrası alarmlar
özetten düşmez ve daha önce özetlenmiş alarmların yeniden teslimi ikinci kez özetlenmez.

**PeriodicReporter** her ``report_interval_minutes`` dakikada son ``report_window_hours`` saat için ``periodic``
raporu üretir, yazar ve ``report.generated`` ile yayınlar.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, RoutingKey
from ..config import Settings
from ..models import AlarmEvent, Report, utcnow
from ..store import ArticleStore
from ..textutil import excerpt
from .builder import ReportBuilder
from .prompts import format_tr, parse_datetime, to_aware

log = logging.getLogger(__name__)

#: Özetlenmiş alarm kimliklerinden hatırlanan son N tanesi (özet sonrası yeniden teslimleri elemek için).
RECENT_IDS_LIMIT = 2000
#: Süre eşiği gözcüsünün tamponu yoklama aralığı (saniye); dış ``stop_event`` de bu aralıkla fark edilir.
DIGEST_POLL_SECONDS = 1.0
#: Periyodik raporda art arda hata olduğunda en fazla bu kadar bekle (dakika cinsinden aralık yine geçerlidir).
_PERIODIC_RETRY_SECONDS = 60.0
#: Süre eşiğiyle tetiklenen özet üretilemezse (depo/broker hatası) yeniden denemeden önce bekleme (saniye).
_DIGEST_RETRY_SECONDS = 60.0
#: Yeniden başlatmada özetlenmemiş alarmları bulmak için depodan çekilen alarm / özet raporu sayısı.
_RECOVERY_ALARMS_FETCH = 500
_RECOVERY_REPORTS_FETCH = 500


@dataclass
class ReportingStats:
    received: int = 0
    buffered: int = 0
    recovered: int = 0
    duplicates: int = 0
    rejected: int = 0
    digests: int = 0
    periodic_reports: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BufferedAlarm:
    """Tampondaki bir alarm: kimlik, yükseltilme anı ve ``news-alarms`` biçimindeki belgesi."""

    alarm_id: str
    raised_at: datetime
    doc: dict[str, Any]


class ReportingConsumer:
    """``Queue.ALARMS`` tüketicisi: alarmları tamponlar ve ``alarm_digest`` raporları üretir."""

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        store: ArticleStore,
        builder: ReportBuilder,
        *,
        poll_seconds: float = DIGEST_POLL_SECONDS,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.store = store
        self.builder = builder
        self.poll_seconds = max(0.01, float(poll_seconds))
        self.buffer: list[BufferedAlarm] = []
        self.last_digest_at: datetime = utcnow()
        #: Tampondaki en eski alarmın tampona girdiği an; tampon boşken ``None``. Süre eşiği buna göre ölçülür.
        self.batch_opened_at: datetime | None = None
        self.stats = ReportingStats()
        self._lock = threading.RLock()
        self._recent_ids: OrderedDict[str, None] = OrderedDict()
        self._retry_not_before = 0.0  # time.monotonic(); başarısız zamanlı özetten sonra bekleme sınırı

    # --- durum ---
    @property
    def buffered(self) -> int:
        with self._lock:
            return len(self.buffer)

    def is_digest_due(self, now: datetime | None = None) -> bool:
        """Tampon sayı eşiğine ulaştı mı ya da tampondaki en eski alarm süre eşiğinden uzun süredir bekliyor mu?"""
        now = now or utcnow()
        with self._lock:
            count = len(self.buffer)
            if count == 0 or self.batch_opened_at is None:
                return False
            if count >= max(1, int(self.settings.report_digest_every)):
                return True
            waited = now - self.batch_opened_at
            return waited >= timedelta(minutes=max(0, int(self.settings.report_digest_minutes)))

    # --- mesaj işleme ---
    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            event = AlarmEvent.model_validate(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz alarm mesajı (alarm.raised): {excerpt(str(exc), 300)}") from exc

        if self._buffer(event.alarm_id, to_aware(event.raised_at), event.to_es_document()):
            log.info(
                "Alarm tampona alındı (%d/%d) [%s] skor=%d %s",
                self.buffered,
                self.settings.report_digest_every,
                event.source,
                event.alarm_score,
                excerpt(event.title, 80),
            )
        else:
            log.info(
                "Alarm zaten tamponda ya da özetlenmiş (yeniden teslim / yeniden başlatma kurtarması), "
                "tampona eklenmedi: %s",
                event.alarm_id,
            )
        self.flush_if_due()

    def _buffer(self, alarm_id: str, raised_at: datetime, doc: dict[str, Any]) -> bool:
        """Alarmı (daha önce özetlenmemiş ve tamponda yoksa) tampona ekler; eklendiyse ``True``."""
        with self._lock:
            if alarm_id in self._recent_ids or any(b.alarm_id == alarm_id for b in self.buffer):
                self.stats.duplicates += 1
                return False
            if not self.buffer:
                self.batch_opened_at = utcnow()
            self.buffer.append(BufferedAlarm(alarm_id=alarm_id, raised_at=raised_at, doc=doc))
            self.stats.buffered += 1
            return True

    def recover_pending(self, *, now: datetime | None = None) -> int:
        """Depodaki son ``report_window_hours`` saatin henüz hiçbir ``alarm_digest``'e girmemiş alarmlarını
        tampona geri alır; özetlenmiş kimlikleri yeniden teslim elemesi için hatırlar. Eklenen alarm sayısını döndürür.

        Tampon süreç belleğinde olduğundan çökme/yeniden başlatma sonrası özetlenmemiş alarmlar ancak böyle
        kurtarılır. Kuyrukta hâlâ bekleyen aynı alarmlar geldiğinde ``alarm_id`` ile elenir.
        """
        now = now or utcnow()
        since = now - timedelta(hours=max(1, int(self.settings.report_window_hours)))
        digests = self.store.list_reports(kind="alarm_digest", size=_RECOVERY_REPORTS_FETCH)
        digested: set[str] = set()
        for report in digests:
            top = report.get("top_alarms") if isinstance(report, Mapping) else None
            for alarm in top if isinstance(top, list) else []:
                alarm_id = str(alarm.get("alarm_id") or "") if isinstance(alarm, Mapping) else ""
                if alarm_id:
                    digested.add(alarm_id)
        fetched = self.store.recent_alarms(since=since, size=_RECOVERY_ALARMS_FETCH)
        candidates: list[BufferedAlarm] = []
        for doc in fetched:
            alarm_id = str(doc.get("alarm_id") or "")
            if not alarm_id or alarm_id in digested:
                continue
            raised_at = parse_datetime(doc.get("raised_at")) or since
            candidates.append(BufferedAlarm(alarm_id=alarm_id, raised_at=raised_at, doc=dict(doc)))
        candidates.sort(key=lambda b: (b.raised_at, b.alarm_id))

        added = 0
        with self._lock:
            for alarm_id in sorted(digested):
                self._remember(alarm_id)
            for candidate in candidates:
                if self._buffer(candidate.alarm_id, candidate.raised_at, candidate.doc):
                    added += 1
            self.stats.recovered += added
        log.info(
            "Yeniden başlatma kontrolü: %d özet raporunda %d özetlenmiş alarm; %s sonrası %d alarmdan %d tanesi "
            "henüz özetlenmemiş, tampona geri alındı",
            len(digests),
            len(digested),
            format_tr(since),
            len(fetched),
            added,
        )
        return added

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
            raised = [b.raised_at for b in events]
            window_start, window_end = min(raised), max(raised)
            try:
                report = self.builder.build(
                    "alarm_digest",
                    window_start,
                    window_end,
                    top_alarms=[b.doc for b in events],
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
            self.batch_opened_at = None
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
        """``q.alarms`` kuyruğunu tüketir; süre eşiği dolunca ve çıkışta tamponu boşaltır. İşlenen mesaj sayısını döndürür.

        Bellek içi broker kuyruk boşalınca döner; o durumda (``stop_event`` set edilmemiş olsa da) döngü biter.
        """
        stop_event = stop_event or threading.Event()
        self.broker.declare_topology()
        log.info(
            "Raporlama tüketicisi başlıyor: kuyruk=%s, özet eşiği=%d alarm / %d dakika",
            Queue.ALARMS,
            self.settings.report_digest_every,
            self.settings.report_digest_minutes,
        )
        try:
            self.recover_pending()
        except Exception as exc:
            self.stats.failures += 1
            log.error("Özetlenmemiş alarmlar depodan geri alınamadı; yalnızca yeni alarmlar özetlenecek: %s", exc)
        processed = 0
        try:
            while not stop_event.is_set():
                remaining = None if max_messages is None else max_messages - processed
                if remaining is not None and remaining <= 0:
                    break
                count, timed_out = self._consume_once(stop_event, remaining)
                processed += count
                if timed_out:
                    self._timed_flush()
                elif not stop_event.is_set():
                    break  # broker kendiliğinden döndü (bellek içi kuyruk boşaldı / max_messages doldu)
        finally:
            try:
                if self.flush() is not None:
                    log.info("Çıkışta bekleyen alarmlar özetlendi")
            except Exception as exc:
                log.error("Çıkışta alarm özeti üretilemedi (%d alarm kaybedildi): %s", self.buffered, exc)
        log.info("Raporlama tüketicisi durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed

    def _consume_once(self, stop_event: threading.Event, max_messages: int | None) -> tuple[int, bool]:
        """Bir tüketim turu: süre eşiği dolunca ya da ``stop_event`` set edilince ``consume``'u durdurur.

        ``(işlenen mesaj sayısı, süre eşiği nedeniyle durdu mu)`` döndürür.
        """
        wake = threading.Event()
        timed_out = threading.Event()

        def watch() -> None:
            while not wake.is_set() and not stop_event.is_set():
                if time.monotonic() >= self._retry_not_before and self.is_digest_due():
                    timed_out.set()
                    break
                wake.wait(self.poll_seconds)
            wake.set()

        watcher = threading.Thread(target=watch, name="report-digest-watch", daemon=True)
        watcher.start()
        try:
            processed = self.broker.consume(
                Queue.ALARMS,
                self.handle,
                prefetch=self.settings.rabbitmq_prefetch,
                stop_event=wake,
                max_messages=max_messages,
            )
        finally:
            wake.set()
            watcher.join()
        return processed, timed_out.is_set()

    def _timed_flush(self) -> None:
        """Süre eşiği dolunca tüketici iş parçacığında özet üretir; hata loglanır ve bir süre yeniden denenmez."""
        try:
            report = self.flush_if_due()
        except Exception as exc:
            self._retry_not_before = time.monotonic() + _DIGEST_RETRY_SECONDS
            log.error(
                "Süre eşiğiyle tetiklenen alarm özeti üretilemedi (%d alarm tamponda kalıyor); %.0fs sonra yeniden "
                "denenecek: %s",
                self.buffered,
                _DIGEST_RETRY_SECONDS,
                exc,
            )
            return
        if report is not None:
            log.info("Süre eşiği (%d dk) doldu, bekleyen alarmlar özetlendi", self.settings.report_digest_minutes)


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
