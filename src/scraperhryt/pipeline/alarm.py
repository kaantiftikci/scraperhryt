"""Alarm katmanı: q.articles.scored → Elasticsearch (her kayıt) → alarm kanalları + q.alarms (alarm olanlar).

Her mesaj için (BUILD_SPEC §C):

1. ``NewsRecord`` çözümlenir (bozuk mesaj → ``Reject`` → ölü mektup), ``stage=alarm`` yapılır.
2. Alarm değilse kayıt ``news-articles``'a yazılır; iş biter (q.alarms'a hiçbir şey gitmez).
3. Alarmsa ``AlarmEvent`` kurulur, ``alarm_id``/``alarmed_at`` kayda işlenir ve kayıt yazılır; kanallar
   bilgilendirilir (hatalar loglanır, asla ölümcül değildir); olay ``alarm.raised`` ile yayınlanır ve
   son olarak ``news-alarms``'a yazılır.

Yeniden teslimde (ack kaybı, retry) aynı ``alarm_id`` depoda zaten varsa ve kaydın ``content_hash``'i
değişmemişse kanal bildirimi ve yayın tekrarlanmaz. Alarm belgesi en SON yazılır; böylece yayın başarısız
olursa yeniden denemede olay yeniden yayınlanır (kaybolmaz), bunun bedeli nadiren çift bildirimdir.

Elasticsearch'e ulaşılamadığında depo ``Retry`` fırlatır; broker mesajı gecikmeli yeniden dener.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from pydantic import ValidationError

from ..alarm_sinks import AlarmSink, build_sinks
from ..broker import Broker, Message, Queue, Reject, Retry, RoutingKey
from ..config import Settings
from ..models import AlarmEvent, NewsRecord, Stage, utcnow
from ..store import ArticleStore
from ..textutil import excerpt

log = logging.getLogger(__name__)

_INDEX_RETRY_MAX_DELAY = 30.0


class Embedder(Protocol):
    """``pipeline.llm.LLM`` ile uyumlu en küçük arayüz (``OllamaClient``/``FakeOllama`` doğrudan kullanılabilir)."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class AlarmStats:
    received: int = 0
    indexed: int = 0
    alarms_raised: int = 0
    duplicates: int = 0  # yeniden teslim edilen, zaten yükseltilmiş alarmlar
    notify_failures: int = 0
    embed_failures: int = 0
    rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class AlarmService:
    """``Queue.ARTICLES_SCORED`` tüketicisi. ``sinks=None`` → ``build_sinks(settings)``; kanal istenmiyorsa ``[]`` verin.

    ``embedder`` verilir ve ``settings.ollama_embedding_model`` doluysa her kayıt için vektör üretilip
    ``index_record(embedding=...)`` ile yazılır (RAG kNN için); vektör üretimi başarısız olursa kayıt
    vektörsüz yazılır.
    """

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        store: ArticleStore,
        sinks: Sequence[AlarmSink] | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.store = store
        self.sinks: list[AlarmSink] = list(sinks) if sinks is not None else build_sinks(settings)
        self.embedder = embedder if settings.ollama_embedding_model else None
        self.stats = AlarmStats()
        if embedder is not None and self.embedder is None:
            log.info("embedder verildi ama OLLAMA_EMBEDDING_MODEL boş; vektör üretimi kapalı")

    # --- yardımcılar ---
    def _embedding_for(self, record: NewsRecord) -> list[float] | None:
        if self.embedder is None:
            return None
        body = (record.content or "")[: self.settings.ollama_max_content_chars]
        text = "\n".join(part for part in (record.title, record.subtitle, body) if part)
        try:
            vectors = self.embedder.embed([text])
        except Exception as exc:
            self.stats.embed_failures += 1
            log.warning("Embedding üretilemedi (kayıt %s), vektörsüz yazılacak: %s", record.id, exc)
            return None
        if not vectors or not vectors[0]:
            self.stats.embed_failures += 1
            log.warning("Embedding boş döndü (kayıt %s), vektörsüz yazılacak", record.id)
            return None
        return [float(x) for x in vectors[0]]

    def _already_raised(self, record: NewsRecord, alarm_id: str) -> bool:
        existing = self.store.get_alarm(alarm_id)
        if existing is None or existing.get("record_id") != record.id:
            return False
        stored = self.store.get_record(record.id)
        return stored is None or stored.get("content_hash") == record.content_hash

    def _notify(self, event: AlarmEvent) -> list[str]:
        notified: list[str] = []
        for sink in self.sinks:
            try:
                sink.send(event)
                notified.append(sink.name)
            except Exception as exc:
                self.stats.notify_failures += 1
                log.error("Alarm kanalı başarısız (%s, alarm %s): %s", sink.name, event.alarm_id, exc)
        return notified

    # --- mesaj işleme ---
    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            record = NewsRecord.from_message(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz haber mesajı (article.scored): {excerpt(str(exc), 300)}") from exc

        record.stage = Stage.ALARM
        if record.processed_at is None:
            record.processed_at = utcnow()
        title = excerpt(record.title, 80)

        if not record.is_alarm:
            self.store.index_record(record, embedding=self._embedding_for(record))
            self.stats.indexed += 1
            log.info("Kayıt indekslendi, alarm yok [%s] skor=%d %s", record.source, record.alarm_score, title)
            return

        event = AlarmEvent.from_record(record)
        if self._already_raised(record, event.alarm_id):
            self.stats.duplicates += 1
            log.info("Alarm zaten yükseltilmiş, yeniden teslim atlandı [%s] alarm=%s %s", record.source, event.alarm_id, title)
            return

        record.alarm_id = event.alarm_id
        record.alarmed_at = event.raised_at
        event.record = record
        self.store.index_record(record, embedding=self._embedding_for(record))
        self.stats.indexed += 1

        event.channels_notified = self._notify(event)
        self.broker.publish(RoutingKey.ALARM_RAISED, event.to_message())
        self.store.index_alarm(event)
        self.stats.alarms_raised += 1
        log.warning(
            "Alarm yükseltildi [%s] skor=%d alarm=%s kanallar=%s → %s | %s",
            record.source,
            record.alarm_score,
            event.alarm_id,
            ",".join(event.channels_notified) or "-",
            Queue.ALARMS,
            title,
        )

    # --- servis döngüsü ---
    def _prepare_indices(self, stop_event: threading.Event) -> bool:
        delay = 1.0
        while not stop_event.is_set():
            try:
                self.store.ensure_indices()
                return True
            except Retry as exc:
                log.warning("Elasticsearch indeksleri hazırlanamadı: %s; %.0fs sonra tekrar denenecek", exc, delay)
                if stop_event.wait(delay):
                    break
                delay = min(delay * 2, _INDEX_RETRY_MAX_DELAY)
        return False

    def run(self, stop_event: threading.Event | None = None, max_messages: int | None = None) -> int:
        """``q.articles.scored`` kuyruğunu tüketir; işlenen mesaj sayısını döndürür."""
        stop_event = stop_event or threading.Event()
        self.broker.declare_topology()
        if not self._prepare_indices(stop_event):
            log.info("Alarm katmanı indeksler hazırlanamadan durduruldu")
            return 0
        log.info(
            "Alarm katmanı başlıyor: kuyruk=%s, eşik=%d, kanallar=%s, embedding=%s",
            Queue.ARTICLES_SCORED,
            self.settings.alarm_threshold,
            ", ".join(s.name for s in self.sinks) or "-",
            "açık" if self.embedder is not None else "kapalı",
        )
        processed = self.broker.consume(
            Queue.ARTICLES_SCORED,
            self.handle,
            prefetch=self.settings.rabbitmq_prefetch,
            stop_event=stop_event,
            max_messages=max_messages,
        )
        log.info("Alarm katmanı durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed
