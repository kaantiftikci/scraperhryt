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

Elasticsearch'e ulaşılamadığında depo ``Unavailable`` (``Retry`` alt tipi) fırlatır. Servis bu durumda mesajı
hemen geri vermez: önce ``_prepare_indices`` ile ES yeniden erişilebilir olana dek bekler (geri basınç; prefetch
kadar mesaj askıda kalır), sonra istisnayı broker'a iletir. Böylece dakikalar süren bir ES kesintisi
``rabbitmq_max_attempts`` bütçesini tüketmez ve hiçbir kayıt ölü mektuba düşmez.

Embedding modeli ``EMBEDDING_DIMS``'ten farklı boyutta vektör üretiyorsa (ör. mxbai-embed-large=1024, nomic=768)
``dense_vector`` eşlemesi belgeyi reddederdi; bu yüzden boyut başlangıçta (``run``) ve kayıt başına denetlenir,
uyuşmayan vektör atılır ve kayıt vektörsüz yazılır.
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
from .dedup import EventClusterer

log = logging.getLogger(__name__)

_INDEX_RETRY_INITIAL_DELAY = 1.0
_INDEX_RETRY_MAX_DELAY = 30.0
_EMBED_PROBE_TEXT = "Bağlantı denemesi"  # başlangıçta embedding boyutunu doğrulamak için gönderilen metin


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
    suppressed: int = 0  # tekrar (duplicate_of) olduğu için bildirimi bastırılan alarmlar
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
        self.clusterer = EventClusterer(settings, store)
        self._stop_event = threading.Event()
        if embedder is not None and self.embedder is None:
            log.info("embedder verildi ama OLLAMA_EMBEDDING_MODEL boş; vektör üretimi kapalı")

    # --- yardımcılar ---
    def _store_rejects_embeddings(self) -> bool:
        """Depo (``ElasticsearchStore.ensure_indices``/``index_record``) eşleme uyuşmazlığında vektör indekslemeyi
        kapatır; o durumda her kayıt için boşa embedding üretmek yerine üretim burada da kapatılır."""
        if self.embedder is None or getattr(self.store, "embeddings_enabled", True):
            return False
        log.warning(
            "Depo vektör indekslemeyi kapattı (embedding eşlemesi ayarlarla uyuşmuyor); vektör üretimi KAPATILDI, "
            "kayıtlar vektörsüz yazılacak"
        )
        self.embedder = None
        return True

    def _embedding_for(self, record: NewsRecord) -> list[float] | None:
        if self.embedder is None or self._store_rejects_embeddings():
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
        vector = [float(x) for x in vectors[0]]
        expected = int(self.settings.embedding_dims)
        if len(vector) != expected:
            self.stats.embed_failures += 1
            log.error(
                "Embedding boyutu %d, EMBEDDING_DIMS=%d (kayıt %s); vektörsüz yazılacak. EMBEDDING_DIMS'i modele "
                "(%s) göre düzeltin",
                len(vector),
                expected,
                record.id,
                self.settings.ollama_embedding_model,
            )
            return None
        return vector

    def _check_embedder(self) -> None:
        """Başlangıçta embedding modelinin ``EMBEDDING_DIMS`` boyutunda vektör ürettiğini doğrular.

        Uyuşmazlıkta vektör üretimi kapatılır (kayıtlar vektörsüz yazılmaya devam eder; her kayıt ES'e
        ulaşmalıdır). Model o an erişilemiyorsa denetim kayıt başına ``_embedding_for``'a bırakılır.
        """
        if self.embedder is None or self._store_rejects_embeddings():
            return
        expected = int(self.settings.embedding_dims)
        model = self.settings.ollama_embedding_model
        try:
            vectors = self.embedder.embed([_EMBED_PROBE_TEXT])
        except Exception as exc:
            log.warning("Embedding modeli (%s) başlangıçta doğrulanamadı: %s; boyut kayıt başına denetlenecek", model, exc)
            return
        got = len(vectors[0]) if vectors and vectors[0] else 0
        if got == expected:
            log.info("Embedding modeli doğrulandı: %s → %d boyut", model, got)
            return
        log.error(
            "Embedding modeli %s %d boyutlu vektör üretiyor, EMBEDDING_DIMS=%d; vektör üretimi KAPATILDI, kayıtlar "
            "vektörsüz yazılacak. EMBEDDING_DIMS=%d yapıp news-articles indeksini yeniden oluşturun",
            model,
            got,
            expected,
            got,
        )
        self.embedder = None

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
        try:
            self._process(record)
        except Retry as exc:
            # ES kesintisi: mesajı hemen geri verip deneme bütçesini tüketmek yerine depo gelene dek bekle (geri basınç),
            # sonra broker gecikmeli yeniden teslim etsin (Unavailable → TRANSIENT_MAX_ATTEMPTS tavanı).
            log.warning("Depo erişilemiyor (kayıt %s): %s; Elasticsearch hazır olana dek bekleniyor", record.id, exc)
            if self._prepare_indices(self._stop_event):
                log.info("Elasticsearch yeniden erişilebilir; kayıt %s broker tarafından yeniden teslim edilecek", record.id)
            raise

    def _process(self, record: NewsRecord) -> None:
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
        embedding = self._embedding_for(record)
        decision = self.clusterer.cluster(record, embedding)
        record.event_id = decision.event_id
        record.duplicate_of = decision.duplicate_of
        event.record = record
        self.store.index_record(record, embedding=embedding)
        self.stats.indexed += 1

        is_duplicate = bool(decision.duplicate_of)
        if is_duplicate and not self.settings.alarm_notify_duplicates:
            event.channels_notified = []
            self.stats.suppressed += 1
            log.info(
                "Tekrar alarm bastırıldı [%s] alarm=%s olay=%s önceki=%s benzerlik=%.2f (%s) | %s",
                record.source,
                event.alarm_id,
                decision.event_id,
                decision.duplicate_of,
                decision.similarity,
                decision.method,
                title,
            )
        else:
            event.channels_notified = self._notify(event)
        self.broker.publish(
            RoutingKey.ALARM_RAISED,
            event.to_message(),
            headers={"x-duplicate": is_duplicate, "x-event-id": decision.event_id},
        )
        self.store.index_alarm(event)
        self.stats.alarms_raised += 1
        log.warning(
            "Alarm yükseltildi [%s] skor=%d alarm=%s olay=%s kanallar=%s → %s | %s",
            record.source,
            record.alarm_score,
            event.alarm_id,
            decision.event_id,
            ",".join(event.channels_notified) or ("bastırıldı" if is_duplicate else "-"),
            Queue.ALARMS,
            title,
        )

    # --- servis döngüsü ---
    def _prepare_indices(self, stop_event: threading.Event) -> bool:
        """``ensure_indices`` başarana (True) ya da ``stop_event`` set edilene (False) dek üstel geri çekilmeyle bekler."""
        delay = _INDEX_RETRY_INITIAL_DELAY
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
        self._stop_event = stop_event
        self.broker.declare_topology()
        if not self._prepare_indices(stop_event):
            log.info("Alarm katmanı indeksler hazırlanamadan durduruldu")
            return 0
        self._check_embedder()
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
