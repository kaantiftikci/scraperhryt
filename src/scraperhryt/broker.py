"""RabbitMQ mesajlaşma katmanı.

Topoloji (tüm servisler ``declare_topology`` ile idempotent olarak oluşturur):

    exchange  news.topic (topic, durable)
    exchange  news.dlx   (fanout, durable)  →  q.dead_letter

    q.articles.raw      ← article.raw        (scraper çıktısı, tüm haberler)
    q.articles.keyword  ← article.keyword    (anahtar kelime eşleşen haberler → LLM'e gidecekler)
    q.articles.scored   ← article.scored     (LLM skoru uygulanmış / uygulanmamış TÜM haberler, aynı nesne)
    q.alarms            ← alarm.raised       (alarm katmanının yükselttiği alarmlar)
    q.reports           ← report.#           (raporlama katmanı çıktıları)

Her ana kuyruğun ``<kuyruk>.retry`` eşi vardır: başarısız mesaj TTL'li retry kuyruğuna yazılır,
TTL dolunca ana exchange'e aynı routing key ile geri düşer. ``x-attempts`` başlığı deneme sayısını taşır;
``rabbitmq_max_attempts`` aşılınca mesaj ölü mektup kuyruğuna (q.dead_letter) gider.

Thread güvenliği: pika BlockingConnection thread-safe değildir. Her thread/işlem kendi Broker örneğini kullanmalıdır.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .config import Settings, get_settings

log = logging.getLogger(__name__)


class RoutingKey(StrEnum):
    ARTICLE_RAW = "article.raw"
    ARTICLE_KEYWORD = "article.keyword"
    ARTICLE_SCORED = "article.scored"
    ALARM_RAISED = "alarm.raised"
    REPORT_GENERATED = "report.generated"
    REPORT_ALARM_DIGEST = "report.alarm_digest"


class Queue(StrEnum):
    ARTICLES_RAW = "q.articles.raw"
    ARTICLES_KEYWORD = "q.articles.keyword"
    ARTICLES_SCORED = "q.articles.scored"
    ALARMS = "q.alarms"
    REPORTS = "q.reports"
    DEAD_LETTER = "q.dead_letter"


# kuyruk → (binding deseni, retry sonrası geri dönüş routing key'i)
QUEUE_BINDINGS: dict[str, tuple[str, str]] = {
    Queue.ARTICLES_RAW: (RoutingKey.ARTICLE_RAW, RoutingKey.ARTICLE_RAW),
    Queue.ARTICLES_KEYWORD: (RoutingKey.ARTICLE_KEYWORD, RoutingKey.ARTICLE_KEYWORD),
    Queue.ARTICLES_SCORED: (RoutingKey.ARTICLE_SCORED, RoutingKey.ARTICLE_SCORED),
    Queue.ALARMS: (RoutingKey.ALARM_RAISED, RoutingKey.ALARM_RAISED),
    Queue.REPORTS: ("report.#", RoutingKey.REPORT_GENERATED),
}


def retry_queue_name(queue: str) -> str:
    return f"{queue}.retry"


def topic_matches(pattern: str, routing_key: str) -> bool:
    """AMQP topic eşleşmesi: '*' tek kelime, '#' sıfır veya daha fazla kelime."""
    pw = pattern.split(".")
    kw = routing_key.split(".")

    def rec(i: int, j: int) -> bool:
        if i == len(pw):
            return j == len(kw)
        if pw[i] == "#":
            return any(rec(i + 1, k) for k in range(j, len(kw) + 1))
        if j >= len(kw):
            return False
        if pw[i] == "*" or pw[i] == kw[j]:
            return rec(i + 1, j + 1)
        return False

    return rec(0, 0)


class Reject(Exception):
    """Mesajı yeniden denemeden ölü mektup kuyruğuna gönder (örn. bozuk/parse edilemeyen mesaj)."""


class Retry(Exception):
    """Mesajı gecikmeli olarak yeniden dene (geçici hata: LLM/ES erişilemiyor vb.)."""


@dataclass
class Message:
    body: dict[str, Any]
    routing_key: str
    headers: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    queue: str = ""
    message_id: str = ""


Handler = Callable[[Message], None]


class Broker(Protocol):
    def declare_topology(self) -> None: ...
    def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None: ...
    def consume(
        self,
        queue: str,
        handler: Handler,
        *,
        prefetch: int | None = None,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ) -> int: ...
    def close(self) -> None: ...


def encode_body(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")


def decode_body(raw: bytes | str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise Reject(f"Mesaj gövdesi JSON nesnesi değil: {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------------------------------------
# Bellek-içi broker (testler ve tek süreçli geliştirme çalıştırması için)
# ---------------------------------------------------------------------------------------------------------


class InMemoryBroker:
    """RabbitMQ ile aynı topoloji semantiğini (topic binding, retry, ölü mektup) bellek içinde taklit eder."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.queues: dict[str, deque[Message]] = defaultdict(deque)
        self.dead_letters: deque[Message] = deque()
        self.published: list[Message] = []  # test gözlemi için
        self._lock = threading.RLock()
        self._bindings: dict[str, str] = {q: b[0] for q, b in QUEUE_BINDINGS.items()}

    def declare_topology(self) -> None:  # idempotent
        for q in QUEUE_BINDINGS:
            self.queues.setdefault(q, deque())
        self.queues.setdefault(Queue.DEAD_LETTER, deque())

    def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
        # Gerçek broker gibi JSON gidiş-dönüşü yaparız ki serileştirme hataları testte yakalansın.
        payload = decode_body(encode_body(body))
        hdrs = dict(headers or {})
        attempts = int(hdrs.get("x-attempts", 0))
        with self._lock:
            msg = Message(body=payload, routing_key=routing_key, headers=hdrs, attempts=attempts, message_id=str(payload.get("id", "")))
            self.published.append(msg)
            for queue, pattern in self._bindings.items():
                if topic_matches(pattern, routing_key):
                    self.queues[queue].append(Message(**{**msg.__dict__, "queue": queue}))

    def consume(
        self,
        queue: str,
        handler: Handler,
        *,
        prefetch: int | None = None,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ) -> int:
        """Kuyruk boşalana (veya max_messages'a / stop_event'e) kadar mesajları işler; işlenen mesaj sayısını döndürür."""
        processed = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if max_messages is not None and processed >= max_messages:
                break
            with self._lock:
                if not self.queues[queue]:
                    break
                msg = self.queues[queue].popleft()
            self._dispatch(queue, msg, handler)
            processed += 1
        return processed

    def _dispatch(self, queue: str, msg: Message, handler: Handler) -> None:
        try:
            handler(msg)
        except Reject as exc:
            log.warning("Mesaj reddedildi (%s): %s", queue, exc)
            msg.headers["x-error"] = str(exc)
            self.dead_letters.append(msg)
        except Exception as exc:  # Retry dahil
            msg.attempts += 1
            msg.headers["x-attempts"] = msg.attempts
            msg.headers["x-error"] = str(exc)
            if msg.attempts >= self.settings.rabbitmq_max_attempts:
                log.error("Mesaj %d denemeden sonra ölü mektuba gitti (%s): %s", msg.attempts, queue, exc)
                self.dead_letters.append(msg)
            else:
                log.warning("Mesaj yeniden denenecek (%s, deneme %d): %s", queue, msg.attempts, exc)
                with self._lock:
                    self.queues[queue].append(msg)

    def size(self, queue: str) -> int:
        with self._lock:
            return len(self.queues[queue])

    def drain(self, queue: str) -> list[Message]:
        with self._lock:
            items = list(self.queues[queue])
            self.queues[queue].clear()
        return items

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------------------------------------
# RabbitMQ broker (pika)
# ---------------------------------------------------------------------------------------------------------


class RabbitMQBroker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._conn = None
        self._ch = None
        self._lock = threading.RLock()
        self._topology_declared = False

    # --- bağlantı yönetimi ---
    def _connect(self) -> None:
        import pika

        params = pika.URLParameters(self.settings.rabbitmq_url)
        params.heartbeat = self.settings.rabbitmq_heartbeat
        params.blocked_connection_timeout = 300
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, 9):
            try:
                self._conn = pika.BlockingConnection(params)
                self._ch = self._conn.channel()
                self._ch.confirm_delivery()
                log.info("RabbitMQ bağlantısı kuruldu (%s)", _redact(self.settings.rabbitmq_url))
                if self._topology_declared:
                    self._declare(self._ch)
                return
            except Exception as exc:  # pragma: no cover - ağ hatası
                last_exc = exc
                log.warning("RabbitMQ bağlantısı kurulamadı (deneme %d): %s; %.0fs sonra tekrar", attempt, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise ConnectionError(f"RabbitMQ'ya bağlanılamadı: {last_exc}")

    def _channel(self):
        if self._ch is None or self._ch.is_closed or self._conn is None or self._conn.is_closed:
            self._connect()
        return self._ch

    def _declare(self, ch) -> None:
        import pika
        from pika.exchange_type import ExchangeType

        s = self.settings
        ch.exchange_declare(exchange=s.rabbitmq_exchange, exchange_type=ExchangeType.topic, durable=True)
        ch.exchange_declare(exchange=s.rabbitmq_dlx, exchange_type=ExchangeType.fanout, durable=True)
        ch.queue_declare(queue=Queue.DEAD_LETTER, durable=True)
        ch.queue_bind(queue=Queue.DEAD_LETTER, exchange=s.rabbitmq_dlx, routing_key="#")
        for queue, (pattern, retry_key) in QUEUE_BINDINGS.items():
            ch.queue_declare(queue=queue, durable=True, arguments={"x-dead-letter-exchange": s.rabbitmq_dlx})
            ch.queue_bind(queue=queue, exchange=s.rabbitmq_exchange, routing_key=pattern)
            ch.queue_declare(
                queue=retry_queue_name(queue),
                durable=True,
                arguments={
                    "x-dead-letter-exchange": s.rabbitmq_exchange,
                    "x-dead-letter-routing-key": retry_key,
                    "x-message-ttl": int(s.rabbitmq_retry_delay_ms),
                },
            )
        _ = pika  # (import yan etkisi için)

    def declare_topology(self) -> None:
        with self._lock:
            self._declare(self._channel())
            self._topology_declared = True
            log.info("RabbitMQ topolojisi hazır: exchange=%s, kuyruklar=%s", self.settings.rabbitmq_exchange, list(QUEUE_BINDINGS))

    # --- yayınlama ---
    def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
        import pika

        payload = encode_body(body)
        props = pika.BasicProperties(
            content_type="application/json",
            content_encoding="utf-8",
            delivery_mode=2,  # kalıcı
            headers=dict(headers or {}),
            message_id=str(body.get("id") or body.get("alarm_id") or body.get("report_id") or ""),
            timestamp=int(time.time()),
        )
        with self._lock:
            for attempt in range(1, 4):
                try:
                    self._channel().basic_publish(
                        exchange=self.settings.rabbitmq_exchange, routing_key=routing_key, body=payload, properties=props
                    )
                    return
                except Exception as exc:  # pragma: no cover - ağ hatası
                    log.warning("Yayınlama başarısız (deneme %d, %s): %s", attempt, routing_key, exc)
                    self._reset()
                    time.sleep(min(2**attempt, 10))
            raise ConnectionError(f"Mesaj yayınlanamadı: {routing_key}")

    def _publish_to_queue(self, ch, queue: str, payload: bytes, headers: dict[str, Any]) -> None:
        import pika

        props = pika.BasicProperties(content_type="application/json", delivery_mode=2, headers=headers, timestamp=int(time.time()))
        ch.basic_publish(exchange="", routing_key=queue, body=payload, properties=props)

    # --- tüketim ---
    def consume(
        self,
        queue: str,
        handler: Handler,
        *,
        prefetch: int | None = None,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ) -> int:
        """Bloklayan tüketici döngüsü. stop_event set edilene (veya max_messages'a) kadar çalışır; bağlantı koparsa yeniden bağlanır."""
        processed = 0
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            try:
                ch = self._channel()
                ch.basic_qos(prefetch_count=prefetch or self.settings.rabbitmq_prefetch)
                counter = {"n": 0}

                def on_message(channel, method, properties, body, _queue=queue, _counter=counter):
                    msg = self._to_message(_queue, method, properties, body)
                    self._handle(channel, method, properties, body, msg, handler)
                    _counter["n"] += 1
                    if max_messages is not None and processed + _counter["n"] >= max_messages:
                        stop_event.set()

                tag = ch.basic_consume(queue=queue, on_message_callback=on_message, auto_ack=False)
                log.info("Tüketim başladı: %s", queue)
                while not stop_event.is_set():
                    self._conn.process_data_events(time_limit=1.0)
                processed += counter["n"]
                try:
                    ch.basic_cancel(tag)
                except Exception:  # pragma: no cover
                    pass
            except Exception as exc:  # pragma: no cover - ağ hatası
                if stop_event.is_set():
                    break
                log.warning("Tüketici bağlantısı koptu (%s): %s; yeniden bağlanılıyor", queue, exc)
                self._reset()
                time.sleep(2)
        return processed

    def _to_message(self, queue: str, method, properties, body: bytes) -> Message:
        headers = dict(getattr(properties, "headers", None) or {})
        attempts = int(headers.get("x-attempts", 0) or 0)
        try:
            data = decode_body(body)
        except Reject:
            raise
        except Exception as exc:
            raise Reject(f"JSON çözümlenemedi: {exc}") from exc
        return Message(
            body=data,
            routing_key=method.routing_key,
            headers=headers,
            attempts=attempts,
            queue=queue,
            message_id=str(getattr(properties, "message_id", "") or ""),
        )

    def _handle(self, ch, method, properties, body: bytes, msg: Message | None, handler: Handler) -> None:
        try:
            if msg is None:
                raise Reject("boş mesaj")
            handler(msg)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Reject as exc:
            log.warning("Mesaj reddedildi → ölü mektup (%s): %s", method.routing_key, exc)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as exc:
            attempts = (msg.attempts if msg else 0) + 1
            headers = dict((msg.headers if msg else {}) or {})
            headers.update({"x-attempts": attempts, "x-error": str(exc)[:500], "x-origin-queue": msg.queue if msg else ""})
            if attempts >= self.settings.rabbitmq_max_attempts:
                log.error("Mesaj %d denemeden sonra ölü mektuba gönderildi (%s): %s", attempts, method.routing_key, exc)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            log.warning("Mesaj retry kuyruğuna alındı (%s, deneme %d): %s", method.routing_key, attempts, exc)
            try:
                self._publish_to_queue(ch, retry_queue_name(msg.queue if msg else method.routing_key), body, headers)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as pub_exc:  # pragma: no cover
                log.error("Retry kuyruğuna yazılamadı, mesaj broker'a iade ediliyor: %s", pub_exc)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    # --- kapanış ---
    def _reset(self) -> None:
        try:
            if self._conn is not None and self._conn.is_open:
                self._conn.close()
        except Exception:  # pragma: no cover
            pass
        self._conn = None
        self._ch = None

    def close(self) -> None:
        with self._lock:
            self._reset()


def _redact(url: str) -> str:
    try:
        from urllib.parse import urlsplit, urlunsplit

        p = urlsplit(url)
        netloc = p.hostname or ""
        if p.port:
            netloc += f":{p.port}"
        if p.username:
            netloc = f"{p.username}:***@{netloc}"
        return urlunsplit((p.scheme, netloc, p.path, "", ""))
    except Exception:  # pragma: no cover
        return "<amqp>"


def make_broker(settings: Settings | None = None, *, in_memory: bool = False) -> Broker:
    settings = settings or get_settings()
    if in_memory or settings.rabbitmq_url in ("memory://", "inmemory://"):
        return InMemoryBroker(settings)
    return RabbitMQBroker(settings)
