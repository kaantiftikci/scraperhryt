"""RabbitMQ mesajlaşma katmanı.

Topoloji (tüm servisler ``declare_topology`` ile idempotent olarak oluşturur):

    exchange  news.topic       (topic, durable)
    exchange  news.topic.retry (topic, durable)   →  <kuyruk>.retry  (aynı binding deseniyle)
    exchange  news.dlx         (fanout, durable)  →  q.dead_letter

    q.articles.raw      ← article.raw        (scraper çıktısı, tüm haberler)
    q.articles.keyword  ← article.keyword    (anahtar kelime eşleşen haberler → LLM'e gidecekler)
    q.articles.scored   ← article.scored     (LLM skoru uygulanmış / uygulanmamış TÜM haberler, aynı nesne)
    q.alarms            ← alarm.raised       (alarm katmanının yükselttiği alarmlar)
    q.reports           ← report.#           (raporlama katmanı çıktıları; tüketicisi yok, uzunluk/TTL ile sınırlı)

Her ana kuyruğun ``<kuyruk>.retry`` eşi vardır: başarısız mesaj ORİJİNAL routing key'i ile retry exchange'ine
(``<rabbitmq_exchange>.retry``) yazılır, TTL dolunca ana exchange'e aynı routing key ile geri düşer
(``report.alarm_digest`` gibi desenle eşleşen anahtarlar değişmez). ``x-attempts`` başlığı deneme sayısını taşır;
tavan aşılınca mesaj ölü mektup kuyruğuna (q.dead_letter) gider. Tavan hatanın türüne bağlıdır:

* ``Reject``                       → hemen ölü mektup (bozuk/parse edilemeyen mesaj)
* ``Unavailable`` (Retry alt tipi) → altyapı geçici olarak erişilemiyor; ``TRANSIENT_MAX_ATTEMPTS`` (240 × 15 s
                                      ≈ 1 saat) — kısa bir Ollama/ES kesintisi kayıtları ölü mektuba düşürmez
* diğer tüm istisnalar (Retry dahil) → ``rabbitmq_max_attempts``

Kuyruk argümanları (``x-message-ttl`` vb.) RabbitMQ'da sonradan değiştirilemez; ``RABBITMQ_RETRY_DELAY_MS``
değişince mevcut ``q.*.retry`` kuyrukları korunur (uyarı loglanır) ve yeni değer ancak kuyruk silinince uygulanır.

Thread güvenliği: pika BlockingConnection thread-safe değildir. Her thread/işlem kendi Broker örneğini
kullanmalıdır. ``RabbitMQBroker`` içinde bağlantıya her erişim ``_lock`` ile seri hale getirilir; işleyici
(handler) ayrı bir iş parçacığında çalışırken bağlantı iş parçacığı heartbeat'leri işlemeye devam eder, böylece
uzun LLM çağrıları (dakikalar) bağlantıyı düşürmez.
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

# İşleyici çalışırken bağlantı iş parçacığının her turda ioloop'u (heartbeat vb.) işlettiği süre (saniye).
HANDLER_PUMP_SECONDS = 0.2
# İşleyici iş parçacığının bitmesinin beklendiği süre; bu sırada kilit serbesttir (işleyici publish yapabilir).
HANDLER_JOIN_SECONDS = 0.05
# ``Unavailable`` (altyapı geçici olarak erişilemiyor) için ölü mektup öncesi deneme tavanı; 0 = sınırsız.
# Varsayılan rabbitmq_retry_delay_ms (15 s) ile ≈ 1 saatlik bir kesinti mesaj kaybetmeden atlatılır.
TRANSIENT_MAX_ATTEMPTS = 240
# q.reports'un tüketicisi yoktur (raporlar news-reports indeksinde de saklanır): broker'da birikmesin.
REPORTS_QUEUE_MAX_LENGTH = 1000
REPORTS_QUEUE_TTL_MS = 7 * 24 * 60 * 60 * 1000
# InMemoryBroker'ın gözlem tamponları (``published`` / ``dead_letters``) bu kadar kaydı tutar; sürekli çalışan
# ``run-all --in-memory`` sürecinde bellek sınırsız büyümesin (tek turluk özet için fazlasıyla yeterli).
PUBLISHED_HISTORY_LIMIT = 10_000
DEAD_LETTER_HISTORY_LIMIT = 1_000


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


# kuyruk → (binding deseni, varsayılan routing key)
# Retry'da mesajın ORİJİNAL routing key'i korunur; yalnızca desene uymayan bir anahtarla gelmişse (örn. varsayılan
# exchange üzerinden elle/yeniden oynatılarak yayınlandıysa) varsayılan anahtar kullanılır ki mesaj kuyruğa dönebilsin.
QUEUE_BINDINGS: dict[str, tuple[str, str]] = {
    Queue.ARTICLES_RAW: (RoutingKey.ARTICLE_RAW, RoutingKey.ARTICLE_RAW),
    Queue.ARTICLES_KEYWORD: (RoutingKey.ARTICLE_KEYWORD, RoutingKey.ARTICLE_KEYWORD),
    Queue.ARTICLES_SCORED: (RoutingKey.ARTICLE_SCORED, RoutingKey.ARTICLE_SCORED),
    Queue.ALARMS: (RoutingKey.ALARM_RAISED, RoutingKey.ALARM_RAISED),
    Queue.REPORTS: ("report.#", RoutingKey.REPORT_GENERATED),
}


def retry_queue_name(queue: str) -> str:
    return f"{queue}.retry"


def retry_exchange_name(settings: Settings) -> str:
    """Retry kuyruklarını besleyen topic exchange (ana exchange'in adından türetilir, ör. news.topic.retry)."""
    return f"{settings.rabbitmq_exchange}.retry"


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


def retry_routing_key(queue: str, routing_key: str) -> str:
    """Retry'da kullanılacak routing key: orijinal anahtar kuyruğun desenine uyuyorsa o, yoksa kuyruğun varsayılanı."""
    binding = QUEUE_BINDINGS.get(queue)
    if binding is None:
        return routing_key
    pattern, default_key = binding
    if routing_key and topic_matches(pattern, routing_key):
        return routing_key
    return str(default_key)


class Reject(Exception):
    """Mesajı yeniden denemeden ölü mektup kuyruğuna gönder (örn. bozuk/parse edilemeyen mesaj)."""


class Retry(Exception):
    """Mesajı gecikmeli olarak yeniden dene; ``rabbitmq_max_attempts`` sonrası ölü mektup."""


class Unavailable(Retry):
    """Bağımlı altyapı (Ollama, Elasticsearch...) geçici olarak erişilemiyor.

    ``Retry`` gibi gecikmeli yeniden denenir ancak tavan ``TRANSIENT_MAX_ATTEMPTS``'tır (0 = sınırsız); kısa bir
    kesinti yüzünden kayıtlar ölü mektuba düşmez. ``except Retry`` ile yakalanmaya devam eder.
    """


def attempt_limit_for(settings: Settings, exc: BaseException) -> int:
    """İstisna türüne göre ölü mektup öncesi deneme tavanı (0 = sınırsız)."""
    if isinstance(exc, Unavailable):
        return max(0, int(TRANSIENT_MAX_ATTEMPTS))
    return max(1, int(settings.rabbitmq_max_attempts))


def attempts_exhausted(settings: Settings, exc: BaseException, attempts: int) -> bool:
    limit = attempt_limit_for(settings, exc)
    return limit > 0 and attempts >= limit


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


def queue_arguments(settings: Settings, queue: str) -> dict[str, Any]:
    """Ana kuyruğun declare argümanları. q.reports tüketilmediği için uzunluk/TTL ile sınırlanır ve DLX almaz
    (süresi dolan raporlar ölü mektup kuyruğunu doldurmasın)."""
    if queue == Queue.REPORTS:
        return {"x-max-length": int(REPORTS_QUEUE_MAX_LENGTH), "x-message-ttl": int(REPORTS_QUEUE_TTL_MS)}
    return {"x-dead-letter-exchange": settings.rabbitmq_dlx}


def retry_queue_arguments(settings: Settings) -> dict[str, Any]:
    """Retry kuyruğu: TTL dolunca ana exchange'e, mesajın kendi routing key'i ile geri düşer."""
    return {
        "x-dead-letter-exchange": settings.rabbitmq_exchange,
        "x-message-ttl": int(settings.rabbitmq_retry_delay_ms),
    }


# ---------------------------------------------------------------------------------------------------------
# Bellek-içi broker (testler ve tek süreçli geliştirme çalıştırması için)
# ---------------------------------------------------------------------------------------------------------


class InMemoryBroker:
    """RabbitMQ ile aynı topoloji semantiğini (topic binding, retry, ölü mektup) bellek içinde taklit eder."""

    def __init__(self, settings: Settings | None = None, *, history_limit: int = PUBLISHED_HISTORY_LIMIT) -> None:
        self.settings = settings or get_settings()
        self.queues: dict[str, deque[Message]] = defaultdict(deque)
        # Gözlem tamponları (test / tek tur özeti): sınırlıdır, en eski kayıtlar düşer.
        self.history_limit = max(1, int(history_limit))
        self.dead_letters: deque[Message] = deque(maxlen=DEAD_LETTER_HISTORY_LIMIT)
        self.published: list[Message] = []
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
            msg = Message(
                body=payload, routing_key=routing_key, headers=hdrs, attempts=attempts, message_id=str(payload.get("id", ""))
            )
            self.published.append(msg)
            if len(self.published) > self.history_limit:
                del self.published[: len(self.published) - self.history_limit]
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
        except Exception as exc:  # Retry / Unavailable dahil
            msg.attempts += 1
            msg.headers["x-attempts"] = msg.attempts
            msg.headers["x-error"] = str(exc)
            if attempts_exhausted(self.settings, exc, msg.attempts):
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

# Tüketiciye teslim edilmiş ama henüz işlenmemiş mesaj: (method, properties, body)
_Delivery = tuple[Any, Any, bytes]


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
                self._ch = self._open_channel()
                log.info("RabbitMQ bağlantısı kuruldu (%s)", _redact(self.settings.rabbitmq_url))
                if self._topology_declared:
                    self._declare()
                return
            except Exception as exc:  # pragma: no cover - ağ hatası
                last_exc = exc
                log.warning("RabbitMQ bağlantısı kurulamadı (deneme %d): %s; %.0fs sonra tekrar", attempt, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise ConnectionError(f"RabbitMQ'ya bağlanılamadı: {last_exc}")

    def _open_channel(self):
        ch = self._conn.channel()
        ch.confirm_delivery()
        return ch

    def _channel(self):
        if self._ch is None or self._ch.is_closed or self._conn is None or self._conn.is_closed:
            self._connect()
        return self._ch

    def _live_channel(self):
        """Boşta kalmış bir bağlantının heartbeat'lerini işler; broker bağlantıyı kapattıysa uyarı/bekleme
        olmadan yeniden bağlanır (yalnızca yayın yapan servisler dakikalarca sessiz kalabilir)."""
        if self._conn is not None and self._conn.is_open:
            try:
                self._conn.process_data_events(time_limit=0)
            except Exception as exc:
                log.info("Boşta kalan RabbitMQ bağlantısı kapanmış (%s); yeniden bağlanılıyor", exc)
                self._reset()
        return self._channel()

    def _declare_queue(self, queue: str, arguments: dict[str, Any]) -> None:
        """Kuyruğu declare eder; farklı argümanlarla zaten varsa (PRECONDITION_FAILED 406) mevcut kuyruğu korur."""
        from pika.exceptions import ChannelClosedByBroker

        try:
            self._ch.queue_declare(queue=queue, durable=True, arguments=arguments)
        except ChannelClosedByBroker as exc:
            if exc.reply_code != 406:
                raise
            log.warning(
                "Kuyruk '%s' farklı argümanlarla zaten var; mevcut kuyruk korunuyor (%s). Yeni argümanları (%s) "
                "uygulamak için kuyruğu silip servisi yeniden başlatın: rabbitmqctl delete_queue %s",
                queue,
                exc.reply_text,
                arguments,
                queue,
            )
            self._ch = self._open_channel()
            self._ch.queue_declare(queue=queue, durable=True, passive=True)

    def _declare(self) -> None:
        from pika.exchange_type import ExchangeType

        s = self.settings
        ch = self._ch
        ch.exchange_declare(exchange=s.rabbitmq_exchange, exchange_type=ExchangeType.topic, durable=True)
        ch.exchange_declare(exchange=retry_exchange_name(s), exchange_type=ExchangeType.topic, durable=True)
        ch.exchange_declare(exchange=s.rabbitmq_dlx, exchange_type=ExchangeType.fanout, durable=True)
        self._declare_queue(Queue.DEAD_LETTER, {})
        self._ch.queue_bind(queue=Queue.DEAD_LETTER, exchange=s.rabbitmq_dlx, routing_key="#")
        for queue, (pattern, _default_key) in QUEUE_BINDINGS.items():
            self._declare_queue(queue, queue_arguments(s, queue))
            self._ch.queue_bind(queue=queue, exchange=s.rabbitmq_exchange, routing_key=pattern)
            retry_queue = retry_queue_name(queue)
            self._declare_queue(retry_queue, retry_queue_arguments(s))
            self._ch.queue_bind(queue=retry_queue, exchange=retry_exchange_name(s), routing_key=pattern)

    def declare_topology(self) -> None:
        with self._lock:
            self._channel()
            self._declare()
            self._topology_declared = True
            log.info(
                "RabbitMQ topolojisi hazır: exchange=%s, retry=%s, kuyruklar=%s",
                self.settings.rabbitmq_exchange,
                retry_exchange_name(self.settings),
                list(QUEUE_BINDINGS),
            )

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
                    self._live_channel().basic_publish(
                        exchange=self.settings.rabbitmq_exchange, routing_key=routing_key, body=payload, properties=props
                    )
                    return
                except Exception as exc:  # pragma: no cover - ağ hatası
                    log.warning("Yayınlama başarısız (deneme %d, %s): %s", attempt, routing_key, exc)
                    self._reset()
                    if attempt < 3:
                        time.sleep(min(2**attempt, 10))
            raise ConnectionError(f"Mesaj yayınlanamadı: {routing_key}")

    def _publish_retry(self, ch, routing_key: str, payload: bytes, headers: dict[str, Any]) -> None:
        """Mesajı orijinal routing key'i ile retry exchange'ine yazar (TTL sonra ana exchange'e geri düşer)."""
        import pika

        props = pika.BasicProperties(
            content_type="application/json", delivery_mode=2, headers=headers, timestamp=int(time.time())
        )
        ch.basic_publish(
            exchange=retry_exchange_name(self.settings), routing_key=routing_key, body=payload, properties=props
        )

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
        """Bloklayan tüketici döngüsü. stop_event set edilene (veya max_messages'a) kadar çalışır; bağlantı koparsa
        yeniden bağlanır. Teslimatlar pika callback'inde yalnızca sıraya alınır; işleme bağlantı iş parçacığında,
        callback dışında yapılır (yeniden giriş yok), işleyici ise ayrı iş parçacığında koşar."""
        processed = 0
        stop_event = stop_event or threading.Event()
        # Mesaj bütçesi dolunca çağıranın stop_event'ine dokunmayız (paylaşılan kapanış olayı olabilir);
        # kendi bayrağımızla çıkarız.
        done = threading.Event()

        def should_stop() -> bool:
            return stop_event.is_set() or done.is_set()

        while not should_stop():
            pending: deque[_Delivery] = deque()

            def on_message(_channel, method, properties, body, _pending=pending):
                _pending.append((method, properties, body))

            try:
                with self._lock:
                    ch = self._channel()
                    ch.basic_qos(prefetch_count=prefetch or self.settings.rabbitmq_prefetch)
                    tag = ch.basic_consume(queue=queue, on_message_callback=on_message, auto_ack=False)
                log.info("Tüketim başladı: %s", queue)
                while not should_stop():
                    self._pump(ch, 1.0)
                    while pending and not should_stop():
                        method, properties, body = pending.popleft()
                        self._process_delivery(ch, queue, method, properties, body, handler)
                        processed += 1
                        if max_messages is not None and processed >= max_messages:
                            done.set()
                self._stop_consumer(ch, tag, pending)
            except Exception as exc:  # pragma: no cover - ağ hatası
                if should_stop():
                    break
                log.warning("Tüketici bağlantısı koptu (%s): %s; yeniden bağlanılıyor", queue, exc)
                self._reset()
                time.sleep(2)
        return processed

    def _pump(self, ch, timeout: float) -> None:
        """ioloop'u kilit altında işletir (heartbeat, teslimat, confirm). Bağlantı/kanal kapandıysa hata verir."""
        with self._lock:
            conn = self._conn
            if conn is None or conn.is_closed or ch.is_closed:
                raise ConnectionError("RabbitMQ bağlantısı/kanalı kapanmış")
            conn.process_data_events(time_limit=timeout)

    def _stop_consumer(self, ch, tag: str, pending: deque[_Delivery]) -> None:
        """Tüketimi iptal eder; teslim edilmiş ama işlenmemiş mesajları broker'a iade eder."""
        with self._lock:
            if ch.is_closed:
                return
            try:
                ch.basic_cancel(tag)
                while pending:
                    method, _properties, _body = pending.popleft()
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            except Exception as exc:  # pragma: no cover - kapanış sırasında ağ hatası
                log.debug("Tüketici iptali tamamlanamadı: %s", exc)

    def _process_delivery(self, ch, queue: str, method, properties, body: bytes, handler: Handler) -> None:
        try:
            msg = self._to_message(queue, method, properties, body)
        except Reject as exc:
            log.warning("Mesaj çözümlenemedi → ölü mektup (%s): %s", method.routing_key, exc)
            with self._lock:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        error = self._run_handler(ch, msg, handler)
        self._settle(ch, method, body, msg, error)

    def _run_handler(self, ch, msg: Message, handler: Handler) -> BaseException | None:
        """İşleyiciyi ayrı iş parçacığında çalıştırır; bu sırada bağlantının ioloop'unu işletmeye devam eder ki
        dakikalar süren LLM/ES çağrıları heartbeat'leri aç bırakıp bağlantıyı düşürmesin. İşleyicinin istisnasını
        (varsa) döndürür."""
        outcome: dict[str, BaseException] = {}

        def work() -> None:
            try:
                handler(msg)
            except BaseException as exc:  # sonuç bağlantı iş parçacığında değerlendirilir
                outcome["error"] = exc

        worker = threading.Thread(target=work, name=f"handler-{msg.queue or 'queue'}", daemon=True)
        worker.start()
        while worker.is_alive():
            try:
                self._pump(ch, HANDLER_PUMP_SECONDS)
            except Exception:
                log.warning(
                    "İşleyici çalışırken RabbitMQ bağlantısı koptu (%s); işleyici bitince mesaj broker tarafından "
                    "yeniden teslim edilecek",
                    msg.routing_key,
                )
                worker.join()
                raise
            worker.join(HANDLER_JOIN_SECONDS)
        return outcome.get("error")

    def _to_message(self, queue: str, method, properties, body: bytes) -> Message:
        headers = dict(getattr(properties, "headers", None) or {})
        try:
            attempts = int(headers.get("x-attempts", 0) or 0)
        except (TypeError, ValueError):
            attempts = 0
        try:
            data = decode_body(body)
        except Reject:
            raise
        except Exception as exc:
            raise Reject(f"JSON çözümlenemedi: {exc}") from exc
        return Message(
            body=data,
            routing_key=str(getattr(method, "routing_key", "") or ""),
            headers=headers,
            attempts=attempts,
            queue=queue,
            message_id=str(getattr(properties, "message_id", "") or ""),
        )

    def _settle(self, ch, method, body: bytes, msg: Message, error: BaseException | None) -> None:
        """İşleyici sonucunu bağlantı iş parçacığında uygular: ack / ölü mektup / retry."""
        with self._lock:
            if ch.is_closed:
                raise ConnectionError(f"Kanal kapandığı için mesaj onaylanamadı ({method.routing_key})")
            if error is None:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            if isinstance(error, Reject):
                log.warning("Mesaj reddedildi → ölü mektup (%s): %s", method.routing_key, error)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            if not isinstance(error, Exception):  # KeyboardInterrupt/SystemExit: mesajı iade et, kapanışı sürdür
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                raise error
            attempts = msg.attempts + 1
            headers = dict(msg.headers)
            headers.update({"x-attempts": attempts, "x-error": str(error)[:500], "x-origin-queue": msg.queue})
            if attempts_exhausted(self.settings, error, attempts):
                log.error(
                    "Mesaj %d denemeden sonra ölü mektuba gönderildi (%s): %s", attempts, method.routing_key, error
                )
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            key = retry_routing_key(msg.queue, msg.routing_key)
            log.warning("Mesaj retry kuyruğuna alındı (%s, deneme %d): %s", key, attempts, error)
            try:
                self._publish_retry(ch, key, body, headers)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as pub_exc:
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
