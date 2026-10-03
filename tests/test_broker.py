"""Broker katmanı birim testleri.

``RabbitMQBroker`` sahte bir pika bağlantısı/kanalı ile (ağ yok) sürülür: zehirli mesaj, işleyici sırasında
heartbeat işletimi, retry exchange'i / routing key korunumu, deneme tavanları (Retry vs Unavailable), 406
PRECONDITION_FAILED toleransı ve boşta kalan yayıncı bağlantısının sessizce yenilenmesi.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest
from pika.exceptions import ChannelClosedByBroker

from scraperhryt import broker as broker_mod
from scraperhryt.broker import (
    QUEUE_BINDINGS,
    InMemoryBroker,
    Message,
    Queue,
    RabbitMQBroker,
    Reject,
    Retry,
    RoutingKey,
    Unavailable,
    attempt_limit_for,
    queue_arguments,
    retry_exchange_name,
    retry_queue_arguments,
    retry_queue_name,
    retry_routing_key,
)
from scraperhryt.config import Settings

pytestmark = pytest.mark.timeout(30)


# ---------------------------------------------------------------------------------------------------------
# sahte pika
# ---------------------------------------------------------------------------------------------------------


@dataclass
class FakeMethod:
    delivery_tag: int
    routing_key: str


@dataclass
class FakeProperties:
    headers: dict[str, Any] | None = None
    message_id: str = ""


@dataclass
class FakeChannel:
    conn: FakeConnection
    is_closed: bool = False
    acks: list[int] = field(default_factory=list)
    nacks: list[tuple[int, bool]] = field(default_factory=list)
    published: list[dict[str, Any]] = field(default_factory=list)
    declared: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    passive_declared: list[str] = field(default_factory=list)
    bound: list[tuple[str, str, str]] = field(default_factory=list)
    exchanges: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    prefetch: int | None = None
    declare_errors: dict[str, Exception] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return not self.is_closed

    def confirm_delivery(self) -> None:
        return None

    def basic_qos(self, prefetch_count: int) -> None:
        self.prefetch = prefetch_count

    def basic_consume(self, queue: str, on_message_callback, auto_ack: bool) -> str:
        assert auto_ack is False
        self.conn.consumer = (self, on_message_callback)
        return f"ctag-{queue}"

    def basic_cancel(self, consumer_tag: str) -> list[Any]:
        self.cancelled.append(consumer_tag)
        self.conn.consumer = None
        return []

    def basic_ack(self, delivery_tag: int) -> None:
        self._check_open()
        self.acks.append(delivery_tag)

    def basic_nack(self, delivery_tag: int, requeue: bool) -> None:
        self._check_open()
        self.nacks.append((delivery_tag, requeue))

    def basic_publish(self, exchange: str, routing_key: str, body: bytes, properties) -> None:
        self._check_open()
        self.published.append(
            {"exchange": exchange, "routing_key": routing_key, "body": body, "headers": dict(properties.headers or {})}
        )

    def queue_declare(self, queue: str, durable: bool, arguments: dict[str, Any] | None = None, passive: bool = False):
        self._check_open()
        if passive:
            self.passive_declared.append(queue)
            return None
        err = self.declare_errors.pop(queue, None)
        if err is not None:
            self.is_closed = True
            raise err
        self.declared.append((queue, dict(arguments or {})))
        return None

    def queue_bind(self, queue: str, exchange: str, routing_key: str) -> None:
        self._check_open()
        self.bound.append((queue, exchange, routing_key))

    def exchange_declare(self, exchange: str, exchange_type, durable: bool) -> None:
        self._check_open()
        self.exchanges.append((exchange, str(getattr(exchange_type, "value", exchange_type))))

    def _check_open(self) -> None:
        if self.is_closed:
            raise ChannelClosedByBroker(504, "kanal kapalı")


class FakeConnection:
    """process_data_events çağrısında sıradaki teslimatları kayıtlı tüketiciye verir ve çağrıları sayar."""

    def __init__(self) -> None:
        self.is_closed = False
        self.consumer: tuple[FakeChannel, Any] | None = None
        self.deliveries: deque[tuple[FakeMethod, FakeProperties, bytes]] = deque()
        self.pump_calls: list[float] = []
        self.channels: list[FakeChannel] = []
        self.pump_error: Exception | None = None  # bir sonraki process_data_events'te fırlatılır (tek sefer)
        self._sleeper = threading.Event()

    @property
    def is_open(self) -> bool:
        return not self.is_closed

    def channel(self) -> FakeChannel:
        ch = FakeChannel(conn=self)
        self.channels.append(ch)
        return ch

    def process_data_events(self, time_limit: float | None = 0) -> None:
        self.pump_calls.append(float(time_limit or 0))
        if self.pump_error is not None:
            err, self.pump_error = self.pump_error, None
            self.is_closed = True
            raise err
        if self.consumer is not None:
            ch, callback = self.consumer
            while self.deliveries and self.consumer is not None:
                method, props, body = self.deliveries.popleft()
                callback(ch, method, props, body)
        if time_limit:
            # time.sleep değil: testler broker.time.sleep'i yamalayabilir, sahte ioloop bundan etkilenmemeli
            self._sleeper.wait(min(time_limit, 0.005))

    def close(self) -> None:
        self.is_closed = True
        for ch in self.channels:
            ch.is_closed = True


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"rabbitmq_max_attempts": 3}
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeRabbit:
    """RabbitMQBroker'ı sahte bağlantılarla bağlar; her _connect yeni bir FakeConnection üretir."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or make_settings()
        self.broker = RabbitMQBroker(self.settings)
        self.connections: list[FakeConnection] = []
        self.broker._connect = self._connect  # type: ignore[method-assign]

    def _connect(self) -> None:
        conn = FakeConnection()
        self.connections.append(conn)
        self.broker._conn = conn
        self.broker._ch = conn.channel()
        if self.broker._topology_declared:
            self.broker._declare()

    @property
    def conn(self) -> FakeConnection:
        return self.connections[-1]

    @property
    def ch(self) -> FakeChannel:
        return self.conn.channels[-1]

    def deliver(self, body: Any, routing_key: str = "article.keyword", headers: dict[str, Any] | None = None) -> int:
        if not self.connections:
            self._connect()
        tag = len(self.conn.deliveries) + 1 + sum(len(c.acks) + len(c.nacks) for c in self.conn.channels)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.conn.deliveries.append((FakeMethod(tag, routing_key), FakeProperties(headers=headers), raw))
        return tag


@pytest.fixture
def rabbit() -> FakeRabbit:
    return FakeRabbit()


@pytest.fixture(autouse=True)
def transient_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_mod, "TRANSIENT_MAX_ATTEMPTS", 6)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(broker_mod.time, "sleep", lambda s: calls.append(s))
    return calls


# ---------------------------------------------------------------------------------------------------------
# saf yardımcılar
# ---------------------------------------------------------------------------------------------------------


def test_retry_routing_key_preserves_original_when_it_matches_pattern() -> None:
    assert retry_routing_key(Queue.REPORTS, RoutingKey.REPORT_ALARM_DIGEST) == RoutingKey.REPORT_ALARM_DIGEST
    assert retry_routing_key(Queue.REPORTS, RoutingKey.REPORT_GENERATED) == RoutingKey.REPORT_GENERATED
    assert retry_routing_key(Queue.ARTICLES_KEYWORD, RoutingKey.ARTICLE_KEYWORD) == RoutingKey.ARTICLE_KEYWORD


def test_retry_routing_key_falls_back_for_foreign_keys() -> None:
    # varsayılan exchange üzerinden elle yayınlanmış mesajın routing key'i kuyruk adıdır → desene uymaz
    assert retry_routing_key(Queue.REPORTS, "q.reports") == RoutingKey.REPORT_GENERATED
    assert retry_routing_key(Queue.ARTICLES_RAW, "") == RoutingKey.ARTICLE_RAW
    assert retry_routing_key("q.unknown", "x.y") == "x.y"


def test_attempt_limits_by_exception_type(monkeypatch: pytest.MonkeyPatch) -> None:
    s = make_settings(rabbitmq_max_attempts=4)
    assert attempt_limit_for(s, ValueError("x")) == 4
    assert attempt_limit_for(s, Retry("x")) == 4
    assert attempt_limit_for(s, Unavailable("x")) == 6
    monkeypatch.setattr(broker_mod, "TRANSIENT_MAX_ATTEMPTS", 0)
    assert attempt_limit_for(s, Unavailable("x")) == 0  # sınırsız
    assert isinstance(Unavailable("x"), Retry)


def test_retry_exchange_name_follows_main_exchange() -> None:
    assert retry_exchange_name(make_settings()) == "news.topic.retry"
    assert retry_exchange_name(make_settings(rabbitmq_exchange="haber")) == "haber.retry"


def test_queue_arguments() -> None:
    s = make_settings(rabbitmq_retry_delay_ms=777)
    assert queue_arguments(s, Queue.ARTICLES_RAW) == {"x-dead-letter-exchange": "news.dlx"}
    reports = queue_arguments(s, Queue.REPORTS)
    assert reports == {"x-max-length": 1000, "x-message-ttl": 604_800_000}
    retry = retry_queue_arguments(s)
    assert retry == {"x-dead-letter-exchange": "news.topic", "x-message-ttl": 777}
    assert "x-dead-letter-routing-key" not in retry


# ---------------------------------------------------------------------------------------------------------
# InMemoryBroker: deneme tavanları
# ---------------------------------------------------------------------------------------------------------


def test_inmemory_retry_dead_letters_after_max_attempts() -> None:
    broker = InMemoryBroker(make_settings(rabbitmq_max_attempts=3))
    broker.publish(RoutingKey.ARTICLE_KEYWORD, {"id": "a"})

    def handler(msg: Message) -> None:
        raise Retry("kalıcı hata")

    assert broker.consume(Queue.ARTICLES_KEYWORD, handler) == 3
    assert len(broker.dead_letters) == 1 and broker.dead_letters[0].attempts == 3


def test_inmemory_unavailable_survives_past_max_attempts() -> None:
    broker = InMemoryBroker(make_settings(rabbitmq_max_attempts=2))
    broker.publish(RoutingKey.ARTICLE_KEYWORD, {"id": "a"})
    calls = {"n": 0}

    def handler(msg: Message) -> None:
        calls["n"] += 1
        if calls["n"] < 4:
            raise Unavailable("Ollama kapalı")

    assert broker.consume(Queue.ARTICLES_KEYWORD, handler) == 4  # 3 başarısız + 1 başarılı
    assert not broker.dead_letters


def test_inmemory_unavailable_eventually_dead_letters_at_transient_ceiling() -> None:
    broker = InMemoryBroker(make_settings(rabbitmq_max_attempts=2))
    broker.publish(RoutingKey.ARTICLE_KEYWORD, {"id": "a"})

    def handler(msg: Message) -> None:
        raise Unavailable("ES kapalı")

    assert broker.consume(Queue.ARTICLES_KEYWORD, handler) == 6
    assert len(broker.dead_letters) == 1 and broker.dead_letters[0].attempts == 6


# ---------------------------------------------------------------------------------------------------------
# RabbitMQBroker.consume
# ---------------------------------------------------------------------------------------------------------


def test_poison_messages_are_dead_lettered_not_reconnected(rabbit: FakeRabbit, no_sleep: list[float]) -> None:
    rabbit.deliver(b"not json")
    rabbit.deliver(b"[1, 2]")
    seen: list[Message] = []

    processed = rabbit.broker.consume(Queue.ARTICLES_KEYWORD, seen.append, max_messages=2)

    assert processed == 2
    assert seen == []
    assert rabbit.ch.nacks == [(1, False), (2, False)]
    assert rabbit.ch.acks == []
    assert len(rabbit.connections) == 1  # yeniden bağlanma yok
    assert no_sleep == []


def test_handler_runs_off_connection_thread_while_heartbeats_are_pumped(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"}, headers={"x-attempts": 0})
    info: dict[str, Any] = {}

    def handler(msg: Message) -> None:
        info["thread"] = threading.current_thread().name
        info["pumps_at_start"] = len(rabbit.conn.pump_calls)
        time.sleep(0.5)
        info["pumps_at_end"] = len(rabbit.conn.pump_calls)
        info["msg"] = msg

    assert rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, max_messages=1) == 1
    assert info["thread"] != threading.current_thread().name
    assert info["pumps_at_end"] - info["pumps_at_start"] >= 2  # işleyici sürerken ioloop işletildi
    assert info["msg"].queue == Queue.ARTICLES_KEYWORD and info["msg"].body == {"id": "x"}
    assert rabbit.ch.acks == [1] and rabbit.ch.nacks == []


def test_handler_may_publish_through_the_same_broker(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"})

    def handler(msg: Message) -> None:
        time.sleep(0.25)
        rabbit.broker.publish(RoutingKey.ARTICLE_SCORED, {"id": msg.body["id"], "stage": "scored"})

    assert rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, max_messages=1) == 1
    assert [p["routing_key"] for p in rabbit.ch.published] == [RoutingKey.ARTICLE_SCORED]
    assert rabbit.ch.published[0]["exchange"] == "news.topic"
    assert rabbit.ch.acks == [1]


def test_reject_dead_letters_immediately(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"})

    def handler(msg: Message) -> None:
        raise Reject("geçersiz")

    rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, max_messages=1)
    assert rabbit.ch.nacks == [(1, False)] and rabbit.ch.acks == [] and rabbit.ch.published == []


def test_failure_goes_to_retry_exchange_with_original_routing_key(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"report_id": "r1"}, routing_key=RoutingKey.REPORT_ALARM_DIGEST, headers={"x-attempts": 1})

    def handler(msg: Message) -> None:
        raise RuntimeError("patladı")

    rabbit.broker.consume(Queue.REPORTS, handler, max_messages=1)
    assert len(rabbit.ch.published) == 1
    pub = rabbit.ch.published[0]
    assert pub["exchange"] == "news.topic.retry"
    assert pub["routing_key"] == RoutingKey.REPORT_ALARM_DIGEST  # report.generated'a yeniden yazılmaz
    assert pub["headers"]["x-attempts"] == 2
    assert pub["headers"]["x-origin-queue"] == Queue.REPORTS
    assert "patladı" in pub["headers"]["x-error"]
    assert json.loads(pub["body"]) == {"report_id": "r1"}
    assert rabbit.ch.acks == [1] and rabbit.ch.nacks == []


def test_retry_uses_default_key_for_messages_published_via_default_exchange(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"}, routing_key="q.articles.keyword")

    def handler(msg: Message) -> None:
        raise Retry("tekrar")

    rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, max_messages=1)
    assert rabbit.ch.published[0]["routing_key"] == RoutingKey.ARTICLE_KEYWORD


def test_generic_failure_dead_letters_after_max_attempts(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"}, headers={"x-attempts": 2})  # 3. deneme = tavan

    def handler(msg: Message) -> None:
        raise Retry("yine olmadı")

    rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, max_messages=1)
    assert rabbit.ch.nacks == [(1, False)] and rabbit.ch.published == []


def test_unavailable_keeps_retrying_beyond_max_attempts(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"}, headers={"x-attempts": 4})  # rabbitmq_max_attempts=3 aşıldı, transient=6 aşılmadı
    rabbit.deliver({"id": "y"}, headers={"x-attempts": 5})  # transient tavana ulaşır → ölü mektup

    def handler(msg: Message) -> None:
        raise Unavailable("Elasticsearch erişilemiyor")

    rabbit.broker.consume(Queue.ARTICLES_SCORED, handler, max_messages=2)
    assert [p["headers"]["x-attempts"] for p in rabbit.ch.published] == [5]
    assert rabbit.ch.acks == [1]
    assert rabbit.ch.nacks == [(2, False)]


def test_unlimited_transient_attempts(rabbit: FakeRabbit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_mod, "TRANSIENT_MAX_ATTEMPTS", 0)
    rabbit.deliver({"id": "x"}, headers={"x-attempts": 10_000})

    def handler(msg: Message) -> None:
        raise Unavailable("hâlâ kapalı")

    rabbit.broker.consume(Queue.ARTICLES_SCORED, handler, max_messages=1)
    assert rabbit.ch.published[0]["headers"]["x-attempts"] == 10_001 and rabbit.ch.nacks == []


def test_malformed_attempts_header_is_tolerated(rabbit: FakeRabbit) -> None:
    rabbit.deliver({"id": "x"}, headers={"x-attempts": "abc"})
    seen: list[Message] = []
    rabbit.broker.consume(Queue.ARTICLES_KEYWORD, seen.append, max_messages=1)
    assert seen[0].attempts == 0 and rabbit.ch.acks == [1]


def test_stop_consumer_returns_undispatched_prefetch_to_broker(rabbit: FakeRabbit) -> None:
    for i in range(3):
        rabbit.deliver({"id": str(i)})
    seen: list[Message] = []

    processed = rabbit.broker.consume(Queue.ARTICLES_KEYWORD, seen.append, max_messages=1)

    assert processed == 1 and [m.body["id"] for m in seen] == ["0"]
    assert rabbit.ch.acks == [1]
    assert rabbit.ch.cancelled == ["ctag-q.articles.keyword"]
    assert rabbit.ch.nacks == [(2, True), (3, True)]


def test_channel_lost_during_handler_triggers_reconnect(rabbit: FakeRabbit, monkeypatch: pytest.MonkeyPatch) -> None:
    rabbit.deliver({"id": "x"})
    stop = threading.Event()
    first_conn = rabbit.conn

    def handler(msg: Message) -> None:
        first_conn.close()  # broker bağlantıyı düşürdü (ör. heartbeat zaman aşımı)

    def fake_sleep(seconds: float) -> None:
        stop.set()  # yeniden bağlanma beklemesi: testi burada sonlandır

    monkeypatch.setattr(broker_mod.time, "sleep", fake_sleep)
    processed = rabbit.broker.consume(Queue.ARTICLES_KEYWORD, handler, stop_event=stop)
    assert processed == 0
    assert first_conn.channels[0].acks == [] and first_conn.channels[0].nacks == []
    assert stop.is_set()


# ---------------------------------------------------------------------------------------------------------
# RabbitMQBroker.publish / declare_topology
# ---------------------------------------------------------------------------------------------------------


def test_publish_renews_idle_connection_without_waiting(rabbit: FakeRabbit, no_sleep: list[float]) -> None:
    rabbit.broker.publish(RoutingKey.ARTICLE_RAW, {"id": "1"})
    stale = rabbit.conn
    stale.pump_error = ConnectionResetError("heartbeat kaçırıldı, broker kapattı")

    rabbit.broker.publish(RoutingKey.ARTICLE_RAW, {"id": "2"})

    assert len(rabbit.connections) == 2
    assert [p["routing_key"] for p in rabbit.ch.published] == [RoutingKey.ARTICLE_RAW]
    assert json.loads(rabbit.ch.published[0]["body"]) == {"id": "2"}
    assert no_sleep == []  # uyarı/bekleme olmadan yeniden bağlandı
    assert stale.pump_calls  # heartbeat için ioloop işletildi


def test_declare_topology_declares_retry_exchange_and_bounded_reports_queue(rabbit: FakeRabbit) -> None:
    rabbit.broker.declare_topology()
    ch = rabbit.ch
    assert ("news.topic.retry", "topic") in ch.exchanges
    declared = dict(ch.declared)
    for queue in QUEUE_BINDINGS:
        assert queue in declared and retry_queue_name(queue) in declared
        assert "x-dead-letter-routing-key" not in declared[retry_queue_name(queue)]
        assert declared[retry_queue_name(queue)]["x-dead-letter-exchange"] == "news.topic"
    assert declared[Queue.REPORTS] == {"x-max-length": 1000, "x-message-ttl": 604_800_000}
    assert declared[Queue.ARTICLES_RAW] == {"x-dead-letter-exchange": "news.dlx"}
    assert (Queue.REPORTS, "news.topic", "report.#") in ch.bound
    assert ("q.reports.retry", "news.topic.retry", "report.#") in ch.bound
    assert ("q.articles.keyword.retry", "news.topic.retry", "article.keyword") in ch.bound
    assert (Queue.DEAD_LETTER, "news.dlx", "#") in ch.bound
    assert rabbit.broker._topology_declared is True


def test_declare_topology_tolerates_inequivalent_queue_arguments(rabbit: FakeRabbit, caplog) -> None:
    rabbit._connect()
    first = rabbit.ch
    first.declare_errors["q.articles.keyword.retry"] = ChannelClosedByBroker(
        406, "PRECONDITION_FAILED - inequivalent arg 'x-message-ttl' for queue 'q.articles.keyword.retry'"
    )

    with caplog.at_level("WARNING", logger="scraperhryt.broker"):
        rabbit.broker.declare_topology()

    assert first.is_closed
    fresh = rabbit.ch
    assert fresh is not first
    assert fresh.passive_declared == ["q.articles.keyword.retry"]
    # kalan kuyruklar yeni kanalda declare/bind edildi
    assert "q.articles.scored" in dict(fresh.declared)
    assert ("q.articles.keyword.retry", "news.topic.retry", "article.keyword") in fresh.bound
    assert any("q.articles.keyword.retry" in rec.message and "x-message-ttl" in rec.message for rec in caplog.records)


def test_declare_topology_raises_on_other_channel_errors(rabbit: FakeRabbit) -> None:
    rabbit._connect()
    rabbit.ch.declare_errors[Queue.ARTICLES_RAW] = ChannelClosedByBroker(403, "ACCESS_REFUSED")
    with pytest.raises(ChannelClosedByBroker):
        rabbit.broker.declare_topology()


# ---------------------------------------------------------------------------------------------------------
# max_messages / gözlem tamponları
# ---------------------------------------------------------------------------------------------------------


def test_max_messages_does_not_set_callers_stop_event(rabbit: FakeRabbit) -> None:
    """Mesaj bütçesi dolunca tüketici çıkar ama paylaşılan kapanış olayına dokunmaz (InMemoryBroker ile aynı)."""
    for i in range(2):
        rabbit.deliver({"id": str(i)})
    shared = threading.Event()
    seen: list[Message] = []

    processed = rabbit.broker.consume(Queue.ARTICLES_KEYWORD, seen.append, stop_event=shared, max_messages=1)

    assert processed == 1 and [m.body["id"] for m in seen] == ["0"]
    assert shared.is_set() is False
    assert rabbit.ch.acks == [1] and rabbit.ch.nacks == [(2, True)]
    assert rabbit.ch.cancelled == ["ctag-q.articles.keyword"]

    broker = InMemoryBroker(make_settings())
    broker.publish(RoutingKey.ARTICLE_RAW, {"id": "a"})
    broker.publish(RoutingKey.ARTICLE_RAW, {"id": "b"})
    assert broker.consume(Queue.ARTICLES_RAW, seen.append, stop_event=shared, max_messages=1) == 1
    assert shared.is_set() is False and broker.size(Queue.ARTICLES_RAW) == 1


def test_inmemory_observation_buffers_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_mod, "DEAD_LETTER_HISTORY_LIMIT", 2)
    broker = InMemoryBroker(make_settings(), history_limit=3)
    for i in range(10):
        broker.publish(RoutingKey.ARTICLE_RAW, {"id": str(i)})

    assert [m.body["id"] for m in broker.published] == ["7", "8", "9"]
    assert broker.size(Queue.ARTICLES_RAW) == 10  # kuyruğun kendisi kırpılmaz

    def reject(msg: Message) -> None:
        raise Reject("bozuk")

    assert broker.consume(Queue.ARTICLES_RAW, reject) == 10
    assert [m.body["id"] for m in broker.dead_letters] == ["8", "9"]
    assert broker.size(Queue.ARTICLES_RAW) == 0


def test_inmemory_default_history_limit_keeps_single_run_summary_intact() -> None:
    broker = InMemoryBroker(make_settings())
    assert broker.history_limit == broker_mod.PUBLISHED_HISTORY_LIMIT >= 3 * make_settings().max_articles_per_run
    assert broker.published == []  # liste semantiği korunur (testler boş liste ile karşılaştırır)
