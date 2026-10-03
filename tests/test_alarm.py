"""Alarm katmanı ve alarm kanalları testleri (InMemoryBroker + InMemoryStore, ağ yok)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from scraperhryt.alarm_sinks import (
    LogSink,
    SinkError,
    TelegramSink,
    WebhookSink,
    build_sinks,
    format_alarm_html,
    format_alarm_text,
)
from scraperhryt.broker import InMemoryBroker, Message, Queue, Retry, RoutingKey, Unavailable
from scraperhryt.config import Settings
from scraperhryt.models import AlarmEvent, LLMVerdict, NewsRecord, Stage
from scraperhryt.pipeline import alarm as alarm_module
from scraperhryt.pipeline.alarm import AlarmService
from scraperhryt.store import InMemoryStore

TITLE = "Bakan, fon soruşturmasında yeni gözaltılar olduğunu açıkladı"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        alarm_threshold=60,
        alarm_log_path=str(tmp_path / "alarms" / "alarms.jsonl"),
        rabbitmq_max_attempts=3,
    )


class RecordingSink:
    name = "recording"

    def __init__(self) -> None:
        self.events: list[AlarmEvent] = []

    def send(self, event: AlarmEvent) -> None:
        self.events.append(event)


class FailingSink:
    name = "failing"

    def send(self, event: AlarmEvent) -> None:
        raise RuntimeError("kanal çöktü")


def make_record(score: int | None = None, *, content: str = "Soruşturma kapsamında üç kişi gözaltına alındı.") -> NewsRecord:
    rec = NewsRecord.new(
        source="hurriyet",
        content_url="https://www.hurriyet.com.tr/gundem/bakan-fon-sorusturmasi-123456",
        title=TITLE,
        subtitle="Soruşturma genişliyor",
        content=content,
        published_at=datetime(2026, 10, 2, 19, 20, tzinfo=UTC),
        category="gundem",
    )
    rec.matched_keywords = ["bakan", "fon"]
    if score is None:
        rec.mark_not_scored()
    else:
        verdict = LLMVerdict(
            model="qwen2.5:7b", alarm_score=score, reason="Fon soruşturmasında bakan düzeyinde gelişme.", summary="Üç gözaltı var."
        )
        rec.apply_verdict(verdict, threshold=60)
    return rec


def scored_message(record: NewsRecord) -> Message:
    return Message(body=record.to_message(), routing_key=RoutingKey.ARTICLE_SCORED, queue=Queue.ARTICLES_SCORED)


def alarm_messages(broker: InMemoryBroker) -> list[Message]:
    return [m for m in broker.published if m.routing_key == RoutingKey.ALARM_RAISED]


# ---------------------------------------------------------------------------------------------------------
# AlarmService
# ---------------------------------------------------------------------------------------------------------


def test_non_alarm_record_is_indexed_only(settings: Settings) -> None:
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    service = AlarmService(settings, broker, store, sinks=[sink])
    record = make_record(score=None)

    service.handle(scored_message(record))

    stored = store.get_record(record.id)
    assert stored is not None and stored["stage"] == Stage.ALARM
    assert stored["is_alarm"] is False and stored["alarm_reason"] == "" and stored["alarm_id"] == ""
    assert broker.size(Queue.ALARMS) == 0 and alarm_messages(broker) == []
    assert store.alarms == {} and sink.events == []
    assert service.stats.indexed == 1 and service.stats.alarms_raised == 0


def test_low_score_record_is_not_alarm(settings: Settings) -> None:
    broker, store = InMemoryBroker(settings), InMemoryStore()
    service = AlarmService(settings, broker, store, sinks=[])
    record = make_record(score=35)
    service.handle(scored_message(record))
    assert store.get_record(record.id)["alarm_score"] == 35
    assert store.alarms == {} and broker.size(Queue.ALARMS) == 0


def test_alarm_record_is_indexed_notified_and_published(settings: Settings) -> None:
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    broker.declare_topology()
    service = AlarmService(settings, broker, store, sinks=[sink])
    record = make_record(score=85)
    broker.publish(RoutingKey.ARTICLE_SCORED, record.to_message())

    assert service.run(max_messages=1) == 1

    stored = store.get_record(record.id)
    assert stored["stage"] == Stage.ALARM and stored["is_alarm"] is True
    assert stored["alarm_id"] and stored["alarmed_at"]
    assert "LLM Özeti: Üç gözaltı var." in stored["alarm_reason"]

    assert list(store.alarms) == [stored["alarm_id"]]
    alarm_doc = store.alarms[stored["alarm_id"]]
    assert alarm_doc["record_id"] == record.id and alarm_doc["alarm_score"] == 85
    assert alarm_doc["channels_notified"] == ["recording"] and alarm_doc["content"] == record.content

    queued = broker.drain(Queue.ALARMS)
    assert len(queued) == 1
    body = queued[0].body
    assert body["alarm_id"] == stored["alarm_id"] and body["record_id"] == record.id
    assert body["record"]["alarm_id"] == stored["alarm_id"] and body["record"]["stage"] == "alarm"
    assert body["channels_notified"] == ["recording"] and body["title"] == TITLE

    assert len(sink.events) == 1 and sink.events[0].alarm_id == stored["alarm_id"]
    assert service.stats.alarms_raised == 1 and service.stats.notify_failures == 0


def test_failing_sink_does_not_prevent_publish(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    service = AlarmService(settings, broker, store, sinks=[FailingSink(), sink])
    record = make_record(score=90)

    with caplog.at_level(logging.ERROR, logger="scraperhryt.pipeline.alarm"):
        service.handle(scored_message(record))

    assert len(alarm_messages(broker)) == 1
    assert alarm_messages(broker)[0].body["channels_notified"] == ["recording"]
    assert len(sink.events) == 1
    assert service.stats.notify_failures == 1 and service.stats.alarms_raised == 1
    assert any("kanal çöktü" in rec.message for rec in caplog.records)


def test_redelivery_is_idempotent(settings: Settings) -> None:
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    service = AlarmService(settings, broker, store, sinks=[sink])
    record = make_record(score=75)
    msg = scored_message(record)

    service.handle(msg)
    service.handle(Message(body=dict(msg.body), routing_key=msg.routing_key, queue=msg.queue, attempts=1))

    assert len(alarm_messages(broker)) == 1
    assert len(sink.events) == 1
    assert len(store.alarms) == 1
    assert service.stats.duplicates == 1 and service.stats.alarms_raised == 1


def test_updated_content_raises_a_new_alarm(settings: Settings) -> None:
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    service = AlarmService(settings, broker, store, sinks=[sink])
    first = make_record(score=75)
    updated = make_record(score=80, content="Güncelleme: gözaltı sayısı beşe çıktı.")
    assert first.id == updated.id and first.content_hash != updated.content_hash

    service.handle(scored_message(first))
    service.handle(scored_message(updated))

    assert len(store.alarms) == 2 and len(alarm_messages(broker)) == 2 and len(sink.events) == 2
    assert store.get_record(first.id)["alarm_score"] == 80


def test_invalid_message_goes_to_dead_letter(settings: Settings) -> None:
    broker, store = InMemoryBroker(settings), InMemoryStore()
    broker.declare_topology()
    service = AlarmService(settings, broker, store, sinks=[])
    broker.publish(RoutingKey.ARTICLE_SCORED, {"foo": "bar"})

    assert service.run() == 1
    assert len(broker.dead_letters) == 1
    assert "Geçersiz haber mesajı" in broker.dead_letters[0].headers["x-error"]
    assert service.stats.rejected == 1 and store.records == {}


def test_store_retry_is_retried_by_broker_then_dead_lettered(settings: Settings) -> None:
    class DownStore(InMemoryStore):
        def index_record(self, record, refresh=False, embedding=None):  # type: ignore[override]
            raise Retry("Elasticsearch erişilemiyor")

    broker, store = InMemoryBroker(settings), DownStore()
    broker.declare_topology()
    service = AlarmService(settings, broker, store, sinks=[])
    broker.publish(RoutingKey.ARTICLE_SCORED, make_record(score=None).to_message())

    processed = service.run()
    assert processed == settings.rabbitmq_max_attempts
    assert len(broker.dead_letters) == 1 and broker.dead_letters[0].attempts == settings.rabbitmq_max_attempts
    assert broker.size(Queue.ALARMS) == 0


def test_run_waits_for_indices_until_stop(settings: Settings) -> None:
    class NeverReadyStore(InMemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.tries = 0

        def ensure_indices(self) -> None:
            self.tries += 1
            raise Retry("ES kapalı")

    broker, store = InMemoryBroker(settings), NeverReadyStore()
    service = AlarmService(settings, broker, store, sinks=[])
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    assert service.run(stop_event=stop) == 0
    assert store.tries >= 1


def test_es_outage_is_unavailable_and_does_not_exhaust_attempt_budget(settings: Settings) -> None:
    """ES kesintisi (Unavailable) rabbitmq_max_attempts bütçesini tüketmez: mesaj kuyrukta kalır, ölü mektup olmaz."""

    class DownStore(InMemoryStore):
        def index_record(self, record, refresh=False, embedding=None):  # type: ignore[override]
            raise Unavailable("Elasticsearch erişilemiyor (index_record)")

    broker, store = InMemoryBroker(settings), DownStore()
    broker.declare_topology()
    service = AlarmService(settings, broker, store, sinks=[])
    broker.publish(RoutingKey.ARTICLE_SCORED, make_record(score=None).to_message())

    rounds = settings.rabbitmq_max_attempts + 2
    assert service.run(max_messages=rounds) == rounds
    assert len(broker.dead_letters) == 0 and broker.size(Queue.ARTICLES_SCORED) == 1
    pending = broker.drain(Queue.ARTICLES_SCORED)[0]
    assert pending.attempts == rounds and "erişilemiyor" in pending.headers["x-error"]


def test_handle_waits_for_store_before_reraising(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Depo Retry fırlatınca servis ES yeniden erişilebilir olana dek bekler (geri basınç), sonra istisnayı iletir."""
    monkeypatch.setattr(alarm_module, "_INDEX_RETRY_INITIAL_DELAY", 0.01)

    class FlakyStore(InMemoryStore):
        def __init__(self, outage_checks: int) -> None:
            super().__init__()
            self.outage_checks = outage_checks
            self.ensure_calls = 0

        def ensure_indices(self) -> None:
            self.ensure_calls += 1
            if self.ensure_calls <= self.outage_checks:
                raise Unavailable("ES kapalı")

        def index_record(self, record, refresh=False, embedding=None):  # type: ignore[override]
            raise Unavailable("Elasticsearch erişilemiyor (index_record)")

    broker, store = InMemoryBroker(settings), FlakyStore(outage_checks=2)
    service = AlarmService(settings, broker, store, sinks=[])
    with pytest.raises(Unavailable, match="index_record"):
        service.handle(scored_message(make_record(score=None)))
    assert store.ensure_calls == 3  # 2 başarısız denetim + 1 başarılı
    assert store.records == {} and broker.published == []

    # kapanış sinyali gelirse bekleme sonsuza dek sürmez; istisna yine iletilir
    stop = threading.Event()
    never = FlakyStore(outage_checks=10**6)
    service2 = AlarmService(settings, broker, never, sinks=[])
    service2._stop_event = stop
    threading.Timer(0.05, stop.set).start()
    with pytest.raises(Unavailable):
        service2.handle(scored_message(make_record(score=None)))
    assert 1 <= never.ensure_calls < 10**6


def test_embedder_vectors_are_stored_and_failures_are_soft(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, ollama_embedding_model="nomic-embed-text", embedding_dims=4, alarm_log_path=str(tmp_path / "a.jsonl"))

    class Embedder:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.texts: list[str] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.texts.extend(texts)
            if self.fail:
                raise ConnectionError("Ollama kapalı")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    broker, store, embedder = InMemoryBroker(settings), InMemoryStore(), Embedder()
    service = AlarmService(settings, broker, store, sinks=[], embedder=embedder)
    record = make_record(score=None)
    service.handle(scored_message(record))
    assert store.embeddings[record.id] == [0.1, 0.2, 0.3, 0.4]
    assert embedder.texts and TITLE in embedder.texts[0]

    broken = Embedder(fail=True)
    store2 = InMemoryStore()
    service2 = AlarmService(settings, broker, store2, sinks=[], embedder=broken)
    service2.handle(scored_message(record))
    assert record.id in store2.records and store2.embeddings == {}
    assert service2.stats.embed_failures == 1

    # embedding modeli ayarlı değilse embedder yok sayılır
    plain = Settings(_env_file=None, alarm_log_path=str(tmp_path / "b.jsonl"))
    service3 = AlarmService(plain, broker, InMemoryStore(), sinks=[], embedder=Embedder())
    assert service3.embedder is None


class DimsEmbedder:
    def __init__(self, dims: int, *, fail: bool = False) -> None:
        self.dims = dims
        self.fail = fail
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("Ollama kapalı")
        return [[0.5] * self.dims for _ in texts]


def test_embedding_with_wrong_dims_is_dropped_per_record(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Model EMBEDDING_DIMS'ten farklı boyut üretirse kayıt vektörsüz yazılır; alarm akışı etkilenmez."""
    settings = Settings(_env_file=None, ollama_embedding_model="mxbai-embed-large", embedding_dims=4, alarm_log_path=str(tmp_path / "a.jsonl"))
    broker, store, sink = InMemoryBroker(settings), InMemoryStore(), RecordingSink()
    service = AlarmService(settings, broker, store, sinks=[sink], embedder=DimsEmbedder(1024))
    record = make_record(score=85)
    with caplog.at_level(logging.ERROR, logger="scraperhryt.pipeline.alarm"):
        service.handle(scored_message(record))
    assert record.id in store.records and store.embeddings == {}
    assert service.stats.embed_failures == 1 and service.stats.alarms_raised == 1
    assert len(sink.events) == 1 and len(alarm_messages(broker)) == 1
    assert any("1024" in r.getMessage() and "EMBEDDING_DIMS=4" in r.getMessage() for r in caplog.records)


def test_run_probes_embedding_dims_at_startup(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """run(): boyut uyuşmazlığında vektör üretimi kapatılır (hata logu); model erişilemezse açık kalır."""
    settings = Settings(_env_file=None, ollama_embedding_model="mxbai-embed-large", embedding_dims=4, alarm_log_path=str(tmp_path / "a.jsonl"))
    broker = InMemoryBroker(settings)
    broker.declare_topology()

    wrong = DimsEmbedder(1024)
    service = AlarmService(settings, broker, InMemoryStore(), sinks=[], embedder=wrong)
    with caplog.at_level(logging.ERROR, logger="scraperhryt.pipeline.alarm"):
        assert service.run() == 0
    assert service.embedder is None and wrong.calls == 1
    assert any("KAPATILDI" in r.getMessage() and "EMBEDDING_DIMS=4" in r.getMessage() for r in caplog.records)

    ok = DimsEmbedder(4)
    service_ok = AlarmService(settings, broker, InMemoryStore(), sinks=[], embedder=ok)
    assert service_ok.run() == 0
    assert service_ok.embedder is ok and ok.calls == 1

    down = DimsEmbedder(1024, fail=True)
    service_down = AlarmService(settings, broker, InMemoryStore(), sinks=[], embedder=down)
    assert service_down.run() == 0
    assert service_down.embedder is down  # erişilemeyen model başlangıçta kapatılmaz; denetim kayıt başına yapılır


class MappingMismatchStore(InMemoryStore):
    """``ElasticsearchStore`` gibi eşleme uyuşmazlığında vektör indekslemeyi kapatmış depo."""

    embeddings_enabled = False


def test_embedder_is_disabled_when_store_rejects_embeddings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Depo (ensure_indices eşleme uzlaştırması) vektörleri kapattıysa kayıt başına boşa embedding üretilmez."""
    settings = Settings(_env_file=None, ollama_embedding_model="nomic-embed-text", embedding_dims=4, alarm_log_path=str(tmp_path / "a.jsonl"))
    broker, store, embedder = InMemoryBroker(settings), MappingMismatchStore(), DimsEmbedder(4)
    service = AlarmService(settings, broker, store, sinks=[], embedder=embedder)
    record = make_record(score=85)
    with caplog.at_level(logging.WARNING, logger="scraperhryt.pipeline.alarm"):
        service.handle(scored_message(record))
    assert record.id in store.records and store.embeddings == {} and embedder.calls == 0
    assert service.embedder is None and service.stats.alarms_raised == 1 and service.stats.embed_failures == 0
    assert any("vektör üretimi KAPATILDI" in r.getMessage() for r in caplog.records)

    # run(): başlangıç denetimi de depo bayrağına bakar; model hiç sorgulanmaz
    broker.declare_topology()
    probe = DimsEmbedder(4)
    service_run = AlarmService(settings, broker, MappingMismatchStore(), sinks=[], embedder=probe)
    assert service_run.run() == 0 and service_run.embedder is None and probe.calls == 0


def test_default_sinks_are_built_from_settings(settings: Settings) -> None:
    service = AlarmService(settings, InMemoryBroker(settings), InMemoryStore())
    assert [s.name for s in service.sinks] == ["log"]


# ---------------------------------------------------------------------------------------------------------
# Kanallar
# ---------------------------------------------------------------------------------------------------------


def test_log_sink_appends_jsonl_and_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "nested" / "dir" / "alarms.jsonl"
    sink = LogSink(path)
    event = AlarmEvent.from_record(make_record(score=85))
    with caplog.at_level(logging.WARNING, logger="scraperhryt.alarm_sinks"):
        sink.send(event)
        sink.send(event)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    doc = json.loads(lines[0])
    assert doc["alarm_id"] == event.alarm_id and doc["alarm_score"] == 85 and "record" not in doc
    assert doc["content"] == event.record.content
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and warnings[0].getMessage().startswith("ALARM [85] ")
    assert event.content_url in warnings[0].getMessage()


def test_format_alarm_text_contains_key_fields() -> None:
    event = AlarmEvent.from_record(make_record(score=85))
    text = format_alarm_text(event)
    assert text.startswith("*ALARM [85/100]* " + TITLE)
    assert "Kaynak: hurriyet" in text and "02.10.2026 22:20" in text  # UTC 19:20 → TRT 22:20
    assert "bakan, fon" in text and "LLM Özeti: Üç gözaltı var." in text
    assert text.rstrip().endswith(event.content_url)


def test_format_alarm_html_escapes_and_respects_limit() -> None:
    rec = make_record(score=85, content="x")
    rec.title = "Bakan <b>&</b> fon"
    rec.alarm_reason = "A & B <script>" + ("çok uzun gerekçe " * 600)
    event = AlarmEvent.from_record(rec)
    html_text = format_alarm_html(event)
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in html_text and "<script>" not in html_text
    assert "&amp; B &lt;script&gt;" in html_text
    assert html_text.startswith("<b>ALARM [85/100]</b>")
    assert html_text.endswith('<a href="https://www.hurriyet.com.tr/gundem/bakan-fon-sorusturmasi-123456">Habere git</a>')
    assert len(html_text) <= 4096


def test_webhook_sink_posts_text_and_event() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"ok": True})

    sink = WebhookSink("https://hooks.example.com/services/T000/B000/secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    event = AlarmEvent.from_record(make_record(score=85))
    sink.send(event)
    assert len(received) == 1 and received[0].method == "POST"
    payload = json.loads(received[0].content)
    assert payload["text"].startswith("*ALARM [85/100]*")
    assert payload["event"]["alarm_id"] == event.alarm_id and payload["event"]["record"]["id"] == event.record_id


def test_webhook_sink_raises_sink_error_without_leaking_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="kapalı")

    url = "https://hooks.example.com/services/T000/B000/secret"
    sink = WebhookSink(url, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(SinkError) as excinfo:
        sink.send(AlarmEvent.from_record(make_record(score=85)))
    assert "HTTP 500" in str(excinfo.value) and "secret" not in str(excinfo.value)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("bağlantı yok", request=request)

    sink2 = WebhookSink(url, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(SinkError) as excinfo2:
        sink2.send(AlarmEvent.from_record(make_record(score=85)))
    assert "ConnectError" in str(excinfo2.value) and "secret" not in str(excinfo2.value)


def test_telegram_sink_sends_html_and_hides_token() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    token = "123456:ABC-SECRET"
    sink = TelegramSink(token, " 987 ", client=httpx.Client(transport=httpx.MockTransport(handler)))
    rec = make_record(score=85)
    rec.title = "Bakan & fon"
    sink.send(AlarmEvent.from_record(rec))
    assert received[0].url == httpx.URL(f"https://api.telegram.org/bot{token}/sendMessage")
    payload = json.loads(received[0].content)
    assert payload["chat_id"] == "987" and payload["parse_mode"] == "HTML"
    assert "Bakan &amp; fon" in payload["text"] and "Habere git" in payload["text"]

    def api_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})

    bad = TelegramSink(token, "1", client=httpx.Client(transport=httpx.MockTransport(api_error)))
    with pytest.raises(SinkError, match="chat not found"):
        bad.send(AlarmEvent.from_record(rec))

    def http_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"ok": False})

    unauthorized = TelegramSink(token, "1", client=httpx.Client(transport=httpx.MockTransport(http_error)))
    with pytest.raises(SinkError) as excinfo:
        unauthorized.send(AlarmEvent.from_record(rec))
    assert "HTTP 401" in str(excinfo.value) and token not in str(excinfo.value)


def test_build_sinks_from_settings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = dict(_env_file=None, alarm_log_path=str(tmp_path / "alarms.jsonl"))
    assert [s.name for s in build_sinks(Settings(**base))] == ["log"]

    full = Settings(**base, alarm_webhook_url="https://hooks.example.com/x", telegram_bot_token="t", telegram_chat_id="c")
    sinks = build_sinks(full)
    assert [s.name for s in sinks] == ["log", "webhook", "telegram"]
    assert isinstance(sinks[0], LogSink) and sinks[0].path == tmp_path / "alarms.jsonl"
    assert isinstance(sinks[1], WebhookSink) and sinks[1].url == "https://hooks.example.com/x"
    assert isinstance(sinks[2], TelegramSink) and sinks[2].chat_id == "c"

    with caplog.at_level(logging.WARNING, logger="scraperhryt.alarm_sinks"):
        half = build_sinks(Settings(**base, telegram_bot_token="t"))
    assert [s.name for s in half] == ["log"]
    assert any("TELEGRAM_CHAT_ID" in r.getMessage() for r in caplog.records)
