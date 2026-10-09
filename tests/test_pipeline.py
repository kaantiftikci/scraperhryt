"""Boru hattı testleri: anahtar kelime filtresi, LLM istemcileri ve skorlama servisi.

Tümü çevrimdışıdır: ``InMemoryBroker`` + ``FakeOllama`` (ve ``OllamaClient`` için ``httpx.MockTransport``).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from scraperhryt.broker import (
    InMemoryBroker,
    Message,
    Queue,
    Retry,
    RoutingKey,
    Unavailable,
    attempt_limit_for,
)
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord, Stage
from scraperhryt.pipeline.keyword_filter import KeywordFilterService
from scraperhryt.pipeline.llm import (
    FakeOllama,
    HeuristicLLM,
    LLMBadOutput,
    LLMUnavailable,
    OllamaClient,
    extract_entities,
    extract_json_object,
    hashed_vector,
    normalize_model_name,
    redact_url,
)
from scraperhryt.pipeline.prompts import (
    STRICT_JSON_REMINDER,
    TOPICS,
    TRUNCATION_MARKER,
    VERDICT_KEYS,
    build_system_prompt,
    build_user_prompt,
    parse_user_prompt,
    truncate_content,
)
from scraperhryt.pipeline.scorer import (
    FALLBACK_MODEL,
    MAX_LLM_ATTEMPTS,
    MAX_UNAVAILABLE_ROUNDS,
    RAW_LIMIT,
    ScoringService,
    build_verdict,
    coerce_bool,
    coerce_score,
    coerce_str_list,
    normalize_topic,
)

IST = timezone(timedelta(hours=3))
PUBLISHED = datetime(2026, 10, 2, 22, 20, tzinfo=IST)

BAKAN_TEXT = (
    "İçişleri Bakanı Ali Yerlikaya, belediyelere yönelik soruşturmanın genişleyeceğini açıkladı. "
    "Operasyon kapsamında 12 kişi tutuklandı."
)
PLAIN_TEXT = "İstanbul'da hafta sonu güneşli bir hava bekleniyor. Vatandaşlar parklara akın etti."


# ---------------------------------------------------------------------------------------------------------
# yardımcılar
# ---------------------------------------------------------------------------------------------------------


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "rabbitmq_url": "memory://",
        "keywords": "bakan,cumhurbaşkanı,fon",
        "alarm_threshold": 60,
        "rabbitmq_max_attempts": 3,
        "llm_score_all": False,
        "ollama_max_content_chars": 6000,
        "embedding_dims": 16,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_record(
    title: str,
    content: str,
    *,
    subtitle: str = "",
    source: str = "hurriyet",
    slug: str = "haber-1",
) -> NewsRecord:
    url = (
        f"https://www.hurriyet.com.tr/gundem/{slug}-{abs(hash(slug)) % 100000}"
        if source == "hurriyet"
        else f"https://12punto.com.tr/gundem/{slug}-{abs(hash(slug)) % 100000}"
    )
    return NewsRecord.new(
        source=source,
        content_url=url,
        title=title,
        subtitle=subtitle,
        content=content,
        published_at=PUBLISHED,
        category="gundem",
    )


def verdict_dict(score: int, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "alarm_score": score,
        "is_alarm": score >= 60,
        "reason": f"Gerekçe: skor {score}; 'bakan' gerçek bir bakanı ifade ediyor.",
        "summary": "Bakan soruşturmanın genişleyeceğini açıkladı. 12 kişi tutuklandı.",
        "topics": ["hukuk", "siyaset"],
        "entities": ["Ali Yerlikaya", "İçişleri Bakanlığı"],
    }
    data.update(extra)
    return data


def raw_message(record: NewsRecord) -> Message:
    return Message(body=record.to_message(), routing_key=RoutingKey.ARTICLE_RAW, queue=Queue.ARTICLES_RAW)


def keyword_message(record: NewsRecord, keywords: list[str] | None = None) -> Message:
    record.matched_keywords = ["bakan"] if keywords is None else keywords
    record.stage = Stage.KEYWORD
    return Message(
        body=record.to_message(), routing_key=RoutingKey.ARTICLE_KEYWORD, queue=Queue.ARTICLES_KEYWORD
    )


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def broker(settings: Settings) -> InMemoryBroker:
    b = InMemoryBroker(settings)
    b.declare_topology()
    return b


# ---------------------------------------------------------------------------------------------------------
# Anahtar kelime filtresi
# ---------------------------------------------------------------------------------------------------------


class TestKeywordFilter:
    def test_hit_routes_to_keyword_queue(self, settings: Settings, broker: InMemoryBroker) -> None:
        svc = KeywordFilterService(settings, broker)
        record = make_record("Bakan Yerlikaya açıklama yaptı", BAKAN_TEXT)
        svc.handle(raw_message(record))

        assert broker.size(Queue.ARTICLES_KEYWORD) == 1
        assert broker.size(Queue.ARTICLES_SCORED) == 0
        msg = broker.queues[Queue.ARTICLES_KEYWORD][0]
        assert msg.routing_key == RoutingKey.ARTICLE_KEYWORD
        out = NewsRecord.from_message(msg.body)
        assert out.id == record.id
        assert out.matched_keywords == ["bakan"]
        assert out.stage == Stage.KEYWORD
        assert out.alarm_score == 0 and out.alarm_reason == "" and not out.is_alarm
        assert svc.stats.hits == 1 and svc.stats.misses == 0

    def test_multiple_keywords_and_suffixes(self, settings: Settings, broker: InMemoryBroker) -> None:
        svc = KeywordFilterService(settings, broker)
        record = make_record(
            "Cumhurbaşkanlığı kararnamesi yayımlandı",
            "Bakanlıklar yeniden yapılandırıldı; fonların denetimi SPK'ya geçti.",
        )
        svc.handle(raw_message(record))
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_KEYWORD][0].body)
        assert out.matched_keywords == ["bakan", "cumhurbaşkanı", "fon"]

    def test_miss_routes_to_scored_queue(self, settings: Settings, broker: InMemoryBroker) -> None:
        svc = KeywordFilterService(settings, broker)
        record = make_record("Hafta sonu hava güzel", PLAIN_TEXT, slug="hava")
        svc.handle(raw_message(record))

        assert broker.size(Queue.ARTICLES_KEYWORD) == 0
        assert broker.size(Queue.ARTICLES_SCORED) == 1
        msg = broker.queues[Queue.ARTICLES_SCORED][0]
        assert msg.routing_key == RoutingKey.ARTICLE_SCORED
        out = NewsRecord.from_message(msg.body)
        assert out.id == record.id
        assert out.stage == Stage.SCORED
        assert out.matched_keywords == []
        assert out.alarm_score == 0
        assert out.is_alarm is False
        assert out.alarm_reason == "" and out.llm_summary == ""
        assert out.llm is None
        assert out.processed_at is not None and out.processed_at.tzinfo is not None
        assert svc.stats.misses == 1 and svc.stats.hits == 0

    def test_llm_score_all_routes_everything_to_keyword_queue(self) -> None:
        settings = make_settings(llm_score_all=True)
        broker = InMemoryBroker(settings)
        broker.declare_topology()
        svc = KeywordFilterService(settings, broker)
        svc.handle(raw_message(make_record("Hafta sonu hava güzel", PLAIN_TEXT, slug="hava")))
        svc.handle(raw_message(make_record("Bakan açıkladı", BAKAN_TEXT, slug="bakan")))

        assert broker.size(Queue.ARTICLES_SCORED) == 0
        assert broker.size(Queue.ARTICLES_KEYWORD) == 2
        bodies = [NewsRecord.from_message(m.body) for m in broker.queues[Queue.ARTICLES_KEYWORD]]
        assert bodies[0].matched_keywords == [] and bodies[0].stage == Stage.KEYWORD
        assert bodies[1].matched_keywords == ["bakan"] and bodies[1].stage == Stage.KEYWORD
        assert svc.stats.forwarded_unmatched == 1 and svc.stats.hits == 1

    def test_invalid_message_is_rejected_to_dead_letter(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        svc = KeywordFilterService(settings, broker)
        broker.publish(RoutingKey.ARTICLE_RAW, {"id": "x", "title": "eksik alanlar"})
        processed = svc.run()
        assert processed == 1
        assert len(broker.dead_letters) == 1
        assert broker.dead_letters[0].attempts == 0  # Reject → yeniden deneme yok
        assert "Geçersiz haber mesajı" in broker.dead_letters[0].headers["x-error"]
        assert svc.stats.rejected == 1
        assert broker.size(Queue.ARTICLES_KEYWORD) == 0 and broker.size(Queue.ARTICLES_SCORED) == 0

    def test_run_consumes_raw_queue(self, settings: Settings, broker: InMemoryBroker) -> None:
        broker.publish(
            RoutingKey.ARTICLE_RAW, make_record("Bakan açıkladı", BAKAN_TEXT, slug="a").to_message()
        )
        broker.publish(RoutingKey.ARTICLE_RAW, make_record("Hava", PLAIN_TEXT, slug="b").to_message())
        svc = KeywordFilterService(settings, broker)
        assert svc.run() == 2
        assert broker.size(Queue.ARTICLES_RAW) == 0
        assert broker.size(Queue.ARTICLES_KEYWORD) == 1
        assert broker.size(Queue.ARTICLES_SCORED) == 1


# ---------------------------------------------------------------------------------------------------------
# JSON çıkarma yardımcısı
# ---------------------------------------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_plain_json(self) -> None:
        assert extract_json_object('{"alarm_score": 5}') == {"alarm_score": 5}

    def test_fenced_json_block(self) -> None:
        text = 'İşte sonuç:\n```json\n{"alarm_score": 85, "is_alarm": true}\n```\nUmarım yardımcı olur.'
        assert extract_json_object(text) == {"alarm_score": 85, "is_alarm": True}

    def test_fence_without_language(self) -> None:
        text = '```\n{"alarm_score": 1, "topics": ["siyaset"]}\n```'
        assert extract_json_object(text)["topics"] == ["siyaset"]

    def test_prefixed_prose_without_fence(self) -> None:
        text = (
            'Değerlendirmem şu şekilde: {"alarm_score": 42, "reason": "metin { içinde } parantez"} ve bitti.'
        )
        assert extract_json_object(text) == {"alarm_score": 42, "reason": "metin { içinde } parantez"}

    def test_first_object_wins_when_multiple(self) -> None:
        text = '{"alarm_score": 10} {"alarm_score": 90}'
        assert extract_json_object(text)["alarm_score"] == 10

    def test_trailing_comma_is_tolerated(self) -> None:
        assert extract_json_object('{"alarm_score": 7, "topics": ["a",],}') == {
            "alarm_score": 7,
            "topics": ["a"],
        }

    def test_list_wrapped_object(self) -> None:
        assert extract_json_object('[{"alarm_score": 3}]') == {"alarm_score": 3}

    @pytest.mark.parametrize(
        "text", ["", "   ", "sadece düz yazı", "[1, 2, 3]", "{bozuk json", "```json\n```"]
    )
    def test_bad_outputs_raise(self, text: str) -> None:
        with pytest.raises(LLMBadOutput):
            extract_json_object(text)


# ---------------------------------------------------------------------------------------------------------
# OllamaClient (httpx.MockTransport ile)
# ---------------------------------------------------------------------------------------------------------


def _client(settings: Settings, handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    return OllamaClient(settings, transport=httpx.MockTransport(handler))


class TestOllamaClient:
    def test_chat_json_sends_expected_payload_and_parses(self, settings: Settings) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content)
            content = json.dumps(verdict_dict(85), ensure_ascii=False)
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": content}, "done": True}
            )

        with _client(settings, handler) as client:
            data = client.chat_json("SİSTEM", "KULLANICI", temperature=0.3, num_ctx=2048)
        assert data["alarm_score"] == 85
        assert seen["path"] == "/api/chat"
        payload = seen["payload"]
        assert payload["model"] == settings.ollama_model
        assert payload["format"] == "json" and payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "SİSTEM"},
            {"role": "user", "content": "KULLANICI"},
        ]
        assert payload["options"] == {"temperature": 0.3, "num_ctx": 2048}

    def test_chat_json_default_options_from_settings(self, settings: Settings) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}})

        with _client(settings, handler) as client:
            client.chat_json("s", "u")
        assert seen["payload"]["options"] == {
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
        }

    def test_chat_json_prose_is_bad_output(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": "Bu bir JSON değil."}}
            )

        with _client(settings, handler) as client, pytest.raises(LLMBadOutput):
            client.chat_json("s", "u")

    def test_chat_json_empty_content_is_bad_output(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"message": {"role": "assistant", "content": ""}, "done_reason": "load"}
            )

        with _client(settings, handler) as client, pytest.raises(LLMBadOutput):
            client.chat_json("s", "u")

    def test_server_error_is_unavailable(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal error")

        with _client(settings, handler) as client, pytest.raises(LLMUnavailable):
            client.chat_json("s", "u")

    def test_model_not_found_is_unavailable(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model 'qwen2.5:7b' not found"})

        with _client(settings, handler) as client, pytest.raises(LLMUnavailable, match="not found"):
            client.chat_json("s", "u")

    def test_connection_error_is_unavailable(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("bağlantı reddedildi", request=request)

        with _client(settings, handler) as client:
            with pytest.raises(LLMUnavailable):
                client.chat_json("s", "u")
            assert client.health() is False
            assert client.model_available() is False

    def test_credentials_in_base_url_are_redacted(self) -> None:
        settings = make_settings(ollama_base_url="http://svc:S3cretT0ken@ollama.internal:11434")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"bağlantı reddedildi: {request.url}", request=request)

        with _client(settings, handler) as client:
            assert client.display_url == "http://svc:***@ollama.internal:11434"
            with pytest.raises(LLMUnavailable) as info:
                client.chat_json("s", "u")
        message = str(info.value)
        assert "S3cretT0ken" not in message
        assert "ollama.internal:11434/api/chat" in message and "svc:***@" in message

    def test_redact_url_helper(self) -> None:
        assert redact_url("http://localhost:11434") == "http://localhost:11434"
        assert redact_url("http://u:p@h:1/x?q=1") == "http://u:***@h:1/x?q=1"
        assert redact_url("http://u@[::1]:1") == "http://u:***@[::1]:1"
        assert redact_url("") == ""

    def test_timeout_is_unavailable(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("zaman aşımı", request=request)

        with _client(settings, handler) as client, pytest.raises(LLMUnavailable, match="zaman aşımı"):
            client.generate_text("s", "u")

    def test_generate_text(self, settings: Settings) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "  Türkçe anlatı.  ", "done": True})

        with _client(settings, handler) as client:
            assert client.generate_text("SİSTEM", "SORU") == "Türkçe anlatı."
        assert seen["path"] == "/api/generate"
        assert seen["payload"]["system"] == "SİSTEM" and seen["payload"]["prompt"] == "SORU"
        assert seen["payload"]["stream"] is False

    def test_embed_new_api(self) -> None:
        settings = make_settings(ollama_embedding_model="nomic-embed-text")
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        with _client(settings, handler) as client:
            vectors = client.embed(["bir", "iki"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert seen["path"] == "/api/embed"
        assert seen["payload"] == {"model": "nomic-embed-text", "input": ["bir", "iki"]}

    def test_embed_falls_back_to_legacy_endpoint_on_404(self, settings: Settings) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            calls.append((request.url.path, payload))
            if request.url.path == "/api/embed":
                return httpx.Response(404, text="404 page not found")
            assert request.url.path == "/api/embeddings"
            value = float(len(payload["prompt"]))
            return httpx.Response(200, json={"embedding": [value, 1.0]})

        with _client(settings, handler) as client:
            vectors = client.embed(["a", "", "abc"])
        assert vectors == [[1.0, 1.0], [0.0, 0.0], [3.0, 1.0]]  # boş metin → sıfır vektör, sıra korunur
        assert [c[0] for c in calls] == ["/api/embed", "/api/embeddings", "/api/embeddings"]

    def test_embed_legacy_mode_is_remembered(self, settings: Settings) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == "/api/embed":
                return httpx.Response(404, text="404 page not found")
            return httpx.Response(200, json={"embedding": [1.0, 2.0]})

        with _client(settings, handler) as client:
            assert client.embed(["a"]) == [[1.0, 2.0]]
            assert client.embed(["b"]) == [[1.0, 2.0]]
        assert paths == ["/api/embed", "/api/embeddings", "/api/embeddings"]  # ikinci çağrı /api/embed'i denemez

    def test_embed_model_not_found_is_not_legacy_fallback(self) -> None:
        settings = make_settings(ollama_embedding_model="nomic-embed-text")
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(404, json={"error": "model 'nomic-embed-text' not found, try pulling it first"})

        with _client(settings, handler) as client, pytest.raises(LLMUnavailable) as info:
            client.embed(["a", "b", "c"])
        assert paths == ["/api/embed"]  # eski uç noktaya düşülmez, her metin için ek istek atılmaz
        assert "/api/embed)" in str(info.value) and "not found" in str(info.value)
        assert "/api/embeddings" not in str(info.value)

    def test_embed_empty_input(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("boş girdi için istek atılmamalı")

        with _client(settings, handler) as client:
            assert client.embed([]) == []

    def test_model_available_and_tags(self, settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/tags" and request.method == "GET"
            return httpx.Response(
                200, json={"models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text:latest"}]}
            )

        with _client(settings, handler) as client:
            assert client.health() is True
            assert client.model_available() is True
            assert client.list_models() == ["qwen2.5:7b", "nomic-embed-text:latest"]
        with _client(make_settings(ollama_model="llama3"), handler) as client:
            assert client.model_available() is False

    def test_normalize_model_name(self) -> None:
        assert normalize_model_name("qwen2.5") == "qwen2.5:latest"
        assert normalize_model_name("qwen2.5:7b") == "qwen2.5:7b"
        assert normalize_model_name("") == ""


# ---------------------------------------------------------------------------------------------------------
# İstemler
# ---------------------------------------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_mentions_rubric_keys_and_topics(self) -> None:
        system = build_system_prompt(60)
        for key in VERDICT_KEYS:
            assert key in system
        for topic in TOPICS:
            assert topic in system
        assert "80-100" in system and "60-79" in system and "30-59" in system and "0-29" in system
        assert "60" in system and "fiil" in system  # eşik ve 'bakan' fiil uyarısı

    def test_user_prompt_contains_fields_and_truncates(self) -> None:
        record = make_record(
            "Bakan açıkladı", "kelime " * 2000, subtitle="Alt başlık burada", source="12punto"
        )
        record.matched_keywords = ["bakan"]
        user = build_user_prompt(record, max_content_chars=500)
        assert "Kaynak: 12punto" in user
        assert "Başlık: Bakan açıkladı" in user
        assert "Alt başlık: Alt başlık burada" in user
        assert "Haber tarihi: 02.10.2026 22:20" in user
        assert "Eşleşen anahtar kelimeler: bakan" in user
        assert TRUNCATION_MARKER in user
        fields = parse_user_prompt(user)
        assert fields["title"] == "Bakan açıkladı" and fields["keywords"] == "bakan"
        assert fields["content"].endswith(TRUNCATION_MARKER)
        assert len(fields["content"]) <= 500 + len(TRUNCATION_MARKER) + 1

    def test_user_prompt_without_truncation_or_keywords(self) -> None:
        record = make_record("Kısa haber", "Kısa içerik.")
        record.published_at = None
        user = build_user_prompt(record, max_content_chars=6000)
        assert TRUNCATION_MARKER not in user
        assert "Haber tarihi: bilinmiyor" in user
        assert "Alt başlık: -" in user
        fields = parse_user_prompt(user)
        assert fields["keywords"] == "" and fields["subtitle"] == "" and fields["content"] == "Kısa içerik."

    def test_truncate_content_marker(self) -> None:
        assert truncate_content("kısa", 100) == "kısa"
        out = truncate_content("a" * 50 + " " + "b" * 50, 60)
        assert out.endswith(TRUNCATION_MARKER) and out.startswith("a" * 50)


# ---------------------------------------------------------------------------------------------------------
# Karar dönüştürme
# ---------------------------------------------------------------------------------------------------------


class TestBuildVerdict:
    def test_coercions_and_clamping(self) -> None:
        verdict = build_verdict(
            {
                "alarm_score": "150/100",
                "is_alarm": "evet",
                "reason": ["  çok ", "önemli  "],
                "summary": None,
                "topics": "Dış Politika; guvenlik; spor; Finans / Fon",
                "entities": [{"name": "Ali Yerlikaya"}, " CHP ", "CHP", ""],
            },
            model="m",
            threshold=60,
            latency_ms=12,
            attempts=2,
            raw="x" * 5000,
        )
        assert verdict.alarm_score == 100 and verdict.is_alarm is True
        assert verdict.reason == "çok önemli" and verdict.summary == ""
        assert verdict.topics == ["dış politika", "güvenlik", "diğer", "finans/fon"]
        assert verdict.entities == ["Ali Yerlikaya", "CHP"]
        assert verdict.latency_ms == 12 and verdict.attempts == 2 and len(verdict.raw) == RAW_LIMIT
        assert verdict.model == "m" and verdict.scored_at.tzinfo is not None

    def test_negative_and_float_scores(self) -> None:
        assert build_verdict({"alarm_score": -5}, model="m", threshold=60).alarm_score == 0
        assert build_verdict({"alarm_score": 72.6}, model="m", threshold=60).alarm_score == 73
        assert coerce_score("yaklaşık 45 puan") == 45

    def test_is_alarm_defaults_to_threshold_policy(self) -> None:
        assert build_verdict({"alarm_score": 70}, model="m", threshold=60).is_alarm is True
        assert build_verdict({"alarm_score": 20}, model="m", threshold=60).is_alarm is False
        assert coerce_bool("hayır", default=True) is False
        assert coerce_bool(None, default=True) is True
        assert coerce_bool(1, default=False) is True

    def test_missing_score_is_bad_output(self) -> None:
        with pytest.raises(LLMBadOutput):
            build_verdict({"is_alarm": True}, model="m", threshold=60)
        with pytest.raises(LLMBadOutput):
            build_verdict({"alarm_score": "yok"}, model="m", threshold=60)
        with pytest.raises(LLMBadOutput):
            build_verdict({"alarm_score": True}, model="m", threshold=60)

    def test_topic_normalization(self) -> None:
        assert normalize_topic("siyaset") == "siyaset"
        assert normalize_topic("  Politika ") == "siyaset"
        assert normalize_topic("Hukuk/Adalet") == "hukuk"
        assert normalize_topic("tamamen alakasız") == "diğer"
        # kanonik ad içerme, "politika" takma adından önce gelir
        assert normalize_topic("ekonomi politikası") == "ekonomi"
        assert normalize_topic("para politikası") == "ekonomi"
        assert normalize_topic("finans politikası") == "finans/fon"
        assert normalize_topic("genel politikası") == "siyaset"  # tamlayanın konusu yoksa "politika" takma adı
        assert normalize_topic("maliye politikası") == "ekonomi"
        assert normalize_topic("güvenlik politikası") == "güvenlik"
        assert normalize_topic("dış politika") == "dış politika"
        assert normalize_topic("sosyal politika") == "sosyal"
        assert normalize_topic("iç politika") == "siyaset"
        assert coerce_str_list("a, b; c\nd") == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------------------------------------
# Skorlama servisi
# ---------------------------------------------------------------------------------------------------------


class TestScoringService:
    def test_alarm_when_score_above_threshold(self, settings: Settings, broker: InMemoryBroker) -> None:
        fake = FakeOllama(responder=lambda system, user: verdict_dict(85))
        svc = ScoringService(settings, broker, fake)
        record = make_record("Bakan Yerlikaya açıklama yaptı", BAKAN_TEXT)
        svc.handle(keyword_message(record))

        assert broker.size(Queue.ARTICLES_SCORED) == 1
        msg = broker.queues[Queue.ARTICLES_SCORED][0]
        assert msg.routing_key == RoutingKey.ARTICLE_SCORED
        out = NewsRecord.from_message(msg.body)
        assert out.id == record.id
        assert out.stage == Stage.SCORED
        assert out.alarm_score == 85 and out.is_alarm is True
        assert "Gerekçe: skor 85" in out.alarm_reason and "LLM Özeti:" in out.alarm_reason
        assert out.llm_summary == "Bakan soruşturmanın genişleyeceğini açıkladı. 12 kişi tutuklandı."
        assert out.matched_keywords == ["bakan"]
        assert out.llm is not None
        assert out.llm.model == "fake" and out.llm.attempts == 1 and out.llm.topics == ["hukuk", "siyaset"]
        assert out.llm.entities == ["Ali Yerlikaya", "İçişleri Bakanlığı"]
        assert json.loads(out.llm.raw)["alarm_score"] == 85
        assert out.processed_at is not None and out.processed_at.tzinfo is not None
        assert svc.stats.scored == 1 and svc.stats.alarms == 1
        # istem içeriği
        system, user = fake.chat_calls[0][1], fake.chat_calls[0][2]
        assert system == build_system_prompt(settings.alarm_threshold)
        assert "Eşleşen anahtar kelimeler: bakan" in user and "Başlık: Bakan Yerlikaya açıklama yaptı" in user

    def test_no_alarm_keeps_verdict_but_empties_alarm_fields(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        fake = FakeOllama(responder=lambda system, user: verdict_dict(20, summary="Rutin açıklama özeti."))
        svc = ScoringService(settings, broker, fake)
        svc.handle(keyword_message(make_record("Bakan açıkladı", BAKAN_TEXT)))

        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.alarm_score == 20 and out.is_alarm is False
        assert out.alarm_reason == "" and out.llm_summary == ""
        assert out.llm is not None and out.llm.summary == "Rutin açıklama özeti."
        assert out.llm.reason.startswith("Gerekçe: skor 20")
        assert svc.stats.alarms == 0 and svc.stats.scored == 1

    def test_threshold_boundary_is_inclusive(self, broker: InMemoryBroker) -> None:
        settings = make_settings(alarm_threshold=60)
        svc = ScoringService(settings, broker, FakeOllama(responder=lambda s, u: verdict_dict(60)))
        out = svc.score_record(make_record("Bakan", BAKAN_TEXT))
        assert out.is_alarm is True and out.alarm_score == 60

    def test_recovers_after_bad_output(self, settings: Settings, broker: InMemoryBroker) -> None:
        calls = {"n": 0}

        def responder(system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return "Elbette, haberi inceledim ve düşüncelerim şöyle..."
            return "Sonuç:\n```json\n" + json.dumps(verdict_dict(70), ensure_ascii=False) + "\n```"

        fake = FakeOllama(responder=responder)
        svc = ScoringService(settings, broker, fake)
        svc.handle(keyword_message(make_record("Bakan", BAKAN_TEXT)))

        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.llm is not None and out.llm.attempts == 2
        assert out.alarm_score == 70 and out.is_alarm is True
        assert len(fake.chat_calls) == 2
        assert not fake.chat_calls[0][2].endswith(STRICT_JSON_REMINDER)
        assert fake.chat_calls[1][2].endswith(STRICT_JSON_REMINDER)
        assert svc.stats.bad_output == 1

    def test_always_bad_output_is_retried_then_published_with_fallback_verdict(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        fake = FakeOllama(responder=lambda system, user: "asla json üretmiyorum")
        svc = ScoringService(settings, broker, fake)
        record = make_record("Bakan Yerlikaya açıklama yaptı", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body)

        # ilk denemeler: bozuk çıktı → Retry (gecikmeli yeniden deneme, x-attempts artar)
        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle, max_messages=1) == 1
        retried = broker.queues[Queue.ARTICLES_KEYWORD][0]
        assert retried.attempts == 1 and "geçerli karar üretemedi" in retried.headers["x-error"]
        assert broker.size(Queue.ARTICLES_SCORED) == 0 and len(broker.dead_letters) == 0

        # son deneme: ölü mektup yerine sezgisel yedek kararla article.scored'a yayınlanır (ES'e ulaşır)
        processed = broker.consume(Queue.ARTICLES_KEYWORD, svc.handle)
        assert processed == settings.rabbitmq_max_attempts - 1
        assert broker.size(Queue.ARTICLES_KEYWORD) == 0
        assert len(broker.dead_letters) == 0
        assert broker.size(Queue.ARTICLES_SCORED) == 1
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.stage == Stage.SCORED and out.matched_keywords == ["bakan"]
        assert out.llm is not None and out.llm.model == FALLBACK_MODEL
        assert out.llm.attempts == settings.rabbitmq_max_attempts
        assert out.llm.reason.startswith(f"LLM skorlaması {settings.rabbitmq_max_attempts} denemede tamamlanamadı")
        assert "geçerli karar üretemedi" in out.llm.reason and "sezgisel yedek" in out.llm.reason
        assert out.llm.summary and out.llm.topics
        assert out.alarm_score == HeuristicLLM(settings).evaluate(title=record.title, content=record.content)[
            "alarm_score"
        ]
        assert out.is_alarm == (out.alarm_score >= settings.alarm_threshold)
        assert len(fake.chat_calls) == settings.rabbitmq_max_attempts * MAX_LLM_ATTEMPTS
        assert svc.stats.exhausted == settings.rabbitmq_max_attempts
        assert svc.stats.fallback == 1 and svc.stats.scored == 1

    def test_unavailable_llm_is_retried_with_attempt_header(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=0)  # süreç içi bekleme kapalı
        record = make_record("Bakan", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body)

        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle, max_messages=1) == 1
        assert broker.size(Queue.ARTICLES_KEYWORD) == 1  # yeniden kuyruğa alındı
        retried = broker.queues[Queue.ARTICLES_KEYWORD][0]
        assert retried.attempts == 1 and retried.headers["x-attempts"] == 1
        assert "LLM erişilemiyor" in retried.headers["x-error"]
        assert len(broker.dead_letters) == 0
        assert svc.stats.unavailable == 1 and svc.stats.waits == 0

        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle, max_messages=1) == 1
        assert broker.queues[Queue.ARTICLES_KEYWORD][0].attempts == 2

        fake.available = True
        fake.responder = lambda system, user: verdict_dict(90)
        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle) == 1
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.is_alarm is True and out.llm is not None and out.llm.model == "fake"

    def test_ollama_outage_is_unavailable_not_counted_against_max_attempts(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        """Ollama kapalı / model yok → Unavailable: rabbitmq_max_attempts (3) aşılsa da ölü mektup yok."""
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=0)
        with pytest.raises(Unavailable, match="LLM erişilemiyor"):
            svc.score_record(make_record("Bakan", BAKAN_TEXT))

        record = make_record("Bakan", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body)
        rounds = settings.rabbitmq_max_attempts * 4
        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle, max_messages=rounds) == rounds
        assert len(broker.dead_letters) == 0
        assert broker.size(Queue.ARTICLES_KEYWORD) == 1
        assert broker.queues[Queue.ARTICLES_KEYWORD][0].attempts == rounds

    def test_unavailable_raised_by_responder_while_ollama_healthy_is_plain_retry(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        """Sunucu ayakta ama çağrı başarısız (zaman aşımı vb.) → sayılan Retry; son denemede yedek karar."""

        def responder(system: str, user: str) -> dict[str, Any]:
            raise LLMUnavailable("okuma zaman aşımı")

        fake = FakeOllama(responder=responder)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=0)
        with pytest.raises(Retry, match="LLM çağrısı başarısız") as info:
            svc.score_record(make_record("Bakan", BAKAN_TEXT))
        assert not isinstance(info.value, Unavailable)
        assert svc.stats.unavailable == 1

        record = make_record("Bakan", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body)
        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle) == settings.rabbitmq_max_attempts
        assert len(broker.dead_letters) == 0 and broker.size(Queue.ARTICLES_SCORED) == 1
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.llm is not None and out.llm.model == FALLBACK_MODEL
        assert "okuma zaman aşımı" in out.llm.reason
        assert svc.stats.fallback == 1

    def test_unavailable_last_attempt_publishes_fallback_verdict(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=0)
        limit = attempt_limit_for(settings, Unavailable("x"))
        assert limit > settings.rabbitmq_max_attempts
        record = make_record("Bakan", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body, headers={"x-attempts": limit - 2})

        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle, max_messages=1) == 1
        assert broker.size(Queue.ARTICLES_SCORED) == 0  # henüz son deneme değil
        assert broker.queues[Queue.ARTICLES_KEYWORD][0].attempts == limit - 1

        assert broker.consume(Queue.ARTICLES_KEYWORD, svc.handle) == 1
        assert len(broker.dead_letters) == 0 and broker.size(Queue.ARTICLES_SCORED) == 1
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.llm is not None
        assert out.llm.model == FALLBACK_MODEL and out.llm.attempts == limit
        assert svc.stats.fallback == 1

    def test_waits_for_ollama_and_rescores_in_process(self, settings: Settings, broker: InMemoryBroker) -> None:
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=5, llm_poll_seconds=0.01)

        def bring_up() -> None:
            fake.responder = lambda system, user: verdict_dict(75)
            fake.available = True

        timer = threading.Timer(0.05, bring_up)
        timer.start()
        try:
            record = make_record("Bakan", BAKAN_TEXT)
            svc.handle(keyword_message(record))  # Retry/Unavailable fırlatmamalı
        finally:
            timer.cancel()

        assert broker.size(Queue.ARTICLES_SCORED) == 1
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.alarm_score == 75 and out.llm is not None and out.llm.model == "fake"
        assert svc.stats.unavailable == 1 and svc.stats.waits == 1 and svc.stats.fallback == 0
        assert len(fake.chat_calls) == 1

    def test_wait_times_out_and_defers_to_broker(self, settings: Settings, broker: InMemoryBroker) -> None:
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=0.05, llm_poll_seconds=0.01)
        with pytest.raises(Unavailable):
            svc.handle(keyword_message(make_record("Bakan", BAKAN_TEXT)))
        assert svc.stats.waits == 0 and broker.size(Queue.ARTICLES_SCORED) == 0

    def test_wait_stops_promptly_on_stop_event(self, settings: Settings, broker: InMemoryBroker) -> None:
        stop = threading.Event()
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, stop_event=stop, llm_wait_seconds=60, llm_poll_seconds=0.5)
        timer = threading.Timer(0.05, stop.set)
        timer.start()
        try:
            with pytest.raises(Unavailable):
                svc.handle(keyword_message(make_record("Bakan", BAKAN_TEXT)))
        finally:
            timer.cancel()
        assert svc.wait_for_llm(max_wait=60) is False  # stop_event set: beklemeden döner

    def test_flapping_ollama_gives_up_after_max_rounds(self, settings: Settings, broker: InMemoryBroker) -> None:
        fake = FakeOllama(available=True)
        health_checks = {"n": 0}

        def responder(system: str, user: str) -> dict[str, Any]:
            raise LLMUnavailable("bağlantı koptu")

        fake.responder = responder
        original_health = fake.health

        def flapping_health() -> bool:  # çağrı anında kapalı, bekleme denetiminde açık
            health_checks["n"] += 1
            return health_checks["n"] % 2 == 0 and original_health()

        fake.health = flapping_health  # type: ignore[method-assign]
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=5, llm_poll_seconds=0.01)
        with pytest.raises(Unavailable):
            svc.handle(keyword_message(make_record("Bakan", BAKAN_TEXT)))
        assert len(fake.chat_calls) == MAX_UNAVAILABLE_ROUNDS
        assert svc.stats.waits == MAX_UNAVAILABLE_ROUNDS - 1

    def test_run_waits_for_ollama_before_consuming(self, settings: Settings, broker: InMemoryBroker) -> None:
        fake = FakeOllama(available=False)
        svc = ScoringService(settings, broker, fake, llm_wait_seconds=5, llm_poll_seconds=0.01)
        record = make_record("Bakan", BAKAN_TEXT)
        broker.publish(RoutingKey.ARTICLE_KEYWORD, keyword_message(record).body)

        def bring_up() -> None:
            fake.responder = lambda system, user: verdict_dict(65)
            fake.available = True

        timer = threading.Timer(0.05, bring_up)
        timer.start()
        try:
            assert svc.run() == 1
        finally:
            timer.cancel()
        assert broker.size(Queue.ARTICLES_KEYWORD) == 0 and len(broker.dead_letters) == 0
        out = NewsRecord.from_message(broker.queues[Queue.ARTICLES_SCORED][0].body)
        assert out.id == record.id and out.is_alarm is True
        assert svc.stats.unavailable == 0  # tüketim Ollama hazır olduktan sonra başladı

    def test_run_with_stop_event_set_does_not_block_on_unavailable_ollama(
        self, settings: Settings, broker: InMemoryBroker
    ) -> None:
        stop = threading.Event()
        stop.set()
        svc = ScoringService(settings, broker, FakeOllama(available=False), llm_wait_seconds=60)
        assert svc.run(stop_event=stop) == 0
        assert svc.stop_event is stop

    def test_invalid_message_is_rejected(self, settings: Settings, broker: InMemoryBroker) -> None:
        svc = ScoringService(settings, broker, FakeOllama())
        broker.publish(RoutingKey.ARTICLE_KEYWORD, {"id": "x"})
        assert svc.run() == 1
        assert len(broker.dead_letters) == 1 and broker.dead_letters[0].attempts == 0
        assert svc.stats.rejected == 1

    def test_content_is_truncated_in_prompt(self, broker: InMemoryBroker) -> None:
        settings = make_settings(ollama_max_content_chars=300)
        fake = FakeOllama(responder=lambda system, user: verdict_dict(10))
        svc = ScoringService(settings, broker, fake)
        svc.score_record(make_record("Bakan", "bakan " * 1000))
        user = fake.chat_calls[0][2]
        assert TRUNCATION_MARKER in user
        assert len(parse_user_prompt(user)["content"]) <= 300 + len(TRUNCATION_MARKER) + 1


# ---------------------------------------------------------------------------------------------------------
# HeuristicLLM ve FakeOllama
# ---------------------------------------------------------------------------------------------------------


class TestHeuristicLLM:
    def test_risky_text_scores_higher_than_plain(self, settings: Settings) -> None:
        llm = HeuristicLLM(settings)
        system = build_system_prompt(settings.alarm_threshold)
        risky = make_record(
            "Bakan hakkında soruşturma: 3 tutuklama",
            "Soruşturma kapsamında eski bakanın danışmanı tutuklandı. Bakan açıklama yapmadı.",
        )
        risky.matched_keywords = ["bakan"]
        plain = make_record("Hafta sonu hava güzel", PLAIN_TEXT, slug="hava")

        high = llm.chat_json(system, build_user_prompt(risky, max_content_chars=6000))
        low = llm.chat_json(system, build_user_prompt(plain, max_content_chars=6000))
        assert set(high) == set(VERDICT_KEYS) and set(low) == set(VERDICT_KEYS)
        assert high["alarm_score"] > low["alarm_score"]
        assert low["alarm_score"] < settings.alarm_threshold and low["is_alarm"] is False
        assert high["is_alarm"] == (high["alarm_score"] >= settings.alarm_threshold)
        assert "hukuk" in high["topics"] and low["topics"] == ["diğer"]
        assert all(t in TOPICS for t in high["topics"] + low["topics"])
        assert "soruşturma" in high["reason"] and "tutukla" in high["reason"]
        assert high["summary"].startswith("Bakan hakkında soruşturma: 3 tutuklama")
        # aynı girdi → aynı çıktı
        assert llm.chat_json(system, build_user_prompt(risky, max_content_chars=6000)) == high

    def test_common_words_do_not_trigger_risk_terms(self, settings: Settings) -> None:
        llm = HeuristicLLM(settings)
        plain = llm.evaluate(title="Hafta sonu hava güzel", content="O zaman parklara gidelim, zamanla hava bozar.")
        baseline = llm.evaluate(title="Hafta sonu hava güzel", content="Parklara gidelim, hava bozar.")
        assert plain["alarm_score"] == baseline["alarm_score"] and plain["topics"] == ["diğer"]
        assert "zam" not in plain["reason"].split("risk terimleri: ", 1)[1].split(";")[0]
        assert llm._risk.find("Zamanında gelmedi, zamanla alıştı. Ben bunu atamam, o da atamaz, atamadı.") == []
        hike = llm.evaluate(title="Doğalgaza zam", content="Zamlar yarın geçerli; zamlı tarife ve zammı açıklandı.")
        assert hike["alarm_score"] > baseline["alarm_score"] and hike["topics"] == ["ekonomi"]
        assert "risk terimleri: zam;" in hike["reason"]  # gerekçede regex değil etiket görünür
        appointed = llm.evaluate(title="Üst düzey atama", content="Atamalar Resmî Gazete'de yayımlandı.")
        assert appointed["topics"] == ["siyaset"] and "atama" in appointed["reason"]
        assert "resmi gazete" in appointed["reason"]

    def test_sports_text_is_penalised(self, settings: Settings) -> None:
        llm = HeuristicLLM(settings)
        sports = llm.evaluate(title="Maçta 3 gol", content="Teknik direktör transfer dönemini değerlendirdi.")
        assert sports["alarm_score"] == 0 and sports["is_alarm"] is False

    def test_verdict_feeds_scoring_service(self, settings: Settings, broker: InMemoryBroker) -> None:
        svc = ScoringService(settings, broker, HeuristicLLM(settings))
        out = svc.score_record(make_record("Bakan", BAKAN_TEXT))
        assert out.llm is not None and out.llm.model == "heuristic"
        assert out.stage == Stage.SCORED and 0 <= out.alarm_score <= 100

    def test_embed_and_text(self, settings: Settings) -> None:
        llm = HeuristicLLM(settings)
        vectors = llm.embed(["bakan açıkladı", "bakan açıkladı", "bambaşka"])
        assert len(vectors) == 3 and all(len(v) == settings.embedding_dims for v in vectors)
        assert vectors[0] == vectors[1] != vectors[2]
        assert abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-9
        assert llm.health() and llm.model_available()
        assert "Sezgisel mod" in llm.generate_text("s", "Rapor yaz")

    def test_extract_entities(self) -> None:
        text = "Bu Kemal Kılıçdaroğlu ile Özgür Özel görüştü. Soruşturma sürüyor. CHP kurultayı toplanıyor."
        assert extract_entities(text) == ["Kemal Kılıçdaroğlu", "Özgür Özel", "CHP"]


class TestFakeOllama:
    def test_defaults_and_call_log(self) -> None:
        fake = FakeOllama(embed_dims=4, model_name="sahte")
        assert fake.model_name == "sahte"
        data = fake.chat_json("s", "u")
        assert set(data) == set(VERDICT_KEYS)
        data["alarm_score"] = 99  # kopya olmalı, varsayılanı bozmamalı
        assert fake.chat_json("s", "u")["alarm_score"] != 99
        assert fake.generate_text("s", "u").startswith("{")
        assert [len(v) for v in fake.embed(["a", "b"])] == [4, 4]
        assert [c[0] for c in fake.calls] == ["chat_json", "chat_json", "generate_text", "embed"]
        assert hashed_vector("", 4) == [0.0] * 4

    def test_unavailable_flag(self) -> None:
        fake = FakeOllama(available=False)
        assert fake.health() is False and fake.model_available() is False
        with pytest.raises(LLMUnavailable):
            fake.embed(["x"])


# ---------------------------------------------------------------------------------------------------------
# Uçtan uca: raw → filtre → skor (bellek içi)
# ---------------------------------------------------------------------------------------------------------


def test_end_to_end_raw_to_scored(settings: Settings, broker: InMemoryBroker) -> None:
    records = [
        make_record("Bakan Yerlikaya: soruşturma genişliyor", BAKAN_TEXT, slug="bakan"),
        make_record(
            "Fon yöneticisi gözaltında",
            "MASAK raporuna göre fonun hesapları donduruldu.",
            slug="fon",
            source="12punto",
        ),
        make_record("Hafta sonu hava güzel", PLAIN_TEXT, slug="hava"),
    ]
    for record in records:
        broker.publish(RoutingKey.ARTICLE_RAW, record.to_message())

    def responder(system: str, user: str) -> dict[str, Any]:
        fields = parse_user_prompt(user)
        return verdict_dict(90 if "fon" in fields["keywords"] else 40, summary=f"Özet: {fields['title']}")

    filter_svc = KeywordFilterService(settings, broker)
    scorer = ScoringService(settings, broker, FakeOllama(responder=responder))
    assert filter_svc.run() == 3
    assert scorer.run() == 2

    scored = [NewsRecord.from_message(m.body) for m in broker.drain(Queue.ARTICLES_SCORED)]
    assert len(scored) == 3
    assert {r.id for r in scored} == {r.id for r in records}
    assert all(r.stage == Stage.SCORED for r in scored)
    by_id = {r.id: r for r in scored}

    plain = by_id[records[2].id]
    assert plain.alarm_score == 0 and plain.llm is None and plain.alarm_reason == ""

    bakan = by_id[records[0].id]
    assert bakan.matched_keywords == ["bakan"] and bakan.alarm_score == 40 and not bakan.is_alarm
    assert bakan.alarm_reason == "" and bakan.llm is not None and bakan.llm.summary.startswith("Özet: Bakan")

    fon = by_id[records[1].id]
    assert fon.matched_keywords == ["fon"] and fon.alarm_score == 90 and fon.is_alarm
    assert "LLM Özeti: Özet: Fon yöneticisi gözaltında" in fon.alarm_reason
    assert fon.llm_summary == "Özet: Fon yöneticisi gözaltında"
    assert broker.size(Queue.ARTICLES_RAW) == 0 and broker.size(Queue.ARTICLES_KEYWORD) == 0
    assert len(broker.dead_letters) == 0
