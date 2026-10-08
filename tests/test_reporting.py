"""Raporlama katmanı testleri: ReportBuilder, QAEngine (RAG), ReportingConsumer / PeriodicReporter ve FastAPI.

Ağ yok: InMemoryStore + InMemoryBroker + FakeOllama + fastapi.testclient.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from scraperhryt.broker import InMemoryBroker, Message, Queue, Reject, Retry, RoutingKey
from scraperhryt.config import Settings
from scraperhryt.models import AlarmEvent, LLMVerdict, NewsRecord, Report, Stage
from scraperhryt.pipeline.llm import (
    FakeOllama,
    HeuristicLLM,
    LLMBadOutput,
    LLMUnavailable,
    OllamaClient,
    hashed_vector,
)
from scraperhryt.reporting import prompts
from scraperhryt.reporting import service as reporting_service
from scraperhryt.reporting.api import create_app, score_class
from scraperhryt.reporting.builder import (
    TEMPLATE_MODEL,
    ReportBuilder,
    alarm_stats,
    alarm_summary,
    build_template_narrative,
    report_id_for,
)
from scraperhryt.reporting.rag import (
    COMBINED_QUERY_TERMS,
    FALLBACK_MODEL,
    NO_EVIDENCE_MODEL,
    QAEngine,
    RankedDoc,
    newest_first,
    question_tokens,
    reciprocal_rank_fusion,
)
from scraperhryt.reporting.service import PeriodicReporter, ReportingConsumer
from scraperhryt.store import InMemoryStore, SearchHit

QUESTION = "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"
FAKE_ANSWER = "Son durum: 01.10.2026 itibarıyla Özel, Kılıçdaroğlu'na uzlaşma mesajı verdi [1]. Daha önce sert açıklama [2]."
FAKE_NARRATIVE = "Yönetici özeti: Pencerede fon soruşturması öne çıktı. Öne çıkan gelişmeler: - Fon soruşturması."
REWRITE = {"search_terms": ["tartışma", "uzlaşma"], "entities": ["Özgür Özel", "Kemal Kılıçdaroğlu", "CHP"]}

NOW = datetime(2026, 10, 2, 20, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------------------
# Yardımcılar / fikstürler
# ---------------------------------------------------------------------------------------------------------


def days_ago(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def make_record(
    slug: str,
    title: str,
    content: str,
    *,
    subtitle: str = "",
    source: str = "hurriyet",
    published: datetime,
    category: str = "gundem",
    score: int | None = None,
    keywords: list[str] | None = None,
) -> NewsRecord:
    host = "www.hurriyet.com.tr/gundem" if source == "hurriyet" else "12punto.com.tr/gundem"
    rec = NewsRecord.new(
        source=source,
        content_url=f"https://{host}/{slug}-{abs(hash(slug)) % 10_000_000}",
        title=title,
        subtitle=subtitle,
        content=content,
        published_at=published,
        category=category,
    )
    rec.matched_keywords = list(keywords or [])
    if score is None:
        rec.mark_not_scored()
    else:
        verdict = LLMVerdict(
            model="qwen2.5:7b",
            alarm_score=score,
            reason="Fon soruşturmasında bakan düzeyinde gelişme; 'fon' gerçek bir yatırım fonunu ifade ediyor.",
            summary="Soruşturma kapsamında yeni gözaltılar yapıldı.",
            topics=["finans/fon", "hukuk"],
            entities=["TMSF"],
        )
        rec.apply_verdict(verdict, threshold=60)
    rec.stage = Stage.ALARM
    return rec


def fake_responder(system: str, user: str) -> dict[str, Any] | str:
    if system == prompts.QUERY_REWRITE_SYSTEM_PROMPT:
        return dict(REWRITE)
    if system == prompts.RAG_SYSTEM_PROMPT:
        return FAKE_ANSWER
    return FAKE_NARRATIVE


def unavailable_responder(system: str, user: str) -> dict[str, Any] | str:
    raise LLMUnavailable("Ollama kapalı")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, alarm_threshold=60, report_digest_every=10, report_digest_minutes=30, rag_top_k=12,
        rag_query_rewrite=True,
    )


@pytest.fixture
def seeded() -> tuple[InMemoryStore, dict[str, NewsRecord], list[AlarmEvent]]:
    """6 haber: 3 Özel/Kılıçdaroğlu (farklı tarihler), 2 fon soruşturması alarmı, 1 spor."""
    store = InMemoryStore()
    records = {
        "oldest": make_record(
            "ozel-kilicdaroglu-gorusme",
            "Özgür Özel ile Kemal Kılıçdaroğlu bir araya geldi",
            "CHP Genel Başkanı Özgür Özel, eski Genel Başkan Kemal Kılıçdaroğlu ile görüşme yaptı. Görüşme iki saat sürdü.",
            subtitle="Parti kulislerinde görüşme konuşuluyor",
            published=days_ago(9),
        ),
        "middle": make_record(
            "kilicdaroglu-sert-aciklama",
            "Kılıçdaroğlu'ndan Özel'e sert açıklama",
            "Kemal Kılıçdaroğlu, Özgür Özel'in kurultay çağrısına sert açıklama ile yanıt verdi.",
            subtitle="Kurultay tartışması büyüyor",
            source="12punto",
            published=days_ago(5),
        ),
        "newest": make_record(
            "ozel-uzlasma-mesaji",
            "Özel'den Kılıçdaroğlu'na uzlaşma mesajı",
            "Özgür Özel, Kemal Kılıçdaroğlu ile yaşanan tartışmanın ardından uzlaşma mesajı verdi ve kapının açık olduğunu söyledi.",
            subtitle="Özel: Kapımız herkese açık",
            published=days_ago(1),
        ),
        "alarm_high": make_record(
            "bakan-fon-sorusturmasi-gozalti",
            "Bakan açıkladı: Fon soruşturmasında yeni gözaltılar",
            "Fon soruşturması kapsamında üç kişi gözaltına alındı. Bakan, sürecin genişleyeceğini söyledi.",
            subtitle="Soruşturma genişliyor",
            published=days_ago(0.5),
            score=85,
            keywords=["bakan", "fon"],
        ),
        "alarm_mid": make_record(
            "fon-sorusturmasi-tmsf",
            "Fon soruşturması: TMSF üç şirkete el koydu",
            "Fon soruşturmasında TMSF üç şirkete el koydu; cumhurbaşkanı konuya ilişkin açıklama yaptı.",
            source="12punto",
            published=days_ago(2),
            score=72,
            keywords=["fon", "cumhurbaşkanı"],
        ),
        "sport": make_record(
            "galatasaray-derbi",
            "Galatasaray derbide üç puanı aldı",
            "Galatasaray, Fenerbahçe ile oynanan derbi maçını iki golle kazandı. Teknik direktör galibiyeti değerlendirdi.",
            source="12punto",
            published=days_ago(3),
            category="spor",
        ),
    }
    events: list[AlarmEvent] = []
    for rec in records.values():
        if rec.is_alarm:
            event = AlarmEvent.from_record(rec)
            event.raised_at = rec.published_at + timedelta(minutes=30)
            rec.alarm_id, rec.alarmed_at = event.alarm_id, event.raised_at
            event.record = rec
            store.index_alarm(event)
            events.append(event)
        store.index_record(rec)
    return store, records, events


def alarm_message(event: AlarmEvent) -> Message:
    return Message(body=event.to_message(), routing_key=RoutingKey.ALARM_RAISED, queue=Queue.ALARMS)


def report_messages(broker: InMemoryBroker, routing_key: str) -> list[Message]:
    return [m for m in broker.published if m.routing_key == routing_key]


class BlockingBroker(InMemoryBroker):
    """RabbitMQ gibi davranır: kuyruk boşalınca dönmez, ``stop_event`` set edilene kadar bekler."""

    def consume(
        self,
        queue: str,
        handler: Any,
        *,
        prefetch: int | None = None,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ) -> int:
        stop_event = stop_event or threading.Event()
        processed = 0
        while not stop_event.is_set():
            processed += super().consume(queue, handler, prefetch=prefetch, stop_event=stop_event)
            stop_event.wait(0.01)
        return processed


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = datetime.now(UTC) + timedelta(seconds=timeout)
    while datetime.now(UTC) < deadline:
        if predicate():
            return True
        threading.Event().wait(0.01)
    return False


# ---------------------------------------------------------------------------------------------------------
# Saf yardımcılar
# ---------------------------------------------------------------------------------------------------------


def test_question_tokens_drops_stopwords_and_keeps_names() -> None:
    assert question_tokens(QUESTION) == ["özgür", "özel", "kemal", "kılıçdaroğlu"]
    assert question_tokens("Son durum ne?") == []
    assert question_tokens("") == []


def test_reciprocal_rank_fusion_merges_and_ranks() -> None:
    a = [SearchHit({"id": "x", "title": "x"}, 3.0), SearchHit({"id": "y", "title": "y"}, 2.0)]
    b = [SearchHit({"id": "y", "title": "y"}, 0.9, highlights={"content": ["<em>y</em> ..."]}), SearchHit({"id": "z"}, 0.5)]
    fused = reciprocal_rank_fusion([a, b])
    assert [item.doc["id"] for item in fused] == ["y", "x", "z"]
    assert fused[0].score == round(1 / 62 + 1 / 61, 6)
    assert fused[0].highlights == {"content": ["<em>y</em> ..."]}
    assert reciprocal_rank_fusion([]) == []


def test_newest_first_puts_undated_last() -> None:
    docs = [
        RankedDoc({"id": "old", "published_at": "2026-09-01T10:00:00+00:00"}, 0.5),
        RankedDoc({"id": "none"}, 0.9),
        RankedDoc({"id": "new", "published_at": "2026-10-01T10:00:00+03:00"}, 0.1),
    ]
    assert [d.doc["id"] for d in newest_first(docs)] == ["new", "old", "none"]


def test_prompt_helpers_format_turkish() -> None:
    assert prompts.format_tr("2026-10-02T19:20:00+00:00") == "02.10.2026 22:20"
    assert prompts.format_tr(None) == "bilinmiyor"
    assert prompts.one_line_reason({"alarm_reason": "Gerekçe cümlesi.\n\nLLM Özeti: Özet metni."}) == "Gerekçe cümlesi."
    assert prompts.one_line_reason({"alarm_reason": "", "llm_summary": "Yalnız özet"}) == "Yalnız özet"
    assert prompts.format_distribution({"12punto": 2, "hurriyet": 4}) == "hurriyet: 4, 12punto: 2"
    assert prompts.peak_hour([{"ts": "a", "count": 1}, {"ts": "b", "count": 3}, {"ts": "c", "count": 2}])["ts"] == "b"
    block = prompts.build_context_block(1, {"source": "hurriyet", "published_at": "2026-10-02T19:20:00+00:00", "title": "Başlık", "subtitle": "Alt", "content": "İçerik " * 400})
    assert block.startswith("[1] (hurriyet, 02.10.2026 22:20) Başlık — Alt — İçerik")
    assert len(block) < 1400


def test_report_id_is_deterministic_and_alarm_sensitive() -> None:
    start, end = NOW - timedelta(hours=24), NOW
    first = report_id_for("periodic", start, end)
    assert first == report_id_for("periodic", start, end)
    assert first.startswith("periodic-20261002T2000-")
    assert report_id_for("alarm_digest", start, end, ["a"]) != report_id_for("alarm_digest", start, end, ["b"])
    assert report_id_for("periodic", start.replace(tzinfo=None), end.replace(tzinfo=None)) == first


def test_alarm_summary_drops_content_and_coerces() -> None:
    summary = alarm_summary({"alarm_id": "a1", "id": "r1", "title": "T", "alarm_score": "77", "content": "gövde", "matched_keywords": ("fon",)})
    assert summary["record_id"] == "r1" and summary["alarm_score"] == 77 and summary["matched_keywords"] == ["fon"]
    assert "content" not in summary


# ---------------------------------------------------------------------------------------------------------
# QAEngine (RAG)
# ---------------------------------------------------------------------------------------------------------


def test_qa_engine_answers_newest_first_with_citations(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    llm = FakeOllama(responder=fake_responder)
    answer = QAEngine(settings, store, llm).ask(QUESTION)

    assert answer.answer == FAKE_ANSWER
    assert answer.model == "fake"
    assert answer.question == QUESTION
    ids = [c.id for c in answer.sources]
    assert ids[0] == records["newest"].id
    assert ids[:3] == [records["newest"].id, records["middle"].id, records["oldest"].id]
    assert records["sport"].id not in ids
    assert answer.retrieved_count == len(answer.sources) >= 3
    assert "Özgür Özel" in answer.search_terms and "tartışma" in answer.search_terms
    assert answer.sources[0].content_url == records["newest"].content_url
    assert answer.sources[0].published_at == records["newest"].published_at

    generate_calls = [c for c in llm.calls if c[0] == "generate_text"]
    assert len(generate_calls) == 1
    system, user = generate_calls[0][1], generate_calls[0][2]
    assert system == prompts.RAG_SYSTEM_PROMPT
    assert "[1] (hurriyet" in user and records["newest"].title in user
    assert user.index(records["newest"].title) < user.index(records["middle"].title) < user.index(records["oldest"].title)
    assert records["sport"].title not in user
    rewrite_calls = llm.chat_calls
    assert len(rewrite_calls) == 1 and rewrite_calls[0][1] == prompts.QUERY_REWRITE_SYSTEM_PROMPT
    assert QUESTION in rewrite_calls[0][2]


def test_qa_engine_fallback_when_llm_unavailable(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    llm = FakeOllama(responder=unavailable_responder)
    answer = QAEngine(settings, store, llm).ask(QUESTION)

    assert answer.model == FALLBACK_MODEL
    # Yedek yanıt 2-3 cümlelik özettir: en yeni haber [1] olarak önce gelir, başlık listesi yoktur.
    assert answer.sources[0].id == records["newest"].id and answer.answer.startswith("Son gelişme (")
    assert "[1]" in answer.answer and answer.answer.index("[1]") < answer.answer.index("[2]")
    assert "[1]" in answer.answer
    assert answer.sources[0].id == records["newest"].id
    assert answer.search_terms == ["özgür", "özel", "kemal", "kılıçdaroğlu"]


def test_qa_engine_bounds_ollama_waits_and_falls_back_on_timeout(seeded) -> None:
    """Ollama skorlayıcıyla meşgulken soru sonsuza dek beklemez: her çağrı kendi süre sınırıyla gider,
    yanıt üretimi zaman aşımına uğrarsa haber metninden çıkarımsal yanıt döner."""
    import json

    import httpx

    from scraperhryt.pipeline.llm import OllamaClient

    store, records, _ = seeded
    s = Settings(_env_file=None, rag_top_k=12, rag_rewrite_timeout=7.0, rag_answer_timeout=11.0, rag_query_rewrite=True)
    waits: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        waits[request.url.path] = request.extensions["timeout"]["read"]
        if request.url.path == "/api/generate":
            raise httpx.ReadTimeout("timed out", request=request)
        content = json.dumps({"search_terms": ["Özgür Özel", "Kılıçdaroğlu"], "entities": ["Özgür Özel"]}, ensure_ascii=False)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done": True})

    with OllamaClient(s, transport=httpx.MockTransport(handler)) as llm:
        answer = QAEngine(s, store, llm).ask(QUESTION)
    assert waits == {"/api/chat": 7.0, "/api/generate": 11.0}
    assert answer.model == FALLBACK_MODEL and answer.sources[0].id == records["newest"].id
    assert answer.answer.startswith("Son gelişme (") and "[1]" in answer.answer


def test_qa_engine_fallback_when_rewrite_is_bad_json(settings: Settings, seeded) -> None:
    store, records, _ = seeded

    def responder(system: str, user: str) -> dict[str, Any] | str:
        if system == prompts.QUERY_REWRITE_SYSTEM_PROMPT:
            raise LLMBadOutput("JSON yok")
        return FAKE_ANSWER

    answer = QAEngine(settings, store, FakeOllama(responder=responder)).ask(QUESTION)
    assert answer.answer == FAKE_ANSWER and answer.sources[0].id == records["newest"].id


def test_qa_engine_says_insufficient_when_nothing_matches(settings: Settings, seeded) -> None:
    store, _, _ = seeded
    llm = FakeOllama(responder=lambda s, u: {"search_terms": ["kriptopara"], "entities": []} if s == prompts.QUERY_REWRITE_SYSTEM_PROMPT else FAKE_ANSWER)
    answer = QAEngine(settings, store, llm).ask("Kriptopara düzenlemesi ne oldu?")
    assert answer.answer.startswith(prompts.INSUFFICIENT_EVIDENCE_TEXT)
    assert answer.sources == [] and answer.retrieved_count == 0
    # LLM sağlıklı ve hiç çağrılmadı: "LLM kullanılamadı" (fallback) işareti yanlış olurdu.
    assert answer.model == NO_EVIDENCE_MODEL and answer.model != FALLBACK_MODEL
    assert not [c for c in llm.calls if c[0] == "generate_text"]


def test_search_queries_are_narrow_and_deduplicated() -> None:
    terms = ["tartışma", "uzlaşma", "kurultay", "açıklama", "görüşme"]
    queries = QAEngine.search_queries(QUESTION, terms, ["Özgür Özel", "Kemal Kılıçdaroğlu", "CHP"])
    assert queries[0] == "tartışma uzlaşma kurultay açıklama"
    assert queries[1:4] == ["Özgür Özel", "Kemal Kılıçdaroğlu", "CHP"]
    assert queries[-1] == "özgür özel kemal kılıçdaroğlu"
    assert all(len(q.split()) <= COMBINED_QUERY_TERMS for q in queries)
    # Varlık yoksa terimler tek tek sorgulanır; soru sözcükleri birleşik sorguyla aynıysa tekrarlanmaz.
    fallback = QAEngine.search_queries(QUESTION, ["özgür", "özel", "kemal", "kılıçdaroğlu"], [])
    assert fallback == ["özgür özel kemal kılıçdaroğlu", "özgür", "özel", "kemal", "kılıçdaroğlu"]
    # 12 terimlik yeniden yazma tek bir dev sorguya (ES'te alan başına %60 eşleşme şartı) dönüşmez.
    many = QAEngine.search_queries("Soru", [f"terim{i}" for i in range(12)], [])
    assert max(len(q.split()) for q in many) == COMBINED_QUERY_TERMS and many[-1] == "soru"
    assert QAEngine.search_queries("ve ile", [], []) == ["ve ile"]


def test_qa_engine_issues_each_search_query_to_store(settings: Settings, seeded) -> None:
    store, records, _ = seeded

    class SpyStore(InMemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def search_records(self, query: str, **kwargs: Any) -> list[SearchHit]:
            self.queries.append(query)
            return super().search_records(query, **kwargs)

    spy = SpyStore()
    spy.records.update(store.records)
    answer = QAEngine(settings, spy, FakeOllama(responder=fake_responder)).ask(QUESTION)
    assert spy.queries == QAEngine.search_queries(QUESTION, REWRITE["search_terms"], REWRITE["entities"])
    assert [c.id for c in answer.sources][:3] == [records["newest"].id, records["middle"].id, records["oldest"].id]
    assert records["sport"].id not in [c.id for c in answer.sources]


def test_qa_engine_respects_since_days_top_k_and_sources(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    engine = QAEngine(settings, store, FakeOllama(responder=fake_responder))
    recent = engine.ask(QUESTION, since_days=3)
    assert [c.id for c in recent.sources] == [records["newest"].id]
    limited = engine.ask(QUESTION, top_k=2)
    assert len(limited.sources) == 2
    punto = engine.ask(QUESTION, sources=["12punto"])
    assert [c.source for c in punto.sources] == ["12punto"] and punto.sources[0].id == records["middle"].id
    with pytest.raises(ValueError):
        engine.ask("   ")


def test_qa_engine_fuses_knn_results_when_embeddings_enabled(seeded) -> None:
    seeded_store, records, _ = seeded
    dims = 64

    class SpyStore(InMemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.knn_calls: list[tuple[int, datetime | None]] = []

        def knn_search(self, vector, k: int, since: datetime | None = None) -> list[SearchHit]:
            self.knn_calls.append((k, since))
            return super().knn_search(vector, k, since)

    store = SpyStore()
    store.records.update(seeded_store.records)
    store.alarms.update(seeded_store.alarms)
    for rec in records.values():
        store.embeddings[rec.id] = hashed_vector(f"{rec.title}\n{rec.content}", dims)
    settings = Settings(_env_file=None, ollama_embedding_model="nomic-embed-text", embedding_dims=dims)
    llm = FakeOllama(responder=fake_responder, embed_dims=dims)

    answer = QAEngine(settings, store, llm).ask(QUESTION, top_k=3)

    assert [c[0] for c in llm.calls if c[0] == "embed"] == ["embed"]
    # aday havuzu top_k × 2 (süzgeç sonrası top_k dolu kalsın)
    assert len(store.knn_calls) == 1 and store.knn_calls[0][0] == 6 and store.knn_calls[0][1] is not None
    assert [c.id for c in answer.sources] == [records["newest"].id, records["middle"].id, records["oldest"].id]
    assert all(c.score > 1 / 61 for c in answer.sources)  # her biri en az iki sıralamada yer aldı (RRF)

    # Embedding modeli kapalıysa kNN hiç çağrılmaz.
    store.knn_calls.clear()
    QAEngine(Settings(_env_file=None), store, FakeOllama(responder=fake_responder, embed_dims=dims)).ask(QUESTION)
    assert store.knn_calls == []


def test_qa_engine_with_heuristic_llm_is_deterministic(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    answer = QAEngine(settings, store, HeuristicLLM(settings)).ask(QUESTION)
    assert answer.model == FALLBACK_MODEL
    assert answer.sources[0].id == records["newest"].id and answer.answer.startswith("Son gelişme (")


# ---------------------------------------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------------------------------------


def test_report_builder_builds_stats_top_alarms_and_narrative(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    llm = FakeOllama(responder=fake_responder)
    report = ReportBuilder(settings, store, llm).build("periodic", days_ago(30), datetime.now(UTC))

    assert isinstance(report, Report) and report.kind == "periodic"
    assert report.stats["total"] == 6 and report.stats["alarms"] == 2
    assert report.narrative == FAKE_NARRATIVE and report.model == "fake"
    assert [a["alarm_score"] for a in report.top_alarms] == [85, 72]
    assert report.top_alarms[0]["title"] == records["alarm_high"].title
    assert report.top_alarms[0]["record_id"] == records["alarm_high"].id
    assert report.report_id.startswith("periodic-")
    prompt_user = [c for c in llm.calls if c[0] == "generate_text"][0][2]
    assert "Toplam haber: 6 | Alarm: 2" in prompt_user and records["alarm_high"].title in prompt_user
    assert "gerçek bir yatırım fonunu" in prompt_user and "LLM Özeti" not in prompt_user


def test_report_builder_template_fallback_when_llm_fails(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    report = ReportBuilder(settings, store, FakeOllama(responder=unavailable_responder)).build(
        "periodic", days_ago(30), datetime.now(UTC)
    )
    assert report.model == TEMPLATE_MODEL
    assert "6 haber" in report.narrative and "2 tanesi" in report.narrative
    assert records["alarm_high"].title in report.narrative
    assert "Yönetici özeti:" in report.narrative and "İzlenmesi gerekenler:" in report.narrative

    heuristic = ReportBuilder(settings, store, HeuristicLLM(settings)).build("adhoc", days_ago(30), datetime.now(UTC))
    assert heuristic.model == TEMPLATE_MODEL and "Sezgisel mod" not in heuristic.narrative

    quiet = ReportBuilder(settings, store, FakeOllama(responder=fake_responder)).build(
        "periodic", days_ago(400), days_ago(300), narrative=False
    )
    assert quiet.model == TEMPLATE_MODEL and quiet.stats["total"] == 0 and "haber bulunmuyor" in quiet.narrative


def test_report_builder_top_alarms_share_the_stats_axis(settings: Settings, seeded) -> None:
    """Haber tarihi pencere dışında, alarmı pencere içinde yükseltilmiş kayıt: istatistik saymaz, liste de saymaz."""
    store, records, _ = seeded
    late = make_record(
        "eski-haber-gec-alarm",
        "Bakan: eski fon dosyası yeniden açıldı",
        "Fon soruşturmasına ilişkin eski bir dosya yeniden açıldı; bakan açıklama yaptı.",
        published=days_ago(40),
        score=90,
        keywords=["bakan", "fon"],
    )
    event = AlarmEvent.from_record(late)
    event.raised_at = days_ago(0.1)  # geç kazındı / yeniden skorlandı: alarm şimdi, haber 40 gün önce
    late.alarm_id, late.alarmed_at = event.alarm_id, event.raised_at
    store.index_alarm(event)
    store.index_record(late)

    report = ReportBuilder(settings, store, FakeOllama(responder=fake_responder)).build(
        "periodic", days_ago(30), datetime.now(UTC)
    )
    assert report.stats["alarms"] == 2 == len(report.top_alarms)
    assert [a["alarm_score"] for a in report.top_alarms] == [85, 72]
    assert event.alarm_id not in {a["alarm_id"] for a in report.top_alarms}
    assert report.stats["avg_alarm_score"] == 78.5

    # Haber tarihi pencereye girince (40 gün geriye) hem sayıya hem listeye girer.
    wide = ReportBuilder(settings, store, FakeOllama(responder=fake_responder)).build(
        "adhoc", days_ago(45), datetime.now(UTC)
    )
    assert wide.stats["alarms"] == 3 == len(wide.top_alarms) and wide.top_alarms[0]["alarm_id"] == event.alarm_id
    assert records["alarm_high"].id == wide.top_alarms[1]["record_id"]


def test_report_builder_rejects_invalid_window(settings: Settings, seeded) -> None:
    store, _, _ = seeded
    builder = ReportBuilder(settings, store, FakeOllama(responder=fake_responder))
    with pytest.raises(ValueError):
        builder.build("periodic", datetime.now(UTC), days_ago(1))
    with pytest.raises(ValueError):
        builder.build("", days_ago(1), datetime.now(UTC))


def test_alarm_digest_stats_derive_from_buffered_alarms(settings: Settings, seeded) -> None:
    store, _, events = seeded

    class NoStatsStore(InMemoryStore):
        def stats(self, since: datetime | None, until: datetime | None = None) -> dict[str, Any]:
            raise AssertionError("alarm özeti depo istatistiği sorgulamamalı")

    llm = FakeOllama(responder=fake_responder)
    builder = ReportBuilder(settings, NoStatsStore(), llm)
    raised = sorted(e.raised_at for e in events)
    docs = [e.to_es_document() for e in events]
    report = builder.build("alarm_digest", raised[0], raised[-1], top_alarms=docs)

    assert report.stats["total"] == 2 and report.stats["alarms"] == 2 and report.stats["avg_alarm_score"] == 78.5
    assert report.stats["by_source"] == {"12punto": 1, "hurriyet": 1}
    assert report.stats["by_keyword"] == {"fon": 2, "bakan": 1, "cumhurbaşkanı": 1}
    assert report.stats["by_category"] == {"gundem": 2}
    assert sum(h["count"] for h in report.stats["by_hour"]) == 2
    assert all(h["alarms"] == h["count"] for h in report.stats["by_hour"])
    assert report.stats == alarm_stats(report.top_alarms, raised[0], raised[-1])
    prompt = [c for c in llm.calls if c[0] == "generate_text"][0][2]
    assert "Özetlenen alarm sayısı: 2 | Ortalama alarm skoru: 78.5 | En yüksek skor: 85" in prompt
    assert "Toplam haber" not in prompt

    offline = builder.build("alarm_digest", raised[0], raised[-1], top_alarms=docs, narrative=False)
    assert "Bu özet 2 alarm içerir; ortalama alarm skoru 78.5, en yüksek skor 85." in offline.narrative
    assert "kaynaklar — 12punto: 1, hurriyet: 1" in offline.narrative or "hurriyet: 1" in offline.narrative


def test_template_narrative_handles_empty_stats() -> None:
    text = build_template_narrative("periodic", NOW - timedelta(hours=1), NOW, {}, [])
    assert "haber bulunmuyor" in text and "alarm üretilmedi" in text


# ---------------------------------------------------------------------------------------------------------
# ReportingConsumer / PeriodicReporter
# ---------------------------------------------------------------------------------------------------------


def test_reporting_consumer_digests_after_n_alarms(seeded) -> None:
    store, _, events = seeded
    settings = Settings(_env_file=None, report_digest_every=2, report_digest_minutes=30)
    broker = InMemoryBroker(settings)
    broker.declare_topology()
    llm = FakeOllama(responder=fake_responder)
    consumer = ReportingConsumer(settings, broker, store, ReportBuilder(settings, store, llm))
    for event in events:
        broker.publish(RoutingKey.ALARM_RAISED, event.to_message())

    processed = consumer.run(max_messages=None)

    assert processed == 2 and consumer.buffered == 0 and consumer.stats.digests == 1
    digests = [r for r in store.reports.values() if r["kind"] == "alarm_digest"]
    assert len(digests) == 1
    digest = digests[0]
    assert digest["model"] == "fake" and digest["narrative"] == FAKE_NARRATIVE
    assert [a["alarm_score"] for a in digest["top_alarms"]] == [85, 72]
    assert {a["alarm_id"] for a in digest["top_alarms"]} == {e.alarm_id for e in events}
    raised = sorted(e.raised_at for e in events)
    assert prompts.parse_datetime(digest["window_start"]) == raised[0]
    assert prompts.parse_datetime(digest["window_end"]) == raised[-1]
    published = report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)
    assert len(published) == 1 and published[0].body["report_id"] == digest["report_id"]
    assert broker.size(Queue.REPORTS) == 1
    assert "Özetlenen alarm sayısı: 2" in [c for c in llm.calls if c[0] == "generate_text"][0][2]


def test_reporting_consumer_time_based_digest_and_dedupe(settings: Settings, seeded) -> None:
    store, _, events = seeded
    broker = InMemoryBroker(settings)
    consumer = ReportingConsumer(settings, broker, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)))

    # Uzun bir sessizlikten sonra gelen ilk alarm tek başına özetlenmez: süre, tampondaki en eski alarmdan ölçülür.
    consumer.last_digest_at = datetime.now(UTC) - timedelta(hours=3)
    assert not consumer.is_digest_due()
    consumer.handle(alarm_message(events[0]))
    assert consumer.buffered == 1 and consumer.stats.digests == 0  # 1 < report_digest_every
    assert consumer.batch_opened_at is not None and datetime.now(UTC) - consumer.batch_opened_at < timedelta(seconds=5)
    assert not consumer.is_digest_due()
    consumer.handle(alarm_message(events[0]))  # yeniden teslim
    assert consumer.buffered == 1 and consumer.stats.duplicates == 1

    consumer.batch_opened_at = datetime.now(UTC) - timedelta(minutes=settings.report_digest_minutes + 1)
    assert consumer.is_digest_due()
    consumer.handle(alarm_message(events[1]))
    assert consumer.buffered == 0 and consumer.stats.digests == 1
    assert len(report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)) == 1

    assert consumer.batch_opened_at is None
    consumer.handle(alarm_message(events[1]))  # özetlendikten sonra yeniden teslim
    assert consumer.buffered == 0 and consumer.stats.duplicates == 2
    assert consumer.flush() is None


def test_reporting_consumer_run_emits_time_based_digest_while_idle(settings: Settings, seeded) -> None:
    """Kuyruk sessizken süre eşiği dolunca özet, yeni alarm beklenmeden ve tüketici durmadan üretilir."""
    store, _, events = seeded
    broker = BlockingBroker(settings)
    builder = ReportBuilder(settings, store, FakeOllama(responder=fake_responder))
    consumer = ReportingConsumer(settings, broker, store, builder, poll_seconds=0.02)
    broker.publish(RoutingKey.ALARM_RAISED, events[1].to_message())  # 47 saat önceki alarm: depodan geri alınmaz
    stop = threading.Event()
    thread = threading.Thread(target=consumer.run, kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: consumer.buffered == 2)  # kuyruktaki 1 + depodan geri alınan 1 (12 saat önceki)
        assert consumer.stats.digests == 0 and not consumer.is_digest_due()

        consumer.batch_opened_at = datetime.now(UTC) - timedelta(minutes=settings.report_digest_minutes + 1)
        assert wait_until(lambda: consumer.stats.digests == 1)
        assert consumer.buffered == 0 and thread.is_alive()
        published = report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)
        assert len(published) == 1 and len(published[0].body["top_alarms"]) == 2

        broker.publish(RoutingKey.ALARM_RAISED, events[1].to_message())  # özet sonrası yeniden teslim
        assert wait_until(lambda: consumer.stats.duplicates == 1)
        assert consumer.buffered == 0 and thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert consumer.stats.digests == 1 and len(report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)) == 1


def test_reporting_consumer_timed_flush_failure_backs_off_and_retries(monkeypatch, seeded) -> None:
    store, _, events = seeded
    settings = Settings(_env_file=None, report_digest_every=10, report_digest_minutes=30)
    monkeypatch.setattr(reporting_service, "_DIGEST_RETRY_SECONDS", 0.2)

    class FailingBlockingBroker(BlockingBroker):
        def __init__(self) -> None:
            super().__init__(settings)
            self.fail = True

        def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
            if self.fail and routing_key == RoutingKey.REPORT_ALARM_DIGEST:
                raise Retry("RabbitMQ erişilemiyor")
            super().publish(routing_key, body, headers)

    broker = FailingBlockingBroker()
    consumer = ReportingConsumer(
        settings, broker, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)), poll_seconds=0.02
    )
    broker.publish(RoutingKey.ALARM_RAISED, events[1].to_message())
    stop = threading.Event()
    thread = threading.Thread(target=consumer.run, kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: consumer.buffered == 2)
        consumer.batch_opened_at = datetime.now(UTC) - timedelta(minutes=31)
        assert wait_until(lambda: consumer.stats.failures == 1)
        assert consumer.buffered == 2 and consumer.stats.digests == 0 and thread.is_alive()
        threading.Event().wait(0.1)
        assert consumer.stats.failures == 1  # bekleme süresi dolmadan yeniden denenmez (sıkı döngü yok)
        broker.fail = False
        assert wait_until(lambda: consumer.stats.digests == 1)
        assert consumer.buffered == 0 and consumer.stats.failures == 1 and thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and len(report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)) == 1


def test_reporting_consumer_recovers_undigested_alarms_on_start(settings: Settings, seeded) -> None:
    """Çökme sonrası: depodaki özetlenmemiş yakın alarmlar tampona geri alınır; özetlenmişler yeniden özetlenmez."""
    store, _, events = seeded
    recent, old = events  # 12 saat önce yükseltilen / 47 saat önce yükseltilen (pencere: report_window_hours=24)
    broker = InMemoryBroker(settings)
    builder = ReportBuilder(settings, store, FakeOllama(responder=fake_responder))

    first = ReportingConsumer(settings, broker, store, builder)
    assert first.recover_pending() == 1
    assert first.buffered == 1 and first.stats.recovered == 1 and first.stats.buffered == 1
    assert first.buffer[0].alarm_id == recent.alarm_id and first.buffer[0].raised_at == recent.raised_at
    assert first.batch_opened_at is not None and not first.is_digest_due()
    first.handle(alarm_message(recent))  # kuyrukta bekleyen aynı alarm
    assert first.buffered == 1 and first.stats.duplicates == 1
    assert first.flush() is not None and first.stats.digests == 1

    second = ReportingConsumer(settings, broker, store, builder)
    assert second.recover_pending() == 0 and second.buffered == 0
    broker.publish(RoutingKey.ALARM_RAISED, recent.to_message())  # yeniden başlatma sonrası yeniden teslim
    broker.publish(RoutingKey.ALARM_RAISED, old.to_message())
    assert second.run() == 2
    assert second.stats.duplicates == 1 and second.stats.buffered == 1 and second.stats.digests == 1
    digests = [r for r in store.reports.values() if r["kind"] == "alarm_digest"]
    assert len(digests) == 2
    assert [{a["alarm_id"] for a in d["top_alarms"]} for d in digests].count({old.alarm_id}) == 1


def test_reporting_consumer_run_survives_recovery_failure(settings: Settings, seeded) -> None:
    store, _, events = seeded

    class BrokenRecovery(InMemoryStore):
        def recent_alarms(self, since: datetime | None = None, size: int = 50) -> list[dict[str, Any]]:
            raise Retry("Elasticsearch erişilemiyor")

    broken = BrokenRecovery()
    broker = InMemoryBroker(settings)
    consumer = ReportingConsumer(settings, broker, broken, ReportBuilder(settings, broken, FakeOllama(responder=fake_responder)))
    broker.publish(RoutingKey.ALARM_RAISED, events[0].to_message())
    assert consumer.run() == 1
    assert consumer.stats.failures == 1 and consumer.stats.recovered == 0 and consumer.stats.digests == 1


def test_reporting_consumer_flushes_on_exit_and_rejects_invalid(settings: Settings, seeded) -> None:
    store, _, events = seeded
    broker = InMemoryBroker(settings)
    consumer = ReportingConsumer(settings, broker, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)))
    broker.publish(RoutingKey.ALARM_RAISED, events[0].to_message())
    assert consumer.run() == 1
    assert consumer.stats.digests == 1 and len(report_messages(broker, RoutingKey.REPORT_ALARM_DIGEST)) == 1
    with pytest.raises(Reject):
        consumer.handle(Message(body={"foo": "bar"}, routing_key=RoutingKey.ALARM_RAISED, queue=Queue.ALARMS))
    assert consumer.stats.rejected == 1


def test_reporting_consumer_keeps_buffer_when_publish_fails(seeded) -> None:
    store, _, events = seeded
    settings = Settings(_env_file=None, report_digest_every=1)

    class FailingBroker(InMemoryBroker):
        def __init__(self) -> None:
            super().__init__(settings)
            self.fail = True

        def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
            if self.fail:
                raise Retry("RabbitMQ erişilemiyor")
            super().publish(routing_key, body, headers)

    broker = FailingBroker()
    consumer = ReportingConsumer(settings, broker, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)))
    with pytest.raises(Retry):
        consumer.handle(alarm_message(events[0]))
    assert consumer.buffered == 1 and consumer.stats.failures == 1
    broker.fail = False
    consumer.handle(alarm_message(events[0]))  # broker yeniden teslim etti
    assert consumer.buffered == 0 and consumer.stats.digests == 1 and consumer.stats.duplicates == 1


def test_periodic_reporter_run_once_publishes(settings: Settings, seeded) -> None:
    store, _, _ = seeded
    broker = InMemoryBroker(settings)
    broker.declare_topology()
    reporter = PeriodicReporter(settings, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)), broker)
    report = reporter.run_once()
    assert report.kind == "periodic" and report.window_end - report.window_start == timedelta(hours=settings.report_window_hours)
    # Son 24 saatte yalnızca 12 saat önceki alarm haberi var (1 gün öncekiler pencerenin hemen dışında).
    assert report.stats["total"] == 1 and report.stats["alarms"] == 1 and report.report_id in store.reports
    assert [a["alarm_score"] for a in report.top_alarms] == [85]
    published = report_messages(broker, RoutingKey.REPORT_GENERATED)
    assert len(published) == 1 and published[0].body["report_id"] == report.report_id
    assert broker.size(Queue.REPORTS) == 1 and reporter.last_report is report


def test_periodic_reporter_run_stops_promptly(seeded) -> None:
    store, _, _ = seeded
    settings = Settings(_env_file=None, report_interval_minutes=60)
    broker = InMemoryBroker(settings)
    reporter = PeriodicReporter(settings, store, ReportBuilder(settings, store, FakeOllama(responder=fake_responder)), broker)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    started = datetime.now(UTC)
    assert reporter.run(stop_event=stop) == 1
    assert datetime.now(UTC) - started < timedelta(seconds=5)
    assert len(report_messages(broker, RoutingKey.REPORT_GENERATED)) == 1


def test_periodic_reporter_run_survives_failures(seeded) -> None:
    store, _, _ = seeded
    settings = Settings(_env_file=None, report_interval_minutes=1)

    class BrokenStore(InMemoryStore):
        def stats(self, since: datetime | None, until: datetime | None = None) -> dict[str, Any]:
            raise Retry("Elasticsearch erişilemiyor")

    reporter = PeriodicReporter(settings, BrokenStore(), ReportBuilder(settings, BrokenStore(), FakeOllama()), InMemoryBroker(settings))
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert reporter.run(stop_event=stop) == 0 and reporter.stats.failures == 1


# ---------------------------------------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings, seeded) -> tuple[TestClient, InMemoryBroker, dict[str, NewsRecord]]:
    store, records, _ = seeded
    broker = InMemoryBroker(settings)
    broker.declare_topology()
    app = create_app(settings, store, FakeOllama(responder=fake_responder), broker=broker)
    return TestClient(app), broker, records


def test_api_health(client) -> None:
    http, _, _ = client
    res = http.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok" and body["store"] is True and body["model"] == "fake" and body["broker_configured"] is True


def test_api_health_reports_unavailable_store(settings: Settings) -> None:
    class DownStore(InMemoryStore):
        def health(self) -> bool:
            return False

    http = TestClient(create_app(settings, DownStore(), FakeOllama(available=False)))
    res = http.get("/health")
    assert res.status_code == 503 and res.json()["status"] == "unavailable" and res.json()["llm"] is False


def test_api_health_bounds_slow_llm_probe(settings: Settings, seeded) -> None:
    store, _, _ = seeded
    release = threading.Event()
    probes: list[int] = []

    class BlackHoleLLM(FakeOllama):
        """Paketleri düşüren Ollama ana makinesi: bağlantı reddedilmez, yanıt da gelmez."""

        def health(self) -> bool:
            probes.append(1)
            release.wait(5)
            return True

    app = create_app(settings, store, BlackHoleLLM())
    app.state.llm_probe.timeout = 0.2
    http = TestClient(app)
    started = datetime.now(UTC)
    res = http.get("/health")
    assert res.status_code == 200 and datetime.now(UTC) - started < timedelta(seconds=2)
    body = res.json()
    assert body["status"] == "degraded" and body["llm"] is False and body["model_available"] is False
    # Süren sondaya katılır; her sağlık isteği yeni bir askıda iş parçacığı başlatmaz.
    assert http.get("/health").json()["llm"] is False and len(probes) == 1
    release.set()
    assert wait_until(lambda: http.get("/health").json()["status"] == "ok")


def test_api_ask_returns_citations(client) -> None:
    http, _, records = client
    res = http.post("/ask", json={"question": QUESTION, "since_days": 14, "top_k": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == FAKE_ANSWER and body["model"] == "fake"
    assert body["sources"][0]["id"] == records["newest"].id
    assert body["sources"][0]["content_url"].startswith("https://www.hurriyet.com.tr/")
    assert body["retrieved_count"] == len(body["sources"]) and "Özgür Özel" in body["search_terms"]
    assert http.post("/ask", json={"question": ""}).status_code == 422
    assert http.post("/ask", json={"question": QUESTION, "top_k": 0}).status_code == 422


def test_api_ask_falls_back_without_llm(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    http = TestClient(create_app(settings, store, FakeOllama(available=False)))
    body = http.post("/ask", json={"question": QUESTION}).json()
    assert body["model"] == FALLBACK_MODEL and body["sources"][0]["id"] == records["newest"].id
    assert body["answer"].startswith("Son gelişme (")


def test_api_ask_marks_no_evidence_distinct_from_llm_outage(settings: Settings, seeded) -> None:
    store, _, _ = seeded

    def responder(system: str, user: str) -> dict[str, Any] | str:
        if system == prompts.QUERY_REWRITE_SYSTEM_PROMPT:
            return {"search_terms": ["kriptopara"], "entities": []}
        return FAKE_ANSWER

    http = TestClient(create_app(settings, store, FakeOllama(responder=responder)))
    body = http.post("/ask", json={"question": "Kriptopara düzenlemesi ne oldu?"}).json()
    assert body["retrieved_count"] == 0 and body["model"] == NO_EVIDENCE_MODEL
    assert body["answer"].startswith(prompts.INSUFFICIENT_EVIDENCE_TEXT)
    assert http.get("/health").json()["llm"] is True


def test_api_search_returns_alarm_docs(client) -> None:
    http, _, records = client
    res = http.get("/articles/search", params={"q": "fon"})
    assert res.status_code == 200
    body = res.json()
    ids = {hit["id"] for hit in body["results"]}
    assert ids == {records["alarm_high"].id, records["alarm_mid"].id}
    assert all(hit["is_alarm"] and hit["alarm_score"] >= 60 for hit in body["results"])
    assert body["results"][0]["snippet"]
    only = http.get("/articles/search", params={"q": "fon", "only_alarms": "true", "min_score": 80, "sources": "hurriyet"}).json()
    assert [hit["id"] for hit in only["results"]] == [records["alarm_high"].id]
    assert http.get("/articles/search", params={"q": "", "size": 3}).json()["count"] == 3
    assert http.get("/articles/search", params={"size": 0}).status_code == 422


def test_api_get_article(client) -> None:
    http, _, records = client
    res = http.get(f"/articles/{records['newest'].id}")
    assert res.status_code == 200 and res.json()["title"] == records["newest"].title
    missing = http.get("/articles/yok-boyle-bir-id")
    assert missing.status_code == 404 and "bulunamadı" in missing.json()["detail"]


def test_api_alarms(client) -> None:
    http, _, records = client
    body = http.get("/alarms").json()
    assert body["count"] == 2 and [a["alarm_score"] for a in body["items"]] == [85, 72]
    assert "content" not in body["items"][0]
    assert body["items"][0]["record_id"] == records["alarm_high"].id
    assert http.get("/alarms", params={"min_score": 80}).json()["count"] == 1
    assert "content" in http.get("/alarms", params={"include_content": "true"}).json()["items"][0]


def test_api_stats(client) -> None:
    http, _, _ = client
    body = http.get("/stats", params={"hours": 24 * 30}).json()
    assert body["hours"] == 720 and body["total"] == 6 and body["alarms"] == 2
    assert body["by_source"] == {"hurriyet": 3, "12punto": 3}
    assert body["top_alarms"][0]["alarm_score"] == 85


def test_api_generate_and_list_reports(client) -> None:
    http, broker, _ = client
    res = http.post("/reports/generate", json={"kind": "adhoc", "hours": 24 * 30})
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "adhoc" and body["stats"]["total"] == 6 and body["narrative"] == FAKE_NARRATIVE
    assert body["published"] is True
    published = report_messages(broker, RoutingKey.REPORT_GENERATED)
    assert len(published) == 1 and published[0].body["report_id"] == body["report_id"]
    listed = http.get("/reports", params={"kind": "adhoc"}).json()
    assert listed["count"] == 1 and listed["items"][0]["report_id"] == body["report_id"]
    assert http.get("/reports", params={"kind": "periodic"}).json()["count"] == 0
    assert http.post("/reports/generate", json={"kind": "bilinmeyen"}).status_code == 422


def test_api_generate_report_returns_stored_report_when_publish_fails(settings: Settings, seeded) -> None:
    store, _, _ = seeded

    class DeadBroker(InMemoryBroker):
        def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
            raise ConnectionError("Mesaj yayınlanamadı: report.generated")

    http = TestClient(create_app(settings, store, FakeOllama(responder=fake_responder), broker=DeadBroker(settings)))
    res = http.post("/reports/generate", json={"kind": "adhoc", "hours": 24 * 30})
    assert res.status_code == 200
    body = res.json()
    assert body["published"] is False and body["report_id"] in store.reports and body["narrative"] == FAKE_NARRATIVE
    assert http.get("/reports", params={"kind": "adhoc"}).json()["count"] == 1
    # Broker verilmemişse de rapor yazılır ve yayınlanmadığı bildirilir.
    no_broker = TestClient(create_app(settings, store, FakeOllama(responder=fake_responder)))
    assert no_broker.post("/reports/generate", json={"kind": "daily", "hours": 24}).json()["published"] is False


def test_api_generate_report_does_not_wait_for_stalled_broker(settings: Settings, seeded) -> None:
    store, _, _ = seeded
    release = threading.Event()
    attempts: list[str] = []

    class StalledBroker(InMemoryBroker):
        """RabbitMQ kapalı: publish yeniden bağlanma döngüsünde dakikalarca bloklanır."""

        def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
            attempts.append(str(body["report_id"]))
            release.wait(5)
            super().publish(routing_key, body, headers)

    broker = StalledBroker(settings)
    app = create_app(settings, store, FakeOllama(responder=fake_responder), broker=broker)
    app.state.publisher.timeout = 0.2
    http = TestClient(app)
    started = datetime.now(UTC)
    first = http.post("/reports/generate", json={"kind": "adhoc", "hours": 24})
    assert first.status_code == 200 and first.json()["published"] is False
    assert datetime.now(UTC) - started < timedelta(seconds=2) and first.json()["report_id"] in store.reports
    # Önceki yayın hâlâ askıdayken yeni istek broker'ı çağırmaz (iş parçacığı birikmez) ama raporu yine yazar.
    second = http.post("/reports/generate", json={"kind": "daily", "hours": 24})
    assert second.status_code == 200 and second.json()["published"] is False and len(attempts) == 1
    assert second.json()["report_id"] in store.reports
    release.set()
    assert wait_until(lambda: len(report_messages(broker, RoutingKey.REPORT_GENERATED)) == 1)
    # Broker toparlanınca yayın yeniden onaylanır.
    third = http.post("/reports/generate", json={"kind": "adhoc", "hours": 48})
    assert third.json()["published"] is True and len(report_messages(broker, RoutingKey.REPORT_GENERATED)) == 2


def test_api_maps_store_errors_to_503(settings: Settings) -> None:
    class BrokenStore(InMemoryStore):
        def search_records(self, query: str, **kwargs: Any) -> list[SearchHit]:
            raise Retry("Elasticsearch erişilemiyor")

    http = TestClient(create_app(settings, BrokenStore(), FakeOllama(responder=fake_responder)))
    res = http.get("/articles/search", params={"q": "fon"})
    assert res.status_code == 503 and "erişilemiyor" in res.json()["detail"]


def test_api_dashboard_renders_html(client) -> None:
    http, _, records = client
    http.post("/reports/generate", json={"kind": "adhoc", "hours": 24 * 30})
    res = http.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    html = res.text
    assert "Alarm" in html and "Soru" in html and "/ask" in html
    assert records["alarm_high"].title in html and records["alarm_high"].content_url in html
    assert "badge critical" in html and "badge important" in html
    assert "adhoc" in html or "İsteğe bağlı rapor" in html
    # Eşleşme yokken pano "dil modeli kullanılamadı" demez; o ileti yalnızca gerçek LLM yedeğine (fallback) aittir.
    assert "eşleşen haber bulunamadı" in html and 'answer.model === "fallback"' in html
    assert score_class(85) == "critical" and score_class(60) == "important" and score_class(30) == "notable" and score_class("x") == "routine"


def test_api_dashboard_shows_error_when_store_fails(settings: Settings) -> None:
    class BrokenStore(InMemoryStore):
        def recent_alarms(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("ES çöktü")

    http = TestClient(create_app(settings, BrokenStore(), FakeOllama()))
    res = http.get("/")
    assert res.status_code == 200 and "Veri alınamadı" in res.text


def test_split_narrative_separates_executive_summary_from_sections() -> None:
    from scraperhryt.reporting.api import split_narrative

    summary, details = split_narrative(
        "Yönetici özeti: Pencerede 134 haber işlendi; 10 tanesi alarm.\nÖne çıkan gelişmeler:\n- [85] Fon\nDağılım: hurriyet 40"
    )
    assert summary == "Pencerede 134 haber işlendi; 10 tanesi alarm."
    assert details.startswith("Öne çıkan gelişmeler:") and "Dağılım: hurriyet 40" in details
    assert split_narrative("**Yönetici özeti:** Kısa özet.\n\n## Dağılım\nx") == ("Kısa özet.", "Dağılım\nx")
    assert split_narrative("") == ("", "")


def test_dashboard_lists_reports_newest_window_first() -> None:
    from datetime import timedelta

    from scraperhryt.models import utcnow

    store, now = InMemoryStore(), utcnow()
    for rid, hours_ago, generated_ago in (("eski", 30, 0), ("yeni", 1, 5), ("orta", 10, 1)):
        store.index_report(
            Report(report_id=rid, kind="periodic", window_start=now - timedelta(hours=hours_ago + 24), window_end=now - timedelta(hours=hours_ago), generated_at=now - timedelta(minutes=generated_ago), narrative=f"Yönetici özeti: rapor-{rid}")
        )
    html = TestClient(create_app(Settings(_env_file=None), store, FakeOllama())).get("/").text
    assert html.index("rapor-yeni") < html.index("rapor-orta") < html.index("rapor-eski")


def test_dashboard_shows_alarm_keywords_and_keyword_placeholder() -> None:
    s = Settings(_env_file=None, keywords="bakan,fon,=TMSF,re:kayy[ıi]m")
    html = TestClient(create_app(s, InMemoryStore(), FakeOllama())).get("/").text
    assert "Alarm anahtar kelimeleri" in html and 'data-kw="TMSF"' in html and "kayy" not in html
    assert "Fon soruşturmasında son durum ne?" in html and "Kılıçdaroğlu arasındaki" not in html


def test_alarm_list_filters_by_keyword() -> None:
    from datetime import UTC, datetime

    store = InMemoryStore()
    for slug, kws in (("a", ["fon"]), ("b", ["bakan", "fon"]), ("c", ["ihale"])):
        rec = make_record(slug, f"Haber {slug}", "metin", published=datetime.now(UTC), score=80, keywords=kws)
        store.index_record(rec)
        store.index_alarm(AlarmEvent.from_record(rec))
    client = TestClient(create_app(Settings(_env_file=None), store, FakeOllama()))
    titles = lambda kw: sorted(i["title"] for i in client.get("/alarms", params={"keyword": kw, "include_duplicates": True}).json()["items"])  # noqa: E731
    assert titles("fon") == ["Haber a", "Haber b"] and titles("İHALE") == ["Haber c"] and titles("=bakan") == ["Haber b"]
    assert 'data-kw="fon"' in client.get("/").text


# ---------------------------------------------------------------------------------------------------------
# Hız: sorgu yeniden yazma kapalı, akışlı yanıt
# ---------------------------------------------------------------------------------------------------------
def test_query_rewrite_is_skipped_by_default(seeded) -> None:
    store, records, _ = seeded
    llm = FakeOllama(responder=fake_responder)
    answer = QAEngine(Settings(_env_file=None), store, llm).ask(QUESTION)
    assert not any(system == prompts.QUERY_REWRITE_SYSTEM_PROMPT for _, system, _ in llm.calls)
    assert answer.answer == FAKE_ANSWER and answer.sources[0].id == records["newest"].id


def _events(res) -> list[dict[str, Any]]:
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def test_ask_stream_sends_sources_tokens_and_final_answer(settings: Settings, seeded) -> None:
    store, records, _ = seeded
    http = TestClient(create_app(settings, store, FakeOllama(responder=fake_responder)))
    res = http.post("/ask/stream", json={"question": QUESTION, "since_days": 14, "top_k": 5})
    assert res.status_code == 200 and res.headers["content-type"].startswith("application/x-ndjson")
    events = _events(res)
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["id"] == records["newest"].id
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1 and "".join(tokens) == FAKE_ANSWER
    final = events[-1]
    assert final["type"] == "answer" and final["answer"]["answer"] == FAKE_ANSWER and final["answer"]["model"] == "fake"
    assert final["answer"] == {**http.post("/ask", json={"question": QUESTION, "since_days": 14, "top_k": 5}).json(), "generated_at": final["answer"]["generated_at"]}
    assert http.post("/ask/stream", json={"question": ""}).status_code == 422


def test_ask_stream_falls_back_when_llm_fails_mid_stream(settings: Settings, seeded) -> None:
    store, _, _ = seeded

    class Broken(FakeOllama):
        def stream_text(self, system, user, *, timeout=None):
            yield "Yarım "
            raise LLMUnavailable("bağlantı koptu")

    qa = QAEngine(settings, store, Broken(responder=fake_responder))
    events = list(qa.stream_answer(qa.prepare(QUESTION)))
    assert [e["type"] for e in events] == ["sources", "token", "answer"]
    assert events[-1]["answer"]["model"] == "fallback" and events[-1]["answer"]["answer"]


def test_ollama_stream_text_yields_pieces_and_raises_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        lines = [{"response": "Merhaba ", "done": False}, {"response": "dünya", "done": False}, {"response": "", "done": True}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    client = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    assert list(client.stream_text("s", "u", timeout=5)) == ["Merhaba ", "dünya"]

    bad = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    try:
        list(bad.stream_text("s", "u", timeout=5))
    except LLMUnavailable as exc:
        assert "500" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("LLMUnavailable bekleniyordu")
