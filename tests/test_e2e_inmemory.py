"""Uçtan uca bellek içi boru hattı testi.

5 sahte haber → anahtar kelime filtresi → LLM skorlama (FakeOllama: içeriğinde 'fon' geçenlere 90, diğerlerine 15)
→ alarm katmanı (InMemoryStore) → rapor tüketicisi → QAEngine. Her aşama BUILD_SPEC'teki routing key / kuyruk
sözleşmesini kullanır; NewsRecord alanları aşamalar boyunca kaybolmadan taşınmalıdır.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scraperhryt.broker import InMemoryBroker, Queue, RoutingKey
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord, Stage
from scraperhryt.pipeline.alarm import AlarmService
from scraperhryt.pipeline.keyword_filter import KeywordFilterService
from scraperhryt.pipeline.llm import FakeOllama
from scraperhryt.pipeline.prompts import parse_user_prompt
from scraperhryt.pipeline.scorer import ScoringService
from scraperhryt.reporting import prompts as report_prompts
from scraperhryt.reporting.builder import ReportBuilder
from scraperhryt.reporting.rag import QAEngine
from scraperhryt.reporting.service import ReportingConsumer
from scraperhryt.store import InMemoryStore
from scraperhryt.textutil import tr_lower

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

# (url, title, subtitle, content, category, tags) — ikisi 'fon', ikisi yalnızca 'bakan', biri anahtar kelimesiz
FIXTURE_ARTICLES: list[tuple[str, str, str, str, str, list[str]]] = [
    (
        "https://www.hurriyet.com.tr/gundem/fon-sorusturmasinda-yeni-gelisme-43000001",
        "Fon soruşturmasında yeni gelişme",
        "Savcılık ek tutuklama talep etti",
        "Fon soruşturması kapsamında savcılık, yatırım fonu yöneticileri hakkında ek tutuklama talep etti. "
        "Bakan açıklama yapmadı.",
        "gundem",
        ["fon", "soruşturma"],
    ),
    (
        "https://12punto.com.tr/ekonomi/yatirim-fonu-skandali-buyuyor-900002",
        "Yatırım fonu skandalı büyüyor",
        "Mağdur sayısı artıyor",
        "Yatırım fonu skandalında mağdur sayısı 10 bini aştı; Sermaye Piyasası Kurulu inceleme başlattı.",
        "ekonomi",
        [],
    ),
    (
        "https://www.hurriyet.com.tr/gundem/bakan-yeni-yol-haritasini-acikladi-43000003",
        "Bakan yeni yol haritasını açıkladı",
        "Eğitimde yeni dönem",
        "Milli Eğitim Bakanı, yeni eğitim-öğretim yılı için yol haritasını paylaştı.",
        "gundem",
        ["eğitim"],
    ),
    (
        "https://12punto.com.tr/siyaset/bakanlik-personel-alimi-900004",
        "Bakanlık personel alımı yapacak",
        "",
        "Bakanlık 500 yeni personel alımı için ilana çıktı; başvurular e-Devlet üzerinden alınacak.",
        "siyaset",
        [],
    ),
    (
        "https://www.hurriyet.com.tr/gundem/derbide-kazanan-cikmadi-43000005",
        "Derbide kazanan çıkmadı",
        "Maç 1-1 bitti",
        "Süper Lig'de oynanan derbi 1-1 berabere sona erdi; taraftarlar stadı erken terk etti.",
        "spor",
        ["spor"],
    ),
]


def make_settings(tmp_path: Any) -> Settings:
    return Settings(
        _env_file=None,
        keywords="bakan,cumhurbaşkanı,fon",
        alarm_threshold=60,
        llm_score_all=False,
        ollama_embedding_model="",
        alarm_webhook_url="",
        telegram_bot_token="",
        telegram_chat_id="",
        alarm_log_path=str(tmp_path / "alarms.jsonl"),
        report_digest_every=100,
        report_digest_minutes=10_000,
        rag_recency_days=14,
        rag_top_k=5,
    )


def make_records() -> list[NewsRecord]:
    records: list[NewsRecord] = []
    for index, (url, title, subtitle, content, category, tags) in enumerate(FIXTURE_ARTICLES):
        records.append(
            NewsRecord.new(
                source="hurriyet" if "hurriyet" in url else "12punto",
                content_url=url,
                title=title,
                subtitle=subtitle,
                content=content,
                published_at=NOW - timedelta(hours=index),
                updated_at=NOW - timedelta(hours=index, minutes=-5),
                category=category,
                author="Test Muhabiri",
                image_url=f"https://img.example/{index}.jpg",
                tags=tags,
            )
        )
    return records


def responder(system: str, user: str) -> dict[str, Any] | str:
    if system == report_prompts.QUERY_REWRITE_SYSTEM_PROMPT:
        return {"search_terms": ["fon", "soruşturma"], "entities": ["savcılık"]}
    if system == report_prompts.RAG_SYSTEM_PROMPT:
        return "Son durum: fon soruşturmasında ek tutuklama talep edildi [1]."
    if "alarm_score" not in system:  # rapor anlatısı
        return "Dönemde fon soruşturması öne çıktı."
    content = parse_user_prompt(user)["content"]
    if "fon" in tr_lower(content):
        return {
            "alarm_score": 90,
            "is_alarm": True,
            "reason": "Fon soruşturması; mali suç ve kamu etkisi yüksek.",
            "summary": "Fonla ilgili ciddi bir gelişme bildirildi.",
            "topics": ["finans/fon", "hukuk"],
            "entities": ["Savcılık"],
        }
    return {
        "alarm_score": 15,
        "is_alarm": False,
        "reason": "Rutin kurumsal haber.",
        "summary": "Rutin haber.",
        "topics": ["diğer"],
        "entities": [],
    }


@pytest.fixture
def pipeline(tmp_path: Any) -> dict[str, Any]:
    settings = make_settings(tmp_path)
    broker = InMemoryBroker(settings)
    broker.declare_topology()
    store = InMemoryStore()
    store.ensure_indices()
    llm = FakeOllama(responder=responder)
    builder = ReportBuilder(settings, store, llm)
    return {
        "settings": settings,
        "broker": broker,
        "store": store,
        "llm": llm,
        "filter": KeywordFilterService(settings, broker),
        "scorer": ScoringService(settings, broker, llm),
        "alarm": AlarmService(settings, broker, store, sinks=[]),
        "reporter": ReportingConsumer(settings, broker, store, builder),
        "records": make_records(),
    }


def run_pipeline(p: dict[str, Any]) -> dict[str, int]:
    broker: InMemoryBroker = p["broker"]
    for record in p["records"]:
        broker.publish(RoutingKey.ARTICLE_RAW, record.to_message(), headers={"x-source": record.source})
    processed = {
        "filter": p["filter"].run(max_messages=None),
        "scorer": p["scorer"].run(max_messages=None),
        "alarm": p["alarm"].run(max_messages=None),
        "reporter": p["reporter"].run(max_messages=None),
    }
    return processed


def published_to(broker: InMemoryBroker, routing_key: str) -> list[dict[str, Any]]:
    return [m.body for m in broker.published if m.routing_key == routing_key]


def test_e2e_routing_and_alarm_policy(pipeline: dict[str, Any]) -> None:
    broker: InMemoryBroker = pipeline["broker"]
    processed = run_pipeline(pipeline)
    assert processed == {"filter": 5, "scorer": 4, "alarm": 5, "reporter": 2}

    assert len(published_to(broker, RoutingKey.ARTICLE_RAW)) == 5
    assert len(published_to(broker, RoutingKey.ARTICLE_KEYWORD)) == 4  # spor haberi eşleşmez
    scored = published_to(broker, RoutingKey.ARTICLE_SCORED)
    assert len(scored) == 5
    assert all(body["stage"] == Stage.SCORED for body in scored)
    alarms = published_to(broker, RoutingKey.ALARM_RAISED)
    assert len(alarms) == 2
    assert broker.size(Queue.ALARMS) == 0  # rapor tüketicisi hepsini aldı
    assert not broker.dead_letters
    fon_titles = {"Fon soruşturmasında yeni gelişme", "Yatırım fonu skandalı büyüyor"}
    assert {body["title"] for body in alarms} == fon_titles
    for body in alarms:
        assert body["record"]["stage"] == Stage.ALARM
        assert body["alarm_id"] == body["record"]["alarm_id"]
        assert "fon" in body["matched_keywords"]
        assert body["alarm_score"] == 90 and body["llm_summary"]

    # anahtar kelimesiz haber LLM'e gitmedi: skor 0, llm yok
    sport = next(body for body in scored if body["category"] == "spor")
    assert sport["matched_keywords"] == [] and sport["alarm_score"] == 0 and sport["llm"] is None


def test_e2e_store_contents_and_field_roundtrip(pipeline: dict[str, Any]) -> None:
    run_pipeline(pipeline)
    store: InMemoryStore = pipeline["store"]
    originals = {r.id: r for r in pipeline["records"]}
    assert set(store.records) == set(originals)
    for record_id, doc in store.records.items():
        src = originals[record_id]
        stored = NewsRecord.from_message({k: v for k, v in doc.items() if k not in ("@timestamp", "content_length")})
        assert stored.stage == Stage.ALARM
        assert stored.processed_at is not None
        assert stored.content_hash == src.content_hash
        for field in ("source", "content_url", "title", "subtitle", "content", "category", "author", "image_url", "tags"):
            assert getattr(stored, field) == getattr(src, field), field
        assert stored.published_at == src.published_at and stored.updated_at == src.updated_at
        if "fon" in tr_lower(stored.content):
            assert stored.is_alarm and stored.alarm_score == 90
            assert stored.alarm_reason.startswith("Fon soruşturması") and "LLM Özeti:" in stored.alarm_reason
            assert stored.llm_summary and stored.alarm_id and stored.alarmed_at is not None
        else:
            assert not stored.is_alarm and stored.alarm_reason == "" and stored.llm_summary == ""
            assert stored.alarm_id == "" and stored.alarmed_at is None
            if stored.matched_keywords:
                assert stored.alarm_score == 15 and stored.llm is not None and stored.llm.summary == "Rutin haber."
    assert len(store.alarms) == 2
    assert {a["record_id"] for a in store.alarms.values()} == {
        r.id for r in originals.values() if "fon" in tr_lower(r.content)
    }


def test_e2e_digest_and_rag(pipeline: dict[str, Any]) -> None:
    run_pipeline(pipeline)
    store: InMemoryStore = pipeline["store"]
    broker: InMemoryBroker = pipeline["broker"]
    reporter: ReportingConsumer = pipeline["reporter"]

    # eşik (100 alarm / 10000 dk) dolmadı ama run() çıkışta tamponu boşaltır → tam bir özet
    assert reporter.stats.buffered == 2 and reporter.stats.digests == 1 and reporter.buffered == 0
    digests = [doc for doc in store.reports.values() if doc["kind"] == "alarm_digest"]
    assert len(digests) == 1
    report = digests[0]
    assert len(report["top_alarms"]) == 2 and report["narrative"]
    assert {a["title"] for a in report["top_alarms"]} == {
        "Fon soruşturmasında yeni gelişme",
        "Yatırım fonu skandalı büyüyor",
    }
    published = published_to(broker, RoutingKey.REPORT_ALARM_DIGEST)
    assert len(published) == 1 and published[0]["report_id"] == report["report_id"]
    assert broker.size(Queue.REPORTS) == 1
    assert reporter.flush() is None  # tampon boş: ikinci özet üretilmez

    answer = QAEngine(pipeline["settings"], store, pipeline["llm"]).ask("Fon soruşturmasında son durum ne?")
    assert answer.retrieved_count >= 1 and answer.sources
    first = answer.sources[0]
    assert first.title == "Fon soruşturmasında yeni gelişme"
    assert first.content_url == FIXTURE_ARTICLES[0][0]
    assert "[1]" in answer.answer and answer.model == "fake"
    assert "Derbide kazanan çıkmadı" not in {c.title for c in answer.sources}
