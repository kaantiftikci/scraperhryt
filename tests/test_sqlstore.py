"""SQL asıl kayıt + Elasticsearch arama indeksi; Ollama TLS ayarları."""

from __future__ import annotations

import json
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from scraperhryt.config import Settings
from scraperhryt.models import AlarmEvent, Feedback, LLMVerdict, NewsRecord, Report
from scraperhryt.pipeline.llm import OllamaClient, ollama_tls_verify
from scraperhryt.sqlstore import RecordStore, SqlRecordStore, make_record_store, reindex_from_sql
from scraperhryt.store import InMemoryStore


def _record(slug: str, score: int) -> NewsRecord:
    rec = NewsRecord.new(
        source="hurriyet",
        content_url=f"https://www.hurriyet.com.tr/gundem/{slug}",
        title=f"Fon soruşturmasında {slug}",
        subtitle="Bakan açıklama yaptı",
        content="Fon soruşturması kapsamında yeni gelişmeler yaşandı.",
        published_at=datetime.now(UTC) - timedelta(hours=1),
    )
    rec.matched_keywords = ["fon"]
    verdict = LLMVerdict(model="test", alarm_score=score, reason="gerekçe", summary="özet", topics=["hukuk"], entities=[])
    rec.apply_verdict(verdict, threshold=60)
    return rec


def _sql(tmp_path: Path) -> SqlRecordStore:
    sql = SqlRecordStore(f"sqlite:///{tmp_path / 'records.db'}")
    sql.ensure_schema()
    return sql


def test_record_store_writes_sql_first_and_searches_elastic(tmp_path: Path) -> None:
    search, sql = InMemoryStore(), _sql(tmp_path)
    store = RecordStore(search, sql)
    rec = _record("ilk", 85)
    store.index_record(rec, embedding=[0.1, 0.2])
    store.index_record(rec)  # upsert: ikinci yazma kopya üretmez
    event = AlarmEvent.from_record(rec)
    store.index_alarm(event)
    store.index_report(
        Report(report_id="r1", kind="periodic", window_start=datetime.now(UTC) - timedelta(hours=1), window_end=datetime.now(UTC))
    )
    store.index_feedback(Feedback(feedback_id="f1", alarm_id=event.alarm_id, label="true_positive"))

    assert sql.counts() == {"articles": 1, "alarms": 1, "reports": 1, "feedback": 1}
    assert sql.get_record(rec.id)["title"] == rec.title and store.get_alarm(event.alarm_id)["record_id"] == rec.id
    # Arama SQL'e değil Elasticsearch'e (burada bellek içi arama deposu) gider
    hits = store.search_records("fon soruşturması")
    assert [h.doc["id"] for h in hits] == [rec.id]
    assert store.recent_alarms() and store.list_reports() and store.health()


def test_reindex_rebuilds_search_index_from_sql(tmp_path: Path) -> None:
    sql = _sql(tmp_path)
    store = RecordStore(InMemoryStore(), sql)
    recs = [_record(f"haber-{i}", 70 + i) for i in range(3)]
    for rec in recs:
        store.index_record(rec, embedding=[1.0, 0.0])
        store.index_alarm(AlarmEvent.from_record(rec))

    fresh = InMemoryStore()  # ES silindi / yeni küme
    done = reindex_from_sql(sql, fresh, batch_size=2)
    assert done == {"articles": 3, "alarms": 3, "reports": 0, "feedback": 0}
    assert {h.doc["id"] for h in fresh.search_records("fon", size=10)} == {r.id for r in recs}
    assert len(fresh.recent_alarms()) == 3


def test_make_record_store_is_plain_elastic_without_database_url() -> None:
    search = InMemoryStore()
    assert make_record_store(Settings(_env_file=None, database_url=""), search) is search
    wrapped = make_record_store(Settings(_env_file=None, database_url="sqlite://"), search)
    assert isinstance(wrapped, RecordStore)


def test_ollama_tls_options_and_path_prefix(tmp_path: Path) -> None:
    assert ollama_tls_verify(Settings(_env_file=None)) is True
    assert ollama_tls_verify(Settings(_env_file=None, ollama_verify_tls=False)) is False
    import certifi

    assert isinstance(ollama_tls_verify(Settings(_env_file=None, ollama_ca_bundle=certifi.where())), ssl.SSLContext)

    # Sunucu bir yol altında yayınlanıyorsa (https://host/llm/) istekler o yolun altına gider.
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps({"ok": 1})}, "done": True})

    s = Settings(_env_file=None, ollama_base_url="https://llm.example/llm/", ollama_model="phi4", ollama_verify_tls=False)
    with OllamaClient(s, transport=httpx.MockTransport(handler)) as client:
        assert client.chat_json("s", "u") == {"ok": 1}
    assert seen == ["/llm/api/chat"]
