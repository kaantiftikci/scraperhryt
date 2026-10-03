"""Doğruluk geliştirmeleri: eşik kuralları, çoklu örnekleme, eş anlamlılar, olay kümeleme, geri bildirim, kalibrasyon,
yeniden skorlama, zaman çizelgesi ve ölü mektup geri oynatma."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scraperhryt.broker import InMemoryBroker, Queue, RoutingKey
from scraperhryt.config import Settings
from scraperhryt.models import AlarmEvent, Feedback, LLMVerdict, NewsRecord, Stage, utcnow
from scraperhryt.pipeline.alarm import AlarmService
from scraperhryt.pipeline.calibration import format_calibration_report, load_golden_set, run_calibration
from scraperhryt.pipeline.dedup import EventClusterer, title_jaccard
from scraperhryt.pipeline.keyword_filter import KeywordFilterService
from scraperhryt.pipeline.llm import FakeOllama
from scraperhryt.pipeline.rescore import Rescorer
from scraperhryt.pipeline.scorer import ScoringService
from scraperhryt.pipeline.thresholds import parse_threshold_rules, resolve_threshold
from scraperhryt.replay import origin_queue
from scraperhryt.reporting.api import create_app
from scraperhryt.reporting.rag import QAEngine
from scraperhryt.store import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]


def make_record(title: str, content: str = "içerik", *, source: str = "hurriyet", keywords: list[str] | None = None, days_ago: float = 0) -> NewsRecord:
    rec = NewsRecord.new(
        source=source,
        content_url=f"https://www.hurriyet.com.tr/gundem/{abs(hash(title)) % 10_000_000}-{len(title)}",
        title=title,
        content=content,
        published_at=utcnow() - timedelta(days=days_ago),
    )
    rec.matched_keywords = keywords or []
    rec.stage = Stage.KEYWORD
    return rec


def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, **kw)


# ---- eşikler ----
def test_threshold_rules_precedence_and_invalid_json() -> None:
    parse_threshold_rules.cache_clear()
    s = settings(alarm_threshold=60, alarm_thresholds_json='{"source:12punto": 70, "category:Spor": 90, "keyword:fon": 50}')
    rec = make_record("x", source="12punto", keywords=["bakan"])
    assert resolve_threshold(s, rec) == 70
    rec.matched_keywords = ["bakan", "fon"]
    assert resolve_threshold(s, rec) == 50
    rec.category = "SPOR"
    assert resolve_threshold(s, rec) == 90
    assert resolve_threshold(settings(alarm_threshold=60, alarm_thresholds_json="{bozuk"), rec) == 60
    assert resolve_threshold(settings(alarm_threshold=60), rec) == 60


# ---- çoklu örnekleme ----
def test_self_consistency_median_confidence_and_needs_review() -> None:
    scores = iter([90, 30, 85])

    def responder(system: str, user: str) -> dict[str, Any]:
        return {"alarm_score": next(scores), "is_alarm": True, "reason": "r", "summary": "s", "confidence": 80}

    s = settings(llm_samples=3, llm_disagreement_threshold=25, alarm_threshold=60)
    svc = ScoringService(s, InMemoryBroker(s), FakeOllama(responder=responder))
    rec = svc.score_record(make_record("Bakan açıkladı", keywords=["bakan"]))
    assert rec.alarm_score == 85 and rec.is_alarm and sorted(rec.score_samples) == [30, 85, 90]
    assert rec.needs_review is True and 0 < rec.confidence < 80 and rec.alarm_threshold_used == 60


def test_single_sample_records_model_confidence() -> None:
    s = settings(llm_samples=1)
    svc = ScoringService(s, InMemoryBroker(s), FakeOllama(responder=lambda sy, u: {"alarm_score": 70, "reason": "r", "summary": "s", "confidence": 42}))
    rec = svc.score_record(make_record("Bakan", keywords=["bakan"]))
    assert rec.confidence == 42 and rec.score_samples == [70] and rec.needs_review is False


# ---- eş anlamlılar ----
def test_keyword_aliases_fold_to_canonical(tmp_path: Path) -> None:
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"fon": ["SPK", "Sermaye Piyasası Kurulu"], "_not": "yorum", "yok": ["x"]}), encoding="utf-8")
    s = settings(keywords="bakan,fon", keyword_aliases_path=str(aliases))
    broker = InMemoryBroker(s)
    svc = KeywordFilterService(s, broker)
    rec = NewsRecord.new(source="12punto", content_url="https://12punto.com.tr/ekonomi/spk-karar-1", title="SPK iki şirkete ceza verdi", content="Sermaye Piyasası Kurulu kararı")
    svc.handle(_msg(rec))
    out = NewsRecord.from_message(broker.drain(Queue.ARTICLES_KEYWORD)[0].body)
    assert out.matched_keywords == ["fon"]


def test_shipped_alias_file_is_valid_and_canonical() -> None:
    s = settings(keywords="bakan,cumhurbaşkanı,fon", keyword_aliases_path=str(ROOT / "config" / "keyword_aliases.json"))
    svc = KeywordFilterService(s, InMemoryBroker(s))
    assert set(svc.aliases.values()) <= {"bakan", "cumhurbaşkanı", "fon"} and len(svc.aliases) >= 10


def _msg(rec: NewsRecord):
    from scraperhryt.broker import Message

    return Message(body=rec.to_message(), routing_key=RoutingKey.ARTICLE_RAW)


# ---- olay kümeleme / tekrar bastırma ----
class _Sink:
    name = "test"

    def __init__(self) -> None:
        self.sent: list[AlarmEvent] = []

    def send(self, event: AlarmEvent) -> None:
        self.sent.append(event)


def _alarm(title: str, score: int = 85) -> NewsRecord:
    rec = make_record(title, keywords=["fon"])
    rec.apply_verdict(LLMVerdict(model="fake", alarm_score=score, reason="r", summary="s"), 60)
    return rec


def test_duplicate_alarm_is_stored_published_but_not_notified() -> None:
    s = settings(alarm_dedup_enabled=True, alarm_dedup_title_jaccard=0.5)
    broker, store, sink = InMemoryBroker(s), InMemoryStore(), _Sink()
    svc = AlarmService(s, broker, store, sinks=[sink])
    first = _alarm("Fon soruşturmasında 20 şüpheliye tutuklama talebi")
    second = _alarm("Fon soruşturmasında 20 şüpheli için tutuklama talebi geldi")
    svc.handle(_scored(first))
    svc.handle(_scored(second))
    alarms = broker.drain(Queue.ALARMS)
    assert len(alarms) == 2 and len(sink.sent) == 1 and svc.stats.suppressed == 1
    dup = store.get_record(second.id)
    assert dup["duplicate_of"] == first.id[:0] + store.get_record(first.id)["alarm_id"]
    assert dup["event_id"] == store.get_record(first.id)["event_id"]
    assert alarms[1].headers["x-duplicate"] is True and alarms[0].headers["x-duplicate"] is False
    assert store.get_alarm(dup["alarm_id"])["duplicate_of"] == dup["duplicate_of"]


def test_unrelated_alarms_get_distinct_events_and_dedup_can_be_disabled() -> None:
    s = settings(alarm_dedup_enabled=True)
    store, sink = InMemoryStore(), _Sink()
    svc = AlarmService(s, InMemoryBroker(s), store, sinks=[sink])
    svc.handle(_scored(_alarm("Fon soruşturmasında tutuklama talebi")))
    svc.handle(_scored(_alarm("Bakan asgari ücret zammını açıkladı")))
    assert len(sink.sent) == 2 and len({d["event_id"] for d in store.records.values()}) == 2
    off = EventClusterer(settings(alarm_dedup_enabled=False), store)
    assert off.cluster(_alarm("Fon soruşturmasında tutuklama talebi")).duplicate_of == ""
    assert title_jaccard("Fon soruşturması tutuklama", "fon soruşturması gözaltı") > 0.3


def _scored(rec: NewsRecord):
    from scraperhryt.broker import Message

    return Message(body=rec.to_message(), routing_key=RoutingKey.ARTICLE_SCORED)


# ---- geri bildirim + API ----
def test_feedback_endpoints_and_stats() -> None:
    s = settings()
    store = InMemoryStore()
    rec = _alarm("Fon soruşturması")
    event = AlarmEvent.from_record(rec)
    rec.alarm_id = event.alarm_id
    store.index_record(rec)
    store.index_alarm(event)
    client = TestClient(create_app(s, store, FakeOllama()))
    r = client.post(f"/alarms/{event.alarm_id}/feedback", json={"label": "false_positive", "note": "fiil"})
    assert r.status_code == 200 and r.json()["record_id"] == rec.id
    assert client.post("/alarms/yok/feedback", json={"label": "true_positive"}).status_code == 404
    assert client.post(f"/alarms/{event.alarm_id}/feedback", json={"label": "true_positive"}).status_code == 200
    stats = client.get("/feedback/stats").json()
    assert stats["total"] == 2 and stats["precision_estimate"] == 0.5
    assert client.get("/feedback", params={"alarm_id": event.alarm_id}).json()["count"] == 2
    listed = client.get("/alarms", params={"include_duplicates": False}).json()
    assert listed["count"] == 1


# ---- kalibrasyon ----
def test_calibration_report_suggests_threshold() -> None:
    items = load_golden_set(ROOT / "config" / "golden_set.jsonl")
    assert len(items) >= 12

    def responder(system: str, user: str) -> dict[str, Any]:
        hi = any(w in user for w in ("soruşturma", "açıkladı", "iddianame", "kararname", "ceza", "operasyon", "zam"))
        return {"alarm_score": 80 if hi else 10, "reason": "r", "summary": "s", "confidence": 70}

    s = settings()
    scorer = ScoringService(s, InMemoryBroker(s), FakeOllama(responder=responder))
    report = run_calibration(scorer, items, current_threshold=60, thresholds=range(0, 101, 10))
    assert len(report.items) == len(items) and report.failures == 0
    best = next(r for r in report.rows if r.threshold == report.best_threshold)
    assert best.f1 >= max(r.f1 for r in report.rows) - 1e-9
    text = format_calibration_report(report)
    assert "Önerilen ALARM_THRESHOLD" in text and "Eşik" in text


# ---- yeniden skorlama ----
def test_rescore_resets_and_republishes_keyword_hits_only() -> None:
    s = settings()
    store, broker = InMemoryStore(), InMemoryBroker(s)
    scored = _alarm("Fon haberi")
    plain = make_record("Spor haberi")
    plain.mark_not_scored()
    store.index_record(scored)
    store.index_record(plain)
    stats = Rescorer(s, store, broker).run(since_days=7, dry_run=True)
    assert stats.selected == 1 and stats.published == 0 and broker.size(Queue.ARTICLES_KEYWORD) == 0
    stats = Rescorer(s, store, broker).run(since_days=7)
    msgs = broker.drain(Queue.ARTICLES_KEYWORD)
    assert stats.published == 1 and msgs[0].headers == {"x-rescore": True, "x-previous-score": 85}
    out = NewsRecord.from_message(msgs[0].body)
    assert out.alarm_score == 0 and out.alarm_reason == "" and out.stage == Stage.KEYWORD and out.id == scored.id


# ---- zaman çizelgesi ----
def test_answer_has_chronological_timeline() -> None:
    s = settings(rag_recency_days=30)
    store = InMemoryStore()
    for i, (title, days) in enumerate([("Özgür Özel Kılıçdaroğlu ile görüştü", 5), ("Kılıçdaroğlu'ndan Özel'e sert sözler", 1), ("Özel: Kılıçdaroğlu ile uzlaşı", 3)]):
        store.index_record(make_record(title, f"içerik {i}", days_ago=days))
    answer = QAEngine(s, store, FakeOllama(responder=lambda sy, u: {"answer": "cevap"})).ask("Özgür Özel ile Kılıçdaroğlu arasındaki son durum ne?")
    dates = [t.date for t in answer.timeline]
    assert len(answer.timeline) == 3 and dates == sorted(dates)
    assert answer.timeline[-1].event.startswith("Kılıçdaroğlu'ndan")


# ---- ölü mektup ----
def test_origin_queue_inference() -> None:
    assert origin_queue({"x-origin-queue": "q.articles.keyword"}) == "q.articles.keyword"
    assert origin_queue({"x-death": [{"queue": b"q.articles.scored.retry", "reason": "expired"}]}) == "q.articles.scored"
    assert origin_queue({}, "q.alarms") == "q.alarms" and origin_queue({}) is None


def test_feedback_model_roundtrip_in_store() -> None:
    store = InMemoryStore()
    fb = Feedback(feedback_id="f1", alarm_id="a1", label="true_positive")
    store.index_feedback(fb)
    assert store.list_feedback(alarm_id="a1")[0]["label"] == "true_positive"
    assert store.feedback_stats()["precision_estimate"] == 1.0
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Feedback(feedback_id="f2", alarm_id="a1", label="true_positive", created_at="bozuk")  # type: ignore[arg-type]
