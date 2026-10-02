"""store.py testleri: bellek içi depo (sıralama, yenilik, filtreler, istatistik) + saf eşleme/sorgu kurucuları
+ sahte istemciyle ElasticsearchStore davranışı (Retry dönüşümü, parametreler, yanıt çözümleme)."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import NotFoundError

from scraperhryt.broker import Retry
from scraperhryt.config import Settings
from scraperhryt.models import AlarmEvent, LLMVerdict, NewsRecord, Report
from scraperhryt.store import (
    ElasticsearchStore,
    InMemoryStore,
    SearchHit,
    build_alarm_mapping,
    build_article_mapping,
    build_report_mapping,
    build_search_query,
    build_stats_aggs,
    make_store,
    tokenize,
)

_counter = itertools.count(1)


def now() -> datetime:
    return datetime.now(UTC)


def make_record(
    title: str,
    content: str = "",
    *,
    subtitle: str = "",
    source: str = "hurriyet",
    published: datetime | None = None,
    score: int | None = None,
    keywords: list[str] | None = None,
    category: str = "gundem",
) -> NewsRecord:
    n = next(_counter)
    host = "www.hurriyet.com.tr/gundem" if source == "hurriyet" else "12punto.com.tr/gundem"
    rec = NewsRecord.new(
        source=source,
        content_url=f"https://{host}/haber-{n}",
        title=title,
        content=content,
        subtitle=subtitle,
        published_at=published or now(),
        category=category,
    )
    rec.matched_keywords = list(keywords or [])
    if score is None:
        rec.mark_not_scored()
    else:
        verdict = LLMVerdict(model="test", alarm_score=score, reason="Gerekçe", summary="Özet metni")
        rec.apply_verdict(verdict, threshold=60)
    return rec


# ---------------------------------------------------------------------------------------------------------
# InMemoryStore: arama
# ---------------------------------------------------------------------------------------------------------


def test_tokenize_is_turkish_lowercase_and_dedupes() -> None:
    assert tokenize("Özgür ÖZEL ile Kılıçdaroğlu'nun İstanbul'da özel görüşmesi") == [
        "özgür", "özel", "ile", "kılıçdaroğlu", "nun", "istanbul", "da", "görüşmesi",
    ]
    assert tokenize("a b c") == []


def test_search_title_match_outranks_content_match() -> None:
    store = InMemoryStore()
    ts = now() - timedelta(hours=2)
    in_title = make_record(
        "Özgür Özel ile Kılıçdaroğlu arasında yeni tartışma",
        "CHP kurultayı öncesi gerilim sürüyor.",
        published=ts,
    )
    in_content = make_record(
        "CHP'de kurultay takvimi netleşti",
        "Özgür Özel ve Kemal Kılıçdaroğlu arasındaki tartışma parti kulislerinde konuşuluyor.",
        published=ts,
    )
    unrelated = make_record("Borsa güne yükselişle başladı", "BIST 100 endeksi arttı.", published=ts)
    for rec in (in_content, unrelated, in_title):
        store.index_record(rec)

    hits = store.search_records("Özgür Özel Kılıçdaroğlu son durum")
    assert [h.doc["id"] for h in hits] == [in_title.id, in_content.id]
    assert all(isinstance(h, SearchHit) and h.score > 0 for h in hits)
    assert hits[0].score > hits[1].score


def test_search_newer_identical_document_ranks_first() -> None:
    store = InMemoryStore()
    old = make_record("Bakan fon soruşturması hakkında açıklama yaptı", "Detaylar...", published=now() - timedelta(days=6))
    new = make_record("Bakan fon soruşturması hakkında açıklama yaptı", "Detaylar...", published=now() - timedelta(hours=1))
    store.index_record(old)
    store.index_record(new)
    hits = store.search_records("bakan fon soruşturması")
    assert [h.doc["id"] for h in hits] == [new.id, old.id]


def test_search_tolerates_turkish_suffixes_by_prefix() -> None:
    store = InMemoryStore()
    rec = make_record("Bakanlık yeni düzenlemeyi duyurdu", "Cumhurbaşkanlığı kararnamesi yayımlandı.")
    store.index_record(rec)
    assert [h.doc["id"] for h in store.search_records("bakan")] == [rec.id]
    assert [h.doc["id"] for h in store.search_records("cumhurbaşkanlığı")] == [rec.id]
    assert store.search_records("yok") == []


def test_search_filters_since_until_sources_only_alarms_min_score() -> None:
    store = InMemoryStore()
    t = now()
    old_alarm = make_record("Bakan görevden alındı", "...", published=t - timedelta(days=10), score=90, source="hurriyet")
    recent_alarm = make_record("Bakan istifa etti", "...", published=t - timedelta(hours=3), score=70, source="12punto")
    recent_plain = make_record("Bakan ziyaret gerçekleştirdi", "...", published=t - timedelta(hours=1), score=20)
    for rec in (old_alarm, recent_alarm, recent_plain):
        store.index_record(rec)

    ids = lambda hits: {h.doc["id"] for h in hits}  # noqa: E731
    assert ids(store.search_records("bakan")) == {old_alarm.id, recent_alarm.id, recent_plain.id}
    assert ids(store.search_records("bakan", since=t - timedelta(days=1))) == {recent_alarm.id, recent_plain.id}
    assert ids(store.search_records("bakan", until=t - timedelta(days=5))) == {old_alarm.id}
    assert ids(store.search_records("bakan", only_alarms=True)) == {old_alarm.id, recent_alarm.id}
    assert ids(store.search_records("bakan", sources=["12punto"])) == {recent_alarm.id}
    assert ids(store.search_records("bakan", min_score=80)) == {old_alarm.id}
    assert ids(store.search_records("bakan", only_alarms=True, since=t - timedelta(days=1), size=1)) == {recent_alarm.id}


def test_search_empty_query_returns_newest_first() -> None:
    store = InMemoryStore()
    a = make_record("Birinci", "x", published=now() - timedelta(hours=5))
    b = make_record("İkinci", "y", published=now() - timedelta(hours=1))
    store.index_record(a)
    store.index_record(b)
    assert [h.doc["id"] for h in store.search_records("")] == [b.id, a.id]
    assert [h.doc["id"] for h in store.search_records("   ", size=1)] == [b.id]


# ---------------------------------------------------------------------------------------------------------
# InMemoryStore: kayıt/alarm/rapor erişimi ve istatistik
# ---------------------------------------------------------------------------------------------------------


def test_index_get_recent_and_reports_roundtrip() -> None:
    store = InMemoryStore()
    t = now()
    plain = make_record("Sıradan haber", "...", published=t - timedelta(hours=4))
    alarm_rec = make_record("Cumhurbaşkanı kararname imzaladı", "...", published=t - timedelta(hours=1), score=85)
    store.index_record(plain)
    store.index_record(alarm_rec)

    assert store.get_record(plain.id)["title"] == "Sıradan haber"
    assert store.get_record("yok") is None
    # upsert: aynı id yeniden yazılınca kayıt sayısı artmaz
    plain.title = "Sıradan haber (güncellendi)"
    store.index_record(plain)
    assert len(store.records) == 2
    assert store.get_record(plain.id)["title"] == "Sıradan haber (güncellendi)"

    assert [d["id"] for d in store.recent_records()] == [alarm_rec.id, plain.id]
    assert [d["id"] for d in store.recent_records(only_alarms=True)] == [alarm_rec.id]
    assert [d["id"] for d in store.recent_records(since=t - timedelta(hours=2))] == [alarm_rec.id]

    event = AlarmEvent.from_record(alarm_rec)
    store.index_alarm(event)
    assert store.get_alarm(event.alarm_id)["record_id"] == alarm_rec.id
    assert store.get_alarm("yok") is None
    assert "record" not in store.get_alarm(event.alarm_id)
    assert [a["alarm_id"] for a in store.recent_alarms()] == [event.alarm_id]
    assert store.recent_alarms(since=t + timedelta(hours=1)) == []

    r1 = Report(report_id="r1", kind="periodic", window_start=t - timedelta(hours=24), window_end=t, generated_at=t - timedelta(minutes=5))
    r2 = Report(report_id="r2", kind="alarm_digest", window_start=t - timedelta(hours=1), window_end=t, generated_at=t)
    store.index_report(r1)
    store.index_report(r2)
    assert [r["report_id"] for r in store.list_reports()] == ["r2", "r1"]
    assert [r["report_id"] for r in store.list_reports(kind="periodic")] == ["r1"]
    assert store.health() is True
    store.refresh()


def test_stats_shape_and_values() -> None:
    store = InMemoryStore()
    base = datetime(2026, 10, 2, 10, 30, tzinfo=UTC)
    docs = [
        make_record("A", "...", published=base, score=90, keywords=["bakan", "fon"], source="hurriyet"),
        make_record("B", "...", published=base + timedelta(minutes=10), score=70, keywords=["bakan"], source="12punto", category="siyaset"),
        make_record("C", "...", published=base + timedelta(hours=1), score=10, keywords=["fon"], source="hurriyet"),
        make_record("D", "...", published=base + timedelta(hours=1, minutes=5), source="hurriyet", category=""),
        make_record("Eski", "...", published=base - timedelta(days=3), score=99, keywords=["bakan"]),
    ]
    for rec in docs:
        store.index_record(rec)

    stats = store.stats(base - timedelta(hours=1), base + timedelta(hours=2))
    assert set(stats) >= {"total", "alarms", "by_source", "by_keyword", "by_category", "avg_alarm_score", "by_hour", "top_alarms"}
    assert stats["total"] == 4
    assert stats["alarms"] == 2
    assert stats["by_source"] == {"hurriyet": 3, "12punto": 1}
    assert stats["by_keyword"] == {"bakan": 2, "fon": 2}
    assert stats["by_category"] == {"gundem": 2, "siyaset": 1}
    assert stats["avg_alarm_score"] == pytest.approx(80.0)
    assert stats["by_hour"] == [
        {"ts": "2026-10-02T10:00:00+00:00", "count": 2, "alarms": 2},
        {"ts": "2026-10-02T11:00:00+00:00", "count": 2, "alarms": 0},
    ]
    assert [a["title"] for a in stats["top_alarms"]] == ["A", "B"]
    assert set(stats["top_alarms"][0]) == {
        "id", "title", "content_url", "source", "alarm_score", "alarm_reason", "llm_summary", "published_at", "matched_keywords",
    }
    assert stats["since"].startswith("2026-10-02T09:30:00")

    empty = store.stats(base + timedelta(days=30), base + timedelta(days=31))
    assert empty["total"] == 0 and empty["alarms"] == 0 and empty["avg_alarm_score"] == 0.0
    assert empty["by_hour"] == [] and empty["top_alarms"] == []


def test_knn_search_in_memory() -> None:
    store = InMemoryStore()
    assert store.knn_search([1.0, 0.0], 5) == []
    t = now()
    a = make_record("A", "...", published=t - timedelta(hours=1))
    b = make_record("B", "...", published=t - timedelta(days=5))
    c = make_record("C", "...", published=t)
    store.index_record(a, embedding=[1.0, 0.0, 0.0])
    store.index_record(b, embedding=[0.9, 0.1, 0.0])
    store.index_record(c, embedding=[0.0, 1.0, 0.0])
    hits = store.knn_search([1.0, 0.0, 0.0], 2)
    assert [h.doc["id"] for h in hits] == [a.id, b.id]
    assert hits[0].score == pytest.approx(1.0)
    assert [h.doc["id"] for h in store.knn_search([1.0, 0.0, 0.0], 5, since=t - timedelta(days=1))] == [a.id, c.id]
    assert store.knn_search([], 5) == []


# ---------------------------------------------------------------------------------------------------------
# Saf kurucular: eşlemeler ve sorgular
# ---------------------------------------------------------------------------------------------------------


def test_build_article_mapping_has_turkish_analyzer_and_optional_embedding() -> None:
    plain = Settings(_env_file=None, ollama_embedding_model="")
    body = build_article_mapping(plain)
    analysis = body["settings"]["analysis"]
    assert analysis["analyzer"]["tr_text"]["tokenizer"] == "standard"
    assert analysis["analyzer"]["tr_text"]["filter"] == ["apostrophe", "turkish_lowercase", "turkish_stop", "turkish_stemmer"]
    assert analysis["filter"]["turkish_stop"] == {"type": "stop", "stopwords": "_turkish_"}
    assert analysis["filter"]["turkish_stemmer"] == {"type": "stemmer", "language": "turkish"}
    props = body["mappings"]["properties"]
    assert "embedding" not in props
    assert props["title"]["analyzer"] == "tr_text" and props["title"]["fields"]["keyword"]["type"] == "keyword"
    for name in ("subtitle", "content", "alarm_reason", "llm_summary"):
        assert props[name] == {"type": "text", "analyzer": "tr_text"}
    for name in ("id", "source", "category", "content_url", "matched_keywords", "stage", "alarm_id", "tags", "author", "content_hash"):
        assert props[name]["type"] == "keyword", name
    for name in ("published_at", "updated_at", "scraped_at", "processed_at", "alarmed_at", "@timestamp"):
        assert props[name]["type"] == "date", name
    assert props["alarm_score"]["type"] == "integer" and props["is_alarm"]["type"] == "boolean"
    llm = props["llm"]["properties"]
    assert llm["model"]["type"] == "keyword" and llm["raw"] == {"type": "text", "index": False}
    assert llm["summary"]["analyzer"] == "tr_text" and llm["scored_at"]["type"] == "date"

    with_vectors = Settings(_env_file=None, ollama_embedding_model="nomic-embed-text", embedding_dims=768)
    emb = build_article_mapping(with_vectors)["mappings"]["properties"]["embedding"]
    assert emb == {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}


def test_build_alarm_and_report_mappings() -> None:
    settings = Settings(_env_file=None)
    alarm_props = build_alarm_mapping(settings)["mappings"]["properties"]
    assert alarm_props["alarm_id"]["type"] == "keyword" and alarm_props["raised_at"]["type"] == "date"
    assert alarm_props["alarm_reason"]["analyzer"] == "tr_text" and alarm_props["content"]["analyzer"] == "tr_text"
    assert alarm_props["channels_notified"]["type"] == "keyword" and alarm_props["acknowledged"]["type"] == "boolean"
    report = build_report_mapping(settings)
    assert "tr_text" in report["settings"]["analysis"]["analyzer"]
    rprops = report["mappings"]["properties"]
    assert rprops["stats"] == {"type": "object", "enabled": False}
    assert rprops["narrative"]["analyzer"] == "tr_text" and rprops["kind"]["type"] == "keyword"


def test_build_search_query_function_score_with_filters() -> None:
    since = datetime(2026, 10, 1, tzinfo=UTC)
    until = datetime(2026, 10, 2, tzinfo=UTC)
    q = build_search_query("Özgür Özel", since=since, until=until, sources=["hurriyet"], only_alarms=True, min_score=60)
    fs = q["function_score"]
    assert fs["score_mode"] == "multiply" and fs["boost_mode"] == "multiply"
    assert fs["functions"] == [{"gauss": {"@timestamp": {"origin": "now", "scale": "3d", "offset": "12h", "decay": 0.5}}}]
    mm = fs["query"]["bool"]["must"][0]["multi_match"]
    assert mm["query"] == "Özgür Özel"
    assert mm["fields"] == ["title^3", "subtitle^2", "content", "llm_summary^2", "alarm_reason"]
    assert mm["type"] == "best_fields" and mm["operator"] == "or"
    assert mm["minimum_should_match"] == "2<60%" and mm["fuzziness"] == "AUTO"
    filters = fs["query"]["bool"]["filter"]
    assert {"range": {"@timestamp": {"gte": since.isoformat(), "lte": until.isoformat()}}} in filters
    assert {"terms": {"source": ["hurriyet"]}} in filters
    assert {"term": {"is_alarm": True}} in filters
    assert {"range": {"alarm_score": {"gte": 60}}} in filters

    bare = build_search_query("  ")["function_score"]["query"]["bool"]
    assert bare["must"] == [{"match_all": {}}] and bare["filter"] == []


def test_build_stats_aggs_shape() -> None:
    aggs = build_stats_aggs()
    assert aggs["by_source"]["terms"]["field"] == "source"
    assert aggs["by_keyword"]["terms"]["field"] == "matched_keywords"
    assert aggs["by_category"]["terms"]["field"] == "category"
    assert aggs["alarms"]["filter"] == {"term": {"is_alarm": True}}
    assert aggs["alarms"]["aggs"]["avg_alarm_score"] == {"avg": {"field": "alarm_score"}}
    top = aggs["alarms"]["aggs"]["top_alarms"]["top_hits"]
    assert top["size"] == 10 and top["sort"][0] == {"alarm_score": {"order": "desc"}}
    assert "llm_summary" in top["_source"]["includes"]
    assert aggs["by_hour"]["date_histogram"] == {"field": "@timestamp", "calendar_interval": "1h"}
    assert aggs["by_hour"]["aggs"]["alarms"]["filter"] == {"term": {"is_alarm": True}}


# ---------------------------------------------------------------------------------------------------------
# ElasticsearchStore: sahte istemci
# ---------------------------------------------------------------------------------------------------------


def _meta(status: int) -> ApiResponseMeta:
    return ApiResponseMeta(
        status=status, http_version="1.1", headers=HttpHeaders(), duration=0.0, node=NodeConfig("http", "localhost", 9200)
    )


class _StubIndices:
    def __init__(self, parent: _StubES) -> None:
        self.parent = parent
        self.existing: set[str] = set()

    def exists(self, *, index: str) -> bool:
        self.parent.calls.append(("exists", {"index": index}))
        return index in self.existing

    def create(self, *, index: str, settings: dict[str, Any], mappings: dict[str, Any]) -> dict[str, Any]:
        self.parent.calls.append(("create", {"index": index, "settings": settings, "mappings": mappings}))
        self.existing.add(index)
        return {"acknowledged": True}

    def refresh(self, *, index: str) -> dict[str, Any]:
        self.parent.calls.append(("refresh", {"index": index}))
        return {}


class _StubES:
    def __init__(self, *, fail_writes: bool = False, search_response: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_writes = fail_writes
        self.indices = _StubIndices(self)
        self.search_response = search_response or {"hits": {"total": {"value": 0}, "hits": []}}

    def index(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("index", kwargs))
        if self.fail_writes:
            raise ESConnectionError("bağlantı reddedildi")
        return {"result": "created"}

    def get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", kwargs))
        raise NotFoundError("not_found", meta=_meta(404), body={"found": False})

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", kwargs))
        return self.search_response

    def ping(self) -> bool:
        return True

    def of(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]


def _es_store(**kwargs: Any) -> tuple[ElasticsearchStore, _StubES]:
    stub = _StubES(**kwargs)
    settings = Settings(_env_file=None, ollama_embedding_model="nomic-embed-text", embedding_dims=3)
    return ElasticsearchStore(settings, client=stub), stub  # type: ignore[arg-type]


def test_es_ensure_indices_creates_each_index_once() -> None:
    store, stub = _es_store()
    store.ensure_indices()
    created = [kw["index"] for kw in stub.of("create")]
    assert created == ["news-articles", "news-alarms", "news-reports"]
    assert "tr_text" in stub.of("create")[0]["settings"]["analysis"]["analyzer"]
    assert stub.of("create")[0]["mappings"]["properties"]["embedding"]["dims"] == 3
    store.ensure_indices()
    assert len(stub.of("create")) == 3


def test_es_writes_use_ids_and_convert_connection_errors_to_retry() -> None:
    store, stub = _es_store()
    rec = make_record("Bakan açıklama yaptı", "...", score=80)
    store.index_record(rec, embedding=[0.1, 0.2, 0.3], refresh=True)
    call = stub.of("index")[0]
    assert call["index"] == "news-articles" and call["id"] == rec.id and call["refresh"] is True
    assert call["document"]["embedding"] == [0.1, 0.2, 0.3] and call["document"]["@timestamp"]

    event = AlarmEvent.from_record(rec)
    store.index_alarm(event)
    report = Report(report_id="rep-1", kind="periodic", window_start=now(), window_end=now())
    store.index_report(report)
    assert [(kw["index"], kw["id"]) for kw in stub.of("index")[1:]] == [("news-alarms", event.alarm_id), ("news-reports", "rep-1")]

    failing, _ = _es_store(fail_writes=True)
    with pytest.raises(Retry):
        failing.index_record(rec)
    with pytest.raises(Retry):
        failing.index_alarm(event)
    with pytest.raises(Retry):
        failing.index_report(report)


def test_es_embedding_dropped_when_embeddings_disabled() -> None:
    stub = _StubES()
    store = ElasticsearchStore(Settings(_env_file=None, ollama_embedding_model=""), client=stub)  # type: ignore[arg-type]
    store.index_record(make_record("X", "y"), embedding=[0.1, 0.2])
    assert "embedding" not in stub.of("index")[0]["document"]
    assert store.knn_search([0.1, 0.2], 5) == []
    assert stub.of("search") == []


def test_es_get_returns_none_on_not_found() -> None:
    store, stub = _es_store()
    assert store.get_record("yok") is None
    assert store.get_alarm("yok") is None
    assert stub.of("get")[0] == {"index": "news-articles", "id": "yok", "source_excludes": ["embedding"]}


def test_es_search_records_passes_function_score_and_parses_hits() -> None:
    response = {
        "hits": {
            "total": {"value": 1},
            "hits": [{"_id": "a", "_score": 2.5, "_source": {"id": "a", "title": "Bakan"}, "highlight": {"title": ["<em>Bakan</em>"]}}],
        }
    }
    store, stub = _es_store(search_response=response)
    since = now() - timedelta(days=1)
    hits = store.search_records("bakan", since=since, only_alarms=True, size=5, min_score=50)
    assert hits == [SearchHit(doc={"id": "a", "title": "Bakan"}, score=2.5, highlights={"title": ["<em>Bakan</em>"]})]
    call = stub.of("search")[0]
    assert call["index"] == "news-articles" and call["size"] == 5
    assert call["source_excludes"] == ["embedding"]
    assert call["sort"] == ["_score", {"@timestamp": {"order": "desc"}}]
    assert "function_score" in call["query"]
    assert {"term": {"is_alarm": True}} in call["query"]["function_score"]["query"]["bool"]["filter"]
    assert call["highlight"]["fields"]["content"]["fragment_size"] == 180


def test_es_recent_and_reports_sorting_and_filters() -> None:
    store, stub = _es_store()
    since = now()
    store.recent_records(since=since, size=7, only_alarms=True)
    store.recent_alarms(since=since, size=3)
    store.list_reports(kind="periodic", size=2)
    rec_call, alarm_call, rep_call = stub.of("search")
    assert rec_call["sort"] == [{"@timestamp": {"order": "desc"}}] and rec_call["size"] == 7
    assert {"term": {"is_alarm": True}} in rec_call["query"]["bool"]["filter"]
    assert alarm_call["index"] == "news-alarms" and alarm_call["sort"] == [{"raised_at": {"order": "desc"}}]
    assert alarm_call["query"]["bool"]["filter"] == [{"range": {"raised_at": {"gte": since.isoformat()}}}]
    assert rep_call["index"] == "news-reports" and rep_call["query"]["bool"]["filter"] == [{"term": {"kind": "periodic"}}]
    assert rep_call["sort"] == [{"generated_at": {"order": "desc"}}]


def test_es_knn_search_builds_knn_clause() -> None:
    store, stub = _es_store(search_response={"hits": {"hits": [{"_score": 0.9, "_source": {"id": "k"}}]}})
    since = now()
    hits = store.knn_search([0.1, 0.2, 0.3], 4, since=since)
    assert [h.doc["id"] for h in hits] == ["k"] and hits[0].score == pytest.approx(0.9)
    call = stub.of("search")[0]
    assert call["knn"] == {
        "field": "embedding",
        "query_vector": [0.1, 0.2, 0.3],
        "k": 4,
        "num_candidates": 50,
        "filter": {"range": {"@timestamp": {"gte": since.isoformat()}}},
    }
    assert call["size"] == 4 and call["source_excludes"] == ["embedding"]
    assert store.knn_search([0.1, 0.2], 4) == []  # boyut uyuşmazlığı → boş liste, sorgu yok
    assert len(stub.of("search")) == 1


def test_es_stats_parses_aggregations() -> None:
    response = {
        "hits": {"total": {"value": 5}, "hits": []},
        "aggregations": {
            "by_source": {"buckets": [{"key": "hurriyet", "doc_count": 3}, {"key": "12punto", "doc_count": 2}]},
            "by_keyword": {"buckets": [{"key": "bakan", "doc_count": 2}]},
            "by_category": {"buckets": [{"key": "gundem", "doc_count": 5}]},
            "alarms": {
                "doc_count": 2,
                "avg_alarm_score": {"value": 77.5},
                "top_alarms": {"hits": {"hits": [{"_source": {"id": "a", "title": "A", "alarm_score": 85}}]}},
            },
            "by_hour": {"buckets": [{"key": 1790000000000, "key_as_string": "x", "doc_count": 5, "alarms": {"doc_count": 2}}]},
        },
    }
    store, stub = _es_store(search_response=response)
    since, until = now() - timedelta(days=1), now()
    stats = store.stats(since, until)
    call = stub.of("search")[0]
    assert call["size"] == 0 and call["track_total_hits"] is True and "by_hour" in call["aggs"]
    assert call["query"]["bool"]["filter"][0]["range"]["@timestamp"] == {"gte": since.isoformat(), "lte": until.isoformat()}
    assert stats["total"] == 5 and stats["alarms"] == 2
    assert stats["by_source"] == {"hurriyet": 3, "12punto": 2} and stats["by_keyword"] == {"bakan": 2}
    assert stats["by_category"] == {"gundem": 5}
    assert stats["avg_alarm_score"] == 77.5
    assert stats["by_hour"] == [{"ts": datetime.fromtimestamp(1790000000, tz=UTC).isoformat(), "count": 5, "alarms": 2}]
    assert stats["top_alarms"] == [{"id": "a", "title": "A", "alarm_score": 85}]

    empty_store, _ = _es_store(search_response={"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}})
    empty = empty_store.stats(None)
    assert empty["total"] == 0 and empty["avg_alarm_score"] == 0.0 and empty["by_hour"] == [] and empty["top_alarms"] == []


def test_es_refresh_and_health() -> None:
    store, stub = _es_store()
    store.refresh()
    assert stub.of("refresh") == [{"index": "news-articles,news-alarms,news-reports"}]
    assert store.health() is True


def test_make_store_selects_backend() -> None:
    assert isinstance(make_store(Settings(_env_file=None), in_memory=True), InMemoryStore)
    assert isinstance(make_store(Settings(_env_file=None, elasticsearch_url="memory://")), InMemoryStore)
    es = make_store(Settings(_env_file=None, elasticsearch_url="http://localhost:9200"))
    assert isinstance(es, ElasticsearchStore)
