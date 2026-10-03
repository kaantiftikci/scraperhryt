"""Elasticsearch deposu ve testler/geliştirme için bellek içi eşdeğeri.

Üç indeks kullanılır (adları ``Settings``'ten gelir):

- ``news-articles``  : boru hattından geçen HER haber (``NewsRecord.to_es_document``), ``id`` ile upsert
- ``news-alarms``    : alarm katmanının yükselttiği olaylar (``AlarmEvent.to_es_document``), ``alarm_id`` ile upsert
- ``news-reports``   : raporlama katmanının ürettiği raporlar (``Report.to_es_document``), ``report_id`` ile upsert

Metin alanları Türkçe analizörle (``tr_text``: standard tokenizer + apostrophe + turkish_lowercase +
turkish_stop + turkish_stemmer) indekslenir; arama BM25 + yenilik (gauss) ağırlıklı ``function_score``
sorgusudur. ``ollama_embedding_model`` ayarlıysa ``embedding`` alanı ``dense_vector`` olarak eklenir ve
``knn_search`` kullanılabilir.

Yazma/okuma işlemlerinde Elasticsearch'e ulaşılamazsa (bağlantı hatası) ya da düğüm HTTP ile yanıt verip henüz
hazır değilse (429 ``es_rejected_execution``/devre kesici, 503 ``unavailable_shards``/``cluster_block``, 502/504)
``broker.Unavailable`` (``Retry`` alt tipi) fırlatılır; böylece tüketici mesajı gecikmeli olarak yeniden dener ve
kısa bir ES kesintisi ``rabbitmq_max_attempts`` bütçesini tüketip kayıtları ölü mektuba düşürmez. Diğer hatalar
(geçersiz istek vb.) olduğu gibi yükselir.

``ensure_indices`` var olan ``news-articles`` indeksinin ``embedding`` eşlemesini ayarlarla uzlaştırır: eşleme
yoksa eklenir (``put_mapping``); boyut/tür uyuşmuyorsa hata loglanır ve vektör indeksleme kapatılır, kayıtlar
vektörsüz yazılmaya devam eder (hiçbir kayıt kaybolmaz).
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from elasticsearch import ApiError as ESApiError
from elasticsearch import BadRequestError, Elasticsearch, NotFoundError
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import TransportError as ESTransportError

from .broker import Unavailable
from .config import Settings, get_settings
from .models import AlarmEvent, Feedback, NewsRecord, Report
from .textutil import tr_lower

log = logging.getLogger(__name__)

TR_ANALYZER = "tr_text"
FEEDBACK_INDEX = "news-feedback"  # alarm geri bildirimleri (doğru/yanlış pozitif)
FEEDBACK_LABELS = ("true_positive", "false_positive", "needs_context")
SEARCH_FIELDS = ["title^3", "subtitle^2", "content", "llm_summary^2", "alarm_reason"]
TOP_ALARM_FIELDS = [
    "id", "title", "content_url", "source", "alarm_score", "alarm_reason", "llm_summary", "published_at",
    "matched_keywords",
]
STATS_SOURCE_BUCKETS = 20   # stats(): by_source en fazla bu kadar kaynak
STATS_KEYWORD_BUCKETS = 50  # stats(): by_keyword / by_category en fazla bu kadar anahtar
STATS_TOP_ALARMS = 10       # stats(): top_alarms uzunluğu
RECENCY_SCALE_DAYS = 3.0  # bellek içi arama: skor × 1/(1 + yaş_gün/3); ES'te gauss(scale=3d)
# ES'in HTTP ile yanıt verdiği ama isteği karşılayamadığı geçici durumlar (ApiError, TransportError DEĞİL): yeniden
# denenmeli, ölü mektup olmamalı. 400 (mapper_parsing vb.) ve 404 gibi kalıcı hatalar kapsam dışıdır.
TRANSIENT_ES_STATUSES = frozenset({429, 502, 503, 504})
_MIN_TOKEN_LEN = 2
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_HIGHLIGHT = {
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "fields": {
        "title": {"number_of_fragments": 0},
        "subtitle": {"number_of_fragments": 0},
        "content": {"fragment_size": 180, "number_of_fragments": 2},
        "llm_summary": {"fragment_size": 180, "number_of_fragments": 1},
    },
}


@dataclass
class SearchHit:
    doc: dict[str, Any]
    score: float
    highlights: dict[str, list[str]] | None = None


class ArticleStore(Protocol):
    """Alarm, raporlama ve API katmanlarının kullandığı depo arayüzü (ES ve bellek içi uygulamalar)."""

    def ensure_indices(self) -> None: ...
    def index_record(
        self, record: NewsRecord, refresh: bool = False, embedding: Sequence[float] | None = None
    ) -> None: ...
    def index_alarm(self, event: AlarmEvent, refresh: bool = False) -> None: ...
    def index_report(self, report: Report, refresh: bool = False) -> None: ...
    def get_record(self, id: str) -> dict[str, Any] | None: ...
    def get_alarm(self, alarm_id: str) -> dict[str, Any] | None: ...
    def search_records(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        sources: Sequence[str] | None = None,
        only_alarms: bool = False,
        size: int = 20,
        min_score: int | None = None,
    ) -> list[SearchHit]: ...
    def recent_records(
        self, since: datetime | None = None, size: int = 50, only_alarms: bool = False
    ) -> list[dict[str, Any]]: ...
    def recent_alarms(self, since: datetime | None = None, size: int = 50) -> list[dict[str, Any]]: ...
    def stats(self, since: datetime | None, until: datetime | None = None) -> dict[str, Any]: ...
    def list_reports(self, kind: str | None = None, size: int = 20) -> list[dict[str, Any]]: ...
    def knn_search(
        self, vector: Sequence[float], k: int, since: datetime | None = None
    ) -> list[SearchHit]: ...
    def refresh(self) -> None: ...
    def health(self) -> bool: ...
    def index_feedback(self, feedback: Feedback, refresh: bool = False) -> None: ...
    def list_feedback(
        self, *, since: datetime | None = None, alarm_id: str | None = None, size: int = 200
    ) -> list[dict[str, Any]]: ...
    def feedback_stats(self, since: datetime | None = None) -> dict[str, Any]: ...
    def find_similar_alarms(
        self, *, title: str, embedding: Sequence[float] | None = None, since: datetime | None = None, size: int = 10
    ) -> list[SearchHit]: ...
    def iter_records(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        only_keyword_hits: bool = False,
        sources: Sequence[str] | None = None,
        batch_size: int = 50,
    ) -> Iterator[dict[str, Any]]: ...


# ---------------------------------------------------------------------------------------------------------
# Tarih yardımcıları
# ---------------------------------------------------------------------------------------------------------


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return _aware(dt).isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        log.debug("Tarih çözümlenemedi: %r", value)
        return None


def _doc_timestamp(doc: dict[str, Any]) -> datetime:
    for key in ("@timestamp", "published_at", "scraped_at", "raised_at", "generated_at"):
        dt = _parse_dt(doc.get(key))
        if dt is not None:
            return dt
    return datetime.fromtimestamp(0, tz=UTC)


def _range_filter(field: str, since: datetime | None, until: datetime | None) -> dict[str, Any] | None:
    bounds: dict[str, str] = {}
    if since is not None:
        bounds["gte"] = _iso(since)
    if until is not None:
        bounds["lte"] = _iso(until)
    return {"range": {field: bounds}} if bounds else None


# ---------------------------------------------------------------------------------------------------------
# İndeks eşlemeleri (saf fonksiyonlar; sunucu gerekmez)
# ---------------------------------------------------------------------------------------------------------

_KEYWORD: dict[str, Any] = {"type": "keyword"}
_DATE: dict[str, Any] = {"type": "date"}
_INT: dict[str, Any] = {"type": "integer"}
_BOOL: dict[str, Any] = {"type": "boolean"}


def _text() -> dict[str, Any]:
    return {"type": "text", "analyzer": TR_ANALYZER}


def _text_with_keyword() -> dict[str, Any]:
    return {"type": "text", "analyzer": TR_ANALYZER, "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}}


def build_index_settings() -> dict[str, Any]:
    """Tek düğümlü kurulum için indeks ayarları + Türkçe ``tr_text`` analizörü."""
    return {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "filter": {
                "turkish_lowercase": {"type": "lowercase", "language": "turkish"},
                "turkish_stop": {"type": "stop", "stopwords": "_turkish_"},
                "turkish_stemmer": {"type": "stemmer", "language": "turkish"},
            },
            "analyzer": {
                TR_ANALYZER: {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["apostrophe", "turkish_lowercase", "turkish_stop", "turkish_stemmer"],
                }
            },
        },
    }


def build_article_mapping(settings: Settings) -> dict[str, Any]:
    """``news-articles`` indeksi: ``NewsRecord.to_es_document`` alanları (+ isteğe bağlı ``embedding``)."""
    properties: dict[str, Any] = {
        "@timestamp": _DATE,
        "schema_version": _INT,
        "id": _KEYWORD,
        "source": _KEYWORD,
        "content_url": _KEYWORD,
        "title": _text_with_keyword(),
        "subtitle": _text(),
        "published_at": _DATE,
        "updated_at": _DATE,
        "content": _text(),
        "content_length": _INT,
        "category": _KEYWORD,
        "author": _KEYWORD,
        "image_url": {"type": "keyword", "index": False},
        "tags": _KEYWORD,
        "language": _KEYWORD,
        "scraped_at": _DATE,
        "content_hash": _KEYWORD,
        "stage": _KEYWORD,
        "matched_keywords": _KEYWORD,
        "alarm_score": _INT,
        "is_alarm": _BOOL,
        "alarm_reason": _text(),
        "llm_summary": _text(),
        "llm": {
            "type": "object",
            "properties": {
                "model": _KEYWORD,
                "alarm_score": _INT,
                "is_alarm": _BOOL,
                "reason": _text(),
                "summary": _text(),
                "topics": _KEYWORD,
                "entities": _KEYWORD,
                "scored_at": _DATE,
                "latency_ms": _INT,
                "attempts": _INT,
                "raw": {"type": "text", "index": False},
            },
        },
        "alarm_id": _KEYWORD,
        "alarmed_at": _DATE,
        "processed_at": _DATE,
    }
    if settings.ollama_embedding_model:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": int(settings.embedding_dims),
            "index": True,
            "similarity": "cosine",
        }
    return {"settings": build_index_settings(), "mappings": {"properties": properties}}


def build_alarm_mapping(settings: Settings) -> dict[str, Any]:
    """``news-alarms`` indeksi: ``AlarmEvent.to_es_document`` alanları."""
    _ = settings  # eşleme ayara bağlı değil; imza diğer eşleme kurucularıyla tutarlı
    properties: dict[str, Any] = {
        "@timestamp": _DATE,
        "alarm_id": _KEYWORD,
        "record_id": _KEYWORD,
        "source": _KEYWORD,
        "content_url": _KEYWORD,
        "title": _text_with_keyword(),
        "subtitle": _text(),
        "published_at": _DATE,
        "alarm_score": _INT,
        "alarm_reason": _text(),
        "llm_summary": _text(),
        "matched_keywords": _KEYWORD,
        "raised_at": _DATE,
        "channels_notified": _KEYWORD,
        "acknowledged": _BOOL,
        "content": _text(),
        "category": _KEYWORD,
    }
    return {"settings": build_index_settings(), "mappings": {"properties": properties}}


def build_report_mapping(settings: Settings) -> dict[str, Any]:
    """``news-reports`` indeksi: ``Report.to_es_document`` alanları (``stats``/``top_alarms`` serbest biçimli)."""
    _ = settings
    properties: dict[str, Any] = {
        "@timestamp": _DATE,
        "report_id": _KEYWORD,
        "kind": _KEYWORD,
        "window_start": _DATE,
        "window_end": _DATE,
        "generated_at": _DATE,
        "stats": {"type": "object", "enabled": False},
        "narrative": _text(),
        "top_alarms": {"type": "object", "enabled": False},
        "model": _KEYWORD,
    }
    return {"settings": build_index_settings(), "mappings": {"properties": properties}}


# ---------------------------------------------------------------------------------------------------------
# Sorgu kurucular (saf fonksiyonlar)
# ---------------------------------------------------------------------------------------------------------


def _article_filters(
    *,
    since: datetime | None,
    until: datetime | None,
    sources: Sequence[str] | None,
    only_alarms: bool,
    min_score: int | None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    ts = _range_filter("@timestamp", since, until)
    if ts is not None:
        filters.append(ts)
    if sources:
        filters.append({"terms": {"source": [s for s in sources if s]}})
    if only_alarms:
        filters.append({"term": {"is_alarm": True}})
    if min_score is not None:
        filters.append({"range": {"alarm_score": {"gte": int(min_score)}}})
    return filters


def build_search_query(
    query: str,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    sources: Sequence[str] | None = None,
    only_alarms: bool = False,
    min_score: int | None = None,
) -> dict[str, Any]:
    """Yenilik ağırlıklı BM25 sorgusu: ``function_score(bool(multi_match + filtreler), gauss(@timestamp))``.

    Boş sorgu metni ``match_all`` olur; sonuç o zaman yalnızca yeniliğe göre sıralanır.
    """
    text = (query or "").strip()
    if text:
        must: dict[str, Any] = {
            "multi_match": {
                "query": text,
                "fields": list(SEARCH_FIELDS),
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "2<60%",
                "fuzziness": "AUTO",
            }
        }
    else:
        must = {"match_all": {}}
    filters = _article_filters(since=since, until=until, sources=sources, only_alarms=only_alarms, min_score=min_score)
    return {
        "function_score": {
            "query": {"bool": {"must": [must], "filter": filters}},
            "functions": [
                {"gauss": {"@timestamp": {"origin": "now", "scale": "3d", "offset": "12h", "decay": 0.5}}}
            ],
            "score_mode": "multiply",
            "boost_mode": "multiply",
        }
    }


def build_stats_aggs() -> dict[str, Any]:
    """``stats()`` için toplulaştırmalar (``_parse_stats_response`` ile çözümlenir)."""
    return {
        "by_source": {"terms": {"field": "source", "size": STATS_SOURCE_BUCKETS}},
        "by_keyword": {"terms": {"field": "matched_keywords", "size": STATS_KEYWORD_BUCKETS}},
        "by_category": {"terms": {"field": "category", "size": STATS_KEYWORD_BUCKETS, "exclude": [""]}},
        "alarms": {
            "filter": {"term": {"is_alarm": True}},
            "aggs": {
                "avg_alarm_score": {"avg": {"field": "alarm_score"}},
                "top_alarms": {
                    "top_hits": {
                        "size": STATS_TOP_ALARMS,
                        "sort": [{"alarm_score": {"order": "desc"}}, {"@timestamp": {"order": "desc"}}],
                        "_source": {"includes": list(TOP_ALARM_FIELDS)},
                    }
                },
            },
        },
        "by_hour": {
            "date_histogram": {"field": "@timestamp", "calendar_interval": "1h", "min_doc_count": 1},
            "aggs": {"alarms": {"filter": {"term": {"is_alarm": True}}}},
        },
    }


def _parse_stats_response(
    res: Any, since: datetime | None, until: datetime | None
) -> dict[str, Any]:
    aggs = res.get("aggregations") or {}

    def buckets(name: str) -> dict[str, int]:
        return {str(b["key"]): int(b["doc_count"]) for b in (aggs.get(name) or {}).get("buckets", [])}

    alarms_agg = aggs.get("alarms") or {}
    avg_raw = (alarms_agg.get("avg_alarm_score") or {}).get("value")
    by_hour = [
        {
            "ts": datetime.fromtimestamp(int(b["key"]) / 1000, tz=UTC).isoformat(),
            "count": int(b["doc_count"]),
            "alarms": int((b.get("alarms") or {}).get("doc_count", 0)),
        }
        for b in (aggs.get("by_hour") or {}).get("buckets", [])
    ]
    top_hits = ((alarms_agg.get("top_alarms") or {}).get("hits") or {}).get("hits", [])
    total = (res.get("hits") or {}).get("total") or {}
    return {
        "since": _iso(since) if since else None,
        "until": _iso(until) if until else None,
        "total": int(total.get("value", 0) if isinstance(total, dict) else total),
        "alarms": int(alarms_agg.get("doc_count", 0)),
        "by_source": buckets("by_source"),
        "by_keyword": buckets("by_keyword"),
        "by_category": buckets("by_category"),
        "avg_alarm_score": round(float(avg_raw), 2) if avg_raw is not None else 0.0,
        "by_hour": by_hour,
        "top_alarms": [dict(h.get("_source") or {}) for h in top_hits],
    }


def _hits_to_search_hits(res: Any) -> list[SearchHit]:
    hits = (res.get("hits") or {}).get("hits", [])
    return [
        SearchHit(
            doc=dict(h.get("_source") or {}),
            score=float(h.get("_score") or 0.0),
            highlights=h.get("highlight"),
        )
        for h in hits
    ]


def _describe(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    return f"{type(exc).__name__}: {message or exc}"


# ---------------------------------------------------------------------------------------------------------
# Elasticsearch deposu
# ---------------------------------------------------------------------------------------------------------


def build_feedback_mapping() -> dict[str, Any]:
    """``news-feedback`` indeksi: ``Feedback.to_es_document`` alanları."""
    properties: dict[str, Any] = {
        "@timestamp": _DATE,
        "feedback_id": _KEYWORD,
        "alarm_id": _KEYWORD,
        "record_id": _KEYWORD,
        "label": _KEYWORD,
        "note": _text(),
        "user": _KEYWORD,
        "channel": _KEYWORD,
        "created_at": _DATE,
    }
    return {"settings": build_index_settings(), "mappings": {"dynamic": False, "properties": properties}}


def summarize_feedback(docs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Geri bildirim belgelerinden etiket sayıları ve kesinlik tahmini (tp / (tp + fp))."""
    counts = {label: 0 for label in FEEDBACK_LABELS}
    for doc in docs:
        label = str(doc.get("label", ""))
        if label in counts:
            counts[label] += 1
    judged = counts["true_positive"] + counts["false_positive"]
    precision = round(counts["true_positive"] / judged, 3) if judged else None
    return {"total": len(docs), **counts, "precision_estimate": precision}


def _alarm_similarity_filters(since: datetime | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [{"term": {"is_alarm": True}}]
    ts = _range_filter("@timestamp", since, None)
    if ts is not None:
        filters.append(ts)
    return filters


class ElasticsearchStore:
    """elasticsearch-py 8 üzerinden üç indeksi yöneten depo. ``client`` testlerde sahte istemci için verilebilir."""

    def __init__(self, settings: Settings, client: Elasticsearch | None = None) -> None:
        self.settings = settings
        self.index_articles = settings.es_index_articles
        self.index_alarms = settings.es_index_alarms
        self.index_reports = settings.es_index_reports
        self.embeddings_enabled = bool(settings.ollama_embedding_model)
        self.es = client or Elasticsearch(
            settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key or None,
            request_timeout=settings.es_request_timeout,
        )
        log.info(
            "Elasticsearch deposu: %s (indeksler: %s, %s, %s; embedding=%s)",
            settings.elasticsearch_url,
            self.index_articles,
            self.index_alarms,
            self.index_reports,
            "açık" if self.embeddings_enabled else "kapalı",
        )

    # --- yardımcılar ---
    def _write(self, what: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        try:
            return fn(**kwargs)
        except (ESConnectionError, ESTransportError) as exc:
            raise Unavailable(f"Elasticsearch erişilemiyor ({what}): {_describe(exc)}") from exc
        except ESApiError as exc:
            # elasticsearch-py 8: ApiError TransportError'ın alt tipi değildir; 429/502/503/504 (parça atama, devre
            # kesici, küme bloğu) düğüm ayağa kalkana dek sürer ve deneme bütçesini tüketmemelidir.
            if exc.status_code in TRANSIENT_ES_STATUSES:
                raise Unavailable(
                    f"Elasticsearch geçici olarak erişilemiyor ({what}, HTTP {exc.status_code}): {_describe(exc)}"
                ) from exc
            raise

    def _create_index(self, name: str, body: dict[str, Any]) -> bool:
        """İndeks yoksa oluşturur; bu çağrıda oluşturulduysa True, zaten varsa False döndürür."""
        if self._write("indices.exists", self.es.indices.exists, index=name):
            return False
        try:
            self._write(
                "indices.create", self.es.indices.create, index=name, settings=body["settings"], mappings=body["mappings"]
            )
        except BadRequestError as exc:
            if "resource_already_exists_exception" in f"{exc.message} {exc.body}":
                log.info("İndeks eşzamanlı olarak başka bir süreç tarafından oluşturuldu: %s", name)
                return False
            raise
        log.info("Elasticsearch indeksi oluşturuldu: %s", name)
        return True

    def _disable_embeddings(self, reason: str) -> None:
        log.error(
            "%s; vektör indeksleme KAPATILDI, kayıtlar vektörsüz yazılacak (kNN arama devre dışı). EMBEDDING_DIMS'i "
            "modele (%s) göre düzeltin ya da %s indeksini yeniden oluşturun",
            reason,
            self.settings.ollama_embedding_model,
            self.index_articles,
        )
        self.embeddings_enabled = False

    def _reconcile_embedding_mapping(self, expected: dict[str, Any]) -> None:
        """Var olan ``news-articles`` indeksinin ``embedding`` eşlemesini ayarlarla uzlaştırır.

        Eşleme yoksa (model indeks oluşturulduktan sonra açıldı) ``put_mapping`` ile eklenir; aksi halde ES alanı
        dinamik olarak ``float`` eşler ve her kNN sorgusu 400 ile sessizce başarısız olurdu. Tür/boyut uyuşmuyorsa
        (ör. EMBEDDING_DIMS değişti ama indeks yeniden oluşturulmadı) her belge 400 ``mapper_parsing_exception``
        alıp ölü mektuba düşerdi; bunun yerine vektör indeksleme kapatılır.
        """
        res = dict(self._write("indices.get_mapping", self.es.indices.get_mapping, index=self.index_articles))
        # yanıt anahtarı gerçek indeks adıdır; takma ad kullanılıyorsa ilk (tek) girdi alınır
        index_body = res.get(self.index_articles) or next(iter(res.values()), {})
        properties = ((index_body or {}).get("mappings") or {}).get("properties") or {}
        current = properties.get("embedding")
        if current is None:
            try:
                self._write(
                    "indices.put_mapping",
                    self.es.indices.put_mapping,
                    index=self.index_articles,
                    properties={"embedding": expected},
                )
            except BadRequestError as exc:
                self._disable_embeddings(
                    f"{self.index_articles} indeksine embedding eşlemesi eklenemedi: {_describe(exc)}"
                )
                return
            log.info(
                "%s indeksine embedding eşlemesi eklendi: dense_vector(dims=%d)", self.index_articles, expected["dims"]
            )
            return
        current_type = current.get("type")
        current_dims = current.get("dims")
        if current_type == expected["type"] and current_dims is not None and int(current_dims) == int(expected["dims"]):
            return
        self._disable_embeddings(
            f"{self.index_articles} indeksinde embedding eşlemesi {current_type}(dims={current_dims}), "
            f"ayarlar {expected['type']}(dims={expected['dims']}) bekliyor"
        )

    # --- ArticleStore ---
    def ensure_indices(self) -> None:
        article_mapping = build_article_mapping(self.settings)
        created = self._create_index(self.index_articles, article_mapping)
        if not created and self.embeddings_enabled:
            self._reconcile_embedding_mapping(article_mapping["mappings"]["properties"]["embedding"])
        self._create_index(self.index_alarms, build_alarm_mapping(self.settings))
        self._create_index(self.index_reports, build_report_mapping(self.settings))
        self._create_index(FEEDBACK_INDEX, build_feedback_mapping())

    def index_record(
        self, record: NewsRecord, refresh: bool = False, embedding: Sequence[float] | None = None
    ) -> None:
        doc = record.to_es_document()
        if embedding is not None:
            expected = int(self.settings.embedding_dims)
            if not self.embeddings_enabled:
                log.warning(
                    "embedding verildi ama vektör indeksleme kapalı (OLLAMA_EMBEDDING_MODEL boş ya da eşleme "
                    "uyuşmazlığı); kayıt %s vektörsüz yazılıyor",
                    record.id,
                )
            elif len(embedding) != expected:
                # dense_vector eşlemesi EMBEDDING_DIMS boyutundadır; uyuşmayan vektör ES'te 400 (mapper_parsing)
                # üretir ve kaydın tamamını kaybettirir. Vektörsüz yazmak her zaman tercih edilir.
                log.error(
                    "Embedding boyutu %d, eşleme %d (EMBEDDING_DIMS); kayıt %s vektörsüz yazılıyor. "
                    "EMBEDDING_DIMS'i modele göre düzeltip %s indeksini yeniden oluşturun",
                    len(embedding),
                    expected,
                    record.id,
                    self.index_articles,
                )
            else:
                doc["embedding"] = [float(x) for x in embedding]
        try:
            self._write("index_record", self.es.index, index=self.index_articles, id=record.id, document=doc, refresh=refresh)
        except BadRequestError as exc:
            # Son savunma hattı: canlı eşleme vektörü reddediyorsa (ensure_indices bu süreçte çalışmadı ya da eşleme
            # sonradan değişti) kayıt vektörsüz yazılır; belge ölü mektuba düşmez.
            if "embedding" not in doc or "mapper_parsing_exception" not in f"{exc.message} {exc.body}":
                raise
            self._disable_embeddings(f"{self.index_articles} eşlemesi embedding vektörünü reddetti: {_describe(exc)}")
            plain = {key: value for key, value in doc.items() if key != "embedding"}
            self._write(
                "index_record", self.es.index, index=self.index_articles, id=record.id, document=plain, refresh=refresh
            )

    def index_alarm(self, event: AlarmEvent, refresh: bool = False) -> None:
        self._write(
            "index_alarm",
            self.es.index,
            index=self.index_alarms,
            id=event.alarm_id,
            document=event.to_es_document(),
            refresh=refresh,
        )

    def index_report(self, report: Report, refresh: bool = False) -> None:
        self._write(
            "index_report",
            self.es.index,
            index=self.index_reports,
            id=report.report_id,
            document=report.to_es_document(),
            refresh=refresh,
        )

    def get_record(self, id: str) -> dict[str, Any] | None:
        try:
            res = self._write("get_record", self.es.get, index=self.index_articles, id=id, source_excludes=["embedding"])
        except NotFoundError:
            return None
        return dict(res["_source"])

    def get_alarm(self, alarm_id: str) -> dict[str, Any] | None:
        try:
            res = self._write("get_alarm", self.es.get, index=self.index_alarms, id=alarm_id)
        except NotFoundError:
            return None
        return dict(res["_source"])

    def search_records(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        sources: Sequence[str] | None = None,
        only_alarms: bool = False,
        size: int = 20,
        min_score: int | None = None,
    ) -> list[SearchHit]:
        body = build_search_query(
            query, since=since, until=until, sources=sources, only_alarms=only_alarms, min_score=min_score
        )
        res = self.es.search(
            index=self.index_articles,
            query=body,
            size=size,
            sort=["_score", {"@timestamp": {"order": "desc"}}],
            source_excludes=["embedding"],
            highlight=_HIGHLIGHT if (query or "").strip() else None,
        )
        return _hits_to_search_hits(res)

    def recent_records(
        self, since: datetime | None = None, size: int = 50, only_alarms: bool = False
    ) -> list[dict[str, Any]]:
        filters = _article_filters(since=since, until=None, sources=None, only_alarms=only_alarms, min_score=None)
        res = self.es.search(
            index=self.index_articles,
            query={"bool": {"filter": filters}} if filters else {"match_all": {}},
            size=size,
            sort=[{"@timestamp": {"order": "desc"}}],
            source_excludes=["embedding"],
        )
        return [h.doc for h in _hits_to_search_hits(res)]

    def recent_alarms(self, since: datetime | None = None, size: int = 50) -> list[dict[str, Any]]:
        ts = _range_filter("raised_at", since, None)
        res = self.es.search(
            index=self.index_alarms,
            query={"bool": {"filter": [ts]}} if ts else {"match_all": {}},
            size=size,
            sort=[{"raised_at": {"order": "desc"}}],
        )
        return [h.doc for h in _hits_to_search_hits(res)]

    def stats(self, since: datetime | None, until: datetime | None = None) -> dict[str, Any]:
        ts = _range_filter("@timestamp", since, until)
        res = self.es.search(
            index=self.index_articles,
            query={"bool": {"filter": [ts]}} if ts else {"match_all": {}},
            size=0,
            aggs=build_stats_aggs(),
            track_total_hits=True,
        )
        return _parse_stats_response(res, since, until)

    def list_reports(self, kind: str | None = None, size: int = 20) -> list[dict[str, Any]]:
        res = self.es.search(
            index=self.index_reports,
            query={"bool": {"filter": [{"term": {"kind": kind}}]}} if kind else {"match_all": {}},
            size=size,
            sort=[{"generated_at": {"order": "desc"}}],
        )
        return [h.doc for h in _hits_to_search_hits(res)]

    def knn_search(self, vector: Sequence[float], k: int, since: datetime | None = None) -> list[SearchHit]:
        if not self.embeddings_enabled or not vector or k <= 0:
            return []
        if len(vector) != int(self.settings.embedding_dims):
            log.warning(
                "kNN atlandı: embedding boyutu %d, beklenen %d (EMBEDDING_DIMS)", len(vector), self.settings.embedding_dims
            )
            return []
        knn: dict[str, Any] = {
            "field": "embedding",
            "query_vector": [float(x) for x in vector],
            "k": int(k),
            "num_candidates": max(50, 5 * int(k)),
        }
        ts = _range_filter("@timestamp", since, None)
        if ts is not None:
            knn["filter"] = ts
        res = self.es.search(index=self.index_articles, knn=knn, size=int(k), source_excludes=["embedding"])
        return _hits_to_search_hits(res)

    # --- geri bildirim ---
    def index_feedback(self, feedback: Feedback, refresh: bool = False) -> None:
        self._write(
            "index_feedback",
            self.es.index,
            index=FEEDBACK_INDEX,
            id=feedback.feedback_id,
            document=feedback.to_es_document(),
            refresh=refresh,
        )

    def list_feedback(
        self, *, since: datetime | None = None, alarm_id: str | None = None, size: int = 200
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        ts = _range_filter("created_at", since, None)
        if ts is not None:
            filters.append(ts)
        if alarm_id:
            filters.append({"term": {"alarm_id": alarm_id}})
        res = self.es.search(
            index=FEEDBACK_INDEX,
            query={"bool": {"filter": filters}} if filters else {"match_all": {}},
            size=size,
            sort=[{"created_at": {"order": "desc"}}],
        )
        return [h.doc for h in _hits_to_search_hits(res)]

    def feedback_stats(self, since: datetime | None = None) -> dict[str, Any]:
        return summarize_feedback(self.list_feedback(since=since, size=1000))

    # --- olay kümeleme / yeniden skorlama yardımcıları ---
    def find_similar_alarms(
        self, *, title: str, embedding: Sequence[float] | None = None, since: datetime | None = None, size: int = 10
    ) -> list[SearchHit]:
        """Pencere içindeki alarmlı kayıtlar arasında benzerleri bulur: embedding varsa kNN (skor=(1+cos)/2), yoksa başlık eşleşmesi."""
        filters = _alarm_similarity_filters(since)
        if embedding and self.embeddings_enabled and len(embedding) == int(self.settings.embedding_dims):
            knn = {
                "field": "embedding",
                "query_vector": [float(x) for x in embedding],
                "k": int(size),
                "num_candidates": max(50, 5 * int(size)),
                "filter": {"bool": {"filter": filters}},
            }
            res = self.es.search(index=self.index_articles, knn=knn, size=int(size), source_excludes=["embedding"])
            return _hits_to_search_hits(res)
        if not title.strip():
            return []
        res = self.es.search(
            index=self.index_articles,
            query={"bool": {"must": [{"match": {"title": {"query": title, "minimum_should_match": "50%"}}}], "filter": filters}},
            size=int(size),
            source_excludes=["embedding"],
        )
        return _hits_to_search_hits(res)

    def iter_records(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        only_keyword_hits: bool = False,
        sources: Sequence[str] | None = None,
        batch_size: int = 50,
    ) -> Iterator[dict[str, Any]]:
        """Kayıtları eskiden yeniye ``search_after`` ile sayfalayarak döndürür (yeniden skorlama için)."""
        filters: list[dict[str, Any]] = []
        ts = _range_filter("@timestamp", since, until)
        if ts is not None:
            filters.append(ts)
        if sources:
            filters.append({"terms": {"source": list(sources)}})
        if only_keyword_hits:
            filters.append({"exists": {"field": "matched_keywords"}})
        query = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        search_after: list[Any] | None = None
        size = max(1, int(batch_size))
        while True:
            kwargs: dict[str, Any] = {
                "index": self.index_articles,
                "query": query,
                "size": size,
                "sort": [{"@timestamp": {"order": "asc"}}, {"id": {"order": "asc"}}],
                "source_excludes": ["embedding"],
            }
            if search_after is not None:
                kwargs["search_after"] = search_after
            res = self.es.search(**kwargs)
            hits = res["hits"]["hits"]
            if not hits:
                return
            for hit in hits:
                yield dict(hit["_source"])
            search_after = hits[-1].get("sort")
            if len(hits) < size or search_after is None:
                return

    def refresh(self) -> None:
        self._write(
            "indices.refresh",
            self.es.indices.refresh,
            index=",".join((self.index_articles, self.index_alarms, self.index_reports)),
        )

    def health(self) -> bool:
        try:
            return bool(self.es.ping())
        except (ESConnectionError, ESTransportError) as exc:
            log.warning("Elasticsearch sağlık kontrolü başarısız: %s", _describe(exc))
            return False


# ---------------------------------------------------------------------------------------------------------
# Bellek içi depo (testler ve --in-memory geliştirme modu)
# ---------------------------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Türkçe küçük harfe çevrilmiş, en az 2 karakterlik kelime parçaları (sıra korunur, tekrarlar atılır)."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(tr_lower(text or "")):
        if len(tok) >= _MIN_TOKEN_LEN and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


_FIELD_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("title", 3.0),
    ("subtitle", 2.0),
    ("llm_summary", 2.0),
    ("content", 1.0),
    ("alarm_reason", 1.0),
)


def _token_matches(query_token: str, field_tokens: set[str]) -> bool:
    """Tam eşleşme ya da (4+ karakterli sorgu parçası için) Türkçe ek toleranslı önek eşleşmesi."""
    if query_token in field_tokens:
        return True
    if len(query_token) >= 4:
        return any(tok.startswith(query_token) for tok in field_tokens)
    return False


def _overlap_score(query_tokens: Sequence[str], doc: dict[str, Any]) -> float:
    score = 0.0
    for field_name, weight in _FIELD_WEIGHTS:
        field_tokens = set(tokenize(str(doc.get(field_name) or "")))
        if not field_tokens:
            continue
        score += weight * sum(1 for q in query_tokens if _token_matches(q, field_tokens))
    return score


def _recency_factor(ts: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / RECENCY_SCALE_DAYS)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryStore:
    """``ArticleStore``'un sözlük tabanlı uygulaması: basit kelime örtüşmesi × yenilik skoru; iş parçacığı güvenli."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.alarms: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.feedback: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def ensure_indices(self) -> None:
        return None

    def index_record(
        self, record: NewsRecord, refresh: bool = False, embedding: Sequence[float] | None = None
    ) -> None:
        with self._lock:
            self.records[record.id] = record.to_es_document()
            if embedding is not None:
                self.embeddings[record.id] = [float(x) for x in embedding]
            else:
                # ES'teki ``index`` tüm _source'u değiştirir: vektörsüz yeniden yazılan kayıt kNN'den düşer
                self.embeddings.pop(record.id, None)

    def index_alarm(self, event: AlarmEvent, refresh: bool = False) -> None:
        with self._lock:
            self.alarms[event.alarm_id] = event.to_es_document()

    def index_report(self, report: Report, refresh: bool = False) -> None:
        with self._lock:
            self.reports[report.report_id] = report.to_es_document()

    def get_record(self, id: str) -> dict[str, Any] | None:
        with self._lock:
            doc = self.records.get(id)
        return dict(doc) if doc is not None else None

    def get_alarm(self, alarm_id: str) -> dict[str, Any] | None:
        with self._lock:
            doc = self.alarms.get(alarm_id)
        return dict(doc) if doc is not None else None

    def _filtered_records(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        sources: Sequence[str] | None,
        only_alarms: bool,
        min_score: int | None,
    ) -> list[tuple[datetime, dict[str, Any]]]:
        wanted = {s for s in (sources or []) if s}
        since_dt = _aware(since) if since else None
        until_dt = _aware(until) if until else None
        out: list[tuple[datetime, dict[str, Any]]] = []
        with self._lock:
            docs = [dict(d) for d in self.records.values()]
        for doc in docs:
            ts = _doc_timestamp(doc)
            if since_dt is not None and ts < since_dt:
                continue
            if until_dt is not None and ts > until_dt:
                continue
            if wanted and doc.get("source") not in wanted:
                continue
            if only_alarms and not doc.get("is_alarm"):
                continue
            if min_score is not None and int(doc.get("alarm_score") or 0) < int(min_score):
                continue
            out.append((ts, doc))
        return out

    def search_records(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        sources: Sequence[str] | None = None,
        only_alarms: bool = False,
        size: int = 20,
        min_score: int | None = None,
    ) -> list[SearchHit]:
        now = datetime.now(UTC)
        query_tokens = tokenize(query)
        scored: list[tuple[float, datetime, str, dict[str, Any]]] = []
        for ts, doc in self._filtered_records(
            since=since, until=until, sources=sources, only_alarms=only_alarms, min_score=min_score
        ):
            base = _overlap_score(query_tokens, doc) if query_tokens else 1.0
            if base <= 0:
                continue
            scored.append((base * _recency_factor(ts, now), ts, str(doc.get("id", "")), doc))
        scored.sort(key=lambda item: (-item[0], -item[1].timestamp(), item[2]))
        return [SearchHit(doc=doc, score=round(score, 6)) for score, _ts, _id, doc in scored[: max(0, size)]]

    def recent_records(
        self, since: datetime | None = None, size: int = 50, only_alarms: bool = False
    ) -> list[dict[str, Any]]:
        items = self._filtered_records(since=since, until=None, sources=None, only_alarms=only_alarms, min_score=None)
        items.sort(key=lambda item: (-item[0].timestamp(), str(item[1].get("id", ""))))
        return [doc for _ts, doc in items[: max(0, size)]]

    def recent_alarms(self, since: datetime | None = None, size: int = 50) -> list[dict[str, Any]]:
        since_dt = _aware(since) if since else None
        with self._lock:
            docs = [dict(d) for d in self.alarms.values()]
        items = [(_parse_dt(d.get("raised_at")) or _doc_timestamp(d), d) for d in docs]
        if since_dt is not None:
            items = [(ts, d) for ts, d in items if ts >= since_dt]
        items.sort(key=lambda item: (-item[0].timestamp(), str(item[1].get("alarm_id", ""))))
        return [doc for _ts, doc in items[: max(0, size)]]

    def stats(self, since: datetime | None, until: datetime | None = None) -> dict[str, Any]:
        items = self._filtered_records(since=since, until=until, sources=None, only_alarms=False, min_score=None)
        by_source: Counter[str] = Counter()
        by_keyword: Counter[str] = Counter()
        by_category: Counter[str] = Counter()
        hours: dict[datetime, list[int]] = {}
        alarm_docs: list[tuple[datetime, dict[str, Any]]] = []
        for ts, doc in items:
            by_source[str(doc.get("source") or "")] += 1
            for kw in doc.get("matched_keywords") or []:
                by_keyword[str(kw)] += 1
            if doc.get("category"):
                by_category[str(doc["category"])] += 1
            hour = ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
            bucket = hours.setdefault(hour, [0, 0])
            bucket[0] += 1
            if doc.get("is_alarm"):
                bucket[1] += 1
                alarm_docs.append((ts, doc))
        alarm_docs.sort(key=lambda item: (-int(item[1].get("alarm_score") or 0), -item[0].timestamp()))
        avg = (
            round(sum(int(d.get("alarm_score") or 0) for _ts, d in alarm_docs) / len(alarm_docs), 2)
            if alarm_docs
            else 0.0
        )

        def ordered(counter: Counter[str], limit: int) -> dict[str, int]:
            # ES terms toplulaştırması gibi: en sık görülen ``limit`` anahtar, sayı azalan / anahtar artan sırada
            return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])

        return {
            "since": _iso(since) if since else None,
            "until": _iso(until) if until else None,
            "total": len(items),
            "alarms": len(alarm_docs),
            "by_source": ordered(by_source, STATS_SOURCE_BUCKETS),
            "by_keyword": ordered(by_keyword, STATS_KEYWORD_BUCKETS),
            "by_category": ordered(by_category, STATS_KEYWORD_BUCKETS),
            "avg_alarm_score": avg,
            "by_hour": [
                {"ts": hour.isoformat(), "count": counts[0], "alarms": counts[1]}
                for hour, counts in sorted(hours.items())
            ],
            # ES ``_source.includes`` gibi: belgede olmayan alan None olarak eklenmez
            "top_alarms": [
                {k: d[k] for k in TOP_ALARM_FIELDS if k in d} for _ts, d in alarm_docs[:STATS_TOP_ALARMS]
            ],
        }

    def list_reports(self, kind: str | None = None, size: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            docs = [dict(d) for d in self.reports.values() if not kind or d.get("kind") == kind]
        docs.sort(key=lambda d: (-_doc_timestamp(d).timestamp(), str(d.get("report_id", ""))))
        return docs[: max(0, size)]

    def knn_search(self, vector: Sequence[float], k: int, since: datetime | None = None) -> list[SearchHit]:
        if not vector or k <= 0:
            return []
        since_dt = _aware(since) if since else None
        with self._lock:
            pairs = [(rid, list(vec), dict(self.records[rid])) for rid, vec in self.embeddings.items() if rid in self.records]
        scored: list[tuple[float, datetime, str, dict[str, Any]]] = []
        for rid, vec, doc in pairs:
            ts = _doc_timestamp(doc)
            if since_dt is not None and ts < since_dt:
                continue
            scored.append((_cosine(vector, vec), ts, rid, doc))
        scored.sort(key=lambda item: (-item[0], -item[1].timestamp(), item[2]))
        return [SearchHit(doc=doc, score=round(sim, 6)) for sim, _ts, _rid, doc in scored[: int(k)]]

    # --- geri bildirim ---
    def index_feedback(self, feedback: Feedback, refresh: bool = False) -> None:
        with self._lock:
            self.feedback[feedback.feedback_id] = feedback.to_es_document()

    def list_feedback(
        self, *, since: datetime | None = None, alarm_id: str | None = None, size: int = 200
    ) -> list[dict[str, Any]]:
        since_dt = _aware(since) if since else None
        with self._lock:
            docs = [dict(d) for d in self.feedback.values()]
        if alarm_id:
            docs = [d for d in docs if d.get("alarm_id") == alarm_id]
        if since_dt is not None:
            docs = [d for d in docs if (_parse_dt(d.get("created_at")) or _doc_timestamp(d)) >= since_dt]
        docs.sort(key=lambda d: str(d.get("created_at", "")), reverse=True)
        return docs[: max(0, size)]

    def feedback_stats(self, since: datetime | None = None) -> dict[str, Any]:
        return summarize_feedback(self.list_feedback(since=since, size=1000))

    # --- olay kümeleme / yeniden skorlama yardımcıları ---
    def find_similar_alarms(
        self, *, title: str, embedding: Sequence[float] | None = None, since: datetime | None = None, size: int = 10
    ) -> list[SearchHit]:
        since_dt = _aware(since) if since else None
        query_tokens = set(tokenize(title))
        with self._lock:
            docs = [dict(d) for d in self.records.values() if d.get("is_alarm")]
            vectors = dict(self.embeddings)
        hits: list[SearchHit] = []
        for doc in docs:
            if since_dt is not None and _doc_timestamp(doc) < since_dt:
                continue
            vec = vectors.get(str(doc.get("id", "")))
            if embedding and vec is not None and len(vec) == len(embedding):
                score = (1.0 + _cosine(embedding, vec)) / 2.0  # ES kNN kosinüs skoru ile aynı ölçek
            else:
                doc_tokens = set(tokenize(str(doc.get("title", ""))))
                union = query_tokens | doc_tokens
                score = len(query_tokens & doc_tokens) / len(union) if union else 0.0
            if score > 0:
                hits.append(SearchHit(doc=doc, score=score))
        hits.sort(key=lambda h: (-h.score, str(h.doc.get("id", ""))))
        return hits[: max(0, size)]

    def iter_records(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        only_keyword_hits: bool = False,
        sources: Sequence[str] | None = None,
        batch_size: int = 50,
    ) -> Iterator[dict[str, Any]]:
        since_dt = _aware(since) if since else None
        until_dt = _aware(until) if until else None
        with self._lock:
            docs = [dict(d) for d in self.records.values()]
        selected = []
        for doc in docs:
            ts = _doc_timestamp(doc)
            if since_dt is not None and ts < since_dt:
                continue
            if until_dt is not None and ts > until_dt:
                continue
            if sources and doc.get("source") not in set(sources):
                continue
            if only_keyword_hits and not doc.get("matched_keywords"):
                continue
            selected.append((ts, str(doc.get("id", "")), doc))
        selected.sort(key=lambda item: (item[0], item[1]))
        yield from (doc for _ts, _id, doc in selected)

    def refresh(self) -> None:
        return None

    def health(self) -> bool:
        return True


def make_store(settings: Settings | None = None, *, in_memory: bool = False) -> ArticleStore:
    """``make_broker`` ile simetrik kurucu: ``in_memory`` ya da ``ELASTICSEARCH_URL=memory://`` ise bellek içi depo."""
    settings = settings or get_settings()
    if in_memory or settings.elasticsearch_url in ("memory://", "inmemory://"):
        return InMemoryStore()
    return ElasticsearchStore(settings)
