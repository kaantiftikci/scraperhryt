"""Kalıcı kayıt katmanı: SQL veritabanı (PostgreSQL; testlerde SQLite) asıl kaynak, Elasticsearch arama indeksi.

Elasticsearch bir veritabanı değil, arama motorudur: eşleme değişikliği, küme sorunu ya da indeks silinmesi verinin
kaybı demek olmamalı. Bu yüzden ``DATABASE_URL`` ayarlıysa her yazma önce SQL'e (tek doğruluk kaynağı), sonra ES'e
gider. **Tüm arama işlemleri Elasticsearch'ten yapılır**; SQL yalnızca kimlikle doğrudan okuma (``get_record``,
``get_alarm``) ve ES'i baştan kurma (``scraperhryt reindex``) için okunur.

Tablolar her belgeyi ES'e giden JSON'la birlikte (``doc``) ve sık süzülen alanları ayrı sütunlarda tutar; embedding
vektörü de saklanır ki ES yeniden kurulurken haberler yeniden vektörleştirilmesin.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine

from .config import Settings
from .models import AlarmEvent, Feedback, NewsRecord, Report
from .store import ArticleStore, SearchHit

log = logging.getLogger(__name__)

_JSON = JSON().with_variant(postgresql.JSONB(), "postgresql")
metadata = MetaData()

articles = Table(
    "news_articles",
    metadata,
    Column("id", String(255), primary_key=True),
    Column("source", String(32), index=True),
    Column("content_url", Text, nullable=False),
    Column("title", Text, nullable=False, default=""),
    Column("published_at", DateTime(timezone=True), index=True),
    Column("alarm_score", Integer, nullable=False, default=0),
    Column("is_alarm", Boolean, nullable=False, default=False, index=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("doc", _JSON, nullable=False),
    Column("embedding", _JSON),
)
alarms = Table(
    "news_alarms",
    metadata,
    Column("alarm_id", String(255), primary_key=True),
    Column("record_id", String(255), index=True),
    Column("raised_at", DateTime(timezone=True), index=True),
    Column("alarm_score", Integer, nullable=False, default=0),
    Column("doc", _JSON, nullable=False),
)
reports = Table(
    "news_reports",
    metadata,
    Column("report_id", String(255), primary_key=True),
    Column("kind", String(32), index=True),
    Column("generated_at", DateTime(timezone=True), index=True),
    Column("doc", _JSON, nullable=False),
)
feedback = Table(
    "news_feedback",
    metadata,
    Column("feedback_id", String(255), primary_key=True),
    Column("alarm_id", String(255), index=True),
    Column("label", String(32)),
    Column("created_at", DateTime(timezone=True), index=True),
    Column("doc", _JSON, nullable=False),
)


def _jsonable(doc: dict[str, Any]) -> dict[str, Any]:
    """ES belgesi zaten ``model_dump(mode="json")``; yine de tarih vb. kalırsa metne çevrilir."""
    return json.loads(json.dumps(doc, ensure_ascii=False, default=str))


class SqlRecordStore:
    """SQLAlchemy Core ile PostgreSQL/SQLite üzerinde idempotent (upsert) kayıt deposu."""

    def __init__(self, url: str, *, engine: Engine | None = None) -> None:
        self.url = url
        self.engine = engine or create_engine(url, pool_pre_ping=True, future=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> SqlRecordStore:
        return cls(settings.database_url)

    def ensure_schema(self) -> None:
        metadata.create_all(self.engine)

    def health(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # sağlık sondası hiçbir zaman istisna fırlatmaz
            log.warning("SQL veritabanına erişilemiyor: %s", exc)
            return False

    def close(self) -> None:
        self.engine.dispose()

    # --- yazma (upsert) ---
    def _upsert(self, table: Table, key: str, values: dict[str, Any]) -> None:
        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            stmt = postgresql.insert(table).values(**values)
        elif dialect == "sqlite":
            stmt = sqlite.insert(table).values(**values)
        else:  # pragma: no cover - desteklenmeyen sürücüde basit sil + ekle
            with self.engine.begin() as conn:
                conn.execute(table.delete().where(table.c[key] == values[key]))
                conn.execute(table.insert().values(**values))
            return
        updates = {name: stmt.excluded[name] for name in values if name != key}
        stmt = stmt.on_conflict_do_update(index_elements=[table.c[key]], set_=updates)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def save_record(self, record: NewsRecord, embedding: Sequence[float] | None = None) -> None:
        values: dict[str, Any] = {
            "id": record.id,
            "source": record.source,
            "content_url": record.content_url,
            "title": record.title,
            "published_at": record.published_at,
            "alarm_score": int(record.alarm_score),
            "is_alarm": bool(record.is_alarm),
            "updated_at": record.scraped_at,
            "doc": _jsonable(record.to_es_document()),
        }
        if embedding is not None:
            values["embedding"] = [float(x) for x in embedding]
        self._upsert(articles, "id", values)

    def save_alarm(self, event: AlarmEvent) -> None:
        self._upsert(
            alarms,
            "alarm_id",
            {
                "alarm_id": event.alarm_id,
                "record_id": event.record_id,
                "raised_at": event.raised_at,
                "alarm_score": int(event.alarm_score),
                "doc": _jsonable(event.to_es_document()),
            },
        )

    def save_report(self, report: Report) -> None:
        self._upsert(
            reports,
            "report_id",
            {
                "report_id": report.report_id,
                "kind": str(report.kind),
                "generated_at": report.generated_at,
                "doc": _jsonable(report.to_es_document()),
            },
        )

    def save_feedback(self, item: Feedback) -> None:
        self._upsert(
            feedback,
            "feedback_id",
            {
                "feedback_id": item.feedback_id,
                "alarm_id": item.alarm_id,
                "label": str(item.label),
                "created_at": item.created_at,
                "doc": _jsonable(item.to_es_document()),
            },
        )

    # --- kimlikle okuma ---
    def _get(self, table: Table, key: str, value: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(table.c.doc).where(table.c[key] == value)).first()
        return dict(row[0]) if row else None

    def get_record(self, id: str) -> dict[str, Any] | None:
        return self._get(articles, "id", id)

    def get_alarm(self, alarm_id: str) -> dict[str, Any] | None:
        return self._get(alarms, "alarm_id", alarm_id)

    def counts(self) -> dict[str, int]:
        from sqlalchemy import func

        out: dict[str, int] = {}
        with self.engine.connect() as conn:
            for name, table in (("articles", articles), ("alarms", alarms), ("reports", reports), ("feedback", feedback)):
                out[name] = int(conn.execute(select(func.count()).select_from(table)).scalar_one())
        return out

    # --- ES'i baştan kurmak için toplu okuma ---
    def iter_rows(self, name: str, batch_size: int = 500) -> Iterator[dict[str, Any]]:
        table = {"articles": articles, "alarms": alarms, "reports": reports, "feedback": feedback}[name]
        key = table.primary_key.columns.values()[0]
        cols = [table.c.doc] + ([table.c.embedding] if name == "articles" else [])
        last: str | None = None
        while True:
            stmt = select(key, *cols).order_by(key).limit(batch_size)
            if last is not None:
                stmt = stmt.where(key > last)
            with self.engine.connect() as conn:
                rows = conn.execute(stmt).all()
            if not rows:
                return
            for row in rows:
                item = {"doc": dict(row[1])}
                if name == "articles":
                    item["embedding"] = row[2]
                yield item
            last = rows[-1][0]
            if len(rows) < batch_size:
                return


class RecordStore:
    """``ArticleStore`` sarmalayıcısı: yazmalar önce SQL'e (asıl kayıt), sonra arama indeksine; aramalar yalnızca ES'ten.

    SQL yazması başarısız olursa istisna yükselir ve mesaj kuyrukta yeniden denenir; ES yazması başarısız olursa da
    öyle (her iki yazma da upsert olduğundan tekrar güvenlidir).
    """

    def __init__(self, search: ArticleStore, sql: SqlRecordStore) -> None:
        self.search = search
        self.sql = sql

    def __getattr__(self, name: str) -> Any:
        # Arama, listeleme, istatistik, kNN, benzer alarm, iter_records... hepsi Elasticsearch'ten.
        return getattr(self.search, name)

    def ensure_indices(self) -> None:
        self.sql.ensure_schema()
        self.search.ensure_indices()

    def health(self) -> bool:
        return bool(self.search.health()) and self.sql.health()

    def index_record(self, record: NewsRecord, refresh: bool = False, embedding: Sequence[float] | None = None) -> None:
        self.sql.save_record(record, embedding)
        self.search.index_record(record, refresh=refresh, embedding=embedding)

    def index_alarm(self, event: AlarmEvent, refresh: bool = False) -> None:
        self.sql.save_alarm(event)
        self.search.index_alarm(event, refresh=refresh)

    def index_report(self, report: Report, refresh: bool = False) -> None:
        self.sql.save_report(report)
        self.search.index_report(report, refresh=refresh)

    def index_feedback(self, feedback: Feedback, refresh: bool = False) -> None:
        self.sql.save_feedback(feedback)
        self.search.index_feedback(feedback, refresh=refresh)

    def get_record(self, id: str) -> dict[str, Any] | None:
        return self.sql.get_record(id) or self.search.get_record(id)

    def get_alarm(self, alarm_id: str) -> dict[str, Any] | None:
        return self.sql.get_alarm(alarm_id) or self.search.get_alarm(alarm_id)


def reindex_from_sql(sql: SqlRecordStore, search: ArticleStore, *, batch_size: int = 500) -> dict[str, int]:
    """Arama indeksini SQL kayıtlarından yeniden kurar (ES silindi, eşleme değişti, yeni küme...)."""
    search.ensure_indices()
    done = {"articles": 0, "alarms": 0, "reports": 0, "feedback": 0}
    for item in sql.iter_rows("articles", batch_size):
        search.index_record(NewsRecord.model_validate(_strip_es_fields(item["doc"])), embedding=item.get("embedding"))
        done["articles"] += 1
    for item in sql.iter_rows("alarms", batch_size):
        doc = item["doc"]
        record_doc = sql.get_record(str(doc.get("record_id") or "")) if doc.get("record_id") else None
        event = _alarm_from_doc(doc, record_doc)
        if event is not None:
            search.index_alarm(event)
            done["alarms"] += 1
    for item in sql.iter_rows("reports", batch_size):
        search.index_report(Report.model_validate(_strip_es_fields(item["doc"])))
        done["reports"] += 1
    for item in sql.iter_rows("feedback", batch_size):
        search.index_feedback(Feedback.model_validate(_strip_es_fields(item["doc"])))
        done["feedback"] += 1
    search.refresh()
    return done


def _strip_es_fields(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if not k.startswith("@") and k not in {"content_length", "embedding"}}


def _alarm_from_doc(doc: dict[str, Any], record_doc: dict[str, Any] | None) -> AlarmEvent | None:
    if record_doc is None:
        log.warning("Alarm %s için haber kaydı SQL'de yok; arama indeksine yazılmadı", doc.get("alarm_id"))
        return None
    fields = set(AlarmEvent.model_fields) - {"record"}
    data = {k: v for k, v in doc.items() if k in fields}
    data["record"] = NewsRecord.model_validate(_strip_es_fields(record_doc))
    return AlarmEvent.model_validate(data)


def make_record_store(settings: Settings, search: ArticleStore) -> ArticleStore:
    """``DATABASE_URL`` boşsa arama deposunu olduğu gibi döndürür (yalnızca ES)."""
    if not settings.database_url:
        return search
    return RecordStore(search, SqlRecordStore.from_settings(settings))  # type: ignore[return-value]


__all__ = ["RecordStore", "SearchHit", "SqlRecordStore", "make_record_store", "reindex_from_sql"]
