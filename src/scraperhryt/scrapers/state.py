"""Görülen haberlerin SQLite kaydı (``SeenStore``): aynı haberi yeniden yayınlamamak, güncellenenleri yakalamak."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from ..models import NewsRecord, utcnow

log = logging.getLogger(__name__)

SeenStatus = Literal["new", "updated", "unchanged"]

_PUBLISHED_TOLERANCE = timedelta(seconds=60)  # saniye düzeyindeki yuvarlama farkları güncelleme sayılmaz

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    content_hash  TEXT NOT NULL DEFAULT '',
    published_at  TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    times_seen    INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_seen_last_seen ON seen(last_seen);
"""


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=utcnow().tzinfo)


class SeenStore:
    """Thread güvenli SQLite (WAL) tabanlı "görüldü" deposu.

    ``status(id, content_hash)`` → ``"new"`` (hiç görülmedi), ``"updated"`` (aynı id, farklı içerik özeti),
    ``"unchanged"``. ``seen_recently`` ise sayfayı hiç çekmeden atlamak için kullanılır: kayıt son
    ``recent_hours`` içinde işaretlenmiş ve (verildiyse) RSS yayın tarihi değişmemişse True döner.
    """

    def __init__(self, path: str | os.PathLike[str], *, recent_hours: float = 6.0) -> None:
        self.path = Path(path)
        self.recent_hours = float(recent_hours)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
        log.debug("SeenStore açıldı: %s", self.path)

    # --- sorgular ---
    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM seen WHERE id = ?", (record_id,)).fetchone()
        return dict(row) if row is not None else None

    def status(self, record_id: str, content_hash: str) -> SeenStatus:
        row = self.get(record_id)
        if row is None:
            return "new"
        return "unchanged" if row["content_hash"] == (content_hash or "") else "updated"

    def seen_recently(
        self,
        record_id: str,
        *,
        published_at: datetime | None = None,
        within_hours: float | None = None,
    ) -> bool:
        """Kayıt son ``within_hours`` (varsayılan ``recent_hours``) içinde işaretlendi mi?

        ``published_at`` (beslemedeki pubDate/modified damgası) verilmiş ve depodaki damgadan farklıysa haber
        güncellenmiş sayılır ve False döner ki sayfa yeniden çekilsin.
        """
        hours = self.recent_hours if within_hours is None else float(within_hours)
        if hours <= 0:
            return False
        row = self.get(record_id)
        if row is None:
            return False
        last_seen = _parse_iso(row["last_seen"])
        if last_seen is None or utcnow() - last_seen > timedelta(hours=hours):
            return False
        if published_at is not None and row["published_at"]:
            stored = _parse_iso(row["published_at"])
            if stored is not None and abs(stored - published_at) > _PUBLISHED_TOLERANCE:
                return False
        return True

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM seen").fetchone()
        return int(row["n"]) if row is not None else 0

    def seen_urls(self, limit: int = 100) -> list[str]:
        """En son görülenden başlayarak URL listesi."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT url FROM seen ORDER BY last_seen DESC, rowid DESC LIMIT ?", (max(0, int(limit)),)
            ).fetchall()
        return [str(row["url"]) for row in rows]

    # --- yazma ---
    def mark(self, record: NewsRecord, *, published_at: datetime | None = None) -> None:
        """Kaydı görüldü olarak işaretler (ilk kez ekler ya da içerik özeti/son görülme bilgisini günceller).

        ``published_at`` verilirse (beslemedeki pubDate/modified damgası) kaydın kendi tarihi yerine o saklanır;
        ``seen_recently`` bir sonraki keşifte aynı besleme damgasıyla karşılaştırır.
        """
        now = utcnow().isoformat()
        stamp = published_at or record.published_at
        published = stamp.isoformat() if stamp is not None else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO seen (id, url, content_hash, published_at, first_seen, last_seen, times_seen)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    url = excluded.url,
                    content_hash = excluded.content_hash,
                    published_at = COALESCE(excluded.published_at, seen.published_at),
                    last_seen = excluded.last_seen,
                    times_seen = seen.times_seen + 1
                """,
                (record.id, record.content_url, record.content_hash or "", published, now, now),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SeenStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
