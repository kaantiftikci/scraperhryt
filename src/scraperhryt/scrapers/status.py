"""Kazıyıcı durumu: son tur, sonraki tur, bekleyen (henüz çekilemeyen) haberler ve uyarı metni.

API (``GET /scraper/status``) ve arayüz bu özeti kullanır. Veriler kazıyıcının SQLite durum dosyasından okunur;
kazıyıcı ayrı süreçte çalışsa da (docker'da paylaşılan ``data`` birimi) aynı dosyayı görür.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models import utcnow
from .state import SeenStore


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def scraper_status(settings: Settings, *, now: datetime | None = None, pending_limit: int = 50) -> dict[str, Any]:
    now = now or utcnow()
    interval = max(1, int(settings.scrape_interval_seconds))
    path = Path(settings.state_db_path)
    if not path.is_file():
        return {
            "available": False,
            "warning": "Kazıyıcı henüz hiç çalışmadı: durum dosyası yok. `scraperhryt scrape` veya `run-all` başlatın.",
            "stale": True,
            "pending_count": 0,
            "pending": [],
            "last_run": None,
            "next_run_at": None,
            "interval_seconds": interval,
            "recent_runs": [],
            "running": False,
            "progress": None,
        }
    store = SeenStore(path)
    try:
        last = store.last_run()
        pending = store.list_pending(limit=pending_limit)
        pending_count = store.pending_count()
        recent = store.recent_runs(limit=12)
        next_run = _dt(store.get_meta("next_run_at"))
        interval = int(store.get_meta("interval_seconds") or interval)
        raw_state = store.get_meta("run_state")
    finally:
        store.close()
    progress = _progress(raw_state, now)
    finished = _dt(last["finished_at"]) if last else None
    age = (now - finished) if finished else None
    running = progress is not None
    stale = not running and (finished is None or age > timedelta(seconds=2 * interval + 60))
    warnings: list[str] = []
    if finished is None and running:
        warnings.append("İlk tarama sürüyor; haberler çekildikçe arama sonuçlarına düşecek.")
    elif finished is None:
        warnings.append("Kazıyıcı henüz bir tur tamamlamadı.")
    elif stale:
        warnings.append(f"Son tarama {_age_text(age)} önce; beklenen aralık {interval // 60} dk. Kazıyıcı çalışmıyor olabilir.")
    if pending_count:
        warnings.append(
            f"{pending_count} haber henüz çekilemedi (hata veya tur bütçesi); "
            + (f"sonraki turda ({next_run.astimezone().strftime('%H:%M')}) yeniden denenecek." if next_run else "bir sonraki turda yeniden denenecek.")
        )
    if last and last.get("errors"):
        warnings.append(f"Son turda {last['errors']} çekme hatası oluştu.")
    return {
        "available": True,
        "stale": stale,
        "warning": " ".join(warnings),
        "last_run": last,
        "last_run_age_seconds": int(age.total_seconds()) if age else None,
        "next_run_at": next_run.isoformat() if next_run else None,
        "interval_seconds": interval,
        "pending_count": pending_count,
        "pending": pending,
        "recent_runs": recent,
        "running": running,
        "progress": progress,
    }


_PROGRESS_STALE = timedelta(minutes=5)  # bu süredir güncellenmeyen "sürüyor" kaydı yarıda kalmış sayılır


def _progress(raw: str | None, now: datetime) -> dict[str, Any] | None:
    """Kazıyıcının yazdığı canlı tur durumu; tur sürmüyorsa ya da kayıt bayatsa None."""
    if not raw:
        return None
    try:
        state = json.loads(raw)
    except ValueError:
        return None
    if not state.get("running"):
        return None
    updated = _dt(state.get("updated_at"))
    if updated is None or now - updated > _PROGRESS_STALE:
        return None
    return {
        "fraction": float(state.get("fraction") or 0.0),
        "source": str(state.get("source") or ""),
        "phase": str(state.get("phase") or ""),
        "published": int(state.get("published") or 0),
        "fetched": int(state.get("fetched") or 0),
        "errors": int(state.get("errors") or 0),
        "started_at": state.get("started_at"),
        "updated_at": state.get("updated_at"),
    }


def _age_text(age: timedelta | None) -> str:
    if age is None:
        return "-"
    secs = int(age.total_seconds())
    if secs < 90:
        return f"{secs} sn"
    if secs < 5400:
        return f"{secs // 60} dk"
    return f"{secs // 3600} sa"
