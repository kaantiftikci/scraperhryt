"""Olay kümeleme ve tekrar alarm bastırma.

Aynı olayı anlatan farklı haberler (iki kaynak, güncellenen metin) tek bir ``event_id`` altında toplanır.
Pencere içinde benzer bir alarm varsa yeni kayıt ``duplicate_of`` ile ona bağlanır; alarm yine depolanır ve
``alarm.raised`` ile yayınlanır, ancak bildirim kanallarına (ayar kapalıysa) gönderilmez.
Benzerlik: embedding kosinüsü (``alarm_dedup_similarity``) ya da başlık kelime Jaccard'ı (``alarm_dedup_title_jaccard``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from ..config import Settings
from ..models import NewsRecord, utcnow
from ..store import ArticleStore, tokenize

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "ve ile de da bir bu şu o için son dakika mi mı mu mü ne ki ya veya ama fakat çok daha en".split()
)


def new_event_id(record_id: str) -> str:
    return "evt-" + hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:16]


def title_tokens(title: str) -> set[str]:
    return {t for t in tokenize(title) if t not in _STOPWORDS and len(t) > 2}


def title_jaccard(a: str, b: str) -> float:
    ta, tb = title_tokens(a), title_tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


@dataclass(frozen=True)
class ClusterDecision:
    event_id: str
    duplicate_of: str = ""
    similarity: float = 0.0
    method: str = "none"  # "embedding" | "title" | "none"


class EventClusterer:
    def __init__(self, settings: Settings, store: ArticleStore) -> None:
        self.settings = settings
        self.store = store

    def cluster(self, record: NewsRecord, embedding: Sequence[float] | None = None) -> ClusterDecision:
        fresh = ClusterDecision(event_id=new_event_id(record.id))
        if not self.settings.alarm_dedup_enabled:
            return fresh
        since = utcnow() - timedelta(hours=max(1, int(self.settings.alarm_dedup_window_hours)))
        try:
            hits = self.store.find_similar_alarms(title=record.title, embedding=embedding, since=since, size=10)
        except Exception as exc:
            log.warning("Benzer alarm araması başarısız, kayıt yeni olay sayılacak (%s): %s", record.id, exc)
            return fresh
        best: ClusterDecision | None = None
        for hit in hits:
            doc = hit.doc
            if str(doc.get("id", "")) == record.id or not doc.get("alarm_id"):
                continue
            jaccard = title_jaccard(record.title, str(doc.get("title", "")))
            cosine = 2.0 * float(hit.score) - 1.0 if embedding is not None else -1.0  # ES/InMemory skoru=(1+cos)/2
            if cosine >= float(self.settings.alarm_dedup_similarity):
                candidate = ClusterDecision(
                    event_id=str(doc.get("event_id") or doc.get("alarm_id")),
                    duplicate_of=str(doc["alarm_id"]),
                    similarity=round(cosine, 4),
                    method="embedding",
                )
            elif jaccard >= float(self.settings.alarm_dedup_title_jaccard):
                candidate = ClusterDecision(
                    event_id=str(doc.get("event_id") or doc.get("alarm_id")),
                    duplicate_of=str(doc["alarm_id"]),
                    similarity=round(jaccard, 4),
                    method="title",
                )
            else:
                continue
            if best is None or candidate.similarity > best.similarity:
                best = candidate
        return best or fresh
