"""Yeniden skorlama: model/prompt/eşik değişince depodaki kayıtları LLM yolundan tekrar geçirir.

Kayıtlar ``article.keyword`` ile yeniden yayınlanır; normal skorlayıcı → alarm katmanı yolu çalışır, böylece
Elasticsearch belgesi aynı id ile güncellenir ve yeni alarmlar olağan şekilde yükselir.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import timedelta

from pydantic import ValidationError

from ..broker import Broker, RoutingKey
from ..config import Settings
from ..models import NewsRecord, Stage, utcnow
from ..store import ArticleStore

log = logging.getLogger(__name__)


@dataclass
class RescoreStats:
    selected: int = 0
    published: int = 0
    skipped: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class Rescorer:
    def __init__(self, settings: Settings, store: ArticleStore, broker: Broker) -> None:
        self.settings = settings
        self.store = store
        self.broker = broker

    def run(
        self,
        *,
        since_days: int | None = None,
        only_keyword_hits: bool = True,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> RescoreStats:
        stats = RescoreStats(dry_run=dry_run)
        since = utcnow() - timedelta(days=since_days) if since_days else None
        for doc in self.store.iter_records(
            since=since, only_keyword_hits=only_keyword_hits, batch_size=self.settings.rescore_batch_size
        ):
            if limit is not None and stats.selected >= limit:
                break
            stats.selected += 1
            try:
                record = NewsRecord.from_message(doc)
            except ValidationError as exc:
                stats.skipped += 1
                log.warning("Kayıt yeniden skorlanamadı (geçersiz belge %s): %s", doc.get("id"), exc)
                continue
            previous = record.alarm_score
            record.llm = None
            record.alarm_score = 0
            record.is_alarm = False
            record.alarm_reason = ""
            record.llm_summary = ""
            record.confidence = 0
            record.needs_review = False
            record.score_samples = []
            record.alarm_id = ""
            record.alarmed_at = None
            record.stage = Stage.KEYWORD
            if dry_run:
                log.info("[dry-run] yeniden skorlanacak: %s (önceki skor %d) %s", record.id, previous, record.title[:70])
                continue
            self.broker.publish(
                RoutingKey.ARTICLE_KEYWORD,
                record.to_message(),
                headers={"x-rescore": True, "x-previous-score": previous},
            )
            stats.published += 1
        log.info("Yeniden skorlama: %s", stats.as_dict())
        return stats
