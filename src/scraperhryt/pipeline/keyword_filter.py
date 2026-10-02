"""Anahtar kelime filtresi: q.articles.raw → q.articles.keyword (eşleşme) / q.articles.scored (eşleşme yok).

Yönlendirme kuralı (BUILD_SPEC §B):
- Eşleşme var → ``matched_keywords`` doldurulur, ``stage=keyword``, ``article.keyword`` ile yayınlanır.
- Eşleşme yok ve ``llm_score_all`` açık → boş ``matched_keywords`` ile yine ``article.keyword``.
- Eşleşme yok → ``mark_not_scored()`` (skor 0, alarm yok) ve doğrudan ``article.scored``.
- Çözümlenemeyen mesaj → ``Reject`` (ölü mektup).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, RoutingKey
from ..config import Settings
from ..models import NewsRecord, Stage
from ..textutil import KeywordHit, KeywordMatcher, excerpt

log = logging.getLogger(__name__)


@dataclass
class FilterStats:
    received: int = 0
    hits: int = 0
    misses: int = 0
    forwarded_unmatched: int = 0  # llm_score_all açıkken eşleşmeden LLM'e yönlendirilenler
    rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class KeywordFilterService:
    """``Queue.ARTICLES_RAW`` tüketicisi; her haberi anahtar kelime listesine göre yönlendirir."""

    def __init__(self, settings: Settings, broker: Broker) -> None:
        self.settings = settings
        self.broker = broker
        self.matcher = KeywordMatcher(settings.keyword_list)
        self.stats = FilterStats()
        if not settings.keyword_list:
            log.warning(
                "Anahtar kelime listesi boş (KEYWORDS); llm_score_all=%s olduğundan haberler %s",
                settings.llm_score_all,
                "doğrudan LLM'e gidecek" if settings.llm_score_all else "skorlanmadan geçecek",
            )

    def classify(self, record: NewsRecord) -> list[KeywordHit]:
        """Başlık + alt başlık + içerik üzerinde eşleşen anahtar kelimeleri (sayı ve örnek bağlamla) döndürür."""
        return self.matcher.find(record.text_for_matching())

    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            record = NewsRecord.from_message(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz haber mesajı (article.raw): {excerpt(str(exc), 300)}") from exc

        hits = self.classify(record)
        title = excerpt(record.title, 80)
        if hits:
            record.matched_keywords = [h.keyword for h in hits]
            record.stage = Stage.KEYWORD
            self.broker.publish(RoutingKey.ARTICLE_KEYWORD, record.to_message())
            self.stats.hits += 1
            log.info(
                "Anahtar kelime eşleşti [%s] %s → %s | %s",
                record.source,
                title,
                Queue.ARTICLES_KEYWORD,
                ", ".join(f"{h.keyword}×{h.count}" for h in hits),
            )
            return

        record.matched_keywords = []
        if self.settings.llm_score_all:
            record.stage = Stage.KEYWORD
            self.broker.publish(RoutingKey.ARTICLE_KEYWORD, record.to_message())
            self.stats.forwarded_unmatched += 1
            log.info(
                "Eşleşme yok, llm_score_all açık: LLM'e yönlendirildi [%s] %s → %s",
                record.source,
                title,
                Queue.ARTICLES_KEYWORD,
            )
            return

        record.mark_not_scored()
        self.broker.publish(RoutingKey.ARTICLE_SCORED, record.to_message())
        self.stats.misses += 1
        log.info("Eşleşme yok [%s] %s → %s (skor 0, alarm yok)", record.source, title, Queue.ARTICLES_SCORED)

    def run(self, stop_event: threading.Event | None = None, max_messages: int | None = None) -> int:
        """``q.articles.raw`` kuyruğunu tüketir; işlenen mesaj sayısını döndürür."""
        self.broker.declare_topology()
        log.info(
            "Anahtar kelime filtresi başlıyor: kuyruk=%s, anahtar kelimeler=%s, llm_score_all=%s",
            Queue.ARTICLES_RAW,
            ", ".join(self.settings.keyword_list) or "-",
            self.settings.llm_score_all,
        )
        processed = self.broker.consume(
            Queue.ARTICLES_RAW,
            self.handle,
            prefetch=self.settings.rabbitmq_prefetch,
            stop_event=stop_event,
            max_messages=max_messages,
        )
        log.info("Anahtar kelime filtresi durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed
