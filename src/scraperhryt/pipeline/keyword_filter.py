"""Anahtar kelime filtresi: q.articles.raw → q.articles.keyword (eşleşme) / q.articles.scored (eşleşme yok).

Yönlendirme kuralı (BUILD_SPEC §B):
- Eşleşme var → ``matched_keywords`` doldurulur, ``stage=keyword``, ``article.keyword`` ile yayınlanır.
- Eşleşme yok ve ``llm_score_all`` açık → boş ``matched_keywords`` ile yine ``article.keyword``.
- Eşleşme yok → ``mark_not_scored()`` (skor 0, alarm yok) ve doğrudan ``article.scored``.
- Çözümlenemeyen mesaj → ``Reject`` (ölü mektup).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, RoutingKey
from ..config import Settings
from ..models import NewsRecord, Stage
from ..textutil import KeywordHit, KeywordMatcher, excerpt
from .preclassifier import Embedder, Preclassifier

log = logging.getLogger(__name__)


def load_keyword_aliases(path: str, canonical: list[str]) -> dict[str, str]:
    """``config/keyword_aliases.json`` → {alias: kanonik}. Dosya yoksa boş; bozuksa uyarı verip boş döner.

    Yalnızca ayarlı anahtar kelimelerin (``canonical``) eş anlamlıları yüklenir; '_' ile başlayan anahtarlar yorumdur.
    """
    if not path:
        return {}
    file = Path(path)
    if not file.is_file():
        log.warning("Anahtar kelime eş anlamlı dosyası bulunamadı (%s); eş anlamlılar kapalı", path)
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON nesnesi bekleniyor")
    except (OSError, ValueError) as exc:
        log.warning("Eş anlamlı dosyası okunamadı (%s), eş anlamlılar kapalı: %s", path, exc)
        return {}
    wanted = {k.strip().lower(): k.strip() for k in canonical}
    aliases: dict[str, str] = {}
    for key, values in data.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        canon = wanted.get(key.strip().lower())
        if canon is None or not isinstance(values, list):
            continue
        for alias in values:
            if isinstance(alias, str) and alias.strip() and alias.strip().lower() != canon.lower():
                aliases[alias.strip()] = canon
    if aliases:
        log.info("Anahtar kelime eş anlamlıları yüklendi: %d alias (%s)", len(aliases), path)
    return aliases


@dataclass
class FilterStats:
    received: int = 0
    hits: int = 0
    misses: int = 0
    forwarded_unmatched: int = 0  # llm_score_all açıkken eşleşmeden LLM'e yönlendirilenler
    prefiltered: int = 0  # ön sınıflandırıcının LLM'e göndermeden elediği eşleşmeler
    rejected: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class KeywordFilterService:
    """``Queue.ARTICLES_RAW`` tüketicisi; her haberi anahtar kelime listesine göre yönlendirir."""

    def __init__(self, settings: Settings, broker: Broker, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.broker = broker
        self.preclassifier = Preclassifier(settings, embedder) if settings.preclassifier_enabled else None
        self.aliases = load_keyword_aliases(settings.keyword_aliases_path, settings.keyword_list)
        self.matcher = KeywordMatcher(list(settings.keyword_list) + list(self.aliases))
        self.stats = FilterStats()
        if not settings.keyword_list:
            log.warning(
                "Anahtar kelime listesi boş (KEYWORDS); llm_score_all=%s olduğundan haberler %s",
                settings.llm_score_all,
                "doğrudan LLM'e gidecek" if settings.llm_score_all else "skorlanmadan geçecek",
            )

    def classify(self, record: NewsRecord) -> list[KeywordHit]:
        """Başlık + alt başlık + içerikte eşleşen anahtar kelimeleri (sayı ve örnek bağlamla) döndürür.

        Eş anlamlı eşleşmeleri kanonik anahtar kelimeye katlanır; sıra ``settings.keyword_list`` sırasıdır.
        """
        merged: dict[str, KeywordHit] = {}
        for hit in self.matcher.find(record.text_for_matching()):
            canon = self.aliases.get(hit.keyword, hit.keyword)
            if canon in merged:
                merged[canon].count += hit.count
                merged[canon].samples = (merged[canon].samples + hit.samples)[:3]
            else:
                merged[canon] = KeywordHit(keyword=canon, count=hit.count, samples=list(hit.samples))
        order = {k: i for i, k in enumerate(self.settings.keyword_list)}
        return sorted(merged.values(), key=lambda h: order.get(h.keyword, len(order)))

    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            record = NewsRecord.from_message(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz haber mesajı (article.raw): {excerpt(str(exc), 300)}") from exc

        hits = self.classify(record)
        title = excerpt(record.title, 80)
        if hits and self.preclassifier is not None:
            decision = self.preclassifier.evaluate(record.text_for_matching(), hits, title=record.title)
            record.relevance = decision.relevance
            if decision.drop:
                record.matched_keywords = [h.keyword for h in hits]
                record.prefilter_reason = decision.reason
                record.mark_not_scored()
                self.broker.publish(RoutingKey.ARTICLE_SCORED, record.to_message())
                self.stats.prefiltered += 1
                log.info(
                    "Ön sınıflandırıcı eledi [%s] %s → %s | %s | %s",
                    record.source,
                    title,
                    Queue.ARTICLES_SCORED,
                    ", ".join(f"{h.keyword}×{h.count}" for h in hits),
                    decision.reason,
                )
                return
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
        log.info(
            "Anahtar kelime filtresi durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict()
        )
        return processed
