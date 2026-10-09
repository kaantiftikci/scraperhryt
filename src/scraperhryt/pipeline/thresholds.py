"""Kaynak / kategori / anahtar kelime bazlı alarm eşikleri (ALARM_THRESHOLDS_JSON).

Örnek: ``{"source:12punto": 70, "category:spor": 90, "keyword:fon": 55}``.
Öncelik: category > keyword > source > varsayılan ``alarm_threshold``. Geçersiz JSON bir kez uyarılır ve yok sayılır.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from ..config import Settings
from ..models import NewsRecord
from ..textutil import tr_lower

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def parse_threshold_rules(raw: str) -> dict[str, int]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON nesnesi bekleniyor")
        rules: dict[str, int] = {}
        for key, value in data.items():
            k = tr_lower(str(key)).strip()
            if ":" not in k:
                raise ValueError(f"anahtar 'source:|category:|keyword:' önekli olmalı: {key!r}")
            rules[k] = max(0, min(100, int(value)))
        return rules
    except (ValueError, TypeError) as exc:
        log.warning("ALARM_THRESHOLDS_JSON geçersiz, varsayılan eşik kullanılacak: %s", exc)
        return {}


def resolve_threshold(settings: Settings, record: NewsRecord) -> int:
    """Kayıt için uygulanacak eşiği döndürür (kategori > anahtar kelime > kaynak > varsayılan)."""
    rules = parse_threshold_rules(settings.alarm_thresholds_json)
    if not rules:
        return int(settings.alarm_threshold)
    category = tr_lower(record.category).strip()
    if category and f"category:{category}" in rules:
        return rules[f"category:{category}"]
    keyword_hits = [rules[f"keyword:{tr_lower(k)}"] for k in record.matched_keywords if f"keyword:{tr_lower(k)}" in rules]
    if keyword_hits:
        return min(keyword_hits)  # en hassas (düşük) eşik kazanır
    source = tr_lower(record.source).strip()
    if f"source:{source}" in rules:
        return rules[f"source:{source}"]
    return int(settings.alarm_threshold)
