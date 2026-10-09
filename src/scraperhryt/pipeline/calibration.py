"""Kalibrasyon: etiketli altın set üzerinde LLM skorlarını ölçer.

Her eşik için kesinlik/duyarlılık/F1 hesaplanır ve en iyi F1'i veren eşik önerilir. ``scraperhryt calibrate``.
Altın set satırı (JSONL): {"title","subtitle","content","matched_keywords":[..],"source","label_is_alarm":bool,
"label_score":int,"note":str}
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models import NewsRecord, Stage

log = logging.getLogger(__name__)


@dataclass
class CalibrationItem:
    title: str
    content: str
    label_is_alarm: bool
    subtitle: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    source: str = "golden"
    label_score: int | None = None
    note: str = ""
    origin: str = "golden"

    def to_record(self, index: int) -> NewsRecord:
        record = NewsRecord.new(
            source=self.source or "golden",
            content_url=f"https://calibration.local/{self.origin}/{index}",
            title=self.title,
            subtitle=self.subtitle,
            content=self.content,
        )
        record.matched_keywords = list(self.matched_keywords)
        record.stage = Stage.KEYWORD
        return record


@dataclass
class ThresholdRow:
    threshold: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "precision": round(self.precision, 3), "recall": round(self.recall, 3), "f1": round(self.f1, 3)}


@dataclass
class CalibrationReport:
    items: list[dict[str, Any]]
    rows: list[ThresholdRow]
    best_threshold: int
    current_threshold: int
    mean_abs_error: float | None
    failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "rows": [r.as_dict() for r in self.rows],
            "best_threshold": self.best_threshold,
            "current_threshold": self.current_threshold,
            "mean_abs_error": self.mean_abs_error,
            "failures": self.failures,
        }


def load_golden_set(path: str | Path) -> list[CalibrationItem]:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"Altın set bulunamadı: {file}")
    items: list[CalibrationItem] = []
    for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
            items.append(
                CalibrationItem(
                    title=str(data["title"]),
                    content=str(data.get("content", "")),
                    subtitle=str(data.get("subtitle", "")),
                    matched_keywords=[str(k) for k in data.get("matched_keywords", [])],
                    source=str(data.get("source", "golden")),
                    label_is_alarm=bool(data["label_is_alarm"]),
                    label_score=int(data["label_score"]) if data.get("label_score") is not None else None,
                    note=str(data.get("note", "")),
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("Altın set satırı %d atlandı: %s", lineno, exc)
    return items


def run_calibration(
    scorer: Any,
    items: Sequence[CalibrationItem],
    *,
    current_threshold: int,
    thresholds: Iterable[int] = range(0, 101, 5),
) -> CalibrationReport:
    """Her örneği ``scorer.score_record`` ile puanlar, eşik tablosunu kurar; en iyi F1 eşiğini önerir."""
    scored: list[dict[str, Any]] = []
    failures = 0
    for i, item in enumerate(items):
        record = item.to_record(i)
        try:
            scorer.score_record(record)
        except Exception as exc:
            failures += 1
            log.warning("Kalibrasyon örneği puanlanamadı (%s): %s", item.title[:60], exc)
            continue
        scored.append(
            {
                "title": item.title,
                "origin": item.origin,
                "label_is_alarm": item.label_is_alarm,
                "label_score": item.label_score,
                "alarm_score": record.alarm_score,
                "confidence": record.confidence,
                "needs_review": record.needs_review,
                "note": item.note,
            }
        )
    rows: list[ThresholdRow] = []
    for t in thresholds:
        tp = sum(1 for s in scored if s["label_is_alarm"] and s["alarm_score"] >= t)
        fp = sum(1 for s in scored if not s["label_is_alarm"] and s["alarm_score"] >= t)
        fn = sum(1 for s in scored if s["label_is_alarm"] and s["alarm_score"] < t)
        tn = sum(1 for s in scored if not s["label_is_alarm"] and s["alarm_score"] < t)
        rows.append(ThresholdRow(int(t), tp, fp, fn, tn))
    best = max(rows, key=lambda r: (round(r.f1, 6), r.precision, -abs(r.threshold - current_threshold))) if rows else None
    errors = [abs(s["alarm_score"] - s["label_score"]) for s in scored if s["label_score"] is not None]
    mae = round(sum(errors) / len(errors), 1) if errors else None
    return CalibrationReport(
        items=scored,
        rows=rows,
        best_threshold=best.threshold if best else current_threshold,
        current_threshold=current_threshold,
        mean_abs_error=mae,
        failures=failures,
    )


def format_calibration_report(report: CalibrationReport) -> str:
    lines = [f"Kalibrasyon: {len(report.items)} örnek puanlandı, {report.failures} başarısız"]
    lines.append(f"{'Başlık':<60} {'Etiket':>6} {'Hedef':>5} {'Skor':>4} {'Güven':>5} İnceleme")
    for s in report.items:
        lines.append(
            f"{s['title'][:58]:<60} {'ALARM' if s['label_is_alarm'] else '-':>6} "
            f"{str(s['label_score'] if s['label_score'] is not None else '-'):>5} {s['alarm_score']:>4} "
            f"{s['confidence']:>5} {'evet' if s['needs_review'] else ''}"
        )
    lines.append("")
    lines.append(f"{'Eşik':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'Kesinlik':>8} {'Duyarlılık':>10} {'F1':>5}")
    for r in report.rows:
        marker = " ◀ öneri" if r.threshold == report.best_threshold else (" (mevcut)" if r.threshold == report.current_threshold else "")
        lines.append(
            f"{r.threshold:>4} {r.tp:>3} {r.fp:>3} {r.fn:>3} {r.tn:>3} {r.precision:>8.2f} {r.recall:>10.2f} {r.f1:>5.2f}{marker}"
        )
    if report.mean_abs_error is not None:
        lines.append(f"\nOrtalama mutlak skor hatası (etiketli hedefe göre): {report.mean_abs_error}")
    lines.append(f"Önerilen ALARM_THRESHOLD: {report.best_threshold} (mevcut: {report.current_threshold})")
    return "\n".join(lines)
