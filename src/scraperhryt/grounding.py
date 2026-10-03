"""Sayısal doğruluk denetimi: LLM metnindeki sayılar kaynak haberlerde geçiyor mu?

Küçük modeller özet yazarken sayıları bozabiliyor ("20 milyar dolar" → "20 milyon dolar", "85 tutuklu" →
"58 tutuklu"). Bu modül bir metindeki her sayıyı (ve varsa büyüklük ekini: bin/milyon/milyar/trilyon) kaynak
metindeki sayılarla karşılaştırır; kaynakta karşılığı olmayan sayıyı taşıyan cümleler atılır.

Hem RAG yanıtlarında (``reporting/rag.py``) hem de alarm skorlamasındaki LLM özetlerinde (``pipeline/scorer.py``)
kullanılır. Yazıyla yazılmış sayılar ("beşinci dalga", "yirmi şüpheli") rakama çevrilerek tanınır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .textutil import split_sentences, tr_lower

MAGNITUDES = ("bin", "milyon", "milyar", "trilyon")
_NUMBER_WORDS = {
    "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
    "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50, "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90,
    "yüz": 100,
    "birinci": 1, "ikinci": 2, "üçüncü": 3, "dördüncü": 4, "beşinci": 5, "altıncı": 6, "yedinci": 7,
    "sekizinci": 8, "dokuzuncu": 9, "onuncu": 10, "yirminci": 20, "otuzuncu": 30, "kırkıncı": 40,
    "ellinci": 50, "yüzüncü": 100,
}
_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)(?:\s*['’]?\s*([a-zçğıöşü]+))?", re.UNICODE)
_CITATION_RE = re.compile(r"\[\d+\]")
_WORD_RE = re.compile(r"[a-zçğıöşüâîû]+", re.UNICODE)


_DATE_RE = re.compile(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}")
_TIME_RE = re.compile(r"\d{1,2}[.:]\d{2}")


def _is_date_or_time(raw: str) -> bool:
    if _DATE_RE.fullmatch(raw) or _TIME_RE.fullmatch(raw):
        return True
    return len(raw) == 4 and raw.isdigit() and raw[:2] in ("19", "20")


def normalize_number(token: str) -> str:
    """"1.500" → "1500" (binlik ayırıcı), "2,22" → "2.22", "15.00" (saat) → "15.00"."""
    token = token.strip(".,")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")
    return token.replace(",", ".")


@dataclass(frozen=True)
class Figure:
    number: str
    magnitude: str = ""

    def __str__(self) -> str:
        return f"{self.number} {self.magnitude}".strip()


def extract_figures(text: str) -> list[Figure]:
    """Metindeki sayılar (atıf numaraları [n] hariç) ve hemen ardından gelen büyüklük sözcüğü."""
    low = tr_lower(_CITATION_RE.sub(" ", text or ""))
    figures: list[Figure] = []
    for match in _NUM_RE.finditer(low):
        raw = match.group(1)
        if _is_date_or_time(raw):
            continue  # tarih/saat/yıl: model farklı biçimde yazabilir ("1 Ekim" ↔ "01.10.2026"); miktar değildir
        nxt = match.group(2) or ""
        magnitude = next((m for m in MAGNITUDES if nxt.startswith(m)), "")
        figures.append(Figure(normalize_number(match.group(1)), magnitude))
    words = _WORD_RE.findall(low)
    for i, word in enumerate(words):
        base = word.split("'")[0]
        value = _NUMBER_WORDS.get(base)
        if value is None:
            # ekli sıra sayısı: "beşinci dalgada", "ikincisi"
            value = next((v for w, v in _NUMBER_WORDS.items() if len(w) >= 6 and base.startswith(w)), None)
        if value is None:
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        magnitude = next((m for m in MAGNITUDES if nxt.startswith(m)), "")
        figures.append(Figure(str(value), magnitude))
    return figures


class SourceFigures:
    """Kaynak metinlerdeki sayıların hızlı arama kümesi."""

    def __init__(self, *texts: str) -> None:
        self.numbers: set[str] = set()
        self.pairs: set[tuple[str, str]] = set()
        for text in texts:
            for fig in extract_figures(text):
                self.numbers.add(fig.number)
                self.numbers.add(fig.number.split(".")[0] if fig.number.endswith(".0") else fig.number)
                if fig.magnitude:
                    self.pairs.add((fig.number, fig.magnitude))

    def supports(self, fig: Figure) -> bool:
        if fig.number not in self.numbers:
            return False
        if fig.magnitude:
            return (fig.number, fig.magnitude) in self.pairs
        return True


def unsupported_figures(sentence: str, sources: SourceFigures) -> list[str]:
    return [str(f) for f in extract_figures(sentence) if not sources.supports(f)]


def ground_text(text: str, sources: SourceFigures) -> tuple[str, list[str]]:
    """Kaynakta karşılığı olmayan sayı içeren cümleleri atar. ``(kalan metin, atılan cümleler)``."""
    kept: list[str] = []
    dropped: list[str] = []
    for sentence in split_sentences(text):
        if unsupported_figures(sentence, sources):
            dropped.append(sentence)
        else:
            kept.append(sentence)
    return " ".join(kept), dropped
