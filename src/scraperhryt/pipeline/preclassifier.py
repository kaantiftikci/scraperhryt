"""Ön sınıflandırıcı: anahtar kelime eşleşen haberleri LLM'e göndermeden önce bariz yanlış pozitifleri eler.

İki katman:
1. **Kural tabanlı** — "bakan" kelimesinin fiil (bakmak → "pencereden bakan adam") kullanımını yakalar.
   Yalnızca TÜM eşleşmeler fiil kullanımıysa ve başka anahtar kelime yoksa eler.
2. **Embedding tabanlı** — metin, ``config/preclassifier_prototypes.json`` içindeki "ilgili" ve "ilgisiz" örnek
   cümlelerin embedding merkezleriyle karşılaştırılır: ``relevance = cos(metin, ilgili) - cos(metin, ilgisiz)``.
   ``relevance < preclassifier_threshold`` ise elenir. Embedding alınamazsa (model yok/erişilemiyor) katman atlanır.

Elenen kayıt yine depolanır (skor 0, alarm yok) ve ``prefilter_reason`` ile neden elendiği izlenebilir.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings
from ..textutil import KeywordHit, tr_lower

log = logging.getLogger(__name__)

_MAX_EMBED_CHARS = 1200
_WORD = r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû']+"
_BAKAN_RE = re.compile(rf"(?:(?P<prev>{_WORD})\s+)?(?P<hit>bakan)(?:\s+(?P<next>{_WORD}))?", re.UNICODE)
# fiil kullanımında "bakan"dan önce gelen kelime çoğunlukla -e/-a (yönelme), -den/-dan (ayrılma) ekli ya da zarftır
_VERBAL_PREV_SUFFIX = re.compile(r"(?:[ae]|y[ae]|n[ae]|[dt][ae]n|[dt]an|le|la|yle|yla|ce|ca|çe|ça|rek|rak|ken|ca)$")
_VERBAL_NEXT = frozenset(
    "adam kadın kişi kişiler çocuk çocuklar genç gençler insan insanlar gözler göz bakışlar vatandaş vatandaşlar "
    "yolcu yolcular sürücü öğrenci öğrenciler kedi köpek bebek anne baba yüz yüzler oda odalar ev evler daire manzara "
    "pencere balkon taraf tarafı yön yönü".split()
)
_MINISTRY_PREV = frozenset(
    "eski yeni sayın içişleri dışişleri adalet maliye sağlık eğitim ulaştırma tarım enerji ticaret sanayi çevre kültür "
    "turizm gençlik spor aile çalışma savunma hazine bakanlığı yardımcısı başbakan kabinedeki ilgili sorumlu".split()
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class PrefilterDecision:
    drop: bool
    relevance: float | None = None
    reason: str = ""
    method: str = "none"  # "rule" | "embedding" | "none"


def is_verbal_bakan(prev: str | None, nxt: str | None) -> bool:
    """'bakan' kelimesinin bağlamından fiil (bakmak) olup olmadığını tahmin eder."""
    prev_l = tr_lower(prev or "").strip("'")
    next_l = tr_lower(nxt or "").strip("'")
    if prev_l in _MINISTRY_PREV or prev_l.endswith(("bakanlığı", "bakanlık")):
        return False
    if next_l in _VERBAL_NEXT:
        return True
    if prev_l and _VERBAL_PREV_SUFFIX.search(prev_l) and prev_l not in {"ve", "ile", "da", "de"}:
        # "pencereden bakan", "gözlerine bakan", "dikkatle bakan"
        return next_l not in {"yardımcısı", "bey", "hanım"} and not (nxt or "")[:1].isupper()
    return False


def rule_based_reason(text: str, hits: Sequence[KeywordHit]) -> str:
    """Tüm eşleşmeler 'bakan' fiil kullanımıysa Türkçe gerekçe döndürür; aksi halde boş."""
    keywords = {tr_lower(h.keyword) for h in hits}
    if keywords != {"bakan"}:
        return ""
    lowered = tr_lower(text)
    occurrences = list(_BAKAN_RE.finditer(lowered))
    if not occurrences:
        return ""
    if all(is_verbal_bakan(m.group("prev"), m.group("next")) for m in occurrences):
        sample = occurrences[0]
        ctx = " ".join(p for p in (sample.group("prev"), "bakan", sample.group("next")) if p)
        return f'kural: "bakan" fiil kullanımı ("{ctx}")'
    return ""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]


def load_prototypes(path: str) -> tuple[list[str], list[str]]:
    file = Path(path)
    if not path or not file.is_file():
        log.warning("Ön sınıflandırıcı örnek dosyası yok (%s); embedding katmanı kapalı", path)
        return [], []
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        rel = [str(x) for x in data.get("relevant", []) if str(x).strip()]
        irr = [str(x) for x in data.get("irrelevant", []) if str(x).strip()]
    except (OSError, ValueError, AttributeError) as exc:
        log.warning("Ön sınıflandırıcı örnek dosyası okunamadı (%s): %s", path, exc)
        return [], []
    if len(rel) < 3 or len(irr) < 3:
        log.warning("Ön sınıflandırıcı için en az 3 ilgili ve 3 ilgisiz örnek gerekir (%s)", path)
        return [], []
    return rel, irr


class Preclassifier:
    """Kural + embedding ön sınıflandırıcı. ``embedder`` None ise yalnızca kural katmanı çalışır."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder
        self.threshold = float(settings.preclassifier_threshold)
        self._relevant, self._irrelevant = load_prototypes(settings.preclassifier_prototypes_path)
        self._centroids: tuple[list[float], list[float]] | None = None
        self._lock = threading.Lock()
        self._disabled_reason = ""

    # --- embedding katmanı ---
    def _ensure_centroids(self) -> tuple[list[float], list[float]] | None:
        if self.embedder is None or not self._relevant or self._disabled_reason:
            return None
        with self._lock:
            if self._centroids is not None:
                return self._centroids
            try:
                rel = self.embedder.embed(self._relevant)
                irr = self.embedder.embed(self._irrelevant)
            except Exception as exc:
                log.warning("Ön sınıflandırıcı örnekleri gömülemedi, embedding katmanı bu kez atlandı: %s", exc)
                return None
            if not rel or not irr or len(rel[0]) != len(irr[0]):
                self._disabled_reason = "örnek vektörleri boş/uyuşmaz"
                log.warning("Ön sınıflandırıcı embedding katmanı kapatıldı: %s", self._disabled_reason)
                return None
            self._centroids = (_centroid(rel), _centroid(irr))
            log.info(
                "Ön sınıflandırıcı hazır: %d ilgili / %d ilgisiz örnek, eşik=%.2f",
                len(self._relevant),
                len(self._irrelevant),
                self.threshold,
            )
            return self._centroids

    def relevance(self, text: str) -> float | None:
        centroids = self._ensure_centroids()
        if centroids is None:
            return None
        try:
            vec = self.embedder.embed([text[:_MAX_EMBED_CHARS]])[0]  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("Metin gömülemedi, ön sınıflandırıcı atlandı: %s", exc)
            return None
        rel, irr = centroids
        if len(vec) != len(rel):
            return None
        return round(_cosine(vec, rel) - _cosine(vec, irr), 4)

    # --- karar ---
    def evaluate(self, text: str, hits: Sequence[KeywordHit]) -> PrefilterDecision:
        reason = rule_based_reason(text, hits)
        if reason:
            return PrefilterDecision(drop=True, reason=reason, method="rule")
        rel = self.relevance(text)
        if rel is None:
            return PrefilterDecision(drop=False)
        if rel < self.threshold:
            return PrefilterDecision(
                drop=True,
                relevance=rel,
                reason=f"önsınıflandırıcı: ilgisiz (relevance={rel:.2f} < {self.threshold:.2f})",
                method="embedding",
            )
        return PrefilterDecision(drop=False, relevance=rel, method="embedding")
