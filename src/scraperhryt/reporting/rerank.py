"""Reranker istemcisi: arama adaylarını soruyla birlikte okuyan çapraz kodlayıcı (ör. bge-reranker-v2-m3).

Kelime araması ve kNN adayları bulur ama sıralamaları soruyla tam örtüşmez; reranker her (soru, haber) çiftini
birlikte okuyup bir ilgililik skoru verir ve modele giden haberler bu skora göre seçilir. Ollama yeniden sıralama
sunmadığından ayrı bir sunucu kullanılır:

- ``llamacpp``: ``llama-server --reranking`` (Mac'te Metal ile hızlı). ``POST /v1/rerank`` →
  ``{"results": [{"index", "relevance_score"}]}`` (Jina/Cohere uyumlu biçim).
- ``tei``: HuggingFace Text Embeddings Inference. ``POST /rerank`` → ``[{"index", "score"}]``.

Reranker erişilemezse ya da hata verirse ``None`` döner ve soru-cevap mevcut sıralamayla devam eder.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import Settings

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.api = (settings.reranker_api or "llamacpp").strip().lower()
        self.base_url = settings.reranker_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=settings.reranker_timeout, transport=transport)

    @classmethod
    def from_settings(cls, settings: Settings) -> Reranker | None:
        return cls(settings) if settings.reranker_url else None

    def close(self) -> None:
        self._client.close()

    def scores(self, query: str, documents: Sequence[str]) -> list[float] | None:
        """``documents`` ile aynı sırada ilgililik skorları; hata durumunda ``None``."""
        docs = [doc[: self.settings.reranker_max_chars] for doc in documents]
        if not docs:
            return []
        try:
            if self.api == "tei":
                resp = self._client.post("/rerank", json={"query": query, "texts": docs, "truncate": True})
                resp.raise_for_status()
                items = resp.json()
                key = "score"
            else:
                payload = {"model": self.settings.reranker_model, "query": query, "documents": docs, "top_n": len(docs)}
                resp = self._client.post("/v1/rerank", json=payload)
                resp.raise_for_status()
                body = resp.json()
                items = body.get("results") if isinstance(body, dict) else body
                key = "relevance_score"
            return _align(items, key, len(docs))
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            log.warning("Reranker kullanılamadı (%s); mevcut sıralamayla devam ediliyor: %s", self.base_url, exc)
            return None


def _align(items: Any, key: str, n: int) -> list[float]:
    if not isinstance(items, list):
        raise ValueError("reranker yanıtı liste değil")
    scores = [float("-inf")] * n
    for item in items:
        index = int(item["index"])
        if 0 <= index < n:
            scores[index] = float(item.get(key, item.get("score", item.get("relevance_score"))))
    if any(s == float("-inf") for s in scores):
        raise ValueError("reranker her aday için skor döndürmedi")
    return scores


def rerank_text(doc: dict[str, Any]) -> str:
    """Reranker'a giden aday metni: başlık, alt başlık ve haberin başı (gazetecinin metni)."""
    parts = [str(doc.get(k) or "").strip() for k in ("title", "subtitle", "content")]
    return "\n".join(p for p in parts if p)
