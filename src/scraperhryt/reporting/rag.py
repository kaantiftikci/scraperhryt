"""RAG soru-cevap motoru: "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?" gibi sorulara, sisteme
en son giren haberlerden yola çıkarak, kaynak göstererek yanıt verir.

Adımlar (``QAEngine.ask``):

1. **Sorgu yeniden yazma** — ``llm.chat_json`` → ``{"search_terms": [...], "entities": [...]}``; LLM hatasında
   sorunun kendi sözcükleri (Türkçe küçük harf, durak sözcükler atılmış) kullanılır.
2. **Hibrit geri getirme** — ``store.search_records`` (BM25 + yenilik) birkaç dar sorguyla (en önemli terimler,
   her varlık adı tek başına, sorunun kendi sözcükleri; bkz. ``QAEngine.search_queries``); ``OLLAMA_EMBEDDING_MODEL``
   ayarlıysa ``llm.embed`` → ``store.knn_search``. Sıralamalar Reciprocal Rank Fusion (k=60) ile birleştirilir;
   kNN isteğe bağlıdır ve hatası sözlüksel aramayı engellemez.
3. **Bağlam** — belgeler en yeniden en eskiye sıralanıp ``[n] (kaynak, tarih) Başlık — alt başlık — içerik``
   bloklarına çevrilir; toplam bağlam ``OLLAMA_NUM_CTX``'e göre sınırlanır.
4. **Yanıt** — ``llm.generate_text`` ile atıflı Türkçe yanıt; LLM erişilemezse en yeni başlıklardan deterministik
   çıkarımsal yanıt (``model="fallback"``). Hiç haber bulunamazsa LLM çağrılmadan ``INSUFFICIENT_EVIDENCE_TEXT``
   döner (``model="none"``; bu bir LLM arızası değildir).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..config import Settings
from ..models import Answer, Citation, TimelineItem, utcnow
from ..pipeline.llm import LLM, HeuristicLLM, LLMError
from ..store import ArticleStore, SearchHit
from ..textutil import KeywordMatcher, excerpt, normalize_ws, tr_lower
from .prompts import (
    CONTEXT_CONTENT_CHARS,
    INSUFFICIENT_EVIDENCE_TEXT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_context_block,
    build_query_rewrite_prompt,
    build_rag_user_prompt,
    flat_text,
    format_tr,
    parse_datetime,
)

log = logging.getLogger(__name__)

#: Reciprocal Rank Fusion sabiti (Cormack ve diğ., 2009'daki standart değer).
RRF_K = 60
#: LLM'siz yedek yanıtta listelenen en yeni haber sayısı.
FALLBACK_HEADLINES = 3
#: ``Answer.model`` değeri: LLM erişilemediği için yanıt haberlerden deterministik olarak derlendi.
FALLBACK_MODEL = "fallback"
#: ``Answer.model`` değeri: eşleşen haber bulunamadığından LLM hiç çağrılmadı (LLM arızası değildir).
NO_EVIDENCE_MODEL = "none"
#: ``Answer.search_terms``'e alınan yeniden yazılmış terim + varlık adı üst sınırı.
MAX_SEARCH_TERMS = 12
#: Birleşik sözlüksel sorguya alınan arama terimi sayısı. ES ``best_fields`` sorgusunda ``minimum_should_match``
#: alan başına uygulanır: uzun bir sorguda haberin tek bir alanda terimlerin %60'ını içermesi gerekir ve ilgili
#: haberler elenir; kısa sorgular bu eşiği kolay aşar.
COMBINED_QUERY_TERMS = 4
#: Tek başına sorgulanan varlık adı (yoksa terim) üst sınırı; her biri ayrı bir depo isteğidir.
MAX_NARROW_QUERIES = 4
#: Bağlam bloğu başına içerik alt sınırı (toplam bütçe çok sıkışırsa bile bu kadar verilir).
MIN_CONTENT_CHARS = 300
#: ``OLLAMA_NUM_CTX`` token → karakter yaklaşık çarpanı (Türkçe metinde ~3-4 karakter/token; istem payı düşülmüş).
_CHARS_PER_CTX_TOKEN = 2.5
_MIN_CONTEXT_BUDGET = 4000
_MIN_TOKEN_LEN = 2
_TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")
_EM_TAG_RE = re.compile(r"</?em>")

#: Soru sözcüklerinden atılan Türkçe durak/soru kalıbı sözcükleri (LLM yeniden yazması başarısızsa kullanılır).
STOPWORDS: frozenset[str] = frozenset(
    {
        "acaba", "ama", "ancak", "arasında", "arasındaki", "bana", "bir", "biraz", "bu", "bunun", "da", "daha",
        "de", "değil", "diye", "durum", "durumda", "durumu", "en", "fakat", "gelişme", "gelişmeler", "gibi",
        "göre", "haber", "haberi", "haberler", "haberlerde", "hakkında", "hangi", "hem", "için", "ile", "ilgili",
        "ise", "kadar", "kim", "kimdir", "konuda", "konusunda", "mi", "mı", "mu", "mü", "midir", "mıdır",
        "mudur", "müdür", "nasıl", "ne", "neden", "nedir", "neler", "nerede", "neydi", "niye", "o", "olan",
        "olarak", "oldu", "olduğu", "olmuş", "olup", "oluyor", "önce", "peki", "son", "sonra", "şey", "şimdi",
        "şu", "var", "ve", "veya", "ya", "yani", "yok", "zaman",
    }
)


@dataclass
class RankedDoc:
    """Birleştirilmiş sıralamadaki tek belge: depo belgesi, RRF skoru ve (varsa) arama vurguları."""

    doc: dict[str, Any]
    score: float
    highlights: dict[str, list[str]] = field(default_factory=dict)


def question_tokens(question: str) -> list[str]:
    """Sorunun arama sözcükleri: Türkçe küçük harf, 2+ karakter, durak sözcükler atılmış, sıra korunur, tekrarsız."""
    out: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(tr_lower(question or "")):
        if len(token) < _MIN_TOKEN_LEN or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def doc_identity(doc: dict[str, Any]) -> str:
    """Belgenin birleştirme anahtarı: ``id``, yoksa ``content_url``, o da yoksa başlık."""
    return str(doc.get("id") or doc.get("content_url") or doc.get("title") or "")


def reciprocal_rank_fusion(rankings: Sequence[Sequence[SearchHit]], *, k: int = RRF_K) -> list[RankedDoc]:
    """Birden çok sıralamayı RRF ile birleştirir: ``skor(d) = Σ 1 / (k + sıra_i(d))``.

    Aynı belge (``doc_identity``) tek kez döner; eşit skorda önce görülen önce gelir. Vurgular (highlight)
    sıralamalar arasında birleştirilir.
    """
    scores: dict[str, float] = {}
    docs: dict[str, RankedDoc] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            identity = doc_identity(hit.doc)
            if not identity:
                continue
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            if identity not in docs:
                docs[identity] = RankedDoc(doc=dict(hit.doc), score=0.0)
                order[identity] = len(order)
            if hit.highlights:
                merged = docs[identity].highlights
                for field_name, fragments in hit.highlights.items():
                    merged.setdefault(field_name, []).extend(str(f) for f in fragments)
    ranked = sorted(scores, key=lambda identity: (-scores[identity], order[identity]))
    out: list[RankedDoc] = []
    for identity in ranked:
        item = docs[identity]
        item.score = round(scores[identity], 6)
        out.append(item)
    return out


def doc_timestamp(doc: dict[str, Any]) -> datetime | None:
    """Haber tarihi; yoksa ``@timestamp`` / kazıma zamanı."""
    for key in ("published_at", "@timestamp", "scraped_at"):
        parsed = parse_datetime(doc.get(key))
        if parsed is not None:
            return parsed
    return None


class QAEngine:
    """Depo + LLM üzerinde RAG soru-cevap. ``FakeOllama`` ve ``HeuristicLLM`` ile de çalışır."""

    def __init__(self, settings: Settings, store: ArticleStore, llm: LLM) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        # HeuristicLLM serbest metin üretemez; sorgu yeniden yazma ve yanıt için deterministik yollar kullanılır.
        self._offline = isinstance(llm, HeuristicLLM)

    # --- ana akış ---
    def ask(
        self,
        question: str,
        *,
        since_days: int | None = None,
        top_k: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> Answer:
        question = flat_text(question)
        if not question:
            raise ValueError("Soru boş olamaz")
        days = self.settings.rag_recency_days if since_days is None else int(since_days)
        size = self.settings.rag_top_k if top_k is None else int(top_k)
        if size <= 0:
            raise ValueError("top_k pozitif bir sayı olmalı")
        now = utcnow()
        since = now - timedelta(days=days) if days > 0 else None
        source_filter = [s.strip() for s in (sources or []) if s and s.strip()] or None

        terms, entities = self.rewrite_query(question)
        search_terms = _dedupe(terms + entities)[:MAX_SEARCH_TERMS]
        queries = self.search_queries(question, terms, entities)
        ranked = self.retrieve(question, queries, since=since, size=size, sources=source_filter)
        relevant = filter_relevant(ranked, terms=terms, entities=entities)
        if len(relevant) < len(ranked):
            log.info("İlgisiz %d belge elendi (soru terimleri metinde geçmiyor)", len(ranked) - len(relevant))
        ordered = newest_first(relevant)
        blocks, citations = self.build_context(ordered)
        log.info(
            "Soru işlendi: %r → terimler=%s, bulunan=%d, pencere=%s gün",
            excerpt(question, 80),
            search_terms,
            len(citations),
            days if days > 0 else "sınırsız",
        )

        if not citations:
            scope = f" (Son {days} günde taranan haberler arasında eşleşme bulunamadı.)" if days > 0 else ""
            answer_text, model = INSUFFICIENT_EVIDENCE_TEXT + scope, NO_EVIDENCE_MODEL
        else:
            answer_text, model = self.generate_answer(question, blocks, ordered, citations, since_days=days)

        return Answer(
            question=question,
            answer=answer_text,
            sources=citations,
            model=model,
            retrieved_count=len(citations),
            search_terms=search_terms,
            timeline=build_timeline(citations),
        )

    # --- 1. sorgu yeniden yazma ---
    def rewrite_query(self, question: str) -> tuple[list[str], list[str]]:
        """``(search_terms, entities)``; LLM yoksa/hata verirse ``(question_tokens(question), [])``."""
        fallback = question_tokens(question)
        if self._offline:
            return fallback, []
        try:
            data = self.llm.chat_json(
                QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt(question), temperature=0.0
            )
        except LLMError as exc:
            log.warning("Sorgu yeniden yazılamadı, soru sözcükleri kullanılacak: %s", exc)
            return fallback, []
        terms = _str_list(data.get("search_terms"))
        entities = _str_list(data.get("entities"))
        if not terms and not entities:
            log.info("LLM arama terimi üretmedi; soru sözcükleri kullanılacak")
            return fallback, []
        return terms, entities

    @staticmethod
    def search_queries(question: str, terms: Sequence[str], entities: Sequence[str]) -> list[str]:
        """Sözlüksel arama sorguları; her biri ayrı ``search_records`` çağrısıdır, sonuçlar RRF ile birleştirilir.

        1. en önemli ``COMBINED_QUERY_TERMS`` arama terimi tek sorguda,
        2. her varlık adı (varlık yoksa her terim) tek başına, en fazla ``MAX_NARROW_QUERIES``,
        3. sorunun kendi sözcükleri (durak sözcükler atılmış; yeniden yazma önemli bir sözcüğü düşürdüyse emniyet).

        Tüm terimleri tek sorguda birleştirmek ES'te ters teper: ``best_fields`` + ``minimum_should_match`` alan
        başına uygulandığından 12 terimlik sorguda haberin tek bir alanda 7+ terim içermesi gerekir ve kişi adını
        taşıyan ilgili haberler elenir. Dar sorgular eşiği kolay aşar; birden çok sorguda görünen haber RRF'te öne
        çıkar. Tekrar eden ve boş sorgular atılır.
        """
        term_list = _dedupe(list(terms))
        narrow = _dedupe(list(entities)) or term_list
        candidates = [" ".join(term_list[:COMBINED_QUERY_TERMS])]
        candidates.extend(narrow[:MAX_NARROW_QUERIES])
        candidates.append(" ".join(question_tokens(question)) or flat_text(question))
        return _dedupe(candidates)

    # --- 2. geri getirme ---
    def retrieve(
        self,
        question: str,
        queries: Sequence[str],
        *,
        since: datetime | None,
        size: int,
        sources: Sequence[str] | None,
    ) -> list[RankedDoc]:
        rankings: list[Sequence[SearchHit]] = []
        for query in queries:
            hits = self.store.search_records(query, since=since, size=size, sources=sources)
            log.debug("Sözlüksel arama %r → %d sonuç", query, len(hits))
            rankings.append(hits)
        knn_hits = self._knn_hits(question, since=since, size=size, sources=sources)
        if knn_hits:
            rankings.append(knn_hits)
        return reciprocal_rank_fusion(rankings)[:size]

    def _knn_hits(
        self, question: str, *, since: datetime | None, size: int, sources: Sequence[str] | None
    ) -> list[SearchHit]:
        if not self.settings.ollama_embedding_model:
            return []
        try:
            vectors = self.llm.embed([question])
        except LLMError as exc:
            log.warning("Soru vektörü üretilemedi; yalnızca sözlüksel arama kullanılacak: %s", exc)
            return []
        vector = vectors[0] if vectors else []
        if not vector or not any(vector):
            return []
        try:
            hits = self.store.knn_search(vector, k=size, since=since)
        except Exception as exc:  # kNN isteğe bağlıdır; depo hatası sözlüksel sonuçları engellememeli
            log.warning("kNN araması başarısız; yalnızca sözlüksel sonuçlar kullanılacak: %s", exc)
            return []
        if sources:
            wanted = set(sources)
            hits = [hit for hit in hits if hit.doc.get("source") in wanted]
        log.debug("kNN araması → %d sonuç", len(hits))
        return hits

    # --- 3. bağlam ---
    def build_context(self, ordered: Sequence[RankedDoc]) -> tuple[list[str], list[Citation]]:
        """``newest_first`` ile sıralanmış belgelerden numaralı bağlam blokları ve aynı sırada atıflar üretir."""
        items = list(ordered)
        per_doc = self._content_budget(len(items))
        blocks: list[str] = []
        citations: list[Citation] = []
        for index, item in enumerate(items, 1):
            doc = item.doc
            blocks.append(build_context_block(index, doc, max_content_chars=per_doc))
            citations.append(
                Citation(
                    id=str(doc.get("id") or doc_identity(doc)),
                    title=flat_text(doc.get("title")) or "(başlıksız)",
                    content_url=str(doc.get("content_url") or ""),
                    source=str(doc.get("source") or ""),
                    published_at=parse_datetime(doc.get("published_at")),
                    score=float(item.score),
                    snippet=make_snippet(doc, item.highlights),
                )
            )
        return blocks, citations

    def _content_budget(self, count: int) -> int:
        if count <= 0:
            return CONTEXT_CONTENT_CHARS
        total = max(_MIN_CONTEXT_BUDGET, int(self.settings.ollama_num_ctx * _CHARS_PER_CTX_TOKEN))
        return min(CONTEXT_CONTENT_CHARS, max(MIN_CONTENT_CHARS, total // count))

    # --- 4. yanıt ---
    def generate_answer(
        self,
        question: str,
        blocks: Sequence[str],
        ordered: Sequence[RankedDoc],
        citations: Sequence[Citation],
        *,
        since_days: int | None,
    ) -> tuple[str, str]:
        """``(yanıt, model)``; LLM yoksa/hata verirse çıkarımsal yedek yanıt ve ``FALLBACK_MODEL``.

        ``ordered`` ve ``citations`` aynı (en yeni önce) sırada olmalıdır; atıf numaraları bu sıraya göredir.
        """
        if self._offline:
            return extractive_answer(ordered, citations), FALLBACK_MODEL
        user = build_rag_user_prompt(question, blocks, since_days=since_days)
        try:
            text = normalize_ws(self.llm.generate_text(RAG_SYSTEM_PROMPT, user))
        except LLMError as exc:
            log.warning("LLM yanıt üretemedi; haberlerden çıkarımsal yedek yanıt derlenecek: %s", exc)
            return extractive_answer(ordered, citations, reason=str(exc)), FALLBACK_MODEL
        if not text:
            log.warning("LLM boş yanıt döndürdü; çıkarımsal yedek yanıt derlenecek")
            return extractive_answer(ordered, citations), FALLBACK_MODEL
        return text, self.llm.model_name


def build_timeline(citations: Sequence[Citation]) -> list[TimelineItem]:
    """Atıflardan kronolojik (eski → yeni) zaman çizelgesi: her kaynak bir olay satırı, [n] numarasıyla."""
    items = [
        TimelineItem(
            date=c.published_at,
            event=c.title,
            source=c.source,
            content_url=c.content_url,
            citation=i,
        )
        for i, c in enumerate(citations, 1)
    ]
    dated = sorted((it for it in items if it.date is not None), key=lambda it: it.date)  # type: ignore[arg-type]
    undated = [it for it in items if it.date is None]
    return dated + undated


def newest_first(ranked: Sequence[RankedDoc]) -> list[RankedDoc]:
    """Belgeleri haber tarihine göre en yeniden en eskiye sıralar (tarihsizler sona, eşitlikte yüksek skor önce)."""
    return sorted(ranked, key=_newest_first_key)


_QUESTION_STOPWORDS = frozenset(
    "ile ve veya ama için gibi kadar göre son durum durumu nedir ne neler nasıl niye neden hangi kim kimdir "
    "mi mı mu mü midir mıdır var yok oldu olan olarak arasında arasındaki hakkında ilgili üzerine şu bu o "
    "haber haberler haberleri gelişme gelişmeler açıklama bugün dün".split()
)


def key_terms(terms: Sequence[str], entities: Sequence[str]) -> list[str]:
    """Soru terimlerinden ilgililik filtresi için anlamlı kökler (kısa/işlevsel kelimeler atılır)."""
    out: list[str] = []
    for phrase in list(entities) + list(terms):
        for word in tr_lower(phrase).replace("'", " ").split():
            if len(word) >= 4 and word not in _QUESTION_STOPWORDS and word not in out:
                out.append(word)
    return out


def filter_relevant(ranked: Sequence[RankedDoc], *, terms: Sequence[str], entities: Sequence[str]) -> list[RankedDoc]:
    """Soru terimlerinden hiçbiri başlık/alt başlık/içerikte geçmeyen belgeleri eler (Türkçe ek toleranslı).

    Özel adlar (``entities``) varsa en az bir özel ad eşleşmesi aranır; yoksa herhangi bir anlamlı terim yeter.
    Terim çıkarılamazsa sıralama olduğu gibi döner.
    """
    entity_terms = key_terms([], entities)
    all_terms = key_terms(terms, entities)
    if not all_terms:
        return list(ranked)
    matcher = KeywordMatcher(entity_terms or all_terms)
    kept: list[RankedDoc] = []
    for item in ranked:
        doc = item.doc
        text = " ".join(flat_text(doc.get(k)) for k in ("title", "subtitle", "content", "llm_summary"))
        if matcher.matches(text):
            kept.append(item)
    return kept


def extractive_answer(ordered: Sequence[RankedDoc], citations: Sequence[Citation], *, reason: str = "") -> str:
    """LLM'siz yedek yanıt: en yeni ``FALLBACK_HEADLINES`` başlık + tarih, ardından en yeni haberin özeti (atıflı)."""
    why = f" ({excerpt(reason, 120)})" if reason else ""
    lines = [f"Dil modeline erişilemediği için{why} yanıt haberlerden doğrudan derlendi (en yeniden en eskiye):"]
    for index, citation in enumerate(list(citations)[:FALLBACK_HEADLINES], 1):
        source = f" ({citation.source})" if citation.source else ""
        lines.append(f"- {format_tr(citation.published_at)} — {citation.title}{source} [{index}]")
    if ordered:
        newest = ordered[0].doc
        detail = flat_text(newest.get("llm_summary")) or flat_text(newest.get("subtitle")) or flat_text(
            newest.get("content")
        )
        if detail:
            lines.append(f"Son gelişme ({format_tr(doc_timestamp(newest))}): {excerpt(detail, 300)} [1]")
    return "\n".join(lines)


def make_snippet(
    doc: Mapping[str, Any], highlights: Mapping[str, Sequence[str]] | None = None, limit: int = 240
) -> str:
    """Atıf parçacığı: önce arama vurgusu (``<em>`` etiketleri temizlenir), yoksa LLM özeti / alt başlık / içerik."""
    for field_name in ("content", "llm_summary", "subtitle"):
        for fragment in (highlights or {}).get(field_name) or []:
            text = flat_text(_EM_TAG_RE.sub("", str(fragment)))
            if text:
                return excerpt(text, limit)
    text = flat_text(doc.get("llm_summary")) or flat_text(doc.get("subtitle")) or flat_text(doc.get("content"))
    return excerpt(text, limit) if text else ""


def _newest_first_key(item: RankedDoc) -> tuple[int, float, float]:
    ts = doc_timestamp(item.doc)
    if ts is None:
        return (1, 0.0, -item.score)
    return (0, -ts.timestamp(), -item.score)


def _str_list(value: Any, limit: int = MAX_SEARCH_TERMS) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return []
    out: list[str] = []
    for item in value:
        text = flat_text(item)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return _dedupe(out)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = tr_lower(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
