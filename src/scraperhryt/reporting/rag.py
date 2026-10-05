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
from ..grounding import SourceFigures, ground_text
from ..models import Answer, Citation, TimelineItem, utcnow
from ..pipeline.llm import LLM, HeuristicLLM, LLMError
from ..store import ArticleStore, SearchHit
from ..textutil import KeywordMatcher, excerpt, normalize_ws, split_sentences, tr_lower
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
        # Görüntülenen arama terimleri: soruda karşılığı olmayan (uydurma olabilecek) varlıklar gösterilmez.
        search_terms = _dedupe(terms + ground_entities(question, entities))[:MAX_SEARCH_TERMS]
        queries = self.search_queries(question, terms, entities)
        # Süzgeç sonrası top_k dolu kalsın diye iki katı aday getirilir (yeniden yazma terimleri gürültü ekleyebilir).
        ranked = self.retrieve(question, queries, since=since, size=size * 2, sources=source_filter)
        promoted = promote_entities(question, [item.doc for item in ranked])
        if promoted:
            log.info("Cümle başındaki özel ad(lar) haberlerden tanındı: %s", promoted)
            entities = _dedupe(list(entities) + promoted)
        relevant = filter_relevant(ranked, terms=terms, entities=entities, question=question)[:size]
        if len(relevant) < min(size, len(ranked)):
            log.info("İlgisiz belgeler elendi: %d adaydan %d ilgili haber kaldı", len(ranked), len(relevant))
        ordered = newest_first(relevant)
        signals = question_signals(question, entities)
        blocks, citations = self.build_context(ordered, signals)
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
            answer_text, model = self.generate_answer(
                question, blocks, ordered, citations, since_days=days, entities=entities
            )

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
                QUERY_REWRITE_SYSTEM_PROMPT,
                build_query_rewrite_prompt(question),
                temperature=0.0,
                timeout=self.settings.rag_rewrite_timeout,
            )
        except LLMError as exc:
            log.warning("Sorgu yeniden yazılamadı, soru sözcükleri kullanılacak: %s", exc)
            return fallback, []
        terms = _str_list(data.get("search_terms"))
        entities = _str_list(data.get("entities"))
        # Soruda geçmeyen varlıklar (bağlamsal "CHP" ya da uydurma "Türkiye Finans Kurumu") yalnızca geri getirmede
        # kullanılır; ilgililik süzgeci sorunun kendisine dayanır (filter_relevant).
        ungrounded = [e for e in entities if e not in ground_entities(question, entities)]
        if ungrounded:
            log.info("Soruda geçmeyen varlık adları yalnızca aramada kullanılacak: %s", ungrounded)
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
    def build_context(
        self, ordered: Sequence[RankedDoc], signals: QuestionSignals | None = None
    ) -> tuple[list[str], list[Citation]]:
        """``newest_first`` ile sıralanmış belgelerden numaralı bağlam blokları ve aynı sırada atıflar üretir.

        Her haberin içeriği baştan kesilmez: tüm metin taranır ve soruyla en ilgili cümleler (ilk iki cümle
        bağlam olarak) bütçe içinde modele verilir. Skorlama aşamasındaki LLM özetleri bağlama konmaz; model
        yalnızca gazetecinin yazdığı metni görür.
        """
        items = list(ordered)
        per_doc = self._content_budget(len(items))
        signals = signals or QuestionSignals()
        blocks: list[str] = []
        citations: list[Citation] = []
        for index, item in enumerate(items, 1):
            doc = item.doc
            context_doc = {k: v for k, v in doc.items() if k != "llm_summary"}
            context_doc["content"] = focused_excerpt(doc, signals, per_doc)
            blocks.append(build_context_block(index, context_doc, max_content_chars=per_doc))
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
        entities: Sequence[str] = (),
    ) -> tuple[str, str]:
        """``(yanıt, model)``; LLM yoksa/hata verirse çıkarımsal yedek yanıt ve ``FALLBACK_MODEL``.

        ``ordered`` ve ``citations`` aynı (en yeni önce) sırada olmalıdır; atıf numaraları bu sıraya göredir.
        """
        def fallback(reason: str = "") -> tuple[str, str]:
            return (
                extractive_answer(ordered, citations, reason=reason, question=question, entities=entities),
                FALLBACK_MODEL,
            )

        if self._offline:
            return fallback()
        user = build_rag_user_prompt(question, blocks, since_days=since_days)
        try:
            text = normalize_ws(self.llm.generate_text(RAG_SYSTEM_PROMPT, user, timeout=self.settings.rag_answer_timeout))
        except LLMError as exc:
            log.warning("LLM yanıt üretemedi; haberlerden çıkarımsal yedek yanıt derlenecek: %s", exc)
            return fallback(str(exc))
        if not text:
            log.warning("LLM boş yanıt döndürdü; çıkarımsal yedek yanıt derlenecek")
            return fallback()
        text = _strip_leading_refusal(text)
        if _looks_like_listing(text) and citations:
            log.info("Model özet yerine haber listesi üretti; kısa çıkarımsal özet verilecek")
            return fallback("model özet yerine haber listesi üretti")
        if _is_refusal(text) and citations:
            # Belgeler ilgililik süzgecinden geçti (soru terimleri metinde var) ama küçük model sentezleyemedi:
            # soruyu yanıtsız bırakmak yerine en yeni ilgili haberlerden çıkarımsal özet ver.
            log.info("Model 'yeterli bilgi yok' dedi ama %d ilgili haber var; çıkarımsal özet verilecek", len(citations))
            return fallback("model ilgili haberleri sentezleyemedi")
        text = condense_answer(text)
        # Sayısal doğruluk: yanıttaki her sayı (ve bin/milyon/milyar eki) kaynak haberlerde geçmeli.
        sources = SourceFigures(
            question,
            format_tr(utcnow()),
            *(
                " ".join(
                    [flat_text(item.doc.get(k)) for k in ("title", "subtitle", "content")]
                    + [format_tr(doc_timestamp(item.doc))]
                )
                for item in ordered
            ),
        )
        grounded, dropped = ground_text(text, sources)
        if dropped:
            log.warning("Kaynakta karşılığı olmayan sayı içeren %d cümle yanıttan çıkarıldı: %s", len(dropped), dropped)
        if not grounded.strip():
            return fallback("modelin yanıtındaki sayılar kaynaklarla uyuşmadı")
        return grounded, self.llm.model_name


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


def _question_words(question: str) -> list[tuple[str, bool]]:
    """Sorudaki anlamlı sözcükler: (Türkçe küçük harf, kısaltma mı). Durak sözcükler ve 1-2 harfliler atılır;
    büyük harfli kısaltmalar (TFF, MHK, SPK) 2+ harf olsa da tutulur."""
    out: list[tuple[str, bool]] = []
    for raw in re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+", question or ""):
        word = tr_lower(raw)
        acronym = len(raw) >= 2 and raw.isupper()
        if word in _QUESTION_STOPWORDS or (len(word) < 3 and not acronym):
            continue
        if all(word != w for w, _ in out):
            out.append((word, acronym))
    return out


_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+(?:['’][A-Za-zÇĞİÖŞÜçğıöşü]+)?")


def question_entities(question: str) -> list[str]:
    """Sorudaki özel adlar: kısaltmalar (TFF, MHK) ve büyük harfle başlayan ardışık sözcük grupları
    ("Özgür Özel", "Kemal Kılıçdaroğlu"). Cümle başındaki tek büyük harfli sözcük ("Fon ...") sayılmaz."""
    tokens = _WORD_RE.findall(question or "")
    out: list[str] = []
    group: list[str] = []
    group_start = [-1]

    def flush() -> None:
        # Cümle başındaki tek sözcük ("Fon ...", "Türkiye'nin ...") büyük harfle yazıldığı için özel ad sayılmaz.
        if group and (len(group) > 1 or group_start[0] != 0):
            out.append(" ".join(group))
        group.clear()

    for index, raw in enumerate(tokens):
        base = re.split(r"['’]", raw)[0]
        if len(base) >= 2 and base.isupper():
            flush()
            out.append(base)
            continue
        is_cap = base[:1].isupper() and tr_lower(base) not in _QUESTION_STOPWORDS
        if is_cap:
            if not group:
                group_start[0] = index
            group.append(base)
        else:
            flush()
    flush()
    # tek sözcüklük cümle başı grubu (ör. "Fon") elendi; aynı sözcüğü tekrarlama
    return [e for i, e in enumerate(out) if e not in out[:i]]


def ground_entities(question: str, entities: Sequence[str]) -> list[str]:
    """Yalnızca sözcükleri soruda geçen varlık adlarını tutar (ek toleranslı: "Kılıçdaroğlu" ↔ "Kılıçdaroğlu'nun")."""
    qwords = [w for w, _ in _question_words(question)]

    def in_question(word: str) -> bool:
        return any(q == word or q.startswith(word) or word.startswith(q) for q in qwords if min(len(q), len(word)) >= 3)

    kept: list[str] = []
    for ent in entities:
        words = [tr_lower(w) for w in re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+", ent)]
        words = [w for w in words if w not in _QUESTION_STOPWORDS]
        if words and all(in_question(w) for w in words):
            kept.append(ent)
    return kept


# Tek başına konu belirtmeyen genel haber sözcüklerinin kökleri: soruda daha belirgin bir sözcük varsa
# süzgeç bunlara dayanmaz ("Akaryakıt fiyatlarında..." → yalnızca "akaryakıt"; "fiyat" Tesla zammını da tutar).
_GENERIC_STEMS = (
    "fiyat", "zam", "ücret", "indirim", "artış", "düşüş", "açıkla", "dedi", "söyle", "karar", "gelişme",
    "durum", "yeni", "konu", "olay", "son", "sonuç", "süreç", "tepki", "iddia", "yorum", "değerlendir",
    "neler", "nasıl", "ne ", "oldu", "olacak", "yapıl", "çıktı", "geldi", "gündem",
    # Betimleyici sözcükler: gazeteci olayı her zaman bu sözcükle anmaz ("TFF ile MHK krizi" haberinde "kriz"
    # geçmeyebilir), bu yüzden süzgeçte zorunlu tutulmaz.
    "kriz", "tartışma", "gerilim", "polemik", "skandal", "sorun", "kavga", "çatışma", "anlaşmazlık",
)
# Unvanlar: "Bakan Fidan", "Başkan Özel" gibi sorularda haber "Dışişleri Bakanı Hakan Fidan" der; ad tek başına aranır.
_TITLE_WORDS = frozenset(
    "bakan bakanı başkan başkanı cumhurbaşkanı cumhurbaşkan genel sayın eski milletvekili vekili vali valisi "
    "belediye başkanvekili lider lideri rektör rektörü prof dr av doç büyükelçi sözcü sözcüsü".split()
)


def _is_generic(word: str) -> bool:
    return any(word.startswith(stem.strip()) for stem in _GENERIC_STEMS)


def _question_matcher(question: str) -> KeywordMatcher | None:
    patterns: list[str] = []
    words = _question_words(question)
    specific = [(w, a) for w, a in words if a or not _is_generic(w)]
    for word, acronym in specific or words:
        patterns.extend(_word_patterns(word, acronym))  # kısaltma: kök + ek; sözcük: kök + çekimler
    return KeywordMatcher(patterns) if patterns else None


def promote_entities(question: str, docs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Cümle başındaki tek büyük harfli sözcük ("Erdoğan ne dedi?") özel ad mı, haberlere bakarak karar verir.

    Sözcük haberlerde cümle ortasında hep büyük harfle geçiyorsa (hiç küçük harfle geçmiyorsa) ve haberlerin
    yarısından azında bulunuyorsa (ayırt ediciyse) özel ad sayılır. "Fon ..." (haberde "fon" küçük) ya da
    "Türkiye'nin ..." (hemen her haberde geçer, ayırt edici değil) özel ad sayılmaz.
    """
    tokens = _WORD_RE.findall(question or "")
    if not tokens or not docs:
        return []
    first = re.split(r"['’]", tokens[0])[0]
    low = tr_lower(first)
    if len(first) < 4 or not first[:1].isupper() or low in _QUESTION_STOPWORDS or _is_generic(low):
        return []
    if any(first in e for e in question_entities(question)):
        return []
    capital = re.compile(rf"\b{re.escape(first)}", re.UNICODE)
    lower = re.compile(rf"\b{re.escape(low)}", re.UNICODE)
    docs_with = 0
    capital_hits = 0
    for doc in docs:
        text = " ".join(flat_text(doc.get(k)) for k in ("title", "subtitle", "content"))
        if lower.search(text):
            return []  # sözcük küçük harfle de geçiyor: cins isim
        hits = len(capital.findall(text))
        if hits:
            docs_with += 1
            capital_hits += hits
    if capital_hits >= 2 and docs_with <= max(1, len(docs) // 2):
        return [first]
    return []


def filter_relevant(
    ranked: Sequence[RankedDoc], *, terms: Sequence[str], entities: Sequence[str], question: str = ""
) -> list[RankedDoc]:
    """Soruyla ilgisiz belgeleri eler (Türkçe ek toleranslı). Ölçüt yalnızca SORUNUN KENDİSİNE dayanır:

    - soruda geçen varlık adları (``ground_entities``) varsa en az biri bütün ifade olarak geçmeli
      (çok kelimeli adın ≥6 harfli soyadı tek başına da yeter);
    - yoksa sorunun anlamlı sözcüklerinden (kısaltmalar dahil) en az biri geçmeli.

    LLM'in ürettiği arama terimleri geri getirmede kullanılır ama süzgeçte kullanılmaz: uydurma bir açılım
    ("Türkiye Finans Kurumu") ilgili haberleri elememeli. ``question`` verilmezse ``terms`` yedek olarak kullanılır.
    """
    grounded = ground_entities(question, entities) if question else list(entities)
    if question and not grounded:
        grounded = question_entities(question)
    phrases: list[str] = []
    for ent in grounded:
        words = [w for w in tr_lower(ent).replace("'", " ").replace("’", " ").split() if w not in _QUESTION_STOPWORDS]
        if not words:
            continue
        names = [w for w in words if w not in _TITLE_WORDS]
        if names and len(names) < len(words):
            # "Bakan Fidan" → "fidan": haberde unvan farklı çekimle ve araya ad girerek geçer.
            phrases.append(" ".join(names))
            continue
        phrases.append(" ".join(words))
        if len(words) > 1 and len(words[-1]) >= 6:
            phrases.append(words[-1])
    if phrases:
        matcher: KeywordMatcher | None = KeywordMatcher(phrases)
    elif question:
        matcher = _question_matcher(question)
    else:
        fallback_terms = key_terms(terms, [])
        matcher = KeywordMatcher(fallback_terms) if fallback_terms else None
    if matcher is None:
        return list(ranked)
    kept: list[RankedDoc] = []
    for item in ranked:
        doc = item.doc
        text = " ".join(flat_text(doc.get(k)) for k in ("title", "subtitle", "content", "llm_summary"))
        if matcher.matches(text):
            kept.append(item)
    if not question or len(kept) < 2:
        return kept
    if phrases:
        # Özel ad sorularında adlar birbirinin alternatifidir (TFF ya da MHK geçen haber ilgilidir). Soru adın
        # yanında bir konu da soruyorsa ("Erdoğan FON SORUŞTURMASI hakkında ne dedi?"), konuya da değinen
        # haberler varken yalnızca adı geçen ilgisiz haberler (aynı kişinin başka bir görüşmesi) elenir.
        topic = question_signals(question, grounded)
        if not topic.specific:
            return kept
        on_topic = [
            item for item in kept
            if any(m.matches(" ".join(flat_text(item.doc.get(k)) for k in ("title", "subtitle", "content")))
                   for m in topic.specific)
        ]
        return on_topic or kept
    # Kapsama eşiği: sorunun birden çok belirgin öğesi varsa, en iyi kapsayan habere göre çok az öğe içeren
    # haberler elenir ("Gazeteci tutuklamaları" → yalnız "gazeteci" geçen film haberi düşer).
    signals = question_signals(question, grounded)
    coverage = [doc_relevance(item.doc, signals) for item in kept]
    best = max(coverage)
    if best <= 0:
        return kept
    return [item for item, cov in zip(kept, coverage, strict=True) if cov >= 0.6 * best]


def doc_relevance(doc: Mapping[str, Any], signals: QuestionSignals) -> float:
    """Haberin (başlık + alt başlık + tüm içerik) soruyu kapsama puanı; her soru öğesi bir kez sayılır."""
    return signals.score(" ".join(flat_text(doc.get(k)) for k in ("title", "subtitle", "content")))



# --- Soru sinyalleri ve cümle puanlama (tüm haber metni üzerinde) --------------------------------------------

_BOILERPLATE_RE = re.compile(
    r"google.{0,20}(takip|tercih)|abone ol|haberin devamı|haberlerimizi|bizi takip|fotoğraf\s*:|kaynak\s*:|"
    r"algoritmaya bırakma|whatsapp kanal|tıklayın|reklam",
    re.IGNORECASE,
)
_MIN_SENTENCE_CHARS = 30
_MAX_SENTENCE_CHARS = 360


@dataclass
class QuestionSignals:
    """Sorudan türetilen eşleştiriciler: özel adlar (ağırlık 3), belirgin sözcükler (2), genel sözcükler (0.5)."""

    entities: list[KeywordMatcher] = field(default_factory=list)
    specific: list[KeywordMatcher] = field(default_factory=list)
    generic: list[KeywordMatcher] = field(default_factory=list)

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        total = 3.0 * sum(1 for m in self.entities if m.matches(text))
        total += 2.0 * sum(1 for m in self.specific if m.matches(text))
        total += 0.5 * sum(1 for m in self.generic if m.matches(text))
        return total

    @property
    def empty(self) -> bool:
        return not (self.entities or self.specific or self.generic)


# Hafif Türkçe kök bulucu için ekler (uzundan kısaya); kök en az 4 harf kalır.
_STEM_SUFFIXES = (
    "ndaki", "ndeki", "ndan", "nden", "ları", "leri", "nın", "nin", "nun", "nün", "nda", "nde",
    "dan", "den", "tan", "ten", "lar", "ler", "da", "de", "ta", "te", "ın", "in", "un", "ün",
    "sı", "si", "su", "sü", "ma", "me", "la", "le", "ı", "i", "u", "ü",
)


def light_stem(word: str) -> str:
    """"tutuklamalarında" → "tutuk", "soruşturmasında" → "soruştur", "fiyatlarında" → "fiyat", "krizinde" → "kriz"."""
    stem = word
    changed = True
    while changed:
        changed = False
        for suffix in _STEM_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def _word_patterns(word: str, acronym: bool) -> list[str]:
    patterns = [word]
    if acronym:
        return patterns
    stem = light_stem(word)
    if stem != word:
        # ≥5 harfli kök alt dize olarak aranır ("tutuk" → tutuklu/tutuklandı); 4 harfli kök ek toleranslı kök olarak.
        patterns.append("~" + stem if len(stem) >= 5 else stem)
    return patterns


def question_signals(question: str, entities: Sequence[str] = ()) -> QuestionSignals:
    signals = QuestionSignals()
    grounded = ground_entities(question, entities) or question_entities(question)
    entity_words: set[str] = set()
    for ent in grounded:
        words = [w for w in tr_lower(ent).replace("'", " ").replace("’", " ").split() if w not in _QUESTION_STOPWORDS]
        names = [w for w in words if w not in _TITLE_WORDS] or words
        if not names:
            continue
        entity_words.update(names)
        alternatives = [" ".join(names)]
        if len(names) > 1 and len(names[-1]) >= 6:
            alternatives.append(names[-1])
        signals.entities.append(KeywordMatcher(alternatives))
    for word, acronym in _question_words(question):
        if word in entity_words:
            continue
        matcher = KeywordMatcher(_word_patterns(word, acronym))
        (signals.generic if (not acronym and _is_generic(word)) else signals.specific).append(matcher)
    return signals


def doc_sentences(doc: Mapping[str, Any]) -> list[str]:
    """Haberin kendi metninden (alt başlık + içerik; LLM özeti DEĞİL) temizlenmiş cümleler, metin sırasıyla."""
    title_key = tr_lower(flat_text(doc.get("title")))[:80]
    out: list[str] = []
    seen: set[str] = set()
    for field_name in ("subtitle", "content"):
        for sentence in split_sentences(flat_text(doc.get(field_name))):
            if len(sentence) < _MIN_SENTENCE_CHARS or _BOILERPLATE_RE.search(sentence):
                continue
            if sentence.count(" - ") >= 2 or len(sentence) > _MAX_SENTENCE_CHARS:
                continue  # madde listesi ("- Ankara: 41,29 lira - İzmir: ...") ya da bölünemeyen uzun blok
            key = tr_lower(sentence)[:80]
            if key in seen or key == title_key:
                continue
            seen.add(key)
            out.append(sentence)
    return out


def focused_excerpt(doc: Mapping[str, Any], signals: QuestionSignals, budget: int) -> str:
    """Uzun haberden soruya en ilgili kısım: ilk iki cümle (bağlam) + soruyla eşleşen en güçlü cümleler,
    metindeki sırasıyla, ``budget`` karakteri aşmadan. Haberin yalnızca başı değil tamamı taranır."""
    sentences = [s for s in split_sentences(flat_text(doc.get("content"))) if not _BOILERPLATE_RE.search(s)]
    if not sentences:
        return ""
    if sum(len(s) + 1 for s in sentences) <= budget:
        return " ".join(sentences)
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: (-(signals.score(sentences[i]) + (1.5 if i < 2 else 0.0)), i),
    )
    chosen: set[int] = set()
    used = 0
    for i in ranked:
        cost = len(sentences[i]) + 1
        if used + cost > budget and chosen:
            continue
        chosen.add(i)
        used += cost
        if used >= budget:
            break
    parts: list[str] = []
    previous = -1
    for i in sorted(chosen):
        if previous >= 0 and i != previous + 1:
            parts.append("…")
        parts.append(sentences[i])
        previous = i
    return " ".join(parts)


def _token_set(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(tr_lower(text)) if len(t) > 2}


def _near_duplicate(a: str, b: str) -> bool:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6


def extractive_summary(
    question: str,
    ordered: Sequence[RankedDoc],
    *,
    entities: Sequence[str] = (),
    max_sentences: int = FALLBACK_HEADLINES,
) -> list[tuple[int, str]]:
    """Haberlerin TAM metninden (alt başlık + içerik) soruya en uygun cümleleri seçer: ``[(atıf no, cümle)]``.

    Habere göre iki kip:
    - **Haber konunun kendisi** (soru bir kişi/kurum sormuyor ya da adı başlıkta geçiyor): haber yazımında en
      önemli bilgi başta verilir (ters piramit), bu yüzden alt başlık/ilk cümle öne çıkar; soru sözcükleri ve
      sayısal bilgi ek puan getirir.
    - **Ad yalnızca metnin içinde geçiyor** (ör. Kaya'nın adı fon krizi haberinin ortasında): yalnızca o adı
      içeren cümleler seçilir, haberin genel girişi değil.

    Önce her habere bir cümle düşer (farklı gelişmeler kapsansın), sonra kalan yer en iyi cümlelerle dolar;
    birbirinin tekrarı olan cümleler atlanır. LLM özetleri kullanılmaz.
    """
    signals = question_signals(question, entities)
    candidates: list[tuple[float, int, int, str]] = []
    for doc_index, item in enumerate(ordered):
        sentences = doc_sentences(item.doc)
        title = flat_text(item.doc.get("title"))
        recency = max(0.0, 1.0 - 0.15 * doc_index)
        about_entity = not signals.entities or any(m.matches(title) for m in signals.entities)
        for sent_index, sentence in enumerate(sentences):
            relevance = signals.score(sentence)
            density = 0.5 if re.search(r"\d", sentence) else 0.0
            if about_entity:
                lead = 3.0 if sent_index == 0 else 1.5 if sent_index == 1 else 0.0
                score = lead + relevance + recency + density
            else:
                if not any(m.matches(sentence) for m in signals.entities):
                    continue
                score = relevance + recency + density + (0.5 if sent_index < 2 else 0.0)
            candidates.append((score, doc_index, sent_index, sentence))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    picks: list[tuple[float, int, int, str]] = []
    covered: set[int] = set()
    for cand in candidates:  # 1. tur: her habere bir cümle
        if cand[1] in covered or any(_near_duplicate(cand[3], p[3]) for p in picks):
            continue
        picks.append(cand)
        covered.add(cand[1])
        if len(picks) >= max_sentences:
            break
    for cand in candidates:  # 2. tur: boş kalan yerler
        if len(picks) >= max_sentences:
            break
        if cand in picks or any(_near_duplicate(cand[3], p[3]) for p in picks):
            continue
        picks.append(cand)
    picks.sort(key=lambda c: (c[1], c[2]))
    return [(doc_index + 1, sentence) for _score, doc_index, _si, sentence in picks]


_REFUSAL_PREFIX_RE = re.compile(r"^\s*elimdeki haberlerde bu konuda yeterli bilgi yok\.?\s*", re.IGNORECASE)


_REFUSAL_ANY_RE = re.compile(
    r"(?:sonuç\s*:\s*)?elimdeki haberlerde bu konuda yeterli bilgi yok\.?(?:\s*\([^)]*\))?", re.IGNORECASE
)
_LIST_LINE_RE = re.compile(r"^\s*(?:\[\d+\]|[-*•]|\d+[.)])\s+", re.MULTILINE)


def _strip_leading_refusal(text: str) -> str:
    """Küçük modeller bazen 'yeterli bilgi yok' cümlesini özetin başına ya da sonuna ekler; içerik varsa atılır."""
    stripped = _REFUSAL_ANY_RE.sub("", text).strip()
    return stripped if stripped and stripped != text and len(stripped) > 40 else text


MAX_ANSWER_SENTENCES = 4
_LABEL_RE = re.compile(
    r"^\s*(?:özet|sonuç|en güncel gelişme(?:ler)?|yanıt|cevap)\s*:\s*", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\s*\[\d+\](?:\[\d+\])*\.?)?\s+|\n+")


def condense_answer(text: str, max_sentences: int = MAX_ANSWER_SENTENCES) -> str:
    """Model yanıtını kısa tek paragrafa indirger: "Özet:" gibi etiketleri atar, tekrarlanan cümleleri eler,
    en fazla ``max_sentences`` cümle tutar (atıf numaraları cümleyle birlikte kalır)."""
    pieces: list[str] = []
    for raw in split_sentences(text):
        sentence = _LABEL_RE.sub("", raw).strip()
        if len(sentence) < 3:
            continue
        pieces.append(sentence)
    kept: list[str] = []
    seen: list[str] = []

    def is_repeat(key: str, other: str) -> bool:
        if key[:60] == other[:60]:
            return True
        shorter, longer = (key, other) if len(key) <= len(other) else (other, key)
        return len(shorter) >= 25 and shorter in longer  # kısa parçalar ("av", "dr") tekrar sayılmaz

    for sentence in pieces:
        key = tr_lower(re.sub(r"\[\d+\]", "", sentence))
        key = re.sub(r"\W+", " ", key).strip()
        if not key or any(is_repeat(key, other) for other in seen):
            continue
        seen.append(key)
        kept.append(sentence)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept) if kept else text.strip()


def _looks_like_listing(text: str) -> bool:
    """Yanıt 2-4 cümlelik özet yerine haber haber liste mi (3+ madde/numaralı satır)?"""
    return len(_LIST_LINE_RE.findall(text)) >= 3


def _is_refusal(text: str) -> bool:
    """Yanıt bir ret mi? Kaynak numarası ([1]) taşıyan yanıt haberlere dayanıyordur, ret sayılmaz
    ("...can kaybına ilişkin bilgi bulunmuyor [1]." geçerli bir yanıttır)."""
    low = tr_lower(text).strip()
    if len(low) >= 220 or re.search(r"\[\d+\]", low):
        return False
    return "yeterli bilgi yok" in low or low.startswith(("bu konuda bilgi bulunmuyor", "bu konuda bilgi bulunmamaktadır"))


def _first_sentence(text: str, limit: int = 220) -> str:
    sentences = split_sentences(flat_text(text))
    if not sentences:
        return ""
    return excerpt(sentences[0], limit).rstrip(".…")


def extractive_answer(
    ordered: Sequence[RankedDoc],
    citations: Sequence[Citation],
    *,
    reason: str = "",
    question: str = "",
    entities: Sequence[str] = (),
) -> str:
    """LLM'siz yedek yanıt: ilgili haberlerin TAM metninden seçilen 2-3 cümlelik atıflı özet."""
    items = list(ordered)
    if not items:
        return INSUFFICIENT_EVIDENCE_TEXT
    picked = extractive_summary(question, items, entities=entities)
    if not picked:
        return INSUFFICIENT_EVIDENCE_TEXT
    parts: list[str] = []
    for position, (number, sentence) in enumerate(picked):
        body = sentence.rstrip(" .…!?")
        if position == 0:
            stamp = format_tr(doc_timestamp(items[number - 1].doc))
            parts.append(f"Son gelişme ({stamp}): {body} [{number}].")
        else:
            parts.append(f"{body} [{number}].")
    text = " ".join(parts)
    if reason:
        text += f" (Bu özet haberlerin metninden doğrudan derlendi: {excerpt(reason, 100)}.)"
    return text


def make_snippet(
    doc: Mapping[str, Any], highlights: Mapping[str, Sequence[str]] | None = None, limit: int = 240
) -> str:
    """Atıf parçacığı: önce haber metnindeki arama vurgusu (``<em>`` etiketleri temizlenir), yoksa içeriğin başı.

    Yalnızca gazetecinin metni kullanılır; skorlama aşamasındaki LLM özeti hatalı sayı içerebileceği için
    parçacığa girmez. Alt başlık kartta ayrıca gösterildiğinden yedek olarak en sona kalır.
    """
    for field_name in ("content", "subtitle"):
        for fragment in (highlights or {}).get(field_name) or []:
            text = flat_text(_EM_TAG_RE.sub("", str(fragment)))
            if text:
                return excerpt(text, limit)
    text = flat_text(doc.get("content")) or flat_text(doc.get("subtitle"))
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
