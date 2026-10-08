"""Reranker (llama.cpp / TEI), ayrı embedding sunucusu ve embedding geri doldurma."""

from __future__ import annotations

import argparse
import json
import ssl
from typing import Any

import httpx
import pytest

from scraperhryt.config import Settings
from scraperhryt.pipeline.llm import FakeOllama, OllamaClient, ollama_tls_verify
from scraperhryt.reporting import prompts
from scraperhryt.reporting.rag import QAEngine, RankedDoc
from scraperhryt.reporting.rerank import Reranker


def _reranker(api: str, handler) -> Reranker:
    s = Settings(_env_file=None, reranker_url="http://rerank.local", reranker_api=api)
    return Reranker(s, transport=httpx.MockTransport(handler))


def test_llamacpp_and_tei_responses_are_aligned_to_documents() -> None:
    def llama(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/rerank" and body["query"] == "soru" and len(body["documents"]) == 3
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.5}]})

    def tei(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank" and len(json.loads(request.content)["texts"]) == 3
        return httpx.Response(200, json=[{"index": 1, "score": 0.7}, {"index": 0, "score": 0.2}, {"index": 2, "score": 0.4}])

    assert _reranker("llamacpp", llama).scores("soru", ["a", "b", "c"]) == [0.1, 0.5, 0.9]
    assert _reranker("tei", tei).scores("soru", ["a", "b", "c"]) == [0.2, 0.7, 0.4]


def test_reranker_failure_returns_none() -> None:
    assert _reranker("llamacpp", lambda r: httpx.Response(503)).scores("soru", ["a"]) is None
    assert _reranker("llamacpp", lambda r: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1}]})).scores("s", ["a", "b"]) is None


def test_qa_engine_keeps_reranker_top_documents() -> None:
    docs = [RankedDoc(doc={"id": str(i), "title": f"haber {i}", "content": "fon"}, score=1.0 / (i + 1)) for i in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["documents"])
        return httpx.Response(200, json={"results": [{"index": i, "relevance_score": float(i)} for i in range(n)]})

    s = Settings(_env_file=None, reranker_url="http://rerank.local")
    from scraperhryt.store import InMemoryStore

    qa = QAEngine(s, InMemoryStore(), FakeOllama(), reranker=Reranker(s, transport=httpx.MockTransport(handler)))
    assert [d.doc["id"] for d in qa.rerank("soru", docs, 2)] == ["4", "3"]
    qa_off = QAEngine(s, InMemoryStore(), FakeOllama(), reranker=None)
    assert [d.doc["id"] for d in qa_off.rerank("soru", docs, 2)] == ["0", "1"]
    failing = QAEngine(s, InMemoryStore(), FakeOllama(), reranker=Reranker(s, transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert [d.doc["id"] for d in failing.rerank("soru", docs, 2)] == ["0", "1"]


def test_ask_uses_reranker_end_to_end() -> None:
    from datetime import UTC, datetime, timedelta

    from scraperhryt.models import NewsRecord
    from scraperhryt.store import InMemoryStore

    store = InMemoryStore()
    for i in range(4):
        rec = NewsRecord.new(source="hurriyet", content_url=f"https://www.hurriyet.com.tr/gundem/fon-{i}", title=f"Fon soruşturmasında gelişme {i}", content=f"Fon soruşturması kapsamında {i}. dalga operasyon yapıldı.", published_at=datetime.now(UTC) - timedelta(hours=i + 1))
        rec.matched_keywords = ["fon"]
        store.index_record(rec)
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        docs = json.loads(request.content)["documents"]
        seen.append(len(docs))
        return httpx.Response(200, json={"results": [{"index": i, "relevance_score": 1.0 if "2. dalga" in d else 0.0} for i, d in enumerate(docs)]})

    def responder(system: str, user: str) -> Any:
        if system == prompts.QUERY_REWRITE_SYSTEM_PROMPT:
            return {"search_terms": ["fon soruşturması"], "entities": []}
        return "Son gelişmede 2. dalga operasyon yapıldı [1]."

    s = Settings(_env_file=None, reranker_url="http://rerank.local", rag_top_k=1)
    qa = QAEngine(s, store, FakeOllama(responder=responder), reranker=Reranker(s, transport=httpx.MockTransport(handler)))
    answer = qa.ask("Fon soruşturmasında son durum ne?")
    assert seen and seen[0] >= 2 and len(answer.sources) == 1 and answer.sources[0].title.endswith("2")


def test_embeddings_go_to_separate_server() -> None:
    hosts: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append((request.url.host, request.url.path))
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}, "done": True})

    s = Settings(_env_file=None, ollama_base_url="https://llm.remote/llm", ollama_embedding_base_url="http://mac-mini:11434", ollama_embedding_model="bge-m3", ollama_verify_tls=False)
    with OllamaClient(s, transport=httpx.MockTransport(handler)) as client:
        client.chat_json("s", "u")
        assert client.embed(["metin"]) == [[0.1, 0.2, 0.3]]
    assert hosts == [("llm.remote", "/llm/api/chat"), ("mac-mini", "/api/embed")]


def test_embed_backfill_writes_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from scraperhryt import cli
    from scraperhryt.models import NewsRecord
    from scraperhryt.store import InMemoryStore

    store = InMemoryStore()
    rec = NewsRecord.new(source="hurriyet", content_url="https://www.hurriyet.com.tr/gundem/x-1", title="Bakan açıkladı", content="metin", published_at=datetime.now(UTC))
    store.index_record(rec)
    written: list[Any] = []
    original = store.index_record

    def spy(record, refresh=False, embedding=None):
        written.append(embedding)
        original(record, refresh=refresh, embedding=embedding)

    store.index_record = spy  # type: ignore[method-assign]
    monkeypatch.setattr(cli, "prepare_store", lambda settings, in_memory: store)

    class FakeClient:
        def __init__(self, settings: Settings) -> None: ...
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def embed(self, texts): return [[0.5, 0.5] for _ in texts]

    monkeypatch.setattr("scraperhryt.pipeline.llm.OllamaClient", FakeClient)
    s = Settings(_env_file=None, ollama_embedding_model="bge-m3")
    assert cli.cmd_embed_backfill(argparse.Namespace(since_days=0, limit=None), s) == 0
    assert written == [[0.5, 0.5]]


def test_ollama_tls_options_and_path_prefix() -> None:
    assert ollama_tls_verify(Settings(_env_file=None)) is True
    assert ollama_tls_verify(Settings(_env_file=None, ollama_verify_tls=False)) is False
    import certifi

    assert isinstance(ollama_tls_verify(Settings(_env_file=None, ollama_ca_bundle=certifi.where())), ssl.SSLContext)

    # Sunucu bir yol altında yayınlanıyorsa (https://host/llm/) istekler o yolun altına gider.
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps({"ok": 1})}, "done": True})

    s = Settings(_env_file=None, ollama_base_url="https://llm.example/llm/", ollama_model="phi4", ollama_verify_tls=False)
    with OllamaClient(s, transport=httpx.MockTransport(handler)) as client:
        assert client.chat_json("s", "u") == {"ok": 1}
    assert seen == ["/llm/api/chat"]


class _ScoreReranker:
    """Belge metnindeki ifadeye göre sabit puan veren sahte reranker (puan ölçeği bilerek ham logit gibi)."""

    def __init__(self, table: dict[str, float], default: float = -8.0) -> None:
        self.table = table
        self.default = default
        self.calls: list[int] = []

    def scores(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(len(documents))
        return [next((v for k, v in self.table.items() if k in d), self.default) for d in documents]


def _rd(i: str, title: str) -> RankedDoc:
    return RankedDoc(doc={"id": i, "title": title, "content": ""}, score=1.0)


def test_select_rescues_gate_document_only_when_reranker_rates_it_above_strict_median() -> None:
    from scraperhryt.store import InMemoryStore

    strict = [_rd("s1", "güçlü A"), _rd("s2", "orta B"), _rd("s3", "zayıf C")]
    synonym, noise = _rd("g1", "eş anlamlı D"), _rd("g2", "gürültü E")
    reranker = _ScoreReranker({"güçlü": 6.0, "orta": 2.0, "zayıf": -3.0, "eş anlamlı": 4.0, "gürültü": 0.5})
    qa = QAEngine(Settings(_env_file=None, rag_rerank_min_score=0), InMemoryStore(), FakeOllama(), reranker=reranker)
    picked = [d.doc["id"] for d in qa.select("soru", strict + [synonym, noise], strict, 10)]
    # g1 (4.0) sıkı haberlerin ortancasından (2.0) yüksek → geri alınır; g2 (0.5) düşük → elenir; sıkı haberler kalır.
    assert picked == ["s1", "g1", "s2", "s3"]
    assert [d.doc["id"] for d in qa.select("soru", strict + [synonym, noise], strict, 2)] == ["s1", "g1"]
    # reranker yoksa ya da hata verirse yalnızca sözcüksel sıkı katman
    off = QAEngine(Settings(_env_file=None), InMemoryStore(), FakeOllama(), reranker=None)
    assert [d.doc["id"] for d in off.select("soru", strict + [synonym], strict, 10)] == ["s1", "s2", "s3"]

    class Failing:
        def scores(self, query, documents):
            return None

    failing = QAEngine(Settings(_env_file=None), InMemoryStore(), FakeOllama(), reranker=Failing())
    assert [d.doc["id"] for d in failing.select("soru", strict + [synonym], strict, 10)] == ["s1", "s2", "s3"]


def test_synonym_worded_news_reaches_answer_with_reranker() -> None:
    """Soru "soruşturma ... ortaya çıkan kişi" der, haber "vurgun ... isim ... ortaya çıktı" der: sözcüksel eşik
    haberi eler, reranker onu en ilgili bulur ve haber kaynaklara girer; konu dışı haber girmez."""
    from datetime import UTC, datetime, timedelta

    from scraperhryt.models import NewsRecord
    from scraperhryt.store import InMemoryStore

    store = InMemoryStore()
    now = datetime.now(UTC)
    items = [
        ("hedef", "Fon vurgununda 2 isim daha ortaya çıktı", "Fon krizinde iki bakanın adı Meclis gündemine taşındı.", 1),
        ("s1", "Fon soruşturmasında Kaya ve eşi ifade verdi", "Fon soruşturması kapsamında iki kişi ifade verdi; yeni ayrıntılar ortaya çıktı.", 5),
        ("s2", "Fon soruşturmasında dezenformasyon uyarısı", "Fon soruşturması hakkında asılsız iddialar yayan kişiler hakkında işlem; belgeler ortaya çıktı.", 6),
        ("s3", "Fon soruşturmasında yeni inceleme", "Fon soruşturmasında üç kişinin hisse hareketleri ortaya çıktı.", 7),
        ("konu-dışı", "Maltepe'de çöken binada kaçak kat ortaya çıktı", "Çöken binada iki kişi hayatını kaybetti.", 2),
    ]
    for slug, title, content, hours in items:
        store.index_record(NewsRecord.new(source="12punto", content_url=f"https://12punto.com.tr/gundem/{slug}", title=title, content=content, published_at=now - timedelta(hours=hours)))
    reranker = _ScoreReranker({"vurgununda 2 isim": 7.0, "Kaya ve eşi": 3.0, "dezenformasyon": 1.0, "yeni inceleme": 2.0, "Maltepe": -6.0})
    s = Settings(_env_file=None, rag_top_k=3)
    answer = QAEngine(s, store, FakeOllama(responder=lambda system, user: "Yanıt [1]."), reranker=reranker).ask(
        "fon soruşturmasında en son ortaya çıkan 2 kişi kim"
    )
    titles = [c.title for c in answer.sources]
    assert "Fon vurgununda 2 isim daha ortaya çıktı" in titles
    assert not any("Maltepe" in t for t in titles)
    # aynı soru reranker olmadan: eski (yalnızca sözcüksel) davranış korunur
    plain = QAEngine(s, store, FakeOllama(responder=lambda system, user: "Yanıt [1]."), reranker=None).ask(
        "fon soruşturmasında en son ortaya çıkan 2 kişi kim"
    )
    assert "Fon vurgununda 2 isim daha ortaya çıktı" not in [c.title for c in plain.sources]


def test_select_drops_documents_below_reranker_floor_but_keeps_best_few() -> None:
    from scraperhryt.store import InMemoryStore

    strict = [_rd(f"s{i}", f"haber {i}") for i in range(5)]
    reranker = _ScoreReranker({"haber 0": 0.9, "haber 1": 0.4, "haber 2": 0.0004, "haber 3": 0.0002, "haber 4": 0.0001})
    s = Settings(_env_file=None, rag_rerank_min_score=0.001, rag_rerank_min_keep=1)
    qa = QAEngine(s, InMemoryStore(), FakeOllama(), reranker=reranker)
    assert [d.doc["id"] for d in qa.select("soru", strict, strict, 10)] == ["s0", "s1"]
    qa.settings = s.model_copy(update={"rag_rerank_min_keep": 3})
    assert [d.doc["id"] for d in qa.select("soru", strict, strict, 10)] == ["s0", "s1", "s2"]


def test_llamacpp_logits_are_converted_to_probabilities() -> None:
    def llama(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 4.0}, {"index": 1, "relevance_score": -9.0}]})

    p_hi, p_lo = _reranker("llamacpp", llama).scores("soru", ["a", "b"])
    assert 0.98 < p_hi < 0.99 and p_lo < 0.001
