"""Bağımsız incelemede doğrulanan, varsayılan ayarları etkileyen hataların regresyon testleri."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from scraperhryt.broker import InMemoryBroker
from scraperhryt.config import Settings
from scraperhryt.models import NewsRecord, utcnow
from scraperhryt.pipeline.llm import FakeOllama
from scraperhryt.reporting.rag import (
    QAEngine,
    RankedDoc,
    _first_sentence,
    _is_refusal,
    condense_answer,
    filter_relevant,
    question_entities,
    split_sentences,
)
from scraperhryt.scrapers.base import DiscoveredLink
from scraperhryt.scrapers.runner import ScrapeRunner
from scraperhryt.scrapers.state import SeenStore
from scraperhryt.store import InMemoryStore

ROOT = Path(__file__).resolve().parents[1]


def rd(title: str, content: str = "") -> RankedDoc:
    return RankedDoc(doc={"title": title, "content": content, "id": title}, score=1.0)


# 1) unvan + kısa soyad
def test_title_with_short_surname_keeps_relevant_news() -> None:
    ranked = [rd("Dışişleri Bakanı Hakan Fidan Bağdat'ta temaslarda bulundu"), rd("Fidan dikim kampanyası")]
    kept = filter_relevant(ranked, terms=[], entities=["Bakan Fidan"], question="Bakan Fidan ne dedi?")
    assert kept and kept[0].doc["title"].startswith("Dışişleri Bakanı Hakan Fidan")
    kept2 = filter_relevant([rd("Bakan Tunç: yargı paketi Meclis'te")], terms=[], entities=[], question="Bakan Tunç yargı paketi hakkında ne dedi?")
    assert len(kept2) == 1


# 2) kısaltmalar özetten cümle silmesin
def test_condense_keeps_sentences_with_abbreviations() -> None:
    text = "Av. Mehmet Kaya savunma yaptı [1]. Mahkeme Avrupa İnsan Hakları kararını dikkate aldı [2]."
    assert condense_answer(text) == text
    text2 = "Prof. Dr. Naci Görür uyardı [1]. Dr. Ayşe Yılmaz da benzer görüş bildirdi [2]."
    assert condense_answer(text2) == text2


# 3) sıra sayıları ve baş harfler cümle sonu değil
def test_first_sentence_does_not_cut_at_ordinals_or_initials() -> None:
    assert _first_sentence("Süper Lig'in 8. haftasında Galatasaray, Fenerbahçe'yi 2-1 yendi. Maç sonrası açıklama.") == "Süper Lig'in 8. haftasında Galatasaray, Fenerbahçe'yi 2-1 yendi"
    assert _first_sentence("CHP'nin 39. Olağan Kurultayı'na ilişkin dava ertelendi.") == "CHP'nin 39. Olağan Kurultayı'na ilişkin dava ertelendi"
    assert _first_sentence("Prof. Dr. Naci Görür deprem uyarısı yaptı. İkinci cümle.") == "Prof. Dr. Naci Görür deprem uyarısı yaptı"
    assert split_sentences("Toplam tutuklu sayısı 85 [1]. Soruşturma sürüyor [2].") == ["Toplam tutuklu sayısı 85 [1].", "Soruşturma sürüyor [2]."]
    assert split_sentences("Saat 15.00'te toplanacak. Karar açıklandı.") == ["Saat 15.00'te toplanacak.", "Karar açıklandı."]


# 4) kaynaklı yanıt ret sayılmaz
def test_answer_with_citation_is_not_a_refusal() -> None:
    assert not _is_refusal("Kahramanmaraş'ta 4,8 büyüklüğünde deprem meydana geldi; can kaybı olup olmadığına ilişkin bilgi bulunmuyor [1].")
    assert _is_refusal("Elimdeki haberlerde bu konuda yeterli bilgi yok.")


# 5) cümle başındaki ekli sözcük özel ad sayılmaz
def test_sentence_initial_word_with_suffix_is_not_an_entity() -> None:
    assert question_entities("Türkiye'nin enflasyonu ne oldu?") == []
    assert question_entities("Erdoğan ne dedi?") == []
    assert question_entities("Bugün Erdoğan ne dedi?") == ["Erdoğan"]
    kept = filter_relevant([rd("TÜİK eylül enflasyonunu açıkladı", "Yıllık enflasyon yüzde 30")], terms=[], entities=[], question="Türkiye'nin enflasyonu ne oldu?")
    assert len(kept) == 1


# 6) genel sözcük ("fiyat") ilgisiz haberleri içeri almaz
def test_generic_words_do_not_admit_unrelated_news() -> None:
    ranked = [
        rd("Motorinde tabela bir kez daha değişecek", "Akaryakıt fiyatlarında indirim bekleniyor"),
        rd("LPG'ye ikinci zam", "Akaryakıt piyasasında otogaz fiyatları 40 lirayı aştı"),
        rd("Tesla Model Y'nin Türkiye fiyatlarına zam geldi", "güncel fiyat listesi"),
        rd("Netflix abonelik ücretlerine zam yaptı", "yeni fiyatlar"),
    ]
    kept = filter_relevant(ranked, terms=[], entities=[], question="Akaryakıt fiyatlarında son durum ne?")
    assert [k.doc["title"] for k in kept] == ["Motorinde tabela bir kez daha değişecek", "LPG'ye ikinci zam"]


def test_fuel_question_end_to_end_cites_only_fuel_news() -> None:
    s = Settings(_env_file=None, rag_recency_days=30)
    store = InMemoryStore()
    for title, content, days in [
        ("Motorinde tabela bir kez daha değişecek", "Akaryakıt fiyatlarında 2,22 TL indirim sonrası 4,95 TL daha indirim bekleniyor.", 0),
        ("LPG'ye 24 saat içinde ikinci zam", "Akaryakıt piyasasında otogaz fiyatları 40 liranın üzerine çıktı.", 1),
        ("Tesla Model Y'nin Türkiye fiyatlarına zam geldi", "Güncel fiyat listesi açıklandı.", 0),
        ("Sigaraya zam geldi mi?", "Resmi fiyat artışı açıklanmadı.", 0),
    ]:
        rec = NewsRecord.new(source="12punto", content_url=f"https://12punto.com.tr/ekonomi/{abs(hash(title))}-1", title=title, content=content, published_at=utcnow() - timedelta(days=days))
        store.index_record(rec)

    def responder(system: str, user: str) -> Any:
        if "search_terms" in system:
            return {"search_terms": ["akaryakıt fiyatları", "zam"], "entities": []}
        return "Motorine 4,95 TL indirim bekleniyor [1]. LPG'ye ikinci zam geldi [2]."

    answer = QAEngine(s, store, FakeOllama(responder=responder)).ask("Akaryakıt fiyatlarında son durum ne?")
    assert sorted(c.title for c in answer.sources) == ["LPG'ye 24 saat içinde ikinci zam", "Motorinde tabela bir kez daha değişecek"]


# 7) Docker imajı config/ klasörünü içerir
def test_dockerfile_ships_config_directory() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY config ./config" in dockerfile
    assert (ROOT / "config" / "keyword_aliases.json").is_file()


# 8) yeni haberler bekleyen listeden önce işlenir; daha önce alınmış haber "bekleyen"e yazılmaz
class _Source:
    name = "12punto"

    def __init__(self, slugs: list[str]) -> None:
        self.slugs = slugs

    def discover(self, client, *, backfill_days=0, limit=None):
        return [DiscoveredLink(url=f"https://12punto.com.tr/gundem/{s}", title_hint=s) for s in self.slugs]

    def fetch_article(self, client, link):
        return NewsRecord.new(source="12punto", content_url=link.url, title=link.title_hint, content="içerik")


def test_fresh_links_are_processed_before_backlog(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "s.sqlite3"), max_articles_per_run=2)
    seen = SeenStore(settings.state_db_path)
    source = _Source([f"eski-haber-{i}" for i in range(1, 7)])
    runner = ScrapeRunner(settings, InMemoryBroker(settings), seen_store=seen, sources=[source])
    runner.run_once()
    assert seen.pending_count() == 4
    source.slugs = ["taze-haber-1"]  # eski bağlantılar beslemeden düştü; yalnızca bekleyen listede duruyorlar
    runner.run_once()
    # Bütçe 2: eski sıralamada bekleyen eski-haber-3/4 bütçeyi tüketir, taze haber ertelenirdi.
    fresh_id = NewsRecord.new(source="12punto", content_url="https://12punto.com.tr/gundem/taze-haber-1", title="x", content="").id
    assert seen.get(fresh_id) is not None


def test_already_ingested_links_are_not_added_to_pending(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, state_db_path=str(tmp_path / "s.sqlite3"), max_articles_per_run=10)
    seen = SeenStore(settings.state_db_path)
    source = _Source([f"haber-{i}" for i in range(1, 6)])
    runner = ScrapeRunner(settings, InMemoryBroker(settings), seen_store=seen, sources=[source])
    runner.run_once()
    assert seen.pending_count() == 0
    with seen._lock:  # tüm haberleri yeniden kontrol zamanı gelmiş gibi göster
        seen._conn.execute("UPDATE seen SET last_seen = ?", ((utcnow() - timedelta(hours=3)).isoformat(),))
    runner.settings = Settings(_env_file=None, state_db_path=settings.state_db_path, max_articles_per_run=2)
    runner.run_once()
    assert seen.pending_count() == 0
