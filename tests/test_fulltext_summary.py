"""Tam metin özetleme ve sayısal doğruluk: özetler başlığa değil haberin tüm içeriğine dayanır."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from scraperhryt.broker import InMemoryBroker
from scraperhryt.config import Settings
from scraperhryt.grounding import SourceFigures, extract_figures, ground_text
from scraperhryt.models import NewsRecord, Stage, utcnow
from scraperhryt.pipeline.llm import FakeOllama
from scraperhryt.pipeline.scorer import ScoringService
from scraperhryt.reporting.rag import (
    QAEngine,
    RankedDoc,
    extractive_answer,
    extractive_summary,
    focused_excerpt,
    question_signals,
)
from scraperhryt.store import InMemoryStore


def doc(title: str, subtitle: str, content: str, days: float = 0, summary: str = "") -> RankedDoc:
    return RankedDoc(
        doc={
            "id": title, "title": title, "subtitle": subtitle, "content": content, "llm_summary": summary,
            "published_at": (utcnow() - timedelta(days=days)).isoformat(), "source": "12punto",
        },
        score=1.0,
    )


KAYA = doc(
    "Türkiye'de 20 milyar dolarlık fon krizi yabancı basında",
    "Uluslararası basın fon krizini manşetlerine taşıdı.",
    "Financial Times krizin 455 bin yatırımcıyı etkilediğini yazdı. SPK 131 fonun tasfiyesine karar verdi. "
    "Haberde eski bakan Fatma Betül Sayan Kaya'nın adının soruşturma dosyasında geçtiği de hatırlatıldı. "
    "Analistler belirsizliğin süreceğini söylüyor.",
    summary="Türkiye'de 20 milyon dolarlık fon krizi yaşanıyor.",  # küçük modelin hatalı özeti
)


def test_extractive_uses_full_content_sentence_matching_the_question() -> None:
    picked = extractive_summary("Fatma Betül Sayan Kaya ile ilgili son iddialar neler?", [KAYA])
    assert picked and "Fatma Betül Sayan Kaya" in picked[0][1]  # lead değil, içerideki ilgili cümle


def test_extractive_never_uses_llm_summary_numbers() -> None:
    answer = extractive_answer([KAYA], [], question="Fon krizinde son durum ne?")
    assert "milyon" not in answer and "[1]" in answer


def test_extractive_covers_each_relevant_article() -> None:
    motorin = doc("Motorinde tabela bir kez daha değişecek", "Motorine 2,22 TL indirimin ardından salı günü 4,95 TL daha indirim bekleniyor.", "Benzinde değişiklik öngörülmüyor. Akaryakıt fiyatları dalgalı seyrediyor.", 0)
    lpg = doc("LPG'ye 24 saat içinde ikinci zam", "Otogaz fiyatları büyükşehirlerde 40 liranın üzerine çıktı.", "Akaryakıt piyasasında LPG'ye iki gün üst üste zam geldi.", 1)
    text = extractive_answer([motorin, lpg], [], question="Akaryakıt fiyatlarında son durum ne?")
    assert "[1]" in text and "[2]" in text and "4,95" in text and "40 lira" in text
    assert text[:10].replace(".", "").isdigit() and text.count("[") <= 3


def test_boilerplate_is_never_selected() -> None:
    noisy = doc("Bakan açıklama yaptı", "Haberlerimizi Google'da takip edin.", "Bakan yeni vergi düzenlemesini açıkladı. Abone olmak için tıklayın.")
    text = extractive_answer([noisy], [], question="Bakan ne açıkladı?")
    assert "Google" not in text and "Abone" not in text and "vergi" in text


def test_focused_excerpt_reaches_deep_relevant_sentences() -> None:
    filler = " ".join(f"Genel değerlendirme cümlesi numara {i} piyasa hakkında bilgi veriyor." for i in range(40))
    content = "Giriş cümlesi burada. İkinci cümle bağlam veriyor. " + filler + " Kemal Kılıçdaroğlu kurultay davası hakkında açıklama yaptı."
    signals = question_signals("Kemal Kılıçdaroğlu ne dedi?", ["Kemal Kılıçdaroğlu"])
    excerpt_text = focused_excerpt({"content": content}, signals, budget=400)
    assert "Kemal Kılıçdaroğlu kurultay davası" in excerpt_text and "Giriş cümlesi" in excerpt_text and len(excerpt_text) < 600


def test_grounding_numbers_and_magnitudes() -> None:
    sources = SourceFigures("Türkiye'de 20 milyar dolarlık fon krizi. Fon soruşturmasında beşinci dalga: 34 kişi. Motorine 2,22 TL indirim.")
    kept, dropped = ground_text("Fon krizi 20 milyon dolara ulaştı [1]. 5'inci dalgada 34 kişiye gözaltı kararı verildi [2]. Motorine 2,22 TL indirim yapıldı [1].", sources)
    assert dropped == ["Fon krizi 20 milyon dolara ulaştı [1]."]
    assert "34 kişiye" in kept and "2,22 TL" in kept
    assert [str(f) for f in extract_figures("Saat 15.00'te, 01.10.2026 tarihinde 2026 yılında 1.500 kişi")] == ["1500"]


def test_llm_answer_with_wrong_figure_is_cleaned_or_replaced() -> None:
    s = Settings(_env_file=None, rag_recency_days=30)
    store = InMemoryStore()
    rec = NewsRecord.new(source="12punto", content_url="https://12punto.com.tr/ekonomi/fon-krizi-1", title="Türkiye'de 20 milyar dolarlık fon krizi", subtitle="455 bin yatırımcı için belirsizlik sürüyor.", content="SPK 131 fonun tasfiyesine karar verdi.", published_at=utcnow())
    store.index_record(rec)

    def responder(system: str, user: str) -> Any:
        if "search_terms" in system:
            return {"search_terms": ["fon krizi"], "entities": []}
        return "Fon krizi 20 milyon dolara ulaştı [1]. SPK 131 fonun tasfiyesine karar verdi [1]."

    answer = QAEngine(s, store, FakeOllama(responder=responder)).ask("Fon krizinde son durum ne?")
    assert answer.answer == "SPK 131 fonun tasfiyesine karar verdi [1]."

    only_wrong = QAEngine(s, store, FakeOllama(responder=lambda sy, u: {"search_terms": ["fon krizi"], "entities": []} if "search_terms" in sy else "Fon krizi 20 milyon dolara ulaştı [1].")).ask("Fon krizinde son durum ne?")
    assert only_wrong.model == "fallback" and "milyon" not in only_wrong.answer


def test_alarm_summary_with_unsupported_figure_falls_back_to_article_text() -> None:
    s = Settings(_env_file=None, alarm_threshold=60)
    rec = NewsRecord.new(source="12punto", content_url="https://12punto.com.tr/ekonomi/fon-2", title="Fon soruşturmasında 20 şüpheli tutuklandı", subtitle="Toplam tutuklu sayısı 85'e çıktı.", content="Savcılık soruşturmanın süreceğini açıkladı.")
    rec.matched_keywords = ["fon"]
    rec.stage = Stage.KEYWORD
    svc = ScoringService(s, InMemoryBroker(s), FakeOllama(responder=lambda sy, u: {"alarm_score": 85, "reason": "Finansal suç soruşturması.", "summary": "Toplam tutuklu sayısı 58'e çıktı.", "confidence": 80}))
    out = svc.score_record(rec)
    assert out.is_alarm and "58" not in out.llm_summary and "85" in out.llm_summary and "LLM Özeti" in out.alarm_reason


def test_light_stem_and_inflected_question_words() -> None:
    from scraperhryt.reporting.rag import light_stem

    assert light_stem("tutuklamalarında") == "tutuk"
    assert light_stem("soruşturmasında") == "soruştur"
    assert light_stem("fiyatlarında") == "fiyat"
    assert light_stem("krizinde") == "kriz"
    assert light_stem("enflasyonu") == "enflasyon"


def test_coverage_threshold_drops_single_word_matches() -> None:
    from scraperhryt.reporting.rag import filter_relevant

    film = doc("Gazeteci Aslı Atasoy'un filmi Altın Koza'da", "Film 2 Ekim'de gösterilecek.", "Festival programı açıklandı.")
    okatan = doc("Gazeteci Derya Okatan tutuklandı", "DİSK Basın-İş duyurdu.", "Gazeteci Okatan Ankara'da gözaltına alınmıştı.")
    ergin = doc("Tutuklu gazeteci Fatih Ergin'den açıklama", "Silivri'de tutuklu bulunan gazeteci tepki gösterdi.", "")
    kept = filter_relevant([film, okatan, ergin], terms=[], entities=[], question="Gazeteci tutuklamalarında son durum ne?")
    assert [k.doc["title"] for k in kept] == ["Gazeteci Derya Okatan tutuklandı", "Tutuklu gazeteci Fatih Ergin'den açıklama"]


def test_sentence_initial_proper_noun_is_promoted_from_documents() -> None:
    from scraperhryt.reporting.rag import promote_entities

    docs = [
        {"title": "Fon soruşturmasında yeni gözaltılar", "content": "Savcılık fon soruşturmasını genişletti."},
        {"title": "Meclis açıldı", "content": "Cumhurbaşkanı Recep Tayyip Erdoğan fon skandalı hakkında konuştu. Erdoğan sorunun çözüleceğini söyledi."},
        {"title": "Avukat çift mercek altında", "content": "Fon soruşturmasında avukat çiftin kazancı dosyaya girdi."},
    ]
    assert promote_entities("Erdoğan fon soruşturması hakkında ne dedi?", docs) == ["Erdoğan"]
    assert promote_entities("Fon soruşturmasında son durum ne?", docs) == []
    everywhere = [{"title": f"Türkiye'de gelişme {i}", "content": "Türkiye ekonomisi büyüdü."} for i in range(4)]
    assert promote_entities("Türkiye'nin enflasyonu ne oldu?", everywhere) == []


def test_list_like_sentences_are_not_used_in_summary() -> None:
    lpg = doc("LPG'ye ikinci zam", "LPG'ye 4,65 lira daha zam geldi.", "Güncel otogaz fiyatları şöyle: - Ankara: 41,29 lira - İstanbul: 41,20 lira - İzmir: 41,20 lira Akaryakıt fiyatları değişiyor.")
    text = extractive_answer([lpg], [], question="Akaryakıt fiyatlarında son durum ne?")
    assert "- Ankara" not in text and "4,65" in text


def test_entity_plus_topic_question_drops_entity_only_news() -> None:
    from scraperhryt.reporting.rag import filter_relevant

    fon = doc("Meclis açıldı", "", "Cumhurbaşkanı Erdoğan fon skandalıyla ilgili sorunun üstesinden geleceklerini söyledi.")
    tugay = doc("Tugay'dan açıklama", "", "Tugay 30 Eylül'de Cumhurbaşkanı Erdoğan ile Külliye'de görüşmüştü.")
    kept = filter_relevant([tugay, fon], terms=[], entities=["Erdoğan"], question="Erdoğan fon soruşturması hakkında ne dedi?")
    assert [k.doc["title"] for k in kept] == ["Meclis açıldı"]
    only_name = filter_relevant([tugay], terms=[], entities=["Erdoğan"], question="Erdoğan fon soruşturması hakkında ne dedi?")
    assert len(only_name) == 1  # konuya değinen haber yoksa adı geçen haber yine kullanılır
