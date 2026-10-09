"""Türkçe metin yardımcıları: ek-toleranslı anahtar kelime eşleme ve regex geri izleme koruması."""

from __future__ import annotations

import time

import pytest

from scraperhryt.textutil import (
    KeywordMatcher,
    compile_keyword,
    excerpt,
    html_to_text,
    normalize_ws,
    tr_fold,
    tr_lower,
)

pytestmark = pytest.mark.timeout(10)


def test_tr_lower_and_fold() -> None:
    assert tr_lower("İSTANBUL Iğdır") == "istanbul ığdır"
    assert tr_fold("Şişli Çağlayan İĞDIR") == "sisli caglayan igdir"


@pytest.mark.parametrize(
    "word",
    [
        "bakan", "bakanlık", "bakan'ın", "bakanlarıyla", "bakanımızın", "bakanlıklarındakilerinkilerden",
        "cumhurbaşkanlığına", "cumhurbaşkanı",
    ],
)
def test_suffix_tolerant_match_accepts_turkish_inflections(word: str) -> None:
    matcher = KeywordMatcher(["bakan", "cumhurbaşkanı"])
    assert matcher.matches(f"Dün {word} açıklama yaptı.") != []


@pytest.mark.parametrize("text", ["tabakanın üstü", "mikrofonu kapattı", "fonksiyon çağrısı", "bakanxyz"])
def test_suffix_tolerant_match_rejects_embedded_or_foreign_tails(text: str) -> None:
    assert KeywordMatcher(["bakan", "fon"]).matches(text) == []


def test_fon_matches_only_real_inflections() -> None:
    matcher = KeywordMatcher(["fon"])
    assert matcher.matches("fonlarından para çekildi") == ["fon"]
    assert matcher.matches("Fon'un yöneticisi") == ["fon"]


def test_suffix_loop_is_possessive_no_catastrophic_backtracking() -> None:
    """Kök + üst üste binen eklerden oluşan uzun bir 'kelime' sonunda eşleşmeyince (x), geri izlemeli ``*`` ile
    süre her iki karakterde ~3-5 kat artardı (n=20 → dakikalar). Sahiplenici döngü sabit sürede bitmelidir."""
    pattern = compile_keyword("bakan")
    assert "*+" in pattern.pattern
    matcher = KeywordMatcher(["bakan", "fon"])
    adversarial = " ".join(
        [
            "bakan" + "ın" * 30 + "x",
            "fon" + "un" * 40 + "x",
            "#bakan" + "larının" * 20 + "q",
            "fonununununununununununx",
        ]
    )
    started = time.perf_counter()
    assert matcher.find(adversarial) == []
    assert time.perf_counter() - started < 1.0
    # Sahiplenici döngü geçerli eşleşmeleri değiştirmez.
    assert matcher.matches("bakan" + "ın" * 30) == ["bakan"]


def test_find_reports_counts_and_samples() -> None:
    hits = KeywordMatcher(["bakan"]).find("Bakan geldi. Bakanlık açıkladı. bakanların görüşü.")
    assert len(hits) == 1 and hits[0].keyword == "bakan" and hits[0].count == 3
    assert len(hits[0].samples) == 3 and "Bakan geldi" in hits[0].samples[0]


def test_exact_substring_and_regex_modes() -> None:
    assert KeywordMatcher(["=bakan"]).matches("bakanlık") == []
    assert KeywordMatcher(["=bakan"]).matches("Bakan geldi") == ["=bakan"]
    assert KeywordMatcher(["~fon"]).matches("mikrofonu") == ["~fon"]
    assert KeywordMatcher([r"re:\bfon\w*"]).matches("fonlar") == [r"re:\bfon\w*"]


def test_folded_text_matches_accentless_spelling() -> None:
    assert KeywordMatcher(["cumhurbaşkanı"]).matches("cumhurbaskani konustu") == ["cumhurbaşkanı"]


def test_html_to_text_and_helpers() -> None:
    text = html_to_text("<p>Birinci</p><script>x()</script><div>İkinci&nbsp;satır</div>")
    assert text == "Birinci\n\nİkinci satır"
    assert normalize_ws("a \t b\n\n\n\nc") == "a b\n\nc"
    assert excerpt("kelime " * 100, limit=30).endswith("…")


def test_default_keywords_cover_agenda_words_with_turkish_suffixes() -> None:
    from scraperhryt.config import Settings

    matcher = KeywordMatcher(Settings(_env_file=None).keyword_list)
    cases = {
        "AK Parti milletvekilleri önergeye ret oyu verdi": "milletvekili",
        "Kanun teklifi Meclis'te kabul edildi": "meclis",
        "Atamalar Cumhurbaşkanlığı kararnamesiyle yapıldı": "kararname",
        "Belediyedeki yolsuzluğa ilişkin iddianame hazırlandı": "yolsuzluk",
        "Yolsuzluğun boyutu ortaya çıktı": "yolsuzluk",
        "Köprü ihalesini konsorsiyum kazandı": "ihale",
    }
    for text, keyword in cases.items():
        assert keyword in matcher.matches(text), text


def test_consonant_softening_only_for_longer_keywords() -> None:
    assert KeywordMatcher(["ittifak"]).matches("İttifağın adayı açıklandı") == ["ittifak"]
    assert KeywordMatcher(["kitap"]).matches("Kitabı yayımlandı") == ["kitap"]
    # kısa kelimeler yumuşatılmaz: "at" → "ad" başka bir kelimedir
    assert KeywordMatcher(["at"]).matches("Adı açıklanmadı") == []
