"""Türkçe metin yardımcıları: küçük harfe çevirme, aksan katlama, HTML→metin ve ek-toleranslı anahtar kelime eşleme."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape

from bs4 import BeautifulSoup

_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
_FOLD_MAP = str.maketrans(
    {
        "ş": "s", "ğ": "g", "ı": "i", "ö": "o", "ü": "u", "ç": "c", "â": "a", "î": "i", "û": "u",
        "Ş": "s", "Ğ": "g", "İ": "i", "Ö": "o", "Ü": "u", "Ç": "c", "Â": "a", "Î": "i", "Û": "u", "I": "i",
    }
)

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def tr_lower(text: str) -> str:
    """Türkçe'ye uygun küçük harf (İ→i, I→ı); Python'un varsayılan lower() davranışındaki 'i̇' sorununu önler."""
    return (text or "").translate(_TR_LOWER_MAP).lower()


def tr_fold(text: str) -> str:
    """Aksanları ASCII'ye katlar ve küçük harfe çevirir (slug/aksansız metin karşılaştırmaları için)."""
    return (text or "").translate(_FOLD_MAP).lower()


def normalize_ws(text: str) -> str:
    text = unescape(text or "").replace("\xa0", " ")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NL_RE.sub("\n\n", text).strip()


_BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "br", "tr", "section", "article")
_DROP_TAGS = ("script", "style", "noscript", "iframe", "svg", "figure", "figcaption", "button", "form", "ins", "aside", "nav", "footer", "header")


def html_to_text(html: str) -> str:
    """HTML parçasını paragraf sınırlarını koruyarak düz metne çevirir."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return normalize_ws(soup.get_text(" "))


# --- Türkçe ek-toleranslı anahtar kelime eşleme -------------------------------------------------------------

# Yaygın çekim/yapım ekleri (kök sonrasında sıfır veya daha fazla kez tekrarlanabilir).
_SUFFIXES = (
    "lar|ler|lık|lik|luk|lük|lığ|liğ|luğ|lüğ|ları|leri|"
    "ndan|nden|dan|den|tan|ten|nda|nde|da|de|ta|te|"
    "nın|nin|nun|nün|ın|in|un|ün|yla|yle|la|le|"
    "sı|si|su|sü|ya|ye|yı|yi|yu|yü|nı|ni|nu|nü|na|ne|"
    "ımız|imiz|umuz|ümüz|mız|miz|muz|müz|nız|niz|nuz|nüz|ım|im|um|üm|"
    "dır|dir|dur|dür|tır|tir|tur|tür|ydı|ydi|ydu|ydü|dı|di|du|dü|tı|ti|tu|tü|"
    "cı|ci|cu|cü|çı|çi|çu|çü|ca|ce|ça|çe|ki|sız|siz|suz|süz|lı|li|lu|lü|"
    "ı|i|u|ü|a|e|m|n"
)
_APOS = r"(?:['’`]?)"
_WORD_CHARS = r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû_]"


def _stem_candidates(keyword: str) -> list[str]:
    """'cumhurbaşkanı' → ['cumhurbaşkanı', 'cumhurbaşkan'] (son iyelik ünlüsü atılarak 'cumhurbaşkanlığı' da yakalanır)."""
    kw = tr_lower(keyword).strip()
    cands = [kw]
    if len(kw) > 3 and kw[-1] in "ıiuü" and kw[-2] not in "aeıioöuü":
        cands.append(kw[:-1])
    return cands


def compile_keyword(keyword: str) -> re.Pattern[str]:
    """Tek bir anahtar kelime için desen üretir.

    - ``re:<regex>``   → ham regex (küçük harfe çevrilmiş metin üzerinde çalışır)
    - ``=<kelime>``    → tam kelime eşleşmesi (ek kabul etmez)
    - ``~<parça>``     → alt dize eşleşmesi
    - ``<kelime>``     → kök + Türkçe ekler (varsayılan)
    """
    kw = keyword.strip()
    if kw.startswith("re:"):
        return re.compile(kw[3:], re.IGNORECASE | re.UNICODE)
    if kw.startswith("="):
        root = re.escape(tr_lower(kw[1:]).strip())
        return re.compile(rf"(?<!{_WORD_CHARS}){root}(?!{_WORD_CHARS})", re.UNICODE)
    if kw.startswith("~"):
        return re.compile(re.escape(tr_lower(kw[1:]).strip()), re.UNICODE)
    alts = "|".join(re.escape(c) for c in sorted(_stem_candidates(kw), key=len, reverse=True))
    return re.compile(
        rf"(?<!{_WORD_CHARS})(?:{alts}){_APOS}(?:{_SUFFIXES})*(?!{_WORD_CHARS})",
        re.UNICODE,
    )


@dataclass
class KeywordHit:
    keyword: str
    count: int
    samples: list[str] = field(default_factory=list)


@dataclass
class KeywordMatcher:
    """Yapılandırılmış anahtar kelimeleri bir metin üzerinde arar; Türkçe ekleri ve aksansız yazımı tolere eder."""

    keywords: list[str]
    _patterns: list[tuple[str, re.Pattern[str], re.Pattern[str] | None]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._patterns = []
        for kw in self.keywords:
            kw = kw.strip()
            if not kw:
                continue
            primary = compile_keyword(kw)
            folded = None
            if not kw.startswith(("re:", "=", "~")) and tr_fold(kw) != tr_lower(kw):
                folded = compile_keyword(tr_fold(kw))
            self._patterns.append((kw, primary, folded))

    def find(self, text: str) -> list[KeywordHit]:
        lowered = tr_lower(text)
        folded_text = tr_fold(text)
        hits: list[KeywordHit] = []
        for kw, primary, folded in self._patterns:
            matches = list(primary.finditer(lowered))
            spans = {m.span() for m in matches}
            if folded is not None:
                for m in folded.finditer(folded_text):
                    if m.span() not in spans:
                        spans.add(m.span())
                        matches.append(m)
            if not matches:
                continue
            samples = []
            for m in matches[:3]:
                s, e = m.span()
                samples.append(normalize_ws(text[max(0, s - 40): min(len(text), e + 40)]).replace("\n", " "))
            hits.append(KeywordHit(keyword=kw, count=len(matches), samples=samples))
        return hits

    def matches(self, text: str) -> list[str]:
        return [h.keyword for h in self.find(text)]


def excerpt(text: str, limit: int = 300) -> str:
    text = normalize_ws(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
