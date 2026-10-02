"""Boru hattı boyunca akan veri modelleri.

Tüm aşamalar (scraper → anahtar kelime → LLM skor → alarm → raporlama) AYNI nesneyi, ``NewsRecord``'u
taşır. Nesne aşama ilerledikçe zenginleşir; alarma gitmeyen kayıtlarda alarm alanları boş kalır.

Alan sırası kullanıcı isteğindeki sırayı izler: content_url, title, subtitle (alt başlık),
published_at (haber tarihi), content (haber içeriği), alarm_score (alarm skoru), alarm_reason (alarm sebebi).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = 1

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "fbclid", "gclid", "igshid", "ref", "referrer", "source",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_url(url: str) -> str:
    """URL'yi kimlik üretimi için normalize eder (http→https, host küçük harf, izleme parametreleri ve # kaldırılır)."""
    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme
    netloc = parts.netloc.lower()
    if netloc.startswith("www.12punto.com.tr"):
        netloc = "12punto.com.tr"
    if netloc == "hurriyet.com.tr":
        netloc = "www.hurriyet.com.tr"
    path = re.sub(r"/{2,}", "/", parts.path) or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if k.lower() not in _TRACKING_PARAMS]
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def article_id_for(url: str) -> str:
    """Kanonik URL'den deterministik, kısa ve çakışmaya dayanıklı kimlik üretir."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:32]


def content_hash_for(*parts: str) -> str:
    joined = "\n".join(p or "" for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


class Stage(StrEnum):
    RAW = "raw"            # scraper çıktısı (q.articles.raw)
    KEYWORD = "keyword"    # anahtar kelime eşleşti (q.articles.keyword)
    SCORED = "scored"      # LLM skoru uygulandı (q.articles.scored)
    ALARM = "alarm"        # alarm katmanı işledi (ES'e yazıldı; alarmsa q.alarms)
    REPORTED = "reported"  # raporlama katmanı işledi


class LLMVerdict(BaseModel):
    """Ollama LLM'in bir haber için ürettiği değerlendirme (ham karar; politika NewsRecord.apply_verdict'te)."""

    model: str
    alarm_score: int = Field(ge=0, le=100)
    is_alarm: bool = False
    reason: str = ""
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    scored_at: datetime = Field(default_factory=utcnow)
    latency_ms: int = 0
    attempts: int = 1
    raw: str = ""  # modelin ham çıktısı (izlenebilirlik için, kısaltılmış)


class NewsRecord(BaseModel):
    """Boru hattındaki birleşik haber nesnesi. Alarm verilsin verilmesin aynı şema kullanılır."""

    schema_version: int = SCHEMA_VERSION
    id: str
    source: str  # "hurriyet" | "12punto"
    content_url: str
    title: str
    subtitle: str = ""  # alt başlık / spot
    published_at: datetime | None = None  # haber tarihi
    updated_at: datetime | None = None
    content: str = ""  # haber içeriği (düz metin)
    category: str = ""
    author: str = ""
    image_url: str = ""
    tags: list[str] = Field(default_factory=list)
    language: str = "tr"
    scraped_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""

    # --- boru hattı zenginleştirmeleri ---
    stage: Stage = Stage.RAW
    matched_keywords: list[str] = Field(default_factory=list)
    alarm_score: int = Field(default=0, ge=0, le=100)  # alarm skoru (0-100)
    is_alarm: bool = False
    alarm_reason: str = ""  # alarm sebebi; alarm verildiyse LLM özeti de burada yer alır, aksi halde boş
    llm_summary: str = ""   # alarm verildiyse LLM özeti, aksi halde boş
    llm: LLMVerdict | None = None  # ham LLM kararı (izlenebilirlik)
    alarm_id: str = ""
    alarmed_at: datetime | None = None
    processed_at: datetime | None = None

    @field_validator("content_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        return (v or "").strip()

    @classmethod
    def new(
        cls,
        *,
        source: str,
        content_url: str,
        title: str,
        content: str,
        subtitle: str = "",
        published_at: datetime | None = None,
        updated_at: datetime | None = None,
        category: str = "",
        author: str = "",
        image_url: str = "",
        tags: list[str] | None = None,
    ) -> NewsRecord:
        url = canonical_url(content_url)
        return cls(
            id=article_id_for(url),
            source=source,
            content_url=url,
            title=(title or "").strip(),
            subtitle=(subtitle or "").strip(),
            published_at=published_at,
            updated_at=updated_at,
            content=(content or "").strip(),
            category=(category or "").strip(),
            author=(author or "").strip(),
            image_url=(image_url or "").strip(),
            tags=list(tags or []),
            content_hash=content_hash_for(title, subtitle, content),
        )

    # --- alarm politikası ---
    def apply_verdict(self, verdict: LLMVerdict, threshold: int) -> None:
        """LLM kararını uygular. Alarm kararı deterministik politikadır: skor >= eşik.

        Alarm verilirse ``alarm_reason`` gerekçe + LLM özetini içerir; verilmezse iki alan da boş kalır.
        """
        self.llm = verdict
        self.alarm_score = int(verdict.alarm_score)
        self.is_alarm = self.alarm_score >= int(threshold)
        if self.is_alarm:
            self.llm_summary = (verdict.summary or "").strip()
            reason = (verdict.reason or "").strip() or "LLM alarm eşiğini aşan skor üretti."
            self.alarm_reason = reason if not self.llm_summary else f"{reason}\n\nLLM Özeti: {self.llm_summary}"
        else:
            self.llm_summary = ""
            self.alarm_reason = ""
        self.stage = Stage.SCORED
        self.processed_at = utcnow()

    def mark_not_scored(self) -> None:
        """Anahtar kelime içermeyen (LLM'e gitmeyen) kayıt: skor 0, alarm yok, alanlar boş."""
        self.alarm_score = 0
        self.is_alarm = False
        self.alarm_reason = ""
        self.llm_summary = ""
        self.stage = Stage.SCORED
        self.processed_at = utcnow()

    # --- serileştirme ---
    def to_message(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_message(cls, data: dict[str, Any]) -> NewsRecord:
        return cls.model_validate(data)

    def to_es_document(self) -> dict[str, Any]:
        doc = self.model_dump(mode="json")
        doc["@timestamp"] = (self.published_at or self.scraped_at).isoformat()
        doc["content_length"] = len(self.content or "")
        return doc

    def text_for_matching(self) -> str:
        return "\n".join(p for p in (self.title, self.subtitle, self.content) if p)


class AlarmEvent(BaseModel):
    """Alarm katmanının q.alarms kuyruğuna ve news-alarms indeksine yazdığı olay."""

    alarm_id: str
    record_id: str
    source: str
    content_url: str
    title: str
    subtitle: str = ""
    published_at: datetime | None = None
    alarm_score: int = 0
    alarm_reason: str = ""
    llm_summary: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    raised_at: datetime = Field(default_factory=utcnow)
    channels_notified: list[str] = Field(default_factory=list)
    acknowledged: bool = False
    record: NewsRecord

    @classmethod
    def from_record(cls, record: NewsRecord) -> AlarmEvent:
        alarm_id = record.alarm_id or hashlib.sha1(f"{record.id}:{record.content_hash}".encode()).hexdigest()[:20]
        return cls(
            alarm_id=alarm_id,
            record_id=record.id,
            source=record.source,
            content_url=record.content_url,
            title=record.title,
            subtitle=record.subtitle,
            published_at=record.published_at,
            alarm_score=record.alarm_score,
            alarm_reason=record.alarm_reason,
            llm_summary=record.llm_summary,
            matched_keywords=list(record.matched_keywords),
            record=record,
        )

    def to_message(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_es_document(self) -> dict[str, Any]:
        doc = self.model_dump(mode="json", exclude={"record"})
        doc["@timestamp"] = self.raised_at.isoformat()
        doc["content"] = self.record.content
        doc["category"] = self.record.category
        return doc


class Report(BaseModel):
    """Raporlama katmanının ürettiği rapor (news-reports indeksi, q.reports kuyruğu)."""

    report_id: str
    kind: str  # "periodic" | "alarm_digest" | "adhoc" | "daily"
    window_start: datetime
    window_end: datetime
    generated_at: datetime = Field(default_factory=utcnow)
    stats: dict[str, Any] = Field(default_factory=dict)
    narrative: str = ""
    top_alarms: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""

    def to_message(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_es_document(self) -> dict[str, Any]:
        doc = self.model_dump(mode="json")
        doc["@timestamp"] = self.generated_at.isoformat()
        return doc


class Citation(BaseModel):
    id: str
    title: str
    content_url: str
    source: str
    published_at: datetime | None = None
    score: float = 0.0
    snippet: str = ""


class Answer(BaseModel):
    """RAG soru-cevap çıktısı."""

    question: str
    answer: str
    sources: list[Citation] = Field(default_factory=list)
    model: str = ""
    retrieved_count: int = 0
    generated_at: datetime = Field(default_factory=utcnow)
    search_terms: list[str] = Field(default_factory=list)
