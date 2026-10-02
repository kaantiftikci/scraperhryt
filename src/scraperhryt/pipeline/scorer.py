"""LLM skorlama servisi: q.articles.keyword → Ollama (Türkçe analist istemi) → q.articles.scored.

Politika: model yalnızca ham kararı (``LLMVerdict``) üretir; alarm kararı ``NewsRecord.apply_verdict`` ile
deterministik olarak (``alarm_score >= alarm_threshold``) uygulanır. Bozuk JSON süreç içinde en fazla
``MAX_LLM_ATTEMPTS`` kez denenir; yine olmazsa veya Ollama erişilemezse ``Retry`` fırlatılır (broker gecikmeli
yeniden dener, ``rabbitmq_max_attempts`` sonrası ölü mektup).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, Retry, RoutingKey
from ..config import Settings
from ..models import LLMVerdict, NewsRecord
from ..textutil import excerpt, normalize_ws, tr_fold, tr_lower
from .llm import LLM, LLMBadOutput, LLMUnavailable
from .prompts import STRICT_JSON_REMINDER, TOPICS, build_system_prompt, build_user_prompt

log = logging.getLogger(__name__)

MAX_LLM_ATTEMPTS = 3  # ilk deneme + bozuk çıktı için 2 tekrar
RAW_LIMIT = 2000  # LLMVerdict.raw üst sınırı (karakter)
MAX_LIST_ITEMS = 20
MAX_TEXT_CHARS = 4000  # reason / summary üst sınırı
SCORER_PREFETCH = 1  # LLM yavaş olduğundan tüketici aynı anda tek mesaj tutar

_SCORE_KEYS = ("alarm_score", "score", "alarmScore", "alarm skoru")
_TRUE_WORDS = {"true", "1", "yes", "evet", "doğru", "dogru", "alarm", "var"}
_FALSE_WORDS = {"false", "0", "no", "hayır", "hayir", "yanlış", "yanlis", "yok", ""}
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_LIST_SPLIT_RE = re.compile(r"[,;\n|]+")

_TOPIC_ALIASES: dict[str, str] = {
    "politika": "siyaset", "siyasi": "siyaset", "ic politika": "siyaset", "hukumet": "siyaset",
    "yonetim": "siyaset", "parti": "siyaset", "secim": "siyaset",
    "ekonomik": "ekonomi", "piyasa": "ekonomi", "ticaret": "ekonomi", "enerji": "ekonomi", "maliye": "ekonomi",
    "adalet": "hukuk", "yargi": "hukuk", "yasal": "hukuk", "hukuki": "hukuk", "suc": "hukuk",
    "adalet-hukuk": "hukuk", "adli": "hukuk",
    "asayis": "güvenlik", "teror": "güvenlik", "savunma": "güvenlik", "polis": "güvenlik", "askeri": "güvenlik",
    "diplomasi": "dış politika", "uluslararasi": "dış politika", "dunya": "dış politika",
    "dis iliskiler": "dış politika", "dis-politika": "dış politika",
    "finans": "finans/fon", "fon": "finans/fon", "bankacilik": "finans/fon", "borsa": "finans/fon",
    "finans-fon": "finans/fon", "finans fon": "finans/fon", "finans / fon": "finans/fon", "yatirim": "finans/fon",
    "toplum": "sosyal", "sosyal politika": "sosyal", "egitim": "sosyal", "saglik": "sosyal", "yasam": "sosyal",
    "cevre": "sosyal", "kultur": "sosyal",
    "other": "diğer", "spor": "diğer", "magazin": "diğer", "genel": "diğer",
}
_TOPIC_BY_FOLD: dict[str, str] = {tr_fold(t): t for t in TOPICS}


# ---------------------------------------------------------------------------------------------------------
# Model çıktısını LLMVerdict'e dönüştürme
# ---------------------------------------------------------------------------------------------------------


def coerce_score(value: Any) -> int:
    """int/float/sayısal dize ('85', '85/100', '85 puan') → 0..100 arasına sıkıştırılmış tam sayı."""
    if isinstance(value, bool):
        raise LLMBadOutput("alarm_score boolean olamaz")
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        m = _NUMBER_RE.search(value)
        if not m:
            raise LLMBadOutput(f"alarm_score sayı içermiyor: {value!r}")
        number = float(m.group(0).replace(",", "."))
    else:
        raise LLMBadOutput(f"alarm_score beklenmeyen türde: {type(value).__name__}")
    if not math.isfinite(number):
        raise LLMBadOutput(f"alarm_score sonlu değil: {value!r}")
    return max(0, min(100, int(round(number))))


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        word = tr_lower(value).strip()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return default


def coerce_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        text = " ".join(coerce_text(v, limit) for v in value)
    elif isinstance(value, dict):
        text = " ".join(f"{k}: {coerce_text(v, limit)}" for k, v in value.items())
    else:
        text = str(value)
    text = normalize_ws(text)
    return text if len(text) <= limit else excerpt(text, limit)


def coerce_str_list(value: Any, limit: int = MAX_LIST_ITEMS) -> list[str]:
    """Liste/dize/sözlük → tekrarsız, boş olmayan dize listesi (sıra korunur)."""
    if value is None:
        return []
    if isinstance(value, str):
        items: list[Any] = _LIST_SPLIT_RE.split(value)
    elif isinstance(value, list | tuple | set):
        items = list(value)
    elif isinstance(value, dict):
        items = [v if isinstance(v, str) else k for k, v in value.items()]
    else:
        items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            item = item.get("name") or item.get("entity") or item.get("value") or ""
        text = normalize_ws(str(item)).replace("\n", " ").strip(" .;,")
        key = tr_lower(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def normalize_topic(topic: str) -> str:
    """Model konusunu sabit listeye eşler; bilinmeyen konu 'diğer' olur."""
    folded = tr_fold(normalize_ws(topic)).strip(" .")
    folded = re.sub(r"\s*/\s*", "/", folded)
    if folded in _TOPIC_BY_FOLD:
        return _TOPIC_BY_FOLD[folded]
    if folded in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[folded]
    for alias, canonical in _TOPIC_ALIASES.items():
        if alias in folded:
            return canonical
    for fold, canonical in _TOPIC_BY_FOLD.items():
        if fold in folded:
            return canonical
    return "diğer"


def normalize_topics(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        topic = normalize_topic(item)
        if topic not in out:
            out.append(topic)
    return out


def build_verdict(
    data: dict[str, Any],
    *,
    model: str,
    threshold: int,
    latency_ms: int = 0,
    attempts: int = 1,
    raw: str = "",
) -> LLMVerdict:
    """Modelin JSON çıktısını doğrulayıp ``LLMVerdict``'e çevirir; zorunlu alan yoksa ``LLMBadOutput``."""
    if not isinstance(data, dict):
        raise LLMBadOutput(f"Model çıktısı JSON nesnesi değil: {type(data).__name__}")
    score_value = next((data[k] for k in _SCORE_KEYS if k in data and data[k] is not None), None)
    if score_value is None:
        raise LLMBadOutput(f"Model çıktısında alarm_score yok (anahtarlar: {sorted(map(str, data))})")
    score = coerce_score(score_value)
    return LLMVerdict(
        model=model,
        alarm_score=score,
        is_alarm=coerce_bool(data.get("is_alarm"), default=score >= threshold),
        reason=coerce_text(data.get("reason")),
        summary=coerce_text(data.get("summary")),
        topics=normalize_topics(coerce_str_list(data.get("topics"))),
        entities=coerce_str_list(data.get("entities")),
        latency_ms=max(0, int(latency_ms)),
        attempts=max(1, int(attempts)),
        raw=raw[:RAW_LIMIT],
    )


# ---------------------------------------------------------------------------------------------------------
# Servis
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ScoringStats:
    received: int = 0
    scored: int = 0
    alarms: int = 0
    rejected: int = 0
    llm_calls: int = 0
    bad_output: int = 0  # çözümlenemeyen model çıktısı (süreç içi tekrar)
    unavailable: int = 0  # Ollama erişilemedi (broker yeniden deneme)
    exhausted: int = 0  # MAX_LLM_ATTEMPTS sonunda hâlâ bozuk çıktı (broker yeniden deneme)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ScoringService:
    """``Queue.ARTICLES_KEYWORD`` tüketicisi; her haberi LLM ile puanlar ve ``article.scored`` ile yayınlar."""

    def __init__(self, settings: Settings, broker: Broker, llm: LLM) -> None:
        self.settings = settings
        self.broker = broker
        self.llm = llm
        self.system_prompt = build_system_prompt(settings.alarm_threshold)
        self.stats = ScoringStats()

    def build_prompts(self, record: NewsRecord) -> tuple[str, str]:
        """(system, user) istem çifti; içerik ``ollama_max_content_chars`` ile kısaltılır."""
        return self.system_prompt, build_user_prompt(
            record, max_content_chars=self.settings.ollama_max_content_chars
        )

    def score_record(self, record: NewsRecord) -> NewsRecord:
        """Kaydı LLM ile puanlar ve ``apply_verdict`` uygular (saf: broker'a yazmaz).

        ``LLMVerdict.latency_ms`` tüm denemelerin toplam süresidir; ``attempts`` başarılı denemenin sırasıdır.
        """
        system, user = self.build_prompts(record)
        prompt = user
        title = excerpt(record.title, 80)
        last_error: LLMBadOutput | None = None
        started = time.perf_counter()
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            self.stats.llm_calls += 1
            try:
                data = self.llm.chat_json(system, prompt)
                verdict = build_verdict(
                    data,
                    model=self.llm.model_name,
                    threshold=self.settings.alarm_threshold,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempt,
                    raw=json.dumps(data, ensure_ascii=False, default=str),
                )
            except LLMUnavailable as exc:
                self.stats.unavailable += 1
                log.warning("LLM erişilemiyor, mesaj yeniden denenecek [%s] %s: %s", record.source, title, exc)
                raise Retry(f"LLM erişilemiyor: {exc}") from exc
            except LLMBadOutput as exc:
                last_error = exc
                self.stats.bad_output += 1
                log.warning(
                    "LLM çıktısı çözümlenemedi (deneme %d/%d) [%s] %s: %s",
                    attempt,
                    MAX_LLM_ATTEMPTS,
                    record.source,
                    title,
                    exc,
                )
                prompt = user + STRICT_JSON_REMINDER
                continue
            record.apply_verdict(verdict, self.settings.alarm_threshold)
            return record
        self.stats.exhausted += 1
        raise Retry(f"LLM {MAX_LLM_ATTEMPTS} denemede geçerli karar üretemedi: {last_error}") from last_error

    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            record = NewsRecord.from_message(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz haber mesajı (article.keyword): {excerpt(str(exc), 300)}") from exc

        record = self.score_record(record)
        self.broker.publish(RoutingKey.ARTICLE_SCORED, record.to_message())
        self.stats.scored += 1
        if record.is_alarm:
            self.stats.alarms += 1
        verdict = record.llm
        log.info(
            "%s [%s] %s → skor=%d eşik=%d model=%s deneme=%d süre=%dms konular=%s anahtar=%s",
            "ALARM" if record.is_alarm else "Skorlandı",
            record.source,
            excerpt(record.title, 80),
            record.alarm_score,
            self.settings.alarm_threshold,
            verdict.model if verdict else "-",
            verdict.attempts if verdict else 0,
            verdict.latency_ms if verdict else 0,
            ", ".join(verdict.topics) if verdict and verdict.topics else "-",
            ", ".join(record.matched_keywords) or "-",
        )

    def run(self, stop_event: threading.Event | None = None, max_messages: int | None = None) -> int:
        """``q.articles.keyword`` kuyruğunu tüketir; işlenen mesaj sayısını döndürür."""
        self.broker.declare_topology()
        log.info(
            "LLM skorlama başlıyor: kuyruk=%s, model=%s, eşik=%d, içerik sınırı=%d karakter",
            Queue.ARTICLES_KEYWORD,
            self.llm.model_name,
            self.settings.alarm_threshold,
            self.settings.ollama_max_content_chars,
        )
        if not self.llm.health():
            log.warning("Ollama şu an erişilemiyor (%s); mesajlar gecikmeli yeniden denenecek", self.settings.ollama_base_url)
        elif not self.llm.model_available():
            log.warning("Model '%s' Ollama'da yüklü görünmüyor; `ollama pull %s` çalıştırın", self.llm.model_name, self.llm.model_name)
        processed = self.broker.consume(
            Queue.ARTICLES_KEYWORD,
            self.handle,
            prefetch=SCORER_PREFETCH,
            stop_event=stop_event,
            max_messages=max_messages,
        )
        log.info("LLM skorlama durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed
