"""LLM skorlama servisi: q.articles.keyword → Ollama (Türkçe analist istemi) → q.articles.scored.

Politika: model yalnızca ham kararı (``LLMVerdict``) üretir; alarm kararı ``NewsRecord.apply_verdict`` ile
deterministik olarak (``alarm_score >= alarm_threshold``) uygulanır. Bozuk JSON süreç içinde en fazla
``MAX_LLM_ATTEMPTS`` kez denenir; yine olmazsa ``Retry`` fırlatılır (broker gecikmeli yeniden dener).

Ollama kesintisi bir mesaj kaybı nedeni DEĞİLDİR: erişilemiyorsa / model yüklü değilse servis önce süreç içinde
(``LLM_WAIT_SECONDS``'a kadar, ``stop_event``'e saygılı) Ollama'nın dönmesini bekler, sonra ``Unavailable`` ile
(``TRANSIENT_MAX_ATTEMPTS`` tavanı) broker'a devreder. Broker'ın son denemesinde hâlâ puanlanamayan haber ölü
mektuba gitmez; ``FALLBACK_MODEL`` etiketli sezgisel yedek kararla ``article.scored``'a yayınlanır ki
Elasticsearch'e yazılsın (bkz. ``ScoringService``).
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import ValidationError

from ..broker import Broker, Message, Queue, Reject, Retry, RoutingKey, Unavailable, attempt_limit_for
from ..config import Settings
from ..models import LLMVerdict, NewsRecord
from ..textutil import excerpt, normalize_ws, tr_fold, tr_lower
from .llm import LLM, HeuristicLLM, LLMBadOutput, LLMUnavailable, redact_url
from .prompts import STRICT_JSON_REMINDER, TOPICS, build_system_prompt, build_user_prompt
from .thresholds import resolve_threshold

log = logging.getLogger(__name__)

MAX_LLM_ATTEMPTS = 3  # ilk deneme + bozuk çıktı için 2 tekrar
RAW_LIMIT = 2000  # LLMVerdict.raw üst sınırı (karakter)
MAX_LIST_ITEMS = 20
MAX_TEXT_CHARS = 4000  # reason / summary üst sınırı
SCORER_PREFETCH = 1  # LLM yavaş olduğundan tüketici aynı anda tek mesaj tutar
# Ollama kesintisinde mesaj broker'a dönmeden önce süreç içinde beklenen azami süre (saniye); 0 = bekleme yok.
LLM_WAIT_SECONDS = 300.0
LLM_POLL_SECONDS = 1.0  # hazırlık denetimleri arası başlangıç aralığı (üstel artar)
LLM_POLL_MAX_SECONDS = 30.0
MAX_UNAVAILABLE_ROUNDS = 3  # bir mesaj için kesinti atlatıp yeniden puanlama tur sayısı (kesintili Ollama)
FALLBACK_MODEL = "heuristic-fallback"  # son denemede sezgisel yedek kararın LLMVerdict.model etiketi

_SCORE_KEYS = ("alarm_score", "score", "alarmScore", "alarm skoru")
_TRUE_WORDS = {"true", "1", "yes", "evet", "doğru", "dogru", "alarm", "var"}
_FALSE_WORDS = {"false", "0", "no", "hayır", "hayir", "yanlış", "yanlis", "yok", ""}
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_LIST_SPLIT_RE = re.compile(r"[,;\n|]+")

_TOPIC_ALIASES: dict[str, str] = {
    "politika": "siyaset",
    "siyasi": "siyaset",
    "ic politika": "siyaset",
    "hukumet": "siyaset",
    "yonetim": "siyaset",
    "parti": "siyaset",
    "secim": "siyaset",
    "ekonomik": "ekonomi",
    "piyasa": "ekonomi",
    "para": "ekonomi",
    "ticaret": "ekonomi",
    "enerji": "ekonomi",
    "maliye": "ekonomi",
    "adalet": "hukuk",
    "yargi": "hukuk",
    "yasal": "hukuk",
    "hukuki": "hukuk",
    "suc": "hukuk",
    "adalet-hukuk": "hukuk",
    "adli": "hukuk",
    "asayis": "güvenlik",
    "teror": "güvenlik",
    "savunma": "güvenlik",
    "polis": "güvenlik",
    "askeri": "güvenlik",
    "diplomasi": "dış politika",
    "uluslararasi": "dış politika",
    "dunya": "dış politika",
    "dis iliskiler": "dış politika",
    "dis-politika": "dış politika",
    "finans": "finans/fon",
    "fon": "finans/fon",
    "bankacilik": "finans/fon",
    "borsa": "finans/fon",
    "finans-fon": "finans/fon",
    "finans fon": "finans/fon",
    "finans / fon": "finans/fon",
    "yatirim": "finans/fon",
    "toplum": "sosyal",
    "sosyal politika": "sosyal",
    "egitim": "sosyal",
    "saglik": "sosyal",
    "yasam": "sosyal",
    "cevre": "sosyal",
    "kultur": "sosyal",
    "other": "diğer",
    "spor": "diğer",
    "magazin": "diğer",
    "genel": "diğer",
}
_TOPIC_BY_FOLD: dict[str, str] = {tr_fold(t): t for t in TOPICS}
# "<konu> politikası" (ekonomi/para/maliye politikası...) konusu tamlayanın konusudur, "siyaset" değil.
_POLICY_OF_RE = re.compile(r"^(.+?)\s+politikas[iı]$")


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


def coerce_confidence(value: Any, default: int = 60) -> int:
    """Modelin 'confidence' alanı (0-100); yoksa/bozuksa varsayılan."""
    try:
        if value is None:
            return default
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


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
    policy_of = _POLICY_OF_RE.match(folded)
    if policy_of:
        subject = normalize_topic(policy_of.group(1))
        if subject != "diğer":
            return subject
    # Önce kanonik ad içerme ("ekonomi politikası" → ekonomi); takma ad araması sonra gelir, yoksa "politika"
    # takma adı kanonik adı içeren her "<konu> politikası" biçimini siyasete çekerdi.
    for fold, canonical in _TOPIC_BY_FOLD.items():
        if fold in folded:
            return canonical
    for alias, canonical in _TOPIC_ALIASES.items():
        if alias in folded:
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
        confidence=coerce_confidence(data.get("confidence")),
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
    unavailable: int = 0  # LLM çağrısı altyapı hatasıyla başarısız (bağlantı, zaman aşımı, model yok...)
    waits: int = 0  # Ollama kesintisi süreç içinde beklenip atlatıldı (mesaj broker'a dönmeden puanlandı)
    exhausted: int = 0  # MAX_LLM_ATTEMPTS sonunda hâlâ bozuk çıktı (broker yeniden deneme)
    fallback: int = 0  # son denemede sezgisel yedek kararla yayınlanan haber (kayıt kaybı önlendi)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class ScoringService:
    """``Queue.ARTICLES_KEYWORD`` tüketicisi; her haberi LLM ile puanlar, ``article.scored`` ile yayınlar.

    Dayanıklılık politikası (anahtar kelime eşleşen haberler asla sessizce kaybolmaz):

    * Ollama erişilemiyor / model yüklü değil → ``Unavailable``. ``handle`` önce ``llm_wait_seconds`` boyunca
      (``stop_event``'e saygılı, üstel geri çekilmeli) Ollama'nın dönmesini bekler ve aynı haberi süreç içinde
      yeniden puanlar; bekleme sonuçsuz kalırsa mesaj broker'a ``Unavailable`` ile döner (tavan
      ``TRANSIENT_MAX_ATTEMPTS``, saatlerce kesinti ölü mektup üretmez).
    * Ollama ayakta ama çağrı başarısız (zaman aşımı, sunucu hatası, bellek yetersiz...) → ``Retry``
      (tavan ``rabbitmq_max_attempts``).
    * Broker'ın SON denemesinde hâlâ puanlanamayan haber ölü mektuba gitmez: ``HeuristicLLM`` ile
      ``FALLBACK_MODEL`` etiketli yedek karar üretilir ve haber yine ``article.scored`` ile yayınlanır ki
      Elasticsearch/alarm/raporlama katmanlarına ulaşsın (hata loglanır, ``stats.fallback`` artar).
    """

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        llm: LLM,
        *,
        stop_event: threading.Event | None = None,
        llm_wait_seconds: float = LLM_WAIT_SECONDS,
        llm_poll_seconds: float = LLM_POLL_SECONDS,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.llm = llm
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.llm_wait_seconds = max(0.0, float(llm_wait_seconds))
        self.llm_poll_seconds = max(0.01, float(llm_poll_seconds))
        self.system_prompt = build_system_prompt(settings.alarm_threshold)
        self.stats = ScoringStats()
        self._fallback = HeuristicLLM(settings)

    def build_prompts(self, record: NewsRecord) -> tuple[str, str]:
        """(system, user) istem çifti; içerik ``ollama_max_content_chars`` ile kısaltılır."""
        return self.system_prompt, build_user_prompt(
            record, max_content_chars=self.settings.ollama_max_content_chars
        )

    # --- Ollama hazırlık denetimi ---
    def llm_ready(self) -> bool:
        """Ollama erişilebilir VE yapılandırılan model yüklü mü?"""
        return bool(self.llm.health() and self.llm.model_available())

    def wait_for_llm(self, max_wait: float | None = None) -> bool:
        """Ollama hazır olana dek bekler (üstel geri çekilme: ``llm_poll_seconds`` → ``LLM_POLL_MAX_SECONDS``).

        ``True`` = hazır; ``False`` = ``max_wait`` (varsayılan ``llm_wait_seconds``) doldu ya da ``stop_event``
        set edildi. ``max_wait`` 0 ise beklemeden tek bir denetim yapılır.
        """
        limit = self.llm_wait_seconds if max_wait is None else max(0.0, float(max_wait))
        deadline = time.monotonic() + limit
        delay = self.llm_poll_seconds
        polls = 0
        while True:
            if self.llm_ready():
                if polls:
                    log.info(
                        "Ollama yeniden hazır (%s, model=%s)",
                        redact_url(self.settings.ollama_base_url),
                        self.llm.model_name,
                    )
                return True
            remaining = deadline - time.monotonic()
            if self.stop_event.is_set() or remaining <= 0:
                return False
            pause = min(delay, remaining)
            if polls == 0:
                log.warning(
                    "Ollama hazır değil (%s, model=%s); en fazla %.0f sn beklenecek, mesaj broker'a dönmeyecek",
                    redact_url(self.settings.ollama_base_url),
                    self.llm.model_name,
                    limit,
                )
            else:
                log.debug("Ollama hâlâ hazır değil; %.1f sn sonra yeniden denetlenecek", pause)
            polls += 1
            self.stop_event.wait(pause)
            delay = min(delay * 2, LLM_POLL_MAX_SECONDS)

    # --- puanlama ---
    def threshold_for(self, record: NewsRecord) -> int:
        """Kayıt için kaynak/kategori/anahtar kelime bazlı eşik (ALARM_THRESHOLDS_JSON), yoksa alarm_threshold."""
        return resolve_threshold(self.settings, record)

    def score_record(self, record: NewsRecord) -> NewsRecord:
        """Kaydı LLM ile puanlar ve ``apply_verdict`` uygular (saf: broker'a yazmaz, beklemez).

        ``llm_samples > 1`` ise aynı haber N kez puanlanır (öz-tutarlılık): medyan skor alınır, örnekler arası
        fark ``llm_disagreement_threshold``'u aşarsa ``needs_review`` işaretlenir ve güven düşer.
        Ollama erişilemiyorsa ``Unavailable``, çağrı başarısızsa ``Retry``; ``MAX_LLM_ATTEMPTS`` denemede geçerli
        karar çıkmazsa ``Retry`` fırlatır.
        """
        system, user = self.build_prompts(record)
        samples = max(1, int(self.settings.llm_samples))
        if samples == 1:
            verdict = self._verdict_once(record, system, user)
        else:
            verdict = self._verdict_sampled(record, system, user, samples)
        threshold = self.threshold_for(record)
        record.apply_verdict(verdict, threshold)
        spread = (max(verdict.samples) - min(verdict.samples)) if len(verdict.samples) > 1 else 0
        record.needs_review = spread > int(self.settings.llm_disagreement_threshold)
        if record.needs_review:
            log.warning(
                "LLM örnekleri uyuşmuyor [%s] %s: skorlar=%s fark=%d → insan incelemesi önerilir",
                record.source,
                excerpt(record.title, 80),
                verdict.samples,
                spread,
            )
        return record

    def _verdict_sampled(self, record: NewsRecord, system: str, user: str, samples: int) -> LLMVerdict:
        verdicts: list[LLMVerdict] = []
        for i in range(samples):
            try:
                verdicts.append(
                    self._verdict_once(record, system, user, temperature=self.settings.llm_sample_temperature)
                )
            except Retry as exc:
                if not verdicts:
                    raise
                log.warning("Örnekleme %d/%d başarısız, eldeki %d örnekle devam: %s", i + 1, samples, len(verdicts), exc)
                break
        scores = [v.alarm_score for v in verdicts]
        median = int(round(statistics.median(scores)))
        spread = max(scores) - min(scores)
        model_conf = int(round(statistics.mean(v.confidence for v in verdicts)))
        confidence = int(round((max(0, 100 - 2 * spread) + model_conf) / 2))
        chosen = min(verdicts, key=lambda v: (abs(v.alarm_score - median), -v.confidence))
        return chosen.model_copy(
            update={
                "alarm_score": median,
                "confidence": confidence,
                "samples": scores,
                "latency_ms": sum(v.latency_ms for v in verdicts),
                "attempts": sum(v.attempts for v in verdicts),
            }
        )

    def _verdict_once(
        self, record: NewsRecord, system: str, user: str, *, temperature: float | None = None
    ) -> LLMVerdict:
        prompt = user
        title = excerpt(record.title, 80)
        last_error: LLMBadOutput | None = None
        started = time.perf_counter()
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            self.stats.llm_calls += 1
            try:
                data = self.llm.chat_json(system, prompt, temperature=temperature)
                verdict = build_verdict(
                    data,
                    model=self.llm.model_name,
                    threshold=self.threshold_for(record),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempt,
                    raw=json.dumps(data, ensure_ascii=False, default=str),
                )
            except LLMUnavailable as exc:
                self.stats.unavailable += 1
                if self.llm_ready():
                    log.warning(
                        "Ollama ayakta ama LLM çağrısı başarısız, mesaj yeniden denenecek [%s] %s: %s",
                        record.source,
                        title,
                        exc,
                    )
                    raise Retry(f"LLM çağrısı başarısız: {exc}") from exc
                log.warning("LLM erişilemiyor [%s] %s: %s", record.source, title, exc)
                raise Unavailable(f"LLM erişilemiyor: {exc}") from exc
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
            verdict.samples = [verdict.alarm_score]
            return verdict
        self.stats.exhausted += 1
        raise Retry(f"LLM {MAX_LLM_ATTEMPTS} denemede geçerli karar üretemedi: {last_error}") from last_error

    def score_with_recovery(self, record: NewsRecord) -> NewsRecord:
        """``score_record`` + Ollama kesintisini süreç içinde atlatma.

        ``Unavailable`` gelirse Ollama'nın dönmesi ``llm_wait_seconds``'a kadar beklenir ve aynı haber yeniden
        puanlanır (en fazla ``MAX_UNAVAILABLE_ROUNDS`` tur); bekleme sonuçsuz kalırsa istisna broker'a geçer.
        """
        title = excerpt(record.title, 80)
        round_no = 1
        while True:
            try:
                return self.score_record(record)
            except Unavailable:
                if self.llm_wait_seconds <= 0 or round_no >= MAX_UNAVAILABLE_ROUNDS or not self.wait_for_llm():
                    raise
                round_no += 1
                self.stats.waits += 1
                log.info(
                    "Ollama kesintisi atlatıldı; haber süreç içinde yeniden puanlanıyor (tur %d/%d) [%s] %s",
                    round_no,
                    MAX_UNAVAILABLE_ROUNDS,
                    record.source,
                    title,
                )

    def fallback_verdict(self, record: NewsRecord, error: Exception, attempts: int) -> LLMVerdict:
        """Son broker denemesinde LLM hâlâ başarısızsa sezgisel yedek karar (``FALLBACK_MODEL``) üretir.

        ``attempts`` broker düzeyindeki deneme sayısıdır; gerekçe hatayı ve yedek olduğunu açıkça belirtir.
        """
        data = self._fallback.evaluate(title=record.title, subtitle=record.subtitle, content=record.content)
        data["reason"] = (
            f"LLM skorlaması {attempts} denemede tamamlanamadı ({excerpt(str(error), 200)}); "
            f"sezgisel yedek değerlendirme uygulandı. {coerce_text(data.get('reason'))}"
        )
        return build_verdict(
            data,
            model=FALLBACK_MODEL,
            threshold=self.threshold_for(record),
            attempts=attempts,
            raw=json.dumps(data, ensure_ascii=False, default=str),
        )

    def handle(self, msg: Message) -> None:
        self.stats.received += 1
        try:
            record = NewsRecord.from_message(msg.body)
        except ValidationError as exc:
            self.stats.rejected += 1
            raise Reject(f"Geçersiz haber mesajı (article.keyword): {excerpt(str(exc), 300)}") from exc

        try:
            record = self.score_with_recovery(record)
        except Retry as exc:
            attempts = msg.attempts + 1
            limit = attempt_limit_for(self.settings, exc)
            if limit <= 0 or attempts < limit:
                raise
            # Son deneme: ölü mektup yerine yedek kararla yayınla ki haber ES'e/alarma/raporlamaya ulaşsın.
            self.stats.fallback += 1
            log.error(
                "SKORLAMA BAŞARISIZ: haber %d denemede LLM ile puanlanamadı, sezgisel yedek kararla yayınlanıyor "
                "[%s] %s: %s",
                attempts,
                record.source,
                excerpt(record.title, 80),
                exc,
            )
            record.apply_verdict(self.fallback_verdict(record, exc, attempts), self.threshold_for(record))
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
        """``q.articles.keyword`` kuyruğunu tüketir; işlenen mesaj sayısını döndürür.

        Tüketime başlamadan önce Ollama'nın hazır olması ``llm_wait_seconds``'a kadar beklenir (soğuk başlangıç,
        model indirme); süre dolarsa tüketim yine başlar ve her mesaj kendi bekleme/yeniden deneme yolunu izler.
        """
        if stop_event is not None:
            self.stop_event = stop_event
        self.broker.declare_topology()
        log.info(
            "LLM skorlama başlıyor: kuyruk=%s, model=%s, eşik=%d, içerik sınırı=%d karakter",
            Queue.ARTICLES_KEYWORD,
            self.llm.model_name,
            self.settings.alarm_threshold,
            self.settings.ollama_max_content_chars,
        )
        if not self.wait_for_llm():
            log.warning(
                "Ollama hazır olmadan tüketime başlanıyor (%s, model=%s); mesajlar Ollama dönene dek bekletilip "
                "gecikmeli yeniden denenecek. Çözüm: `ollama serve` ve `ollama pull %s`",
                redact_url(self.settings.ollama_base_url),
                self.llm.model_name,
                self.llm.model_name,
            )
        processed = self.broker.consume(
            Queue.ARTICLES_KEYWORD,
            self.handle,
            prefetch=SCORER_PREFETCH,
            stop_event=self.stop_event,
            max_messages=max_messages,
        )
        log.info("LLM skorlama durdu: %d mesaj işlendi, istatistik=%s", processed, self.stats.as_dict())
        return processed
