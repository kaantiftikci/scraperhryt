"""LLM istemcileri: Ollama HTTP API, test için FakeOllama ve çevrimdışı/deterministik HeuristicLLM.

Üçü de ``LLM`` protokolünü uygular; ScoringService, ReportBuilder ve QAEngine yalnızca bu protokole
bağımlıdır.

Hata sınıfları:
- ``LLMUnavailable``: bağlantı hatası, zaman aşımı, 5xx veya sunucunun reddettiği istek → çağıran
  ``Retry`` üretir.
- ``LLMBadOutput``: model çıktısı JSON olarak çözümlenemedi → çağıran süreç içinde tekrar dener.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config import Settings
from ..textutil import KeywordMatcher, excerpt, normalize_ws, tr_lower
from .prompts import TOPICS, TRUNCATION_MARKER, parse_user_prompt

log = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM katmanı hatalarının ortak atası."""


class LLMUnavailable(LLMError):
    """Ollama'ya erişilemiyor (bağlantı, zaman aşımı, 5xx) veya istek sunucu tarafından reddedildi."""


class LLMBadOutput(LLMError):
    """Model çıktısı beklenen biçimde değil (JSON çözümlenemedi, zorunlu alan yok)."""


# ---------------------------------------------------------------------------------------------------------
# JSON çıkarma
# ---------------------------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_lenient(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if fixed == candidate:
            raise
        return json.loads(fixed)


def _as_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return None


def _balanced_objects(text: str) -> list[str]:
    """Metindeki dengeli ``{...}`` bloklarını (dize içi süslü parantezleri yok sayarak) sırayla verir."""
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def extract_json_object(text: str) -> dict[str, Any]:
    """Model çıktısından ilk JSON nesnesini çıkarır.

    Sırasıyla: doğrudan çözümleme → markdown kod blokları → metin içindeki dengeli ``{...}`` blokları.
    Hiçbiri JSON nesnesi vermezse ``LLMBadOutput`` fırlatır.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMBadOutput("Model boş çıktı üretti")
    stripped = text.strip()
    candidates: list[str] = [stripped]
    candidates.extend(block.strip() for block in _FENCE_RE.findall(stripped))
    candidates.extend(_balanced_objects(stripped))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = _as_object(_loads_lenient(candidate))
        except (json.JSONDecodeError, RecursionError):
            continue
        if obj is not None:
            return obj
    raise LLMBadOutput(f"Çıktıda JSON nesnesi bulunamadı: {excerpt(stripped, 160)!r}")


# ---------------------------------------------------------------------------------------------------------
# Ortak protokol
# ---------------------------------------------------------------------------------------------------------


class LLM(Protocol):
    """Skorlama, raporlama ve RAG katmanlarının kullandığı asgari LLM arayüzü."""

    @property
    def model_name(self) -> str: ...

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> bool: ...

    def model_available(self) -> bool: ...


def normalize_model_name(name: str) -> str:
    """'qwen2.5' → 'qwen2.5:latest' (Ollama etiketi olmayan adları :latest ile listeler)."""
    name = (name or "").strip()
    if not name:
        return ""
    return name if ":" in name else f"{name}:latest"


def redact_url(url: str) -> str:
    """URL'deki kimlik bilgisini gizler: ``http://u:gizli@host:11434`` → ``http://u:***@host:11434``.

    Günlük satırları, ``LLMUnavailable`` metinleri ve bunlardan türeyen alarm gerekçeleri için; ters vekil
    arkasındaki Ollama'ya temel kimlik doğrulamayla bağlanılırken parola ES'e/webhook'a sızmamalıdır.
    """
    try:
        parts = urlsplit(url or "")
        if not parts.username and not parts.password:
            return url
        host = parts.hostname or ""
        if ":" in host:  # IPv6
            host = f"[{host}]"
        if parts.port:
            host += f":{parts.port}"
        if parts.username:
            host = f"{parts.username}:***@{host}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except ValueError:
        return "<url>"


# ---------------------------------------------------------------------------------------------------------
# Ollama istemcisi
# ---------------------------------------------------------------------------------------------------------


class OllamaClient:
    """Ollama HTTP API istemcisi (``/api/chat``, ``/api/generate``, ``/api/embed``, ``/api/tags``)."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._model = settings.ollama_model
        self._embedding_model = settings.ollama_embedding_model or settings.ollama_model
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._display_url = redact_url(self._base_url)  # hata/günlük metinlerinde yalnızca bu kullanılır
        self._legacy_embed = False  # sunucuda /api/embed yoksa (eski sürüm) bir kez tespit edilip hatırlanır
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=settings.ollama_timeout,
            transport=transport,
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        )

    # --- yaşam döngüsü ---
    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model

    @property
    def display_url(self) -> str:
        """Kimlik bilgisi gizlenmiş temel URL (günlük ve hata mesajları için)."""
        return self._display_url

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- HTTP yardımcıları ---
    def _request(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None, timeout: float | None = None
    ) -> httpx.Response:
        request_timeout: float | httpx.Timeout = timeout if timeout is not None else self._client.timeout
        try:
            return self._client.request(method, path, json=json_body, timeout=request_timeout)
        except httpx.TimeoutException as exc:
            raise LLMUnavailable(f"Ollama zaman aşımı ({method} {path}): {self._describe(exc)}") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"Ollama'ya bağlanılamadı ({self._display_url}{path}): {self._describe(exc)}") from exc

    def _describe(self, exc: BaseException) -> str:
        """İstisna metni; httpx ham URL'yi içeriyorsa kimlik bilgisi gizlenmiş hâliyle değiştirilir."""
        text = str(exc)
        if self._display_url != self._base_url and self._base_url in text:
            text = text.replace(self._base_url, self._display_url)
        return text

    def _parse(self, resp: httpx.Response, path: str) -> dict[str, Any]:
        body_excerpt = excerpt(resp.text, 300)
        if resp.status_code >= 500:
            raise LLMUnavailable(f"Ollama sunucu hatası HTTP {resp.status_code} ({path}): {body_excerpt}")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"Ollama isteği reddetti HTTP {resp.status_code} ({path}): {body_excerpt}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMUnavailable(f"Ollama JSON olmayan yanıt döndürdü ({path}): {body_excerpt}") from exc
        if not isinstance(data, dict):
            raise LLMUnavailable(f"Ollama beklenmeyen yanıt türü döndürdü ({path}): {type(data).__name__}")
        if data.get("error"):
            raise LLMUnavailable(f"Ollama hata döndürdü ({path}): {data['error']}")
        return data

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return self._parse(self._request("POST", path, json_body=payload, timeout=timeout), path)

    def _options(self, temperature: float | None, num_ctx: int | None) -> dict[str, Any]:
        return {
            "temperature": self.settings.ollama_temperature if temperature is None else float(temperature),
            "num_ctx": int(num_ctx or self.settings.ollama_num_ctx),
        }

    # --- LLM protokolü ---
    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": "json",
            "stream": False,
            "options": self._options(temperature, num_ctx),
        }
        data = self._post("/api/chat", payload, timeout)
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMBadOutput(
                f"Ollama /api/chat boş içerik döndürdü (done_reason={data.get('done_reason')!r})"
            )
        return extract_json_object(content)

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> str:
        payload = {
            "model": self._model,
            "prompt": user,
            "system": system,
            "stream": False,
            "options": self._options(temperature, num_ctx),
        }
        data = self._post("/api/generate", payload, timeout)
        response = data.get("response")
        if not isinstance(response, str):
            raise LLMBadOutput("Ollama /api/generate 'response' alanı döndürmedi")
        return response.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Metinleri vektörler; boş metinler için sıfır vektör döner (sıra ve uzunluk korunur)."""
        indexed = [(i, t) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
        vectors = self._embed_batch([t for _, t in indexed]) if indexed else []
        if len(vectors) != len(indexed):
            raise LLMBadOutput(f"Ollama {len(indexed)} metin için {len(vectors)} vektör döndürdü")
        dims = len(vectors[0]) if vectors else self.settings.embedding_dims
        result: list[list[float]] = [[0.0] * dims for _ in texts]
        for (i, _), vec in zip(indexed, vectors, strict=True):
            result[i] = vec
        return result

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._legacy_embed:
            return [self._embed_legacy(text) for text in texts]
        resp = self._request("POST", "/api/embed", json_body={"model": self._embedding_model, "input": texts})
        # Ollama eksik model için de 404 döner ama gövdesi {"error": "model ... not found"} olur; yalnızca uç
        # noktanın kendisi yoksa (düz "404 page not found", <0.3.4) eski /api/embeddings'e düşülür.
        if resp.status_code == 404 and not _json_error_body(resp):
            log.info("Ollama /api/embed bulunamadı (eski sürüm); /api/embeddings uç noktasına düşülüyor")
            self._legacy_embed = True
            return [self._embed_legacy(text) for text in texts]
        data = self._parse(resp, "/api/embed")
        return _coerce_vectors(data.get("embeddings"), "/api/embed")

    def _embed_legacy(self, text: str) -> list[float]:
        data = self._post("/api/embeddings", {"model": self._embedding_model, "prompt": text})
        return _coerce_vectors([data.get("embedding")], "/api/embeddings")[0]

    def list_models(self) -> list[str]:
        data = self._parse(self._request("GET", "/api/tags"), "/api/tags")
        models = data.get("models")
        if not isinstance(models, list):
            return []
        return [str(m.get("name", "")) for m in models if isinstance(m, dict) and m.get("name")]

    def health(self) -> bool:
        try:
            self.list_models()
        except LLMError as exc:
            log.warning("Ollama sağlık kontrolü başarısız: %s", exc)
            return False
        return True

    def model_available(self) -> bool:
        try:
            names = {normalize_model_name(n) for n in self.list_models()}
        except LLMError as exc:
            log.warning("Ollama model listesi alınamadı: %s", exc)
            return False
        wanted = normalize_model_name(self._model)
        if wanted not in names:
            log.warning(
                "Ollama'da model bulunamadı: %s (mevcut: %s)", self._model, ", ".join(sorted(names)) or "-"
            )
            return False
        return True



def _json_error_body(resp: httpx.Response) -> bool:
    """Yanıt gövdesi Ollama'nın ``{"error": "..."}`` biçiminde bir hata nesnesi mi?"""
    try:
        data = resp.json()
    except ValueError:
        return False
    return isinstance(data, dict) and bool(data.get("error"))

def _coerce_vectors(raw: Any, path: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise LLMBadOutput(f"Ollama {path} vektör listesi döndürmedi")
    vectors: list[list[float]] = []
    for vec in raw:
        if not isinstance(vec, list) or not vec:
            raise LLMBadOutput(f"Ollama {path} geçersiz vektör döndürdü")
        try:
            vectors.append([float(x) for x in vec])
        except (TypeError, ValueError) as exc:
            raise LLMBadOutput(f"Ollama {path} sayısal olmayan vektör döndürdü: {exc}") from exc
    return vectors


# ---------------------------------------------------------------------------------------------------------
# Deterministik yardımcılar (FakeOllama ve HeuristicLLM ortak)
# ---------------------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")


def hashed_vector(text: str, dims: int) -> list[float]:
    """Metni 'hashing trick' ile sabit boyutlu, L2-normalize edilmiş bir vektöre çevirir (deterministik)."""
    dims = max(1, int(dims))
    vec = [0.0] * dims
    for token in _TOKEN_RE.findall(tr_lower(text or "")):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------------------------------------
# Test için sahte istemci
# ---------------------------------------------------------------------------------------------------------

Responder = Callable[[str, str], dict[str, Any] | str]

DEFAULT_FAKE_VERDICT: dict[str, Any] = {
    "alarm_score": 10,
    "is_alarm": False,
    "reason": "Sahte model: varsayılan düşük skor.",
    "summary": "Sahte model özeti.",
    "topics": ["diğer"],
    "entities": [],
}


class FakeOllama:
    """``LLM`` protokolünü bellek içinde taklit eder.

    ``responder(system, user)`` bir sözlük (doğrudan karar) veya dize (modelin ham çıktısı; JSON çıkarma
    uygulanır, dolayısıyla düz yazı ``LLMBadOutput`` üretir) döndürebilir ya da istisna fırlatabilir.
    ``available=False`` ile tüm çağrılar ``LLMUnavailable`` fırlatır.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        embed_dims: int = 8,
        model_name: str = "fake",
        *,
        available: bool = True,
    ) -> None:
        self.responder = responder
        self.embed_dims = embed_dims
        self._model_name = model_name
        self.available = available
        self.calls: list[tuple[str, str, str]] = []  # (yöntem, system, user)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def chat_calls(self) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[0] == "chat_json"]

    def _ensure_available(self) -> None:
        if not self.available:
            raise LLMUnavailable("FakeOllama erişilemez olarak ayarlandı")

    def _respond(self, system: str, user: str) -> dict[str, Any] | str:
        if self.responder is None:
            return copy.deepcopy(DEFAULT_FAKE_VERDICT)
        return self.responder(system, user)

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self._ensure_available()
        self.calls.append(("chat_json", system, user))
        out = self._respond(system, user)
        if isinstance(out, dict):
            return copy.deepcopy(out)
        if isinstance(out, str):
            return extract_json_object(out)
        raise LLMBadOutput(f"FakeOllama responder desteklenmeyen tür döndürdü: {type(out).__name__}")

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> str:
        self._ensure_available()
        self.calls.append(("generate_text", system, user))
        out = self._respond(system, user)
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            return json.dumps(out, ensure_ascii=False)
        raise LLMBadOutput(f"FakeOllama responder desteklenmeyen tür döndürdü: {type(out).__name__}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_available()
        self.calls.append(("embed", "", "\n".join(texts)))
        return [hashed_vector(t, self.embed_dims) for t in texts]

    def health(self) -> bool:
        return self.available

    def model_available(self) -> bool:
        return self.available


# ---------------------------------------------------------------------------------------------------------
# Sezgisel (LLM'siz) değerlendirici — `--fake-llm` geliştirme modu
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskTerm:
    term: str  # KeywordMatcher deseni (kök + Türkçe ekler, '=' tam, 're:' regex)
    weight: int
    topic: str
    label: str = ""  # gerekçe metninde gösterilecek ad (regex desenleri için); boşsa term'den türetilir

    @property
    def display(self) -> str:
        return self.label or _display_term(self.term)


RISK_TERMS: tuple[RiskTerm, ...] = (
    # hukuk / yargı
    RiskTerm("tutukla", 14, "hukuk"),
    RiskTerm("gözaltı", 12, "hukuk"),
    RiskTerm("soruşturma", 12, "hukuk"),
    RiskTerm("iddianame", 10, "hukuk"),
    RiskTerm("yolsuzluk", 12, "hukuk"),
    RiskTerm("rüşvet", 12, "hukuk"),
    RiskTerm("kara para", 14, "hukuk"),
    RiskTerm("dolandırıcı", 12, "hukuk"),
    RiskTerm("hapis", 8, "hukuk"),
    RiskTerm("mahkeme", 6, "hukuk"),
    RiskTerm("yargıtay", 6, "hukuk"),
    RiskTerm("anayasa mahkemesi", 8, "hukuk"),
    # finans / fon
    RiskTerm("=tmsf", 14, "finans/fon"),
    RiskTerm("=masak", 14, "finans/fon"),
    RiskTerm("=spk", 12, "finans/fon"),
    RiskTerm("kayyum", 12, "finans/fon"),
    RiskTerm("iflas", 10, "finans/fon"),
    RiskTerm("konkordato", 8, "finans/fon"),
    RiskTerm("merkez bankası", 10, "finans/fon"),
    RiskTerm("faiz", 8, "finans/fon"),
    RiskTerm("borsa", 6, "finans/fon"),
    RiskTerm("yatırım fonu", 8, "finans/fon"),
    # siyaset / yönetim
    RiskTerm("kararname", 12, "siyaset"),
    RiskTerm("re:resm[iî] gazete", 10, "siyaset", label="resmi gazete"),
    RiskTerm("kabine", 8, "siyaset"),
    RiskTerm("istifa", 10, "siyaset"),
    RiskTerm("görevden al", 10, "siyaset"),
    RiskTerm("=tbmm", 6, "siyaset"),
    RiskTerm("kanun teklifi", 8, "siyaset"),
    RiskTerm("yasa", 5, "siyaset"),
    RiskTerm("seçim", 6, "siyaset"),
    # Ek zinciri fiil olumsuzlarını da ("atamam", "atamaz", "atamadı") kabul ederdi; ad çekimleri kalır.
    RiskTerm(r"re:\batama(?!m\b|m[ıi]ş|mak|maz|z|d[ıi]|l[ıi]|yan|y[ıi]p|yacak|yor)\w*", 6, "siyaset", label="atama"),
    RiskTerm("genel başkan", 5, "siyaset"),
    # güvenlik
    RiskTerm("operasyon", 8, "güvenlik"),
    RiskTerm("saldırı", 10, "güvenlik"),
    RiskTerm("patlama", 10, "güvenlik"),
    RiskTerm("terör", 10, "güvenlik"),
    RiskTerm("=ohal", 14, "güvenlik"),
    RiskTerm("şehit", 8, "güvenlik"),
    # dış politika
    RiskTerm("=nato", 8, "dış politika"),
    RiskTerm("dışişleri", 8, "dış politika"),
    RiskTerm("büyükelçi", 6, "dış politika"),
    RiskTerm("zirve", 5, "dış politika"),
    RiskTerm("yaptırım", 8, "dış politika"),
    # ekonomi
    # "zam" ek zinciriyle çok sık geçen "zaman/zamanla/zamanında"yı da yakalardı; (?!an) ile dışlanır.
    RiskTerm(r"re:\bzam(?!an)\w*", 6, "ekonomi", label="zam"),
    RiskTerm("enflasyon", 6, "ekonomi"),
    RiskTerm("bütçe", 6, "ekonomi"),
    RiskTerm("vergi", 6, "ekonomi"),
    RiskTerm("asgari ücret", 8, "ekonomi"),
    RiskTerm("ihale", 8, "ekonomi"),
    # sosyal
    RiskTerm("deprem", 8, "sosyal"),
    RiskTerm("=sel", 6, "sosyal"),
    RiskTerm("grev", 6, "sosyal"),
    RiskTerm("eğitim", 3, "sosyal"),
)

# Spor/magazin işaretleri skoru düşürür (anahtar kelime yan anlamda geçmiş olabilir).
NEGATIVE_TERMS: tuple[str, ...] = (
    "maç",
    "=gol",
    "transfer",
    "magazin",
    "dizi",
    "konser",
    "futbol",
    "şampiyon",
    "galibiyet",
    "teknik direktör",
    "ünlü",
    "burç",
)

KEYWORD_TOPICS: dict[str, str] = {
    "bakan": "siyaset",
    "cumhurbaşkanı": "siyaset",
    "fon": "finans/fon",
}

_HEURISTIC_BASE = 8
_KEYWORD_POINTS = 12
_KEYWORD_CAP = 36
_RISK_CAP = 50
_TITLE_BONUS = 6
_NEGATIVE_POINTS = 10
_NEGATIVE_CAP = 30

_CAP_WORD = r"[A-ZÇĞİÖŞÜ][\wçğıöşüâîû'’.-]*"
_ENTITY_RE = re.compile(rf"(?<![\w'’]){_CAP_WORD}(?:[ \t]+{_CAP_WORD})*")
_ENTITY_STOP = {
    "bu",
    "şu",
    "o",
    "ve",
    "ile",
    "ama",
    "fakat",
    "ancak",
    "son",
    "yeni",
    "bir",
    "da",
    "de",
    "ya",
    "hem",
    "ne",
    "ki",
    "için",
    "çok",
    "daha",
    "en",
    "her",
    "ilk",
    "dün",
    "bugün",
    "yarın",
    "flaş",
    "açıklama",
}
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def extract_entities(text: str, limit: int = 10) -> list[str]:
    """Büyük harfle başlayan sözcük dizilerini (kişi/kurum adayları) sırayla, tekrarsız döndürür.

    Cümle başındaki tek büyük harfli sözcükler ('Soruşturma', 'Hafta') özel ad sayılmaz; tek sözcük yalnızca
    kısaltmaysa ('TBMM', 'CHP') alınır. Baştaki bağlaç/işaret sözcükleri ('Bu', 'Son') atılır.
    """
    found: list[str] = []
    seen: set[str] = set()
    for m in _ENTITY_RE.finditer(normalize_ws(text or "").replace("\n", " ")):
        words = m.group(0).strip(" .'’-").split()
        while words and tr_lower(words[0]) in _ENTITY_STOP:
            words.pop(0)
        if not words or (len(words) == 1 and not _looks_like_acronym(words[0])):
            continue
        candidate = " ".join(words)
        key = tr_lower(candidate)
        if key in seen:
            continue
        seen.add(key)
        found.append(candidate)
        if len(found) >= limit:
            break
    return found


def first_sentences(text: str, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Metnin ilk birkaç cümlesini (özet amaçlı) döndürür."""
    flat = normalize_ws(text or "").replace("\n", " ")
    if not flat:
        return ""
    pieces = [p.strip() for p in _SENTENCE_RE.split(flat) if p.strip()]
    out = " ".join(pieces[:max_sentences])
    return excerpt(out, max_chars) if len(out) > max_chars else out


class HeuristicLLM:
    """Gerçek model olmadan deterministik karar üretir (``--fake-llm`` geliştirme modu).

    Skor; eşleşen anahtar kelime sayısı, risk terimlerinin ağırlıkları, başlıkta anahtar kelime bonusu ve
    spor/magazin cezasından hesaplanır. Çıktı şeması gerçek modelinkiyle aynıdır.
    """

    model_name_value = "heuristic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._keywords = KeywordMatcher(settings.keyword_list)
        self._risk = KeywordMatcher([t.term for t in RISK_TERMS])
        self._risk_index = {t.term: t for t in RISK_TERMS}
        self._negative = KeywordMatcher(list(NEGATIVE_TERMS))

    @property
    def model_name(self) -> str:
        return self.model_name_value

    # --- saf değerlendirme ---
    def evaluate(self, *, title: str = "", subtitle: str = "", content: str = "") -> dict[str, Any]:
        """Başlık/alt başlık/içerikten rubrik şemasında karar üretir (saf, deterministik)."""
        text = "\n".join(p for p in (title, subtitle, content) if p)
        kw_hits = self._keywords.find(text)
        risk_hits = self._risk.find(text)
        neg_hits = self._negative.find(text)

        score = float(_HEURISTIC_BASE)
        score += min(_KEYWORD_CAP, _KEYWORD_POINTS * len(kw_hits))
        score += min(_RISK_CAP, sum(self._risk_index[h.keyword].weight for h in risk_hits))
        if title and self._keywords.find(title):
            score += _TITLE_BONUS
        score -= min(_NEGATIVE_CAP, _NEGATIVE_POINTS * len(neg_hits))
        alarm_score = _clamp_score(score)

        topic_weights: Counter[str] = Counter()
        for h in kw_hits:
            topic_weights[KEYWORD_TOPICS.get(tr_lower(h.keyword), "siyaset")] += _KEYWORD_POINTS
        for h in risk_hits:
            term = self._risk_index[h.keyword]
            topic_weights[term.topic] += term.weight
        ranked = sorted(topic_weights.items(), key=lambda kv: (-kv[1], TOPICS.index(kv[0])))
        topics = [t for t, _ in ranked[:3]] or ["diğer"]

        kw_text = ", ".join(f"{h.keyword} ({h.count} kez)" for h in kw_hits) or "yok"
        risk_text = ", ".join(self._risk_index[h.keyword].display for h in risk_hits) or "yok"
        neg_text = ", ".join(_display_term(h.keyword) for h in neg_hits) or "yok"
        reason = (
            f"Sezgisel değerlendirme (LLM bağlı değil): anahtar kelimeler: {kw_text}; "
            f"risk terimleri: {risk_text}; spor/magazin işaretleri: {neg_text}. "
            f"Hesaplanan skor {alarm_score}/100."
        )
        summary = (
            first_sentences(_join_sentences(title, content.replace(TRUNCATION_MARKER, "")))
            or "Özet üretilemedi."
        )
        return {
            "alarm_score": alarm_score,
            "is_alarm": alarm_score >= self.settings.alarm_threshold,
            "reason": reason,
            "summary": summary,
            "topics": topics,
            "entities": extract_entities(" ".join(p for p in (title, subtitle) if p)),
        }

    # --- LLM protokolü ---
    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        fields = parse_user_prompt(user)
        if not any((fields["title"], fields["subtitle"], fields["content"])):
            return self.evaluate(content=user)
        return self.evaluate(title=fields["title"], subtitle=fields["subtitle"], content=fields["content"])

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        timeout: float | None = None,
    ) -> str:
        return (
            "Sezgisel mod: gerçek bir dil modeli bağlı olmadığı için serbest metin üretilmedi. "
            f"İstem özeti: {excerpt(user, 400)}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hashed_vector(t, self.settings.embedding_dims) for t in texts]

    def health(self) -> bool:
        return True

    def model_available(self) -> bool:
        return True


def _display_term(term: str) -> str:
    if term.startswith("re:"):
        return term[3:]
    return term.lstrip("=~")


def _looks_like_acronym(word: str) -> bool:
    """'TBMM', 'CHP', 'AK' gibi tamamı büyük harfli kısaltmalar tek başına varlık sayılır."""
    letters = [ch for ch in word if ch.isalpha()]
    return len(letters) >= 2 and all(ch.isupper() for ch in letters)


def _join_sentences(*parts: str) -> str:
    """Parçaları, her birinin sonunda noktalama olacak şekilde tek metinde birleştirir."""
    out: list[str] = []
    for part in parts:
        text = normalize_ws(part or "").replace("\n", " ").strip()
        if not text:
            continue
        if text[-1] not in ".!?…:":
            text += "."
        out.append(text)
    return " ".join(out)
