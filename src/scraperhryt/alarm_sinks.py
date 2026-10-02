"""Alarm kanalları (sink'ler): JSONL/log dosyası, genel JSON webhook ve Telegram.

Her kanal ``AlarmSink`` arayüzünü uygular: ``name`` ve ``send(event)``. Gönderim hatası ``SinkError`` (ya da
``httpx`` kaynaklı bir istisna) olarak yükselir; alarm katmanı bunları loglar ve asla ölümcül saymaz.
Gizli bilgiler (Telegram bot token'ı, webhook URL'sinin yolu) hata mesajlarına yazılmaz.
"""

from __future__ import annotations

import html
import json
import logging
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from .config import Settings
from .models import AlarmEvent

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
TELEGRAM_MAX_LEN = 4096
_REASON_LIMIT = 1500
_TR_TZ = timezone(timedelta(hours=3), "TRT")  # Türkiye 2016'dan beri sabit UTC+3 kullanır


class SinkError(Exception):
    """Alarm kanalına gönderim başarısız (ağ hatası, HTTP hata kodu, API 'ok: false')."""


class AlarmSink(Protocol):
    name: str

    def send(self, event: AlarmEvent) -> None: ...


# ---------------------------------------------------------------------------------------------------------
# Metin biçimleme
# ---------------------------------------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return ""
    return text[: limit - 1].rstrip() + "…"


def _fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_TR_TZ).strftime("%d.%m.%Y %H:%M")


def _reason(event: AlarmEvent) -> str:
    # alarm_reason, apply_verdict politikası gereği LLM özetini de içerir; boşsa özet tek başına gösterilir.
    return (event.alarm_reason or "").strip() or (event.llm_summary or "").strip() or "Gerekçe belirtilmedi."


def format_alarm_text(event: AlarmEvent) -> str:
    """Webhook/Slack/Mattermost için markdown benzeri düz metin özeti."""
    keywords = ", ".join(event.matched_keywords) or "-"
    lines = [
        f"*ALARM [{event.alarm_score}/100]* {event.title.strip()}",
        f"Kaynak: {event.source} · Haber tarihi: {_fmt_date(event.published_at)} · Anahtar kelimeler: {keywords}",
    ]
    if event.subtitle.strip():
        lines.append(f"_{event.subtitle.strip()}_")
    lines.append(_truncate(_reason(event), _REASON_LIMIT))
    lines.append(event.content_url)
    return "\n".join(lines)


def _fit_html(plain: str, budget: int) -> str:
    """Düz metni HTML-kaçışlı haliyle ``budget`` karaktere sığdırır (kaçış dizileri ortadan kesilmez)."""
    text = plain
    while True:
        escaped = html.escape(text, quote=True)
        if len(escaped) <= budget or not text:
            return escaped
        text = _truncate(text, max(0, int(len(text) * 0.9) - 1))


def format_alarm_html(event: AlarmEvent, max_len: int = TELEGRAM_MAX_LEN) -> str:
    """Telegram ``parse_mode=HTML`` için kaçışlı HTML mesaj (toplam uzunluk ``max_len`` ile sınırlı)."""
    esc = lambda s: html.escape((s or "").strip(), quote=True)  # noqa: E731 - kısa yerel takma ad
    keywords = ", ".join(event.matched_keywords) or "-"
    header = f"<b>ALARM [{event.alarm_score}/100]</b> <b>{_fit_html(event.title, 300)}</b>"
    meta = (
        f"<i>Kaynak:</i> {esc(event.source)} · <i>Haber tarihi:</i> {_fmt_date(event.published_at)} · "
        f"<i>Anahtar kelimeler:</i> {_fit_html(keywords, 200)}"
    )
    link = f'<a href="{esc(event.content_url)}">Habere git</a>'
    parts = [header, meta]
    if event.subtitle.strip():
        parts.append(f"<i>{_fit_html(event.subtitle, 300)}</i>")
    fixed_len = sum(len(p) for p in parts) + len(link) + len(parts) + 1  # satır sonları dahil
    budget = max(0, max_len - fixed_len)
    reason = _fit_html(_truncate(_reason(event), _REASON_LIMIT), budget)
    if reason:
        parts.append(reason)
    parts.append(link)
    return "\n".join(parts)


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/…" if parts.scheme and parts.netloc else "<url>"


def _http_error_text(exc: Exception, *secrets: str) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        text = f"HTTP {exc.response.status_code}"
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# ---------------------------------------------------------------------------------------------------------
# Kanallar
# ---------------------------------------------------------------------------------------------------------


class LogSink:
    """Her alarmı WARNING seviyesinde loglar ve ``path`` dosyasına bir JSON satırı ekler (dizinler oluşturulur)."""

    name = "log"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def send(self, event: AlarmEvent) -> None:
        log.warning("ALARM [%d] %s — %s", event.alarm_score, event.title, event.content_url)
        line = json.dumps(event.to_es_document(), ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class WebhookSink:
    """Genel JSON webhook: ``{"text": <özet>, "event": <AlarmEvent>}`` gövdesiyle POST (Slack/Discord/Mattermost uyumlu)."""

    name = "webhook"

    def __init__(self, url: str, *, timeout: float = DEFAULT_TIMEOUT, client: httpx.Client | None = None) -> None:
        self.url = url.strip()
        self.timeout = timeout
        self._client = client

    def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(url, json=payload, timeout=self.timeout)
        return httpx.post(url, json=payload, timeout=self.timeout)

    def send(self, event: AlarmEvent) -> None:
        payload = {"text": format_alarm_text(event), "event": event.to_message()}
        try:
            response = self._post(self.url, payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SinkError(
                f"Webhook gönderimi başarısız ({_redact_url(self.url)}): {_http_error_text(exc, self.url)}"
            ) from exc
        log.info("Alarm webhook'a gönderildi (%s): %s", _redact_url(self.url), event.alarm_id)


class TelegramSink:
    """Telegram Bot API ``sendMessage`` (parse_mode=HTML). Token hiçbir log/hata mesajına yazılmaz."""

    name = "telegram"
    api_base = "https://api.telegram.org"

    def __init__(
        self, token: str, chat_id: str, *, timeout: float = DEFAULT_TIMEOUT, client: httpx.Client | None = None
    ) -> None:
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.timeout = timeout
        self._client = client

    @property
    def url(self) -> str:
        return f"{self.api_base}/bot{self.token}/sendMessage"

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(self.url, json=payload, timeout=self.timeout)
        return httpx.post(self.url, json=payload, timeout=self.timeout)

    def send(self, event: AlarmEvent) -> None:
        payload = {
            "chat_id": self.chat_id,
            "text": format_alarm_html(event),
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }
        try:
            response = self._post(payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SinkError(f"Telegram gönderimi başarısız: {_http_error_text(exc, self.token)}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise SinkError("Telegram yanıtı JSON değil") from exc
        if not isinstance(body, dict) or not body.get("ok"):
            description = body.get("description") if isinstance(body, dict) else body
            raise SinkError(f"Telegram API hatası: {description}")
        log.info("Alarm Telegram'a gönderildi (chat %s): %s", self.chat_id, event.alarm_id)


def build_sinks(settings: Settings) -> list[AlarmSink]:
    """Ayarlara göre kanal listesi: her zaman ``LogSink``; webhook ve Telegram yapılandırılmışsa eklenir."""
    sinks: list[AlarmSink] = [LogSink(settings.alarm_log_path)]
    if settings.alarm_webhook_url.strip():
        sinks.append(WebhookSink(settings.alarm_webhook_url))
    token, chat_id = settings.telegram_bot_token.strip(), settings.telegram_chat_id.strip()
    if token and chat_id:
        sinks.append(TelegramSink(token, chat_id))
    elif token or chat_id:
        log.warning("Telegram kanalı için TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID birlikte gerekli; kanal devre dışı")
    log.info("Alarm kanalları: %s", ", ".join(s.name for s in sinks))
    return sinks
