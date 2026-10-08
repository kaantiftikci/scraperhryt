"""scraperhryt komut satırı arayüzü.

Alt komutlar:

    setup     RabbitMQ topolojisini ve Elasticsearch indekslerini oluşturur, Ollama modelini denetler
    check     RabbitMQ / Elasticsearch / Ollama erişilebilirlik raporu (gecikmelerle)
    scrape    Hürriyet Gündem + 12punto kazıyıcısı (tek tur ya da sürekli)
    filter    Anahtar kelime filtresi (q.articles.raw → q.articles.keyword / q.articles.scored)
    score     Ollama LLM skorlama (q.articles.keyword → q.articles.scored)
    alarm     Alarm katmanı (q.articles.scored → Elasticsearch + kanallar + q.alarms)
    report    Raporlama katmanı (alarm özetleri + periyodik raporlar)
    api       FastAPI servisi (arama, alarmlar, raporlar, /ask, gösterge paneli)
    ask       RAG soru-cevap: son haberlere dayanarak bir soruyu yanıtlar
    run-all   Tüm katmanları tek süreçte iş parçacıkları olarak çalıştırır (geliştirme / küçük kurulum)

Her alt komut ihtiyaç duyduğu modülleri tembel (lazy) olarak içe aktarır; ``--help`` hiçbir servise ya da
modüle bağımlı değildir. Ortak ayarlar ``Settings`` (ortam değişkenleri / .env) üzerinden gelir; komut satırı
seçenekleri yalnızca ilgili alanları geçersiz kılar.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from . import __version__
from .broker import QUEUE_BINDINGS, Broker, InMemoryBroker, Message, Queue, make_broker, topic_matches
from .config import Settings
from .logging_setup import configure, parse_level
from .models import Answer, Report, utcnow

log = logging.getLogger(__name__)

PROG = "scraperhryt"
SOURCE_CHOICES = ("hurriyet", "12punto")
ISTANBUL = ZoneInfo("Europe/Istanbul")

JOIN_TIMEOUT_SECONDS = 15.0  # kapanışta iş parçacıklarına tanınan toplam süre
IDLE_POLL_SECONDS = 0.5  # bellek içi kuyruk boşken yoklama aralığı
PROBE_TIMEOUT_SECONDS = 5.0  # `check` komutunun servis başına zaman aşımı
DEFAULT_IDLE_EXIT_SECONDS = 30.0  # run-all --once (RabbitMQ): kuyruklar bu kadar süre boş kalınca çık
TOP_ALARMS = 5
SCORER_PREFETCH = 1  # LLM yavaş olduğundan skorlayıcı aynı anda tek mesaj alır
# Ortam değişkeni: 1/true ise `setup` ve `check` Ollama sorununu hata değil uyarı sayar (compose'da --ollama-optional
# bayrağını geçmenin yolu; .env.example'a bakın).
OLLAMA_OPTIONAL_ENV = "OLLAMA_OPTIONAL"
# Ortam değişkeni: `setup` Ollama (sunucu + model) hazır olana dek en çok bu kadar saniye bekler; 0 = beklemez.
# `docker compose --profile ollama up` ilk açılışta modeli indirirken `setup` servisinin hata vermemesi için
# .env içinde OLLAMA_WAIT_SECONDS=900 gibi bir değer verin (bkz. .env.example).
OLLAMA_WAIT_ENV = "OLLAMA_WAIT_SECONDS"
OLLAMA_WAIT_POLL_SECONDS = 5.0  # `setup` Ollama'yı beklerken sondalar arası aralık
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "evet"})

# (argparse alanı, Settings alanı): komut satırı seçenekleri ayarları bu eşlemeyle geçersiz kılar.
_SETTING_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("log_level", "log_level"),
    ("source", "sources"),
    ("backfill_days", "backfill_days"),
    ("interval", "scrape_interval_seconds"),
    ("host", "api_host"),
    ("port", "api_port"),
)

Handler = Callable[[Message], None]


# ---------------------------------------------------------------------------------------------------------
# Küçük yardımcılar
# ---------------------------------------------------------------------------------------------------------


def env_flag(name: str) -> bool:
    """Ortam değişkeni 1/true/yes/on/evet ise ``True`` (büyük/küçük harf duyarsız); yoksa ya da başka değerse ``False``."""
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def env_float(name: str, default: float) -> float:
    """Ortam değişkenini sayıya çevirir; yoksa ya da sayı değilse (uyarı loglanır) ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r sayı değil; %s kullanılıyor", name, raw, default)
        return default


def build_settings(args: argparse.Namespace) -> Settings:
    """Ortam/.env ayarlarını yükler; komut satırında verilen seçenekler ilgili alanları geçersiz kılar."""
    overrides: dict[str, Any] = {}
    for arg_name, field_name in _SETTING_OVERRIDES:
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    return Settings(**overrides)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Sütun genişliklerini hizalayan sade metin tablo."""
    table = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in table:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    lines = [fmt(list(headers)), "  ".join("-" * width for width in widths)]
    lines.extend(fmt(row) for row in table)
    return "\n".join(lines)


def fmt_dt(value: datetime | str | None) -> str:
    """Tarihi İstanbul saatiyle ``gg.aa.yyyy SS:DD`` biçiminde gösterir; bilinmiyorsa '-'."""
    if value is None or value == "":
        return "-"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=ISTANBUL)
    return value.astimezone(ISTANBUL).strftime("%d.%m.%Y %H:%M")


def fmt_latency(ms: float | None) -> str:
    return "-" if ms is None else f"{ms:.0f} ms"


def describe_exc(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def redact_url(url: str) -> str:
    """Bağlantı dizesindeki parolayı gizler (günlük ve tablo çıktıları için)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<url>"
    netloc = parts.hostname or ""
    if parts.port:
        netloc += f":{parts.port}"
    if parts.username:
        netloc = f"{parts.username}:***@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def close_llm(llm: Any) -> None:
    """``OllamaClient`` HTTP havuzunu kapatır; kapatma gerektirmeyen (sezgisel/sahte) LLM'lerde hiçbir şey yapmaz."""
    from .pipeline.llm import OllamaClient

    if isinstance(llm, OllamaClient):
        llm.close()


# ---------------------------------------------------------------------------------------------------------
# Servis sondaları (setup / check)
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    name: str
    ok: bool
    detail: str
    latency_ms: float | None = None
    required: bool = True

    @property
    def label(self) -> str:
        if self.ok:
            return "OK"
        return "HATA" if self.required else "UYARI"


def probe_rabbitmq(settings: Settings, timeout: float = PROBE_TIMEOUT_SECONDS) -> ServiceStatus:
    """Tek denemeyle AMQP bağlantısı ve kanal açar; gecikmeyi ölçer."""
    import pika

    target = redact_url(settings.rabbitmq_url)
    params = pika.URLParameters(settings.rabbitmq_url)
    params.connection_attempts = 1
    params.socket_timeout = timeout
    params.blocked_connection_timeout = timeout
    started = time.perf_counter()
    try:
        connection = pika.BlockingConnection(params)
        try:
            connection.channel().close()
        finally:
            connection.close()
    except Exception as exc:
        return ServiceStatus("RabbitMQ", False, f"bağlanılamadı ({target}): {describe_exc(exc)}")
    latency = (time.perf_counter() - started) * 1000
    return ServiceStatus("RabbitMQ", True, f"{target} bağlantı ve kanal açıldı", latency)


def probe_elasticsearch(settings: Settings, timeout: float = PROBE_TIMEOUT_SECONDS) -> ServiceStatus:
    """``GET /`` ve ``GET /_cluster/health`` ile sürüm ve küme durumunu alır."""
    import httpx

    base = settings.elasticsearch_url.rstrip("/")
    headers = {"Authorization": f"ApiKey {settings.elasticsearch_api_key}"} if settings.elasticsearch_api_key else {}
    started = time.perf_counter()
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=timeout) as client:
            info = client.get("/")
            info.raise_for_status()
            health = client.get("/_cluster/health")
            health.raise_for_status()
            version = str(info.json().get("version", {}).get("number", "?"))
            body = health.json()
    except (httpx.HTTPError, ValueError) as exc:
        return ServiceStatus("Elasticsearch", False, f"erişilemiyor ({base}): {describe_exc(exc)}")
    latency = (time.perf_counter() - started) * 1000
    status = str(body.get("status", "?"))
    cluster = str(body.get("cluster_name", "?"))
    detail = f"küme '{cluster}' durum={status}, sürüm {version} ({base})"
    if status not in ("green", "yellow"):
        return ServiceStatus("Elasticsearch", False, detail + " — küme kırmızı, yazma işlemleri başarısız olabilir", latency)
    return ServiceStatus("Elasticsearch", True, detail, latency)


def probe_ollama(
    settings: Settings, *, required: bool = True, timeout: float = PROBE_TIMEOUT_SECONDS
) -> ServiceStatus:
    """Ollama sunucusunu (``/api/tags``) ve yapılandırılan modelin yüklü olup olmadığını denetler.

    Sonda istekleri ``ollama_timeout`` (dakikalar; LLM çağrıları için) yerine ``timeout`` ile sınırlanır: paketleri
    düşüren (reddetmeyen) bir ana makinede ``check``/``setup``/servis başlangıcı dakikalarca askıda kalmasın.
    """
    from .pipeline.llm import OllamaClient

    base = settings.ollama_base_url
    model = settings.ollama_model
    client = OllamaClient(settings.model_copy(update={"ollama_timeout": float(timeout)}))
    try:
        started = time.perf_counter()
        healthy = client.health()
        latency = (time.perf_counter() - started) * 1000
        if not healthy:
            return ServiceStatus(
                "Ollama", False, f"erişilemiyor ({base}); başlatmak için: ollama serve", None, required
            )
        available = client.model_available()
    finally:
        client.close()
    if not available:
        return ServiceStatus(
            "Ollama", False, f"sunucu çalışıyor ({base}) ama '{model}' yüklü değil; ollama pull {model}", latency, required
        )
    return ServiceStatus("Ollama", True, f"model '{model}' yüklü ({base})", latency, required)


def wait_for_ollama(
    settings: Settings,
    *,
    required: bool = True,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    max_wait: float = 0.0,
    poll: float = OLLAMA_WAIT_POLL_SECONDS,
) -> ServiceStatus:
    """``probe_ollama``'yı Ollama hazır olana ya da ``max_wait`` saniye dolana dek yineler (``setup`` için).

    Sunucunun henüz ayağa kalkmadığı ya da modelin indirilmekte olduğu durumları (``docker compose --profile ollama``
    ilk açılışı) kapsar; ``max_wait`` 0 ise tek sonda yapılır. Son sondanın durumu döner.
    """
    status = probe_ollama(settings, required=required, timeout=timeout)
    if status.ok or max_wait <= 0:
        return status
    deadline = time.monotonic() + max_wait
    log.warning("Ollama hazır değil: %s; en çok %.0f sn beklenecek (%s)", status.detail, max_wait, OLLAMA_WAIT_ENV)
    waited = 0.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning("Ollama %.0f sn içinde hazır olmadı: %s", max_wait, status.detail)
            return ServiceStatus(
                status.name, False, f"{status.detail} ({max_wait:.0f} sn beklendi)", status.latency_ms, required
            )
        pause = min(poll, remaining)
        time.sleep(pause)
        waited += pause
        status = probe_ollama(settings, required=required, timeout=timeout)
        if status.ok:
            log.info("Ollama %.0f sn sonra hazır: %s", waited, status.detail)
            return status
        log.debug("Ollama hâlâ hazır değil (%.0f sn geçti): %s", waited, status.detail)


# ---------------------------------------------------------------------------------------------------------
# LLM / depo kurucuları
# ---------------------------------------------------------------------------------------------------------


def warn_ollama(problem: str, settings: Settings) -> None:
    log.warning(
        "UYARI: %s. Çözüm: Ollama'yı başlatın (`ollama serve` ya da `docker compose --profile ollama up -d`) "
        "ve modeli indirin (`ollama pull %s`). Kullanılan adres: OLLAMA_BASE_URL=%s",
        problem,
        settings.ollama_model,
        settings.ollama_base_url,
    )


def build_llm(settings: Settings, *, fake: bool, fallback: bool) -> Any:
    """``--fake-llm`` → HeuristicLLM; aksi halde OllamaClient. Ollama hazır değilse uyarır ama istemciyi
    korur: skorlayıcı ``LLMUnavailable``'da mesajı gecikmeli yeniden dener, Ollama ayağa kalkınca kendiliğinden
    toparlanır. Yalnızca ``fallback=True`` (run-all ``--llm-fallback``) ise bu çalıştırma için HeuristicLLM'e
    geri düşülür."""
    from .pipeline.llm import HeuristicLLM, OllamaClient

    if fake:
        log.info("Sezgisel LLM kullanılıyor (--fake-llm): skorlar anahtar kelime ve risk terimi sayımına dayanır")
        return HeuristicLLM(settings)
    client = OllamaClient(settings)
    # Hazırlık denetimi kısa sondayla yapılır (``ollama_timeout`` dakikalar sürebilir; askıda kalan bir ana makine
    # servis başlangıcını geciktirmesin). Dönen istemci LLM çağrıları için tam zaman aşımını korur.
    status = probe_ollama(settings, timeout=PROBE_TIMEOUT_SECONDS)
    if status.ok:
        log.info("Ollama hazır: %s, model=%s", settings.ollama_base_url, settings.ollama_model)
        return client
    warn_ollama(status.detail, settings)
    if not fallback:
        log.warning("Ollama hazır olana dek LLM gerektiren mesajlar gecikmeli yeniden denenecek")
        return client
    client.close()
    log.warning(
        "--llm-fallback: bu çalıştırma boyunca LLM yerine sezgisel değerlendirici (HeuristicLLM) kullanılacak; "
        "alarm skorları yaklaşık olacaktır, Ollama sonradan ayağa kalksa da kullanılmaz"
    )
    return HeuristicLLM(settings)


def prepare_store(settings: Settings, *, in_memory: bool) -> Any:
    """Depoyu kurar ve indekslerin var olduğundan emin olur (Elasticsearch'te idempotent)."""
    from .store import make_store

    store = make_store(settings, in_memory=in_memory)
    store.ensure_indices()
    return store


# ---------------------------------------------------------------------------------------------------------
# Sinyal yönetimi ve iş parçacığı denetimi
# ---------------------------------------------------------------------------------------------------------


@contextmanager
def signal_scope(stop: threading.Event) -> Iterator[None]:
    """SIGINT/SIGTERM'de ``stop`` olayını set eder; ikinci sinyalde beklemeden çıkar. Çıkışta eski işleyicileri geri yükler."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handle(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        if stop.is_set():
            log.warning("%s tekrar alındı; beklemeden çıkılıyor", name)
            raise KeyboardInterrupt
        log.info("%s alındı; servisler düzgün biçimde durduruluyor...", name)
        stop.set()

    previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class Supervisor:
    """Adlandırılmış iş parçacıklarını başlatır; biri çökerse kaydeder ve herkesi durdurur."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.threads: list[threading.Thread] = []
        self.failures: list[str] = []
        self._lock = threading.Lock()

    def spawn(self, name: str, target: Callable[[], Any]) -> threading.Thread:
        thread = threading.Thread(target=self._guard, args=(name, target), name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        log.info("'%s' iş parçacığı başlatıldı", name)
        return thread

    def _guard(self, name: str, target: Callable[[], Any]) -> None:
        try:
            target()
        except (Exception, SystemExit):
            log.exception("'%s' iş parçacığı beklenmeyen hatayla çöktü; tüm servisler durduruluyor", name)
            with self._lock:
                self.failures.append(name)
            self.stop.set()
        else:
            log.info("'%s' iş parçacığı sonlandı", name)

    def wait(self) -> None:
        """Durdurma olayı set edilene ya da tüm iş parçacıkları bitene kadar bekler."""
        while not self.stop.is_set() and any(thread.is_alive() for thread in self.threads):
            self.stop.wait(IDLE_POLL_SECONDS)

    def join(self, timeout: float = JOIN_TIMEOUT_SECONDS) -> None:
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                log.warning("'%s' iş parçacığı %.0fs içinde durmadı; arka planda bırakılıyor", thread.name, timeout)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def alive(self) -> list[str]:
        """Hâlâ çalışan iş parçacıklarının adları (``join`` zaman aşımından sonra)."""
        return [thread.name for thread in self.threads if thread.is_alive()]


def shutdown_timeout_for(settings: Settings, llm: Any) -> float:
    """Kapanışta iş parçacıklarına tanınan süre. Skorlayıcı bir Ollama çağrısının ortasında olabilir ve çağrı
    süreç içi tüm denemeleriyle ``MAX_LLM_ATTEMPTS × ollama_timeout`` sürebilir; o bitmeden broker/LLM istemcisi
    kapatılırsa mesaj ikinci kez skorlanıp yayınlanır. Sezgisel/sahte LLM'de kısa sabit süre yeter."""
    from .pipeline.llm import OllamaClient
    from .pipeline.scorer import MAX_LLM_ATTEMPTS

    if isinstance(llm, OllamaClient):
        return max(JOIN_TIMEOUT_SECONDS, MAX_LLM_ATTEMPTS * float(settings.ollama_timeout))
    return JOIN_TIMEOUT_SECONDS


class PipelineCounters:
    """run-all tüketicilerinin işlediği mesaj sayıları ve son etkinlik zamanı (boşta kalma tespiti için)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.processed: Counter[str] = Counter()
        self.in_flight = 0
        self.last_activity = time.monotonic()

    def begin(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.last_activity = time.monotonic()

    def end(self, queue: str, success: bool) -> None:
        with self._lock:
            self.in_flight -= 1
            self.last_activity = time.monotonic()
            if success:
                self.processed[queue] += 1

    def is_idle(self, seconds: float) -> bool:
        with self._lock:
            return self.in_flight == 0 and (time.monotonic() - self.last_activity) >= seconds

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.processed)


def tracked_handler(queue: str, handler: Handler, counters: PipelineCounters) -> Handler:
    """İşleyiciyi ``counters`` ile sarar: her mesaj için başlangıç/bitiş ve başarı sayımı (boşta kalma tespiti)."""

    def tracked(msg: Message) -> None:
        counters.begin()
        success = False
        try:
            handler(msg)
            success = True
        finally:
            counters.end(queue, success)

    return tracked


def consume_loop(
    broker: Broker,
    queue: str,
    handler: Handler,
    stop: threading.Event,
    counters: PipelineCounters,
    *,
    prefetch: int | None = None,
    on_idle: Callable[[], Any] | None = None,
) -> None:
    """``stop`` set edilene kadar kuyruğu tüketir. RabbitMQ'da ``consume`` zaten bloklar; bellek içi broker
    kuyruk boşalınca döndüğünden kısa aralıklarla yeniden yoklanır ve her boş turda ``on_idle`` (varsa) çağrılır
    (ör. süre eşiği dolan alarm özetini üretmek için). ``on_idle`` hatası loglanır, döngüyü durdurmaz."""
    tracked = tracked_handler(queue, handler, counters)
    while not stop.is_set():
        processed = broker.consume(queue, tracked, prefetch=prefetch, stop_event=stop)
        if processed == 0:
            if on_idle is not None:
                try:
                    on_idle()
                except Exception as exc:
                    log.error("Boşta işlem başarısız (%s): %s", queue, describe_exc(exc))
            if stop.wait(IDLE_POLL_SECONDS):
                break


def start_api_server(app: Any, settings: Settings, supervisor: Supervisor) -> Any:
    """uvicorn'u kendi ``Server`` nesnesiyle bir iş parçacığında başlatır; ``server.should_exit`` ile durdurulur."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        log_level=parse_level(settings.log_level),
    )
    server = uvicorn.Server(config)
    supervisor.spawn("api", server.run)
    log.info("API: http://%s:%d (gösterge paneli: / , belgeler: /docs)", settings.api_host, settings.api_port)
    return server


# ---------------------------------------------------------------------------------------------------------
# Çıktı biçimlendiriciler
# ---------------------------------------------------------------------------------------------------------


def format_answer(answer: Answer, *, show_sources: bool = False) -> str:
    lines = [
        f"Soru: {answer.question}",
        f"Model: {answer.model or '-'} | bulunan haber: {answer.retrieved_count} | "
        f"arama terimleri: {', '.join(answer.search_terms) or '-'}",
        "",
        answer.answer.strip() or "(cevap üretilemedi)",
    ]
    if not show_sources:
        if answer.sources:
            lines.extend(["", f"({len(answer.sources)} haber kaynak alındı; kaynakları görmek için --sources)"])
        return "\n".join(lines)
    if answer.timeline:
        lines.extend(["", "Zaman çizelgesi (eski → yeni):"])
        for item in answer.timeline:
            marker = f"[{item.citation}] " if item.citation else ""
            lines.append(f"  {fmt_dt(item.date):<17} {marker}{item.event}")
    if answer.sources:
        lines.extend(["", "Kaynaklar:"])
        for index, citation in enumerate(answer.sources, 1):
            meta = " | ".join(part for part in (citation.source, fmt_dt(citation.published_at)) if part and part != "-")
            lines.append(f"  [{index}] {citation.title} ({meta})" if meta else f"  [{index}] {citation.title}")
            lines.append(f"      {citation.content_url}")
    return "\n".join(lines)


def describe_report(report: Report | None) -> str:
    if report is None:
        return "Rapor üretilmedi (pencerede veri yok)."
    stats = report.stats if isinstance(report.stats, dict) else {}
    lines = [
        f"Rapor: {report.report_id} ({report.kind})",
        f"Pencere: {fmt_dt(report.window_start)} – {fmt_dt(report.window_end)} | üretim: {fmt_dt(report.generated_at)}",
        f"Haber: {stats.get('total', 0)} | Alarm: {stats.get('alarms', 0)} | Model: {report.model or '-'}",
        "",
        report.narrative.strip() or "(anlatı yok)",
    ]
    return "\n".join(lines)


def published_counts(broker: InMemoryBroker) -> dict[str, int]:
    """Bellek içi broker'da yayınlanan mesajları bağlı oldukları kuyruklara göre sayar."""
    counts: Counter[str] = Counter()
    for msg in broker.published:
        for queue, (pattern, _retry_key) in QUEUE_BINDINGS.items():
            if topic_matches(pattern, msg.routing_key):
                counts[str(queue)] += 1
    return dict(counts)


def top_alarms(store: Any, since: datetime, limit: int = TOP_ALARMS) -> list[dict[str, Any]]:
    """Depodaki (bu çalıştırmada üretilen) alarmları skora göre sıralar."""
    # recent_alarms haber tarihine göre süzer; "bu çalıştırmada üretilen" için alarm zamanına (raised_at) bakılır.
    from .reporting.prompts import parse_datetime

    docs = [
        doc
        for doc in store.recent_alarms(size=max(limit * 20, 200))
        if (raised := parse_datetime(doc.get("raised_at"))) is not None and raised >= since
    ]
    ranked = sorted(docs, key=lambda doc: int(doc.get("alarm_score", 0) or 0), reverse=True)
    return ranked[:limit]


def print_run_summary(
    *,
    title: str,
    scrape_summary: str,
    queue_counts: dict[str, int],
    count_label: str,
    alarm_count: int,
    alarms: list[dict[str, Any]],
    dead_letters: int | None,
) -> None:
    rows = [(str(queue), queue_counts.get(str(queue), 0)) for queue in QUEUE_BINDINGS]
    dead = "bilinmiyor (RabbitMQ yönetim panelinde q.dead_letter kuyruğuna bakın)" if dead_letters is None else str(dead_letters)
    lines = [
        f"=== {title} ===",
        f"Kazıma: {scrape_summary}",
        "",
        render_table(("Kuyruk", count_label), rows),
        "",
        f"Ölü mektup (q.dead_letter): {dead}",
        f"Alarm sayısı: {alarm_count}",
    ]
    if alarms:
        lines.append(f"En yüksek {len(alarms)} alarm:")
        for index, doc in enumerate(alarms, 1):
            score = int(doc.get("alarm_score", 0) or 0)
            lines.append(f"  {index}. [{score:>3}] {doc.get('title', '-')} — {doc.get('content_url', '-')}")
    else:
        lines.append("Alarm üretilmedi.")
    print("\n".join(lines))


def print_ask_answer(settings: Settings, store: Any, llm: Any, question: str) -> bool:
    """run-all --ask: boru hattı bittikten sonra aynı süreçteki depo üzerinde RAG sorusunu yanıtlar ve yazdırır."""
    from .reporting.rag import QAEngine

    try:
        answer = QAEngine(settings, store, llm).ask(question)
    except Exception as exc:
        log.error("Soru yanıtlanamadı (%r): %s", question, describe_exc(exc))
        return False
    print()
    print("=== run-all --ask ===")
    print(format_answer(answer))
    return True


# ---------------------------------------------------------------------------------------------------------
# Alt komutlar
# ---------------------------------------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace, settings: Settings) -> int:
    from .store import make_store

    rows: list[tuple[str, str, str]] = []
    failed = False

    directories = sorted({str(Path(settings.state_db_path).parent), str(Path(settings.alarm_log_path).parent)})
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    rows.append(("Veri dizini", "OK", ", ".join(directories)))

    broker = make_broker(settings)
    try:
        broker.declare_topology()
        queues = ", ".join(str(queue) for queue in QUEUE_BINDINGS)
        rows.append(
            (
                "RabbitMQ topolojisi",
                "OK",
                f"{redact_url(settings.rabbitmq_url)} exchange={settings.rabbitmq_exchange} dlx={settings.rabbitmq_dlx}; "
                f"kuyruklar: {queues} (+ .retry eşleri, {Queue.DEAD_LETTER})",
            )
        )
    except Exception as exc:
        failed = True
        rows.append(("RabbitMQ topolojisi", "HATA", f"{redact_url(settings.rabbitmq_url)}: {describe_exc(exc)}"))
    finally:
        broker.close()

    try:
        make_store(settings).ensure_indices()
        rows.append(
            (
                "Elasticsearch indeksleri",
                "OK",
                f"{settings.elasticsearch_url}: {settings.es_index_articles}, {settings.es_index_alarms}, "
                f"{settings.es_index_reports}",
            )
        )
    except Exception as exc:
        failed = True
        rows.append(("Elasticsearch indeksleri", "HATA", f"{settings.elasticsearch_url}: {describe_exc(exc)}"))

    ollama = wait_for_ollama(
        settings, required=not args.ollama_optional, timeout=args.timeout, max_wait=max(0.0, args.ollama_wait)
    )
    rows.append(("Ollama", ollama.label, ollama.detail))
    if settings.reranker_url:
        from .reporting.rerank import Reranker

        reranker = Reranker(settings)
        try:
            ok = reranker.scores("bakan açıklaması", ["Bakan yeni düzenlemeyi açıkladı."]) is not None
        finally:
            reranker.close()
        # Reranker isteğe bağlıdır: erişilemezse soru-cevap mevcut sıralamayla çalışır, kurulum başarısız sayılmaz.
        rows.append(("Reranker", "OK" if ok else "UYARI", f"{settings.reranker_url} ({settings.reranker_api})"))

    print(render_table(("Bileşen", "Durum", "Ayrıntı"), rows))
    if not ollama.ok:
        warn_ollama(ollama.detail, settings)
        if ollama.required:
            failed = True
            print(
                "Ollama hazır değil: skorlayıcı anahtar kelime eşleşen haberleri değerlendiremez ve yeniden denemeler "
                "tükenince bunlar ölü mektuba düşer. Ollama'yı konteynerlerden erişilebilir biçimde başlatın "
                "(ana makinede: OLLAMA_HOST=0.0.0.0 ollama serve; ya da docker compose --profile ollama up -d). "
                f"Ollama'sız kurulum (ör. --fake-llm) için: --ollama-optional ya da {OLLAMA_OPTIONAL_ENV}=1."
            )
    if failed:
        print("Kurulum tamamlanamadı: yukarıdaki HATA satırlarını giderip `scraperhryt setup` komutunu yineleyin.")
        return 1
    print("Kurulum tamam.")
    return 0


def cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    statuses = [
        probe_rabbitmq(settings, timeout=args.timeout),
        probe_elasticsearch(settings, timeout=args.timeout),
        probe_ollama(settings, required=not args.ollama_optional, timeout=args.timeout),
    ]
    rows = [(status.name, status.label, fmt_latency(status.latency_ms), status.detail) for status in statuses]
    print(render_table(("Servis", "Durum", "Gecikme", "Ayrıntı"), rows))
    down = [status.name for status in statuses if status.required and not status.ok]
    if down:
        print(f"Zorunlu servis(ler) erişilemez: {', '.join(down)}")
        return 1
    print("Tüm zorunlu servisler erişilebilir.")
    return 0


def cmd_scrape(args: argparse.Namespace, settings: Settings) -> int:
    from .scrapers.runner import ScrapeRunner, build_sources
    from .scrapers.state import SeenStore

    broker = make_broker(settings)
    stop = threading.Event()
    runner = ScrapeRunner(settings, broker, seen_store=SeenStore(settings.state_db_path), sources=build_sources(settings))
    try:
        with signal_scope(stop):
            if args.once:
                stats = runner.run_once()
                print(stats.summary())
            else:
                runner.run_forever(stop_event=stop)
    finally:
        runner.close()
        broker.close()
    return 0


def cmd_filter(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline.keyword_filter import KeywordFilterService

    broker = make_broker(settings)
    embedder = build_llm(settings, fake=False, fallback=False) if settings.preclassifier_enabled and settings.ollama_embedding_model else None
    stop = threading.Event()
    try:
        with signal_scope(stop):
            KeywordFilterService(settings, broker, embedder=embedder).run(stop_event=stop)
    finally:
        broker.close()
        if embedder is not None:
            close_llm(embedder)
    return 0


def cmd_score(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline.scorer import ScoringService

    llm = build_llm(settings, fake=args.fake_llm, fallback=False)
    broker = make_broker(settings)
    stop = threading.Event()
    try:
        with signal_scope(stop):
            ScoringService(settings, broker, llm).run(stop_event=stop)
    finally:
        broker.close()
        close_llm(llm)
    return 0


def cmd_alarm(args: argparse.Namespace, settings: Settings) -> int:
    from .alarm_sinks import build_sinks
    from .pipeline.alarm import AlarmService
    from .store import make_store

    embedder = None
    if settings.ollama_embedding_model:
        from .pipeline.llm import OllamaClient

        embedder = OllamaClient(settings)
        log.info("Alarm katmanı embedding üretecek: model=%s", settings.ollama_embedding_model)
    broker = make_broker(settings)
    store = make_store(settings)  # indeksleri AlarmService.run kendisi (ES hazır olana dek bekleyerek) oluşturur
    stop = threading.Event()
    try:
        with signal_scope(stop):
            AlarmService(settings, broker, store, build_sinks(settings), embedder=embedder).run(stop_event=stop)
    finally:
        broker.close()
        if embedder is not None:
            embedder.close()
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    from .reporting.builder import ReportBuilder
    from .reporting.service import PeriodicReporter, ReportingConsumer

    llm = build_llm(settings, fake=args.fake_llm, fallback=False)
    try:
        store = prepare_store(settings, in_memory=False)
    except Exception as exc:
        log.error("Elasticsearch deposu hazırlanamadı: %s", describe_exc(exc))
        close_llm(llm)
        return 1
    builder = ReportBuilder(settings, store, llm)

    if args.once:
        broker = make_broker(settings)
        try:
            broker.declare_topology()
            report = PeriodicReporter(settings, store, builder, broker).run_once()
            print(describe_report(report))
        finally:
            broker.close()
            close_llm(llm)
        return 0

    stop = threading.Event()
    supervisor = Supervisor(stop)
    consumer_broker = make_broker(settings)
    periodic_broker = make_broker(settings)
    consumer = ReportingConsumer(settings, consumer_broker, store, builder)
    periodic = PeriodicReporter(settings, store, builder, periodic_broker)
    try:
        with signal_scope(stop):
            supervisor.spawn("report-consumer", functools.partial(consumer.run, stop_event=stop))
            supervisor.spawn("periodic-reporter", functools.partial(periodic.run, stop_event=stop))
            supervisor.wait()
    finally:
        stop.set()
        supervisor.join()
        consumer_broker.close()
        periodic_broker.close()
        close_llm(llm)
    return 0 if supervisor.ok else 1


def cmd_api(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .reporting.api import create_app

    llm = build_llm(settings, fake=args.fake_llm, fallback=False)
    try:
        store = prepare_store(settings, in_memory=False)
    except Exception as exc:
        log.error("Elasticsearch deposu hazırlanamadı: %s", describe_exc(exc))
        close_llm(llm)
        return 1
    broker = make_broker(settings)
    try:
        broker.declare_topology()  # idempotent; POST /reports/generate'in yayınladığı exchange kurulmuş olsun
    except Exception as exc:
        log.warning(
            "RabbitMQ topolojisi oluşturulamadı (%s): %s. API yine başlıyor; `scraperhryt setup` çalıştırılana dek "
            "POST /reports/generate raporu yazar ama kuyruğa yayınlayamayabilir",
            redact_url(settings.rabbitmq_url),
            describe_exc(exc),
        )
    app = create_app(settings, store, llm, broker=broker)
    log.info("API başlıyor: http://%s:%d", settings.api_host, settings.api_port)
    try:
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_config=None,
            log_level=parse_level(settings.log_level),
        )
    finally:
        broker.close()
        close_llm(llm)
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from .reporting.rag import QAEngine

    llm = build_llm(settings, fake=args.fake_llm, fallback=False)
    try:
        store = prepare_store(settings, in_memory=False)
    except Exception as exc:
        log.error("Elasticsearch deposu hazırlanamadı: %s", describe_exc(exc))
        close_llm(llm)
        return 1
    try:
        answer = QAEngine(settings, store, llm).ask(args.question, since_days=args.since_days, top_k=args.top_k)
    except Exception as exc:
        log.error("Soru yanıtlanamadı: %s", describe_exc(exc))
        return 1
    finally:
        close_llm(llm)
    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print(format_answer(answer, show_sources=bool(getattr(args, "sources", False))))
    return 0


def cmd_calibrate(args: argparse.Namespace, settings: Settings) -> int:
    """Altın set (+ isteğe bağlı geri bildirim) üzerinde LLM skorlarını ölçer, eşik önerir."""
    from .broker import InMemoryBroker
    from .pipeline.calibration import (
        feedback_items,
        format_calibration_report,
        load_golden_set,
        run_calibration,
    )
    from .pipeline.scorer import ScoringService

    try:
        items = load_golden_set(args.golden or settings.golden_set_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    store = None
    if args.from_feedback:
        try:
            store = prepare_store(settings, in_memory=False)
            items += feedback_items(store.list_feedback(size=1000), store)
        except Exception as exc:
            log.error("Geri bildirimler okunamadı: %s", describe_exc(exc))
            return 1
    if not items:
        log.error("Kalibrasyon için örnek yok")
        return 1
    llm = build_llm(settings, fake=args.fake_llm, fallback=False)
    try:
        scorer = ScoringService(settings, InMemoryBroker(settings), llm)
        report = run_calibration(scorer, items, current_threshold=settings.alarm_threshold)
    finally:
        close_llm(llm)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) if args.json else format_calibration_report(report))
    return 0 if report.failures < len(items) else 1


def cmd_rescore(args: argparse.Namespace, settings: Settings) -> int:
    """Depodaki kayıtları article.keyword ile yeniden yayınlar (model/prompt/eşik değişiminden sonra)."""
    from .pipeline.rescore import Rescorer

    try:
        store = prepare_store(settings, in_memory=False)
    except Exception as exc:
        log.error("Elasticsearch deposu hazırlanamadı: %s", describe_exc(exc))
        return 1
    broker = make_broker(settings)
    try:
        if not args.dry_run:
            broker.declare_topology()
        stats = Rescorer(settings, store, broker).run(
            since_days=args.since_days, only_keyword_hits=not args.all, limit=args.limit, dry_run=args.dry_run
        )
    except Exception as exc:
        log.error("Yeniden skorlama başarısız: %s", describe_exc(exc))
        return 1
    finally:
        broker.close()
    print(
        f"Yeniden skorlama{' (dry-run)' if stats.dry_run else ''}: seçilen={stats.selected} "
        f"yayınlanan={stats.published} atlanan={stats.skipped} → {Queue.ARTICLES_KEYWORD}"
    )
    return 0


def cmd_embed_backfill(args: argparse.Namespace, settings: Settings) -> int:
    """Kayıtlı haberlere (embedding açılmadan önce gelenler dahil) vektör üretip Elasticsearch'e yazar."""
    from datetime import timedelta

    from .models import NewsRecord
    from .pipeline.llm import LLMError, OllamaClient

    if not settings.ollama_embedding_model:
        log.error("OLLAMA_EMBEDDING_MODEL boş: önce embedding modelini ayarlayın (ör. bge-m3)")
        return 1
    try:
        store = prepare_store(settings, in_memory=False)
    except Exception as exc:
        log.error("Depo hazırlanamadı: %s", describe_exc(exc))
        return 1
    since = utcnow() - timedelta(days=args.since_days) if args.since_days else None
    done = failed = 0
    with OllamaClient(settings) as client:
        for doc in store.iter_records(since=since):
            if args.limit and done >= args.limit:
                break
            record = NewsRecord.model_validate(
                {k: v for k, v in doc.items() if not k.startswith("@") and k not in ("content_length", "embedding")}
            )
            body = (record.content or "")[: settings.ollama_max_content_chars]
            text = "\n".join(part for part in (record.title, record.subtitle, body) if part)
            try:
                vector = client.embed([text])[0]
            except LLMError as exc:
                failed += 1
                log.warning("Vektör üretilemedi (%s): %s", record.id, exc)
                continue
            store.index_record(record, embedding=vector)
            done += 1
            if done % 50 == 0:
                log.info("%d habere vektör yazıldı", done)
    print(f"Embedding tamamlandı: {done} haber vektörlendi, {failed} hata (model={settings.ollama_embedding_model})")
    return 0 if failed == 0 else 1


def cmd_replay(args: argparse.Namespace, settings: Settings) -> int:
    """q.dead_letter mesajlarını köken kuyruklarına geri oynatır."""
    from .broker import RabbitMQBroker
    from .replay import format_replay, replay_dead_letters

    broker = make_broker(settings)
    if not isinstance(broker, RabbitMQBroker):
        log.error("Ölü mektup geri oynatma yalnızca RabbitMQ ile çalışır (RABBITMQ_URL)")
        return 1
    try:
        broker.declare_topology()
        stats = replay_dead_letters(broker, limit=args.limit, dry_run=args.dry_run, target_queue=args.to)
    except Exception as exc:
        log.error("Geri oynatma başarısız: %s", describe_exc(exc))
        return 1
    finally:
        broker.close()
    print(format_replay(stats))
    return 0


def cmd_feedback(args: argparse.Namespace, settings: Settings) -> int:
    """Bir alarmı doğru/yanlış pozitif olarak etiketler (news-feedback)."""
    import hashlib

    from .models import Feedback, utcnow

    try:
        store = prepare_store(settings, in_memory=False)
        alarm = store.get_alarm(args.alarm_id)
        if alarm is None:
            log.error("Alarm bulunamadı: %s", args.alarm_id)
            return 1
        stamp = utcnow()
        fb = Feedback(
            feedback_id=hashlib.sha1(f"{args.alarm_id}:{args.label}:{stamp.isoformat()}".encode()).hexdigest()[:20],
            alarm_id=args.alarm_id,
            record_id=str(alarm.get("record_id", "")),
            label=args.label,
            note=args.note or "",
            user=args.user or "",
            channel="cli",
            created_at=stamp,
        )
        store.index_feedback(fb, refresh=True)
    except Exception as exc:
        log.error("Geri bildirim kaydedilemedi: %s", describe_exc(exc))
        return 1
    print(f"Geri bildirim kaydedildi: {fb.feedback_id} alarm={fb.alarm_id} etiket={fb.label}")
    return 0


def run_all_once_in_memory(
    *,
    settings: Settings,
    broker: InMemoryBroker,
    store: Any,
    stop: threading.Event,
    make_runner: Callable[[], Any],
    stages: Sequence[tuple[str, Any]],
    periodic: Any,
    started_at: datetime,
    llm: Any = None,
    question: str | None = None,
) -> int:
    """Tek tur: kazı, ardından kuyrukları sırayla boşalt (filtre → skor → alarm → rapor), özet yazdır.

    ``question`` verilirse özetten sonra aynı depo üzerinde RAG yanıtı da yazdırılır (``--ask``).
    """
    runner = make_runner()
    try:
        stats = runner.run_once()
    finally:
        runner.close()
    log.info(stats.summary())
    for label, service in stages:
        if stop.is_set():
            log.warning("Durdurma istendi; '%s' aşaması atlandı", label)
            break
        processed = service.run(stop_event=stop, max_messages=None)
        log.info("%s: %d mesaj işlendi", label, processed)
    if not stop.is_set():
        print(describe_report(periodic.run_once()))
        print()
    counts = published_counts(broker)
    print_run_summary(
        title="run-all özeti (tek tur, bellek içi)",
        scrape_summary=stats.summary(),
        queue_counts=counts,
        count_label="Yayınlanan",
        alarm_count=counts.get(str(Queue.ALARMS), 0),
        alarms=top_alarms(store, started_at),
        dead_letters=len(broker.dead_letters),
    )
    if question and not print_ask_answer(settings, store, llm, question):
        return 1
    return 0


def flush_pending_alarms(consumer: Any) -> None:
    """Rapor tüketicisi durunca tampondaki (kuyruktan onaylanmış) alarmları özetler; hata loglanır."""
    try:
        if consumer.flush() is not None:
            log.info("Çıkışta bekleyen alarmlar özetlendi")
    except Exception as exc:
        log.error("Çıkışta alarm özeti üretilemedi (%d alarm kaybedildi): %s", consumer.buffered, describe_exc(exc))


def cmd_run_all(args: argparse.Namespace, settings: Settings) -> int:
    from .alarm_sinks import build_sinks
    from .pipeline.alarm import AlarmService
    from .pipeline.keyword_filter import KeywordFilterService
    from .pipeline.scorer import ScoringService
    from .reporting.builder import ReportBuilder
    from .reporting.service import PeriodicReporter, ReportingConsumer
    from .scrapers.runner import ScrapeRunner, build_sources
    from .scrapers.state import SeenStore

    serve_api = not args.no_api and not args.once
    started_at = utcnow()
    stop = threading.Event()
    if args.ask and not args.once:
        log.warning("--ask yalnızca --once ile çalışır; sürekli modda soru yok sayılıyor (POST /ask kullanın)")

    llm = build_llm(settings, fake=args.fake_llm, fallback=bool(args.llm_fallback))
    try:
        store = prepare_store(settings, in_memory=bool(args.in_memory))
    except Exception as exc:
        log.error("Depo hazırlanamadı: %s", describe_exc(exc))
        close_llm(llm)
        return 1

    # --in-memory (ya da RABBITMQ_URL=memory://) → tek bir InMemoryBroker herkes tarafından paylaşılır.
    # RabbitMQ'da pika bağlantıları iş parçacığı güvenli olmadığından her iş parçacığı kendi broker'ını kurar ve
    # işi bitince KENDİSİ kapatır; ana iş parçacığı yalnızca kendi kurduğu broker'ları (``main_brokers``) kapatır.
    setup_broker = make_broker(settings, in_memory=bool(args.in_memory))
    shared_broker = setup_broker if isinstance(setup_broker, InMemoryBroker) else None
    main_brokers: list[Broker] = [setup_broker]

    def broker_for() -> Broker:
        return shared_broker if shared_broker is not None else make_broker(settings)

    def make_runner(broker: Broker) -> ScrapeRunner:
        seen = SeenStore(":memory:") if shared_broker is not None else SeenStore(settings.state_db_path)
        return ScrapeRunner(settings, broker, seen_store=seen, sources=build_sources(settings))

    log.info(
        "run-all başlıyor: broker=%s, depo=%s, tek tur=%s, LLM=%s, API=%s, kaynaklar=%s, anahtar kelimeler=%s, eşik=%d",
        "bellek içi" if shared_broker is not None else f"RabbitMQ ({redact_url(settings.rabbitmq_url)})",
        type(store).__name__,
        "evet" if args.once else "hayır",
        "sezgisel" if args.fake_llm else settings.ollama_model,
        "açık" if serve_api else "kapalı",
        settings.sources,
        settings.keywords,
        settings.alarm_threshold,
    )
    setup_broker.declare_topology()
    if shared_broker is None:
        setup_broker.close()

    builder = ReportBuilder(settings, store, llm)
    filter_broker, scorer_broker, alarm_broker, consumer_broker, periodic_broker = (broker_for() for _ in range(5))
    filter_service = KeywordFilterService(
        settings, filter_broker, embedder=llm if settings.ollama_embedding_model else None
    )
    # Tüketiciler ``.run()`` yerine ``consume_loop`` ile sürüldüğünden paylaşılan durdurma olayını kurucuda alırlar;
    # aksi halde Ollama/ES kesintisinde ``wait_for_llm`` / ``_prepare_indices`` kendi (hiç set edilmeyen) olayını
    # bekler ve SIGINT/SIGTERM kapanışı join zaman aşımına dek askıda kalırdı.
    scorer = ScoringService(settings, scorer_broker, llm, stop_event=stop)
    alarm = AlarmService(settings, alarm_broker, store, build_sinks(settings), embedder=llm)
    alarm._stop_event = stop  # AlarmService kurucusunda stop_event parametresi yok; run() ile aynı etkiyi verir
    consumer = ReportingConsumer(settings, consumer_broker, store, builder)
    periodic = PeriodicReporter(settings, store, builder, periodic_broker)

    exit_code = 0
    supervisor = Supervisor(stop)
    counters = PipelineCounters()
    api_server: Any = None

    def consume_worker(broker: Broker, queue: str, handler: Handler, prefetch: int) -> None:
        try:
            consume_loop(broker, queue, handler, stop, counters, prefetch=prefetch)
        finally:
            broker.close()

    def reporter_worker() -> None:
        """q.alarms tüketicisi. RabbitMQ'da ``ReportingConsumer.run`` sürer: süre eşiği gözcüsü ve çıkışta tampon
        boşaltma onun içindedir (işleyici sayaçlarla sarılır ki boşta kalma tespiti ve özet sayımı çalışsın).
        Bellek içi broker kuyruk boşalınca döndüğünden ``consume_loop`` ile yoklanır; süre eşiği boş turlarda
        denetlenir ve tampon çıkışta elle boşaltılır."""
        queue = str(Queue.ALARMS)
        try:
            if shared_broker is None:
                consumer.handle = tracked_handler(queue, consumer.handle, counters)
                consumer.run(stop_event=stop)
            else:
                try:
                    consume_loop(
                        consumer_broker,
                        queue,
                        consumer.handle,
                        stop,
                        counters,
                        prefetch=settings.rabbitmq_prefetch,
                        on_idle=consumer.flush_if_due,
                    )
                finally:
                    flush_pending_alarms(consumer)
        finally:
            consumer_broker.close()

    def periodic_worker() -> None:
        try:
            periodic.run(stop_event=stop)
        finally:
            periodic_broker.close()

    def scrape_forever() -> None:
        broker = broker_for()
        try:
            runner = make_runner(broker)
            try:
                runner.run_forever(stop_event=stop)
            finally:
                runner.close()
        finally:
            broker.close()

    try:
        with signal_scope(stop):
            if args.once and shared_broker is not None:
                return run_all_once_in_memory(
                    settings=settings,
                    broker=shared_broker,
                    store=store,
                    stop=stop,
                    make_runner=functools.partial(make_runner, shared_broker),
                    stages=(
                        ("anahtar kelime filtresi", filter_service),
                        ("LLM skorlama", scorer),
                        ("alarm katmanı", alarm),
                        ("rapor tüketicisi", consumer),
                    ),
                    periodic=periodic,
                    started_at=started_at,
                    llm=llm,
                    question=args.ask,
                )

            workers: tuple[tuple[str, Broker, str, Handler, int], ...] = (
                ("filter", filter_broker, str(Queue.ARTICLES_RAW), filter_service.handle, settings.rabbitmq_prefetch),
                ("scorer", scorer_broker, str(Queue.ARTICLES_KEYWORD), scorer.handle, SCORER_PREFETCH),
                ("alarm", alarm_broker, str(Queue.ARTICLES_SCORED), alarm.handle, settings.rabbitmq_prefetch),
            )
            alarm._check_embedder()  # bağımsız `alarm` komutuyla aynı başlangıç denetimi (indeksler zaten hazır)
            for name, broker, queue, handler, prefetch in workers:
                supervisor.spawn(name, functools.partial(consume_worker, broker, queue, handler, prefetch))
            supervisor.spawn("reporter", reporter_worker)
            if not args.once:
                supervisor.spawn("periodic", periodic_worker)  # tek turda periyodik rapor kuyruklar boşalınca üretilir
            if serve_api:
                from .reporting.api import create_app

                api_broker = broker_for()
                main_brokers.append(api_broker)
                app = create_app(settings, store, llm, broker=api_broker)
                api_server = start_api_server(app, settings, supervisor)

            if args.once:
                runner_broker = broker_for()
                main_brokers.append(runner_broker)
                runner = make_runner(runner_broker)
                try:
                    stats = runner.run_once()
                finally:
                    runner.close()
                log.info(stats.summary())
                log.info(
                    "Kazıma turu bitti; kuyruklar %.0fs boyunca boş kalınca çıkılacak (--idle-timeout)",
                    args.idle_timeout,
                )
                while not stop.is_set():
                    if counters.is_idle(args.idle_timeout):
                        log.info("Boru hattı boşta; run-all sonlanıyor")
                        stop.set()
                        break
                    if not any(thread.is_alive() for thread in supervisor.threads):
                        break
                    stop.wait(IDLE_POLL_SECONDS)
                # Özet, rapor tüketicisinin çıkışta tamponu boşaltmasını (alarm özeti) da kapsasın.
                supervisor.join(shutdown_timeout_for(settings, llm))
                main_brokers.append(periodic_broker)  # tek turda periyodik rapor ana iş parçacığında üretilir
                try:
                    print(describe_report(periodic.run_once()))
                    print()
                except Exception as exc:
                    log.error("Periyodik rapor üretilemedi: %s", describe_exc(exc))
                print_run_summary(
                    title="run-all özeti (tek tur)",
                    scrape_summary=stats.summary(),
                    queue_counts=counters.snapshot(),
                    count_label="İşlenen",
                    alarm_count=counters.snapshot().get(str(Queue.ALARMS), 0),
                    alarms=top_alarms(store, started_at),
                    dead_letters=None,
                )
                if args.ask and not print_ask_answer(settings, store, llm, args.ask):
                    exit_code = 1
            else:
                supervisor.spawn("scraper", scrape_forever)
                supervisor.wait()
    finally:
        stop.set()
        if api_server is not None:
            api_server.should_exit = True
        timeout = shutdown_timeout_for(settings, llm)
        log.info("İş parçacıkları bekleniyor (en çok %.0fs; süren bir LLM çağrısı varsa bitmesi beklenir)", timeout)
        supervisor.join(timeout)
        for broker in main_brokers:
            broker.close()
        still_running = supervisor.alive
        if still_running:
            # Tüketici iş parçacıkları kendi broker'larını kapatır; LLM istemcisi onlar kullanırken kapatılmaz
            # (süreç çıkışında bırakılır), aksi halde süren çağrı hata verir ve mesaj yeniden skorlanırdı.
            log.warning(
                "Hâlâ çalışan iş parçacıkları var (%s); LLM istemcisi süreç çıkışına bırakıldı",
                ", ".join(still_running),
            )
        else:
            close_llm(llm)
        if not supervisor.ok:
            exit_code = 1
            log.error("run-all hatayla sonlandı; çöken iş parçacıkları: %s", ", ".join(supervisor.failures))
        else:
            log.info("run-all durdu")
    return exit_code


# ---------------------------------------------------------------------------------------------------------
# Argüman ayrıştırıcı
# ---------------------------------------------------------------------------------------------------------


def _add_fake_llm(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="Ollama yerine deterministik sezgisel değerlendirici kullan (geliştirme/test)",
    )


def _add_ollama_optional(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ollama-optional",
        action="store_true",
        default=env_flag(OLLAMA_OPTIONAL_ENV),
        help=(
            "Ollama erişilemez ya da model yüklü değilse hata yerine uyarı ver (çıkış kodunu etkilemez; --fake-llm "
            f"kullanımı için). Ortam değişkeni {OLLAMA_OPTIONAL_ENV}=1 de aynı etkiyi yapar (docker compose için)"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Hürriyet Gündem + 12punto haber izleme boru hattı: kazıma → anahtar kelime → LLM alarm skoru → "
        "RabbitMQ → alarm → Elasticsearch → raporlama → RAG soru-cevap.",
        epilog=(
            "Örnekler:\n"
            "  scraperhryt check\n"
            "  scraperhryt setup\n"
            "  scraperhryt run-all --once --in-memory --fake-llm\n"
            "  scraperhryt scrape --once --source hurriyet\n"
            '  scraperhryt ask "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?" --since-days 7\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="SEVİYE",
        help="Günlük seviyesi (DEBUG, INFO, WARNING, ERROR); varsayılan LOG_LEVEL ayarı",
    )
    sub = parser.add_subparsers(dest="command", metavar="KOMUT", required=True)

    p = sub.add_parser(
        "setup",
        help="RabbitMQ topolojisi + Elasticsearch indeksleri + Ollama model denetimi (Ollama hazır değilse hata; "
        "--ollama-optional ile uyarı)",
    )
    p.add_argument(
        "--timeout", type=float, default=PROBE_TIMEOUT_SECONDS, help="Ollama sondası zaman aşımı (saniye)"
    )
    p.add_argument(
        "--ollama-wait",
        type=float,
        default=env_float(OLLAMA_WAIT_ENV, 0.0),
        metavar="SN",
        help=(
            "Ollama (sunucu + model) hazır olana dek en çok bu kadar saniye bekle; 0 = bekleme. Ortam değişkeni "
            f"{OLLAMA_WAIT_ENV} de aynı etkiyi yapar (docker compose --profile ollama ilk açılışında model "
            "indirilirken kurulumun hata vermemesi için)"
        ),
    )
    _add_ollama_optional(p)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("check", help="RabbitMQ / Elasticsearch / Ollama erişilebilirlik raporu (gecikmelerle)")
    p.add_argument("--timeout", type=float, default=PROBE_TIMEOUT_SECONDS, help="servis başına zaman aşımı (saniye)")
    _add_ollama_optional(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("scrape", help="haber kazıyıcı (article.raw yayınlar)")
    p.add_argument("--once", action="store_true", help="tek tur çalış ve çık")
    p.add_argument("--source", choices=SOURCE_CHOICES, default=None, help="yalnızca bu kaynağı tara")
    p.add_argument("--backfill-days", type=int, default=None, metavar="N", help="12punto arşivinden N gün geriye tara")
    p.add_argument("--interval", type=int, default=None, metavar="SN", help="turlar arası bekleme (saniye)")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("filter", help="anahtar kelime filtresi (q.articles.raw tüketicisi)")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("score", help="Ollama LLM skorlama (q.articles.keyword tüketicisi)")
    _add_fake_llm(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("alarm", help="alarm katmanı (q.articles.scored tüketicisi → ES + kanallar + q.alarms)")
    p.set_defaults(func=cmd_alarm)

    p = sub.add_parser("report", help="raporlama katmanı (alarm özetleri + periyodik raporlar)")
    p.add_argument("--once", action="store_true", help="tek periyodik rapor üret, yazdır ve çık")
    _add_fake_llm(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("api", help="FastAPI servisi (uvicorn)")
    p.add_argument("--host", default=None, help="dinlenecek adres (varsayılan API_HOST)")
    p.add_argument("--port", type=int, default=None, help="port (varsayılan API_PORT)")
    _add_fake_llm(p)
    p.set_defaults(func=cmd_api)

    p = sub.add_parser("ask", help="RAG soru-cevap: son haberlere göre bir soruyu yanıtla")
    p.add_argument("question", help="soru (tırnak içinde)")
    p.add_argument("--since-days", type=int, default=None, metavar="N", help="yalnızca son N günün haberleri")
    p.add_argument("--top-k", type=int, default=None, metavar="N", help="bağlama alınacak en fazla haber sayısı")
    p.add_argument("--json", action="store_true", help="yanıtı JSON olarak yazdır")
    p.add_argument("--sources", action="store_true", help="özetin altında kaynakları ve zaman çizelgesini de yazdır")
    _add_fake_llm(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("calibrate", help="altın set üzerinde LLM skorlarını ölç, eşik öner (config/golden_set.jsonl)")
    p.add_argument("--golden", help="altın set JSONL yolu (varsayılan GOLDEN_SET_PATH)")
    p.add_argument("--from-feedback", action="store_true", help="Elasticsearch'teki insan geri bildirimlerini de örnek olarak ekle")
    p.add_argument("--fake-llm", action="store_true", help="Ollama yerine sezgisel değerlendirici")
    p.add_argument("--json", action="store_true", help="JSON çıktı")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("rescore", help="depodaki kayıtları yeniden skorlanmak üzere q.articles.keyword'e yayınla")
    p.add_argument("--since-days", type=int, default=7, help="son N gün (varsayılan 7)")
    p.add_argument("--all", action="store_true", help="anahtar kelime eşleşmeyenleri de dahil et")
    p.add_argument("--limit", type=int, help="en çok N kayıt")
    p.add_argument("--dry-run", action="store_true", help="yayınlamadan say")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("embed-backfill", help="kayıtlı haberlere embedding vektörü üret (anlamsal arama için)")
    p.add_argument("--since-days", type=int, default=30, help="son N gün (0 = hepsi, varsayılan 30)")
    p.add_argument("--limit", type=int, help="en çok N haber")
    p.set_defaults(func=cmd_embed_backfill)

    p = sub.add_parser("replay-dead-letters", help="q.dead_letter mesajlarını köken kuyruklarına geri oynat")
    p.add_argument("--limit", type=int, help="en çok N mesaj")
    p.add_argument("--dry-run", action="store_true", help="listele, kuyruğa dokunma")
    p.add_argument("--to", help="köken bilinmiyorsa hedef kuyruk (ör. q.articles.keyword)")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("feedback", help="bir alarmı doğru/yanlış pozitif olarak etiketle")
    p.add_argument("alarm_id")
    p.add_argument("--label", required=True, choices=["true_positive", "false_positive", "needs_context"])
    p.add_argument("--note", help="açıklama")
    p.add_argument("--user", help="etiketleyen")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("run-all", help="tüm katmanları tek süreçte çalıştır")
    p.add_argument(
        "--once",
        action="store_true",
        help="kazıyıcıyı tek tur çalıştır, kuyruklar boşalınca özet yazdırıp çık (API başlatılmaz)",
    )
    p.add_argument(
        "--in-memory",
        action="store_true",
        help="RabbitMQ/Elasticsearch yerine bellek içi broker ve depo kullan (geliştirme)",
    )
    _add_fake_llm(p)
    p.add_argument(
        "--llm-fallback",
        action="store_true",
        help="Ollama başlangıçta hazır değilse (sunucu kapalı / model yüklü değil) bu çalıştırma boyunca sezgisel "
        "değerlendiriciye geri düş. Varsayılan: Ollama istemcisi korunur, LLM gerektiren mesajlar Ollama hazır olana "
        "dek gecikmeli yeniden denenir",
    )
    p.add_argument("--no-api", action="store_true", help="API sunucusunu başlatma")
    p.add_argument("--backfill-days", type=int, default=None, metavar="N", help="12punto arşivinden N gün geriye tara")
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_EXIT_SECONDS,
        metavar="SN",
        help="--once ile RabbitMQ modunda: kuyruklar bu kadar saniye boş kalınca çık",
    )
    p.add_argument(
        "--ask",
        default=None,
        metavar="SORU",
        help="--once ile: boru hattı bitince aynı süreçteki depo üzerinde RAG sorusunu yanıtla ve yazdır",
    )
    p.set_defaults(func=cmd_run_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = build_settings(args)
    except ValidationError as exc:
        print(f"Ayar hatası (ortam değişkenleri / .env): {exc}", file=sys.stderr)
        return 2
    configure(settings.log_level)
    threading.main_thread().name = "main"
    log.debug("Komut: %s | ayarlar: ortam=%s, kaynaklar=%s", args.command, settings.environment, settings.sources)
    try:
        return int(args.func(args, settings))
    except KeyboardInterrupt:
        log.warning("Kesildi; çıkılıyor")
        return 130
    except Exception:
        log.exception("'%s' komutu beklenmeyen bir hatayla sonlandı", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
