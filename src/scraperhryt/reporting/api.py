"""FastAPI uygulaması: arama, alarmlar, raporlar, istatistik, RAG soru-cevap ve Jinja2 panosu.

``create_app(settings, store, llm, broker=None)`` uygulamayı kurar; CLI ``api`` ve ``run-all`` komutları bunu
uvicorn ile sunar. Depo/LLM hataları Türkçe ``detail`` alanı taşıyan HTTP 4xx/5xx yanıtlarına çevrilir:

- geçersiz parametre → 400 / 422 (pydantic doğrulaması)
- bulunamadı → 404
- Elasticsearch / LLM erişilemiyor → 503; Elasticsearch isteği reddetti / LLM bozuk çıktı → 502
- beklenmeyen hata → 500 (ayrıntı günlükte)

İki uç nokta dış bağımlılığın askıda kalmasına karşı süre bütçelidir: ``POST /reports/generate`` raporu
``news-reports``'a yazdıktan sonra kuyruğa yayını en fazla ``REPORT_PUBLISH_TIMEOUT_SECONDS`` bekler ve yayın
başarısızsa raporu yine ``published=false`` ile döndürür (``ReportPublisher``); ``GET /health`` LLM sondasını
``LLM_PROBE_TIMEOUT_SECONDS`` ile sınırlar (``LLMProbe``), ``OLLAMA_TIMEOUT`` üretim için boyutlanmıştır.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from elasticsearch import ApiError as ESApiError
from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import TransportError as ESTransportError
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .. import __version__
from ..broker import Broker, Retry, RoutingKey
from ..config import Settings
from ..models import Answer, Report, utcnow
from ..pipeline.llm import LLM, LLMBadOutput, LLMUnavailable
from ..store import ArticleStore, SearchHit
from ..textutil import excerpt
from .builder import ReportBuilder
from .prompts import (
    flat_text,
    format_tr,
    kind_label,
    one_line_reason,
    parse_datetime,
    to_aware,
)
from .rag import QAEngine, make_snippet

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DASHBOARD_HOURS = 24
DASHBOARD_ALARMS = 20
DASHBOARD_REPORTS = 10
MAX_WINDOW_HOURS = 24 * 366
MAX_SINCE_DAYS = 3660
#: ``POST /reports/generate``: rapor yazıldıktan sonra kuyruğa yayın için en fazla beklenen süre (saniye).
#: ``RabbitMQBroker.publish`` sunucu kapalıyken dakikalarca yeniden bağlanmayı dener; istek bunu beklemez.
REPORT_PUBLISH_TIMEOUT_SECONDS = 10.0
#: ``GET /health``: LLM sondası (``llm.health`` + ``llm.model_available``) için süre bütçesi (saniye). Ollama
#: istemcisinin varsayılan zaman aşımı üretim için boyutlanmıştır (``OLLAMA_TIMEOUT``, 180 s); paketleri düşüren
#: (reddetmeyen) bir ana makinede sağlık ucu o kadar askıda kalmamalı.
LLM_PROBE_TIMEOUT_SECONDS = 5.0
ReportKind = Literal["adhoc", "periodic", "daily"]
T = TypeVar("T")


# ---------------------------------------------------------------------------------------------------------
# İstek / yanıt modelleri
# ---------------------------------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    store: bool
    llm: bool
    model: str
    model_available: bool
    broker_configured: bool
    version: str
    time: datetime


class ArticleHit(BaseModel):
    id: str
    title: str
    subtitle: str = ""
    content_url: str
    source: str
    category: str = ""
    published_at: datetime | None = None
    alarm_score: int = 0
    is_alarm: bool = False
    alarm_reason: str = ""
    llm_summary: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    score: float = 0.0
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    since_days: int | None
    count: int
    results: list[ArticleHit]


class AlarmsResponse(BaseModel):
    count: int
    items: list[dict[str, Any]]


class ReportsResponse(BaseModel):
    count: int
    items: list[dict[str, Any]]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000, description="Türkçe soru")
    since_days: int | None = Field(default=None, ge=1, le=MAX_SINCE_DAYS, description="Varsayılan RAG_RECENCY_DAYS")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Varsayılan RAG_TOP_K")
    sources: list[str] | None = Field(default=None, description="Kaynak filtresi, örn. ['hurriyet']")


class GenerateReportRequest(BaseModel):
    kind: ReportKind = "adhoc"
    hours: int | None = Field(default=None, ge=1, le=MAX_WINDOW_HOURS, description="Varsayılan REPORT_WINDOW_HOURS")
    window_end: datetime | None = Field(default=None, description="Pencere sonu (varsayılan: şimdi)")
    narrative: bool = Field(default=True, description="False ise LLM çağrılmaz, şablon anlatı üretilir")


class GeneratedReport(Report):
    """``POST /reports/generate`` yanıtı: rapor (her durumda ``news-reports``'a yazılmıştır) + yayın durumu."""

    published: bool = Field(
        default=False, description="report.generated ile kuyruğa yayınlandı mı (False: broker yok/erişilemedi)"
    )


# ---------------------------------------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------------------------------------


def score_class(score: Any) -> str:
    """Pano rozetleri: 80+ kritik, 60+ önemli, 30+ dikkat, altı rutin."""
    try:
        value = int(score or 0)
    except (TypeError, ValueError):
        value = 0
    if value >= 80:
        return "critical"
    if value >= 60:
        return "important"
    if value >= 30:
        return "notable"
    return "routine"


def hit_to_article(hit: SearchHit) -> ArticleHit:
    doc = hit.doc
    return ArticleHit(
        id=str(doc.get("id") or ""),
        title=flat_text(doc.get("title")),
        subtitle=flat_text(doc.get("subtitle")),
        content_url=str(doc.get("content_url") or ""),
        source=str(doc.get("source") or ""),
        category=str(doc.get("category") or ""),
        published_at=parse_datetime(doc.get("published_at")),
        alarm_score=_as_int(doc.get("alarm_score")),
        is_alarm=bool(doc.get("is_alarm")),
        alarm_reason=str(doc.get("alarm_reason") or ""),
        llm_summary=str(doc.get("llm_summary") or ""),
        matched_keywords=[str(k) for k in (doc.get("matched_keywords") or [])],
        score=float(hit.score or 0.0),
        snippet=make_snippet(doc, hit.highlights),
    )


def parse_sources(value: str | None) -> list[str] | None:
    items = [part.strip() for part in (value or "").split(",") if part.strip()]
    return items or None


def guarded(what: str, fn: Callable[[], T]) -> T:
    """Depo/LLM hatalarını Türkçe ayrıntılı HTTP hatalarına çevirir."""
    try:
        return fn()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Geçersiz istek ({what}): {exc}") from exc
    except Retry as exc:
        raise HTTPException(status_code=503, detail=f"Depo erişilemiyor ({what}): {exc}") from exc
    except (ESConnectionError, ESTransportError) as exc:
        raise HTTPException(status_code=503, detail=f"Elasticsearch erişilemiyor ({what}): {_describe(exc)}") from exc
    except ESApiError as exc:
        raise HTTPException(
            status_code=502, detail=f"Elasticsearch isteği başarısız ({what}): {_describe(exc)}"
        ) from exc
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"LLM erişilemiyor ({what}): {exc}") from exc
    except LLMBadOutput as exc:
        raise HTTPException(status_code=502, detail=f"LLM geçersiz çıktı üretti ({what}): {exc}") from exc
    except Exception as exc:
        log.exception("API hatası (%s)", what)
        raise HTTPException(status_code=500, detail=f"Beklenmeyen hata ({what}): {type(exc).__name__}") from exc


class ReportPublisher:
    """Raporu ``report.generated`` ile süre bütçesi içinde yayınlar; yayın başarısızlığı isteği düşürmez.

    Rapor zaten ``news-reports``'a yazılmıştır, kuyruk yalnızca bildirimdir. ``RabbitMQBroker.publish`` sunucu
    kapalıyken kilit altında dakikalarca yeniden bağlanmayı dener; bu yüzden yayın daemon iş parçacığında yapılır
    ve ``timeout`` dolunca istek sonucu beklemeden döner. Önceki yayın hâlâ sürüyorsa (broker erişilemiyor)
    yenisi başlatılmaz: her istek yeni bir askıda iş parçacığı biriktirmez.
    """

    def __init__(self, broker: Broker, timeout: float = REPORT_PUBLISH_TIMEOUT_SECONDS) -> None:
        self.broker = broker
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def publish(self, report: Report) -> bool:
        """Yayın ``timeout`` içinde onaylandıysa ``True``; hata, zaman aşımı veya meşgul broker'da ``False``."""
        outcome: dict[str, BaseException] = {}
        abandoned = threading.Event()  # istek yayını beklemekten vazgeçti; sonuç yalnızca günlüğe yazılır

        def work() -> None:
            try:
                self.broker.publish(RoutingKey.REPORT_GENERATED, report.to_message())
            except Exception as exc:
                outcome["error"] = exc
                if abandoned.is_set():
                    log.error("Rapor %s arka planda da kuyruğa yayınlanamadı: %s", report.report_id, exc)
            else:
                if abandoned.is_set():
                    log.info("Rapor %s gecikmeli olarak kuyruğa yayınlandı", report.report_id)

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                log.error(
                    "Rapor %s yazıldı ama yayınlanmadı: önceki rapor yayını hâlâ sürüyor (RabbitMQ erişilemiyor "
                    "olabilir)",
                    report.report_id,
                )
                return False
            worker = threading.Thread(target=work, name="report-publish", daemon=True)
            self._worker = worker
            worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            abandoned.set()
            log.error(
                "Rapor %s yazıldı ama %.0f sn içinde kuyruğa yayınlanamadı; yayın arka planda sürüyor",
                report.report_id,
                self.timeout,
            )
            return False
        error = outcome.get("error")
        if error is not None:
            log.error("Rapor %s yazıldı ama kuyruğa yayınlanamadı: %s", report.report_id, error)
            return False
        log.info("İsteğe bağlı rapor yayınlandı: %s (%s)", report.report_id, report.kind)
        return True


class LLMProbe:
    """``llm.health()`` + ``llm.model_available()`` sondasını süre bütçesiyle, aynı anda en fazla bir kez koşturur.

    Sonda daemon iş parçacığında çalışır; ``timeout`` içinde bitmezse ``(False, False)`` döner. Sonraki
    ``/health`` istekleri yeni sonda başlatmak yerine süren sondaya katılır; yanıt vermeyen bir Ollama ana
    makinesi her sağlık isteğinde yeni bir askıda iş parçacığı biriktirmez.
    """

    def __init__(self, llm: LLM, timeout: float = LLM_PROBE_TIMEOUT_SECONDS) -> None:
        self.llm = llm
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._result: dict[str, tuple[bool, bool]] = {}

    def _run(self, result: dict[str, tuple[bool, bool]]) -> None:
        healthy = _safe_bool(self.llm.health, "LLM sağlık kontrolü")
        available = healthy and _safe_bool(self.llm.model_available, "LLM model kontrolü")
        result["value"] = (healthy, available)

    def probe(self) -> tuple[bool, bool]:
        """``(LLM erişilebilir mi, model yüklü mü)``; bütçe aşılırsa ``(False, False)``."""
        with self._lock:
            worker, result = self._worker, self._result
            if worker is None or not worker.is_alive():
                result = {}
                worker = threading.Thread(target=self._run, args=(result,), name="llm-health-probe", daemon=True)
                self._worker, self._result = worker, result
                worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            log.warning("LLM sağlık sondası %.0f sn içinde yanıt vermedi; LLM erişilemez sayılıyor", self.timeout)
            return False, False
        return result.get("value", (False, False))


def generate_report(
    builder: ReportBuilder,
    store: ArticleStore,
    publisher: ReportPublisher | None,
    *,
    kind: str,
    hours: int,
    window_end: datetime | None = None,
    narrative: bool = True,
) -> GeneratedReport:
    """Raporu kurar, ``news-reports``'a yazar ve ``publisher`` verilmişse ``report.generated`` ile yayınlar.

    Yayın başarısızlığı (broker kapalı, zaman aşımı) raporu geçersiz kılmaz: rapor depodadır ve yanıt
    ``published=False`` taşır. Aksi halde istemci başarılı bir isteği hata sanıp yeniden dener ve aynı rapor
    tekrar tekrar üretilirdi.
    """
    end = to_aware(window_end) if window_end is not None else utcnow()
    report = builder.build(kind, end - timedelta(hours=hours), end, narrative=narrative)
    store.index_report(report, refresh=True)
    if publisher is None:
        log.info("İsteğe bağlı rapor yazıldı (broker yok, yayınlanmadı): %s (%s)", report.report_id, kind)
        published = False
    else:
        published = publisher.publish(report)
    return GeneratedReport(**report.model_dump(), published=published)


# ---------------------------------------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------------------------------------


def create_app(settings: Settings, store: ArticleStore, llm: LLM, broker: Broker | None = None) -> FastAPI:
    app = FastAPI(
        title="scraperhryt API",
        version=__version__,
        description=(
            "Hürriyet Gündem + 12punto haber izleme boru hattı: arama, alarmlar, raporlar, istatistik ve "
            "RAG soru-cevap. Pano için `/` adresine gidin."
        ),
    )
    builder = ReportBuilder(settings, store, llm)
    qa = QAEngine(settings, store, llm)
    publisher = ReportPublisher(broker) if broker is not None else None
    llm_probe = LLMProbe(llm)
    app.state.settings = settings
    app.state.store = store
    app.state.llm = llm
    app.state.broker = broker
    app.state.builder = builder
    app.state.qa = qa
    app.state.publisher = publisher
    app.state.llm_probe = llm_probe

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["tr_dt"] = format_tr
    templates.env.filters["score_class"] = score_class
    templates.env.filters["kind_label"] = kind_label

    # --- sağlık ---
    @app.get("/health", response_model=HealthResponse, summary="Depo ve LLM sağlık özeti")
    def health() -> HealthResponse | JSONResponse:
        store_ok = _safe_bool(store.health, "depo sağlık kontrolü")
        llm_ok, model_ok = llm_probe.probe()
        status: Literal["ok", "degraded", "unavailable"]
        if store_ok and model_ok:
            status = "ok"
        elif store_ok:
            status = "degraded"
        else:
            status = "unavailable"
        body = HealthResponse(
            status=status,
            store=store_ok,
            llm=llm_ok,
            model=llm.model_name,
            model_available=model_ok,
            broker_configured=broker is not None,
            version=__version__,
            time=utcnow(),
        )
        if not store_ok:
            return JSONResponse(status_code=503, content=body.model_dump(mode="json"))
        return body

    # --- haberler ---
    @app.get("/articles/search", response_model=SearchResponse, summary="Türkçe BM25 + yenilik ağırlıklı arama")
    def search_articles(
        q: Annotated[str, Query(max_length=500, description="Arama metni; boşsa yalnızca yeniliğe göre sıralanır")] = "",
        since_days: Annotated[int | None, Query(ge=1, le=MAX_SINCE_DAYS, description="Son N gün")] = None,
        sources: Annotated[str | None, Query(description="Virgülle ayrılmış kaynaklar: hurriyet,12punto")] = None,
        only_alarms: Annotated[bool, Query(description="Yalnızca alarm üretmiş haberler")] = False,
        min_score: Annotated[int | None, Query(ge=0, le=100, description="En düşük alarm skoru")] = None,
        size: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> SearchResponse:
        since = utcnow() - timedelta(days=since_days) if since_days else None
        hits = guarded(
            "haber arama",
            lambda: store.search_records(
                q, since=since, sources=parse_sources(sources), only_alarms=only_alarms, size=size, min_score=min_score
            ),
        )
        results = [hit_to_article(hit) for hit in hits]
        return SearchResponse(query=q, since_days=since_days, count=len(results), results=results)

    @app.get("/articles/{article_id}", summary="Tek haber (news-articles belgesi)")
    def get_article(article_id: Annotated[str, PathParam(min_length=1, max_length=128)]) -> dict[str, Any]:
        doc = guarded("haber okuma", lambda: store.get_record(article_id))
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Haber bulunamadı: {article_id}")
        return doc

    # --- alarmlar ---
    @app.get("/alarms", response_model=AlarmsResponse, summary="Son alarmlar (news-alarms)")
    def list_alarms(
        since_hours: Annotated[int | None, Query(ge=1, le=MAX_WINDOW_HOURS, description="Son N saat")] = None,
        min_score: Annotated[int | None, Query(ge=0, le=100)] = None,
        size: Annotated[int, Query(ge=1, le=200)] = 20,
        include_content: Annotated[bool, Query(description="Haber gövdesini de döndür")] = False,
        include_duplicates: Annotated[bool, Query(description="Tekrar (duplicate_of dolu) alarmları da listele")] = False,
        needs_review: Annotated[bool | None, Query(description="Yalnızca insan incelemesi önerilenler")] = None,
        event_id: Annotated[str | None, Query(description="Belirli bir olay kümesi")] = None,
    ) -> AlarmsResponse:
        since = utcnow() - timedelta(hours=since_hours) if since_hours else None
        fetch = size if (min_score is None and include_duplicates and needs_review is None and not event_id) else max(size * 5, 100)
        docs = guarded("alarm listesi", lambda: store.recent_alarms(since=since, size=fetch))
        items: list[dict[str, Any]] = []
        for doc in docs:
            if min_score is not None and _as_int(doc.get("alarm_score")) < min_score:
                continue
            if not include_duplicates and doc.get("duplicate_of"):
                continue
            if needs_review is not None and bool(doc.get("needs_review")) != needs_review:
                continue
            if event_id and doc.get("event_id") != event_id:
                continue
            item = dict(doc)
            if not include_content:
                item.pop("content", None)
            items.append(item)
            if len(items) >= size:
                break
        return AlarmsResponse(count=len(items), items=items)

    # --- raporlar ---
    @app.get("/reports", response_model=ReportsResponse, summary="Son raporlar (news-reports)")
    def list_reports(
        kind: Annotated[str | None, Query(max_length=40, description="periodic | alarm_digest | adhoc")] = None,
        size: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> ReportsResponse:
        docs = guarded("rapor listesi", lambda: store.list_reports(kind=kind or None, size=size))
        return ReportsResponse(count=len(docs), items=[dict(doc) for doc in docs])

    @app.post("/reports/generate", response_model=GeneratedReport, summary="Anında rapor üret, kaydet ve yayınla")
    def generate(payload: GenerateReportRequest) -> GeneratedReport:
        return guarded(
            "rapor üretme",
            lambda: generate_report(
                builder,
                store,
                publisher,
                kind=payload.kind,
                hours=payload.hours or settings.report_window_hours,
                window_end=payload.window_end,
                narrative=payload.narrative,
            ),
        )

    # --- istatistik ---
    @app.get("/stats", summary="Pencere istatistikleri (toplam, alarm, dağılımlar, saatlik seri, en yüksek alarmlar)")
    def stats(hours: Annotated[int, Query(ge=1, le=MAX_WINDOW_HOURS)] = DASHBOARD_HOURS) -> dict[str, Any]:
        now = utcnow()
        data = guarded("istatistik", lambda: store.stats(now - timedelta(hours=hours), now))
        return {"hours": hours, **dict(data)}

    # --- soru-cevap ---
    @app.post("/ask", response_model=Answer, summary="RAG soru-cevap: en yeni haberlerden atıflı Türkçe yanıt")
    def ask(payload: AskRequest) -> Answer:
        return guarded(
            "soru-cevap",
            lambda: qa.ask(payload.question, since_days=payload.since_days, top_k=payload.top_k, sources=payload.sources),
        )

    # --- pano ---
    @app.get("/scraper/status", summary="Kazıyıcı durumu: son/sonraki tur, çekilemeyen (bekleyen) haberler, uyarı")
    def scraper_status_endpoint() -> dict[str, Any]:
        from ..scrapers.status import scraper_status

        return guarded("kazıyıcı durumu", lambda: scraper_status(settings))

    @app.get("/ara", include_in_schema=False)
    def search_page() -> RedirectResponse:
        """Eski arama adresi: arayüz tek sayfada birleştirildi."""
        return RedirectResponse(url="/", status_code=307)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request) -> HTMLResponse:
        error: str | None = None
        alarms: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        try:
            alarms = [_alarm_view(doc) for doc in store.recent_alarms(size=DASHBOARD_ALARMS)]
            reports = [_report_view(doc) for doc in store.list_reports(size=DASHBOARD_REPORTS)]
            # Ekranda pencere gösterildiği için en yeni pencere (bitişi) önce; eşitse en son üretilen.
            reports.sort(key=lambda r: (_sort_ts(r["window_end"]), _sort_ts(r["generated_at"])), reverse=True)
        except Exception as exc:
            log.error("Arayüz verisi alınamadı: %s", exc)
            error = f"Veri alınamadı: {type(exc).__name__}: {exc}"
        context = {"settings": settings, "alarms": alarms, "reports": reports, "error": error}
        return templates.TemplateResponse(request, "app.html", context)

    return app


# ---------------------------------------------------------------------------------------------------------
# Pano görünümleri
# ---------------------------------------------------------------------------------------------------------


def _alarm_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alarm_id": str(doc.get("alarm_id") or ""),
        "title": flat_text(doc.get("title")) or "(başlıksız)",
        "subtitle": flat_text(doc.get("subtitle")),
        "content_url": str(doc.get("content_url") or ""),
        "source": str(doc.get("source") or "-"),
        "alarm_score": _as_int(doc.get("alarm_score")),
        "reason": one_line_reason(doc, limit=320),
        "llm_summary": excerpt(flat_text(doc.get("llm_summary")), 400),
        "published_at": doc.get("published_at"),
        "raised_at": doc.get("raised_at"),
        "matched_keywords": [str(k) for k in (doc.get("matched_keywords") or [])],
        "channels": [str(c) for c in (doc.get("channels_notified") or [])],
        "needs_review": bool(doc.get("needs_review")),
        "duplicate_of": str(doc.get("duplicate_of") or ""),
    }


def _report_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    stats = doc.get("stats") if isinstance(doc.get("stats"), Mapping) else {}
    top = doc.get("top_alarms") if isinstance(doc.get("top_alarms"), list) else []
    return {
        "report_id": str(doc.get("report_id") or ""),
        "kind": str(doc.get("kind") or ""),
        "window_start": doc.get("window_start"),
        "window_end": doc.get("window_end"),
        "generated_at": doc.get("generated_at"),
        "model": str(doc.get("model") or "-"),
        "narrative": str(doc.get("narrative") or ""),
        **dict(zip(("summary", "details"), split_narrative(str(doc.get("narrative") or "")), strict=True)),
        "total": _as_int(stats.get("total")),
        "alarms": _as_int(stats.get("alarms")),
        "top_alarm_count": len(top),
    }


_MD_MARKUP_RE = re.compile(r"(\*\*|__|^#{1,6}\s*)", re.M)
_SUMMARY_HEAD_RE = re.compile(r"^\s*yönetici özeti\s*:?\s*", re.I)
_SECTION_RE = re.compile(r"^\s*(yönetici özeti|öne çıkan gelişmeler|dağılım|izlenmesi gerekenler|not)\b", re.I)


def split_narrative(text: str) -> tuple[str, str]:
    """Rapor anlatısını arayüz için ayırır: (yönetici özeti, geri kalanı). Markdown işaretleri temizlenir.

    "Yönetici özeti" başlığı varsa onun paragrafı, yoksa ilk paragraf özet olur; kalan bölümler katlanır.
    """
    clean = _MD_MARKUP_RE.sub("", text or "").strip()
    if not clean:
        return "", ""
    paragraphs: list[str] = []
    for line in clean.splitlines():
        if not line.strip():
            paragraphs.append("")
        elif _SECTION_RE.match(line) or not paragraphs:
            paragraphs.append(line.strip())
        else:
            paragraphs[-1] = f"{paragraphs[-1]}\n{line.strip()}".strip()
    paragraphs = [p for p in paragraphs if p]
    index = next((i for i, p in enumerate(paragraphs) if _SUMMARY_HEAD_RE.match(p)), 0)
    head = paragraphs[index]
    summary = _SUMMARY_HEAD_RE.sub("", head, count=1).strip()
    rest = paragraphs[:index] + paragraphs[index + 1 :]
    if not summary and rest:  # başlık tek başına bir satırsa özet sonraki paragraftır
        summary = rest.pop(index if index < len(rest) else 0)
    lines = summary.splitlines()
    if len(lines) > 1 and _SUMMARY_HEAD_RE.match(lines[0] + ":"):
        summary = " ".join(lines[1:])
    return " ".join(summary.split()), "\n\n".join(rest)


def _sort_ts(value: Any) -> float:
    parsed = parse_datetime(value)
    return parsed.timestamp() if parsed else 0.0


def _safe_bool(fn: Callable[[], bool], what: str) -> bool:
    try:
        return bool(fn())
    except Exception as exc:
        log.warning("%s başarısız: %s", what, exc)
        return False


def _describe(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    return f"{type(exc).__name__}: {message or exc}"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
