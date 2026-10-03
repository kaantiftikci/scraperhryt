"""Rapor üretici: pencere istatistikleri + en yüksek alarmlar + Türkçe LLM anlatısı (şablon yedekli).

``ReportBuilder.build`` depodan ``stats(since, until)`` ve ``recent_alarms`` alır, alarmları skora göre sıralar,
``llm.generate_text`` ile anlatı üretir; LLM erişilemez ya da bozuk çıktı verirse deterministik Türkçe şablon
anlatı kullanılır ve ``Report.model`` ``"template"`` olur. ``report_id`` pencere (+ alarm kimlikleri) üzerinden
deterministiktir; aynı rapor yeniden üretildiğinde ``news-reports``'ta üstüne yazılır.

Alarm özetinde (``top_alarms`` verildiğinde) istatistikler depodan değil, özetlenen alarm listesinden türetilir
(``alarm_stats``): depo istatistikleri haberin yayın tarihine (``@timestamp``) göre pencerelenir, alarmlar ise
yükseltilme anına göre toplanır; iki eksen örtüşmediğinden depo sayıları özetteki alarmlarla çelişirdi.
Depodan kurulan (periyodik / isteğe bağlı) raporlarda ise alarm listesi de haberin yayın tarihine göre seçilir
(``_top_alarms_from_store``); böylece anlatıdaki "Alarm: N" sayısı ve listelenen alarmlar aynı ekseni paylaşır.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..config import Settings
from ..models import Report, utcnow
from ..pipeline.llm import LLM, HeuristicLLM, LLMError
from ..store import ArticleStore
from ..textutil import normalize_ws
from .prompts import (
    alarm_ratio_text,
    build_report_system_prompt,
    build_report_user_prompt,
    flat_text,
    format_alarm_lines,
    format_distribution,
    format_tr,
    kind_label,
    parse_datetime,
    peak_hour,
    to_aware,
)

log = logging.getLogger(__name__)

#: Anlatı LLM yerine şablonla üretildiğinde ``Report.model`` değeri.
TEMPLATE_MODEL = "template"
#: Depodan seçilen (alarm özeti dışındaki) raporlarda ``top_alarms`` uzunluğu.
TOP_ALARMS_LIMIT = 10
#: Pencere içindeki alarmları skora göre sıralayabilmek için depodan çekilen alarm sayısı.
_RECENT_ALARMS_FETCH = 500
_SUMMARY_TEXT_FIELDS = (
    "alarm_id",
    "record_id",
    "title",
    "subtitle",
    "content_url",
    "source",
    "category",
    "published_at",
    "raised_at",
    "alarm_reason",
    "llm_summary",
)
_SUMMARY_LIST_FIELDS = ("matched_keywords", "channels_notified")


def alarm_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """``news-alarms`` belgesini (ya da ``AlarmEvent.to_es_document`` çıktısını) rapora girecek sözlüğe indirger.

    Haber gövdesi (``content``) rapora alınmaz; tarihler ISO dizesi olarak kalır (JSON uyumlu).
    """
    out: dict[str, Any] = {}
    for key in _SUMMARY_TEXT_FIELDS:
        value = doc.get(key)
        if isinstance(value, datetime):
            value = to_aware(value).isoformat()
        out[key] = "" if value is None else str(value)
    for key in _SUMMARY_LIST_FIELDS:
        value = doc.get(key)
        out[key] = [str(item) for item in value] if isinstance(value, list | tuple) else []
    out["alarm_score"] = _as_int(doc.get("alarm_score"))
    if not out["record_id"]:
        out["record_id"] = str(doc.get("id") or "")
    return out


def rank_alarms(alarms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Alarm sözlüklerini skora göre azalan (eşitlikte yeni olan önce) sıralar; her biri ``alarm_summary`` olur."""
    summaries = [alarm_summary(alarm) for alarm in alarms]
    return sorted(summaries, key=_alarm_rank_key)


def alarm_stats(
    alarms: Sequence[Mapping[str, Any]], window_start: datetime, window_end: datetime
) -> dict[str, Any]:
    """Alarm özeti istatistikleri: ``store.stats`` ile aynı anahtarlar, ama yalnızca verilen alarmlardan türetilir.

    ``total`` ve ``alarms`` özetlenen alarm sayısıdır; dağılımlar kaynak / eşleşen anahtar kelime / kategori
    üzerinden, ``by_hour`` yükseltilme saatine göre sayılır. Böylece anlatıdaki her sayı aynı listeyi anlatır.
    """
    by_source: Counter[str] = Counter()
    by_keyword: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    hours: Counter[datetime] = Counter()
    scores: list[int] = []
    for alarm in alarms:
        by_source[flat_text(alarm.get("source"))] += 1
        keywords = alarm.get("matched_keywords")
        for keyword in keywords if isinstance(keywords, list | tuple) else []:
            by_keyword[str(keyword)] += 1
        category = flat_text(alarm.get("category"))
        if category:
            by_category[category] += 1
        raised = parse_datetime(alarm.get("raised_at") or alarm.get("published_at"))
        if raised is not None:
            hours[raised.astimezone(UTC).replace(minute=0, second=0, microsecond=0)] += 1
        scores.append(_as_int(alarm.get("alarm_score")))

    def ordered(counter: Counter[str]) -> dict[str, int]:
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "since": to_aware(window_start).isoformat(),
        "until": to_aware(window_end).isoformat(),
        "total": len(scores),
        "alarms": len(scores),
        "by_source": ordered(by_source),
        "by_keyword": ordered(by_keyword),
        "by_category": ordered(by_category),
        "avg_alarm_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "by_hour": [
            {"ts": hour.isoformat(), "count": count, "alarms": count} for hour, count in sorted(hours.items())
        ],
    }


def report_id_for(
    kind: str, window_start: datetime, window_end: datetime, alarm_ids: Sequence[str] = ()
) -> str:
    """``<kind>-<pencere sonu UTC YYYYMMDDTHHMM>-<8 hex>``; aynı tür+pencere(+alarmlar) → aynı kimlik."""
    start, end = to_aware(window_start), to_aware(window_end)
    seed = "|".join((kind, start.isoformat(), end.isoformat(), ",".join(sorted(alarm_ids))))
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f"{kind}-{end.astimezone(UTC):%Y%m%dT%H%M}-{digest}"


def build_template_narrative(
    kind: str,
    window_start: datetime,
    window_end: datetime,
    stats: Mapping[str, Any],
    top_alarms: Sequence[Mapping[str, Any]],
) -> str:
    """LLM'siz, deterministik Türkçe anlatı: aynı dört bölüm, istatistiklerden doldurulmuş cümleler."""
    total = _as_int(stats.get("total"))
    alarms = _as_int(stats.get("alarms"))
    avg = stats.get("avg_alarm_score")
    avg_text = f"{float(avg):.1f}" if isinstance(avg, int | float) and alarms > 0 else "-"
    lines = [f"{kind_label(kind)} — {format_tr(window_start)} ile {format_tr(window_end)} arası (Türkiye saati)."]

    if kind == "alarm_digest":
        top_score = max((_as_int(alarm.get("alarm_score")) for alarm in top_alarms), default=0)
        lines.append(
            f"Yönetici özeti: Bu özet {len(top_alarms)} alarm içerir; ortalama alarm skoru {avg_text}, "
            f"en yüksek skor {top_score}."
        )
    elif total == 0:
        lines.append("Yönetici özeti: Bu pencerede işlenmiş haber bulunmuyor.")
    else:
        lines.append(
            f"Yönetici özeti: Pencerede {total} haber işlendi; {alarms} tanesi ({alarm_ratio_text(total, alarms)}) "
            f"alarm eşiğini aştı. Alarmların ortalama skoru: {avg_text}."
        )

    alarm_lines = format_alarm_lines(top_alarms)
    if alarm_lines:
        lines.append("Öne çıkan gelişmeler:")
        lines.extend(f"- {line}" for line in alarm_lines)
    else:
        lines.append("Öne çıkan gelişmeler: bu pencerede alarm üretilmedi.")

    lines.append(
        "Dağılım: kaynaklar — "
        f"{format_distribution(stats.get('by_source'))}; anahtar kelimeler — "
        f"{format_distribution(stats.get('by_keyword'))}; kategoriler — "
        f"{format_distribution(stats.get('by_category'))}."
    )
    peak = peak_hour(stats.get("by_hour"))
    if peak is not None:
        lines.append(
            f"En yoğun saat: {format_tr(peak.get('ts'))} ({_as_int(peak.get('count'))} haber, "
            f"{_as_int(peak.get('alarms'))} alarm)."
        )

    watch = [flat_text(alarm.get("title")) for alarm in list(top_alarms)[:3] if flat_text(alarm.get("title"))]
    if watch:
        lines.append("İzlenmesi gerekenler: " + "; ".join(watch) + ".")
    else:
        lines.append("İzlenmesi gerekenler: izlenmesi gereken yeni başlık yok.")
    lines.append("Not: Bu anlatı dil modeline erişilemediği için şablonla üretildi.")
    return "\n".join(lines)


class ReportBuilder:
    """``build(kind, window_start, window_end)`` → ``Report`` (istatistik + en yüksek alarmlar + anlatı)."""

    def __init__(self, settings: Settings, store: ArticleStore, llm: LLM) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm
        # HeuristicLLM serbest metin üretemez ("Sezgisel mod" sabit cümlesi); şablon anlatı daha kullanışlıdır.
        self._offline = isinstance(llm, HeuristicLLM)

    def build(
        self,
        kind: str,
        window_start: datetime,
        window_end: datetime,
        *,
        narrative: bool = True,
        top_alarms: Sequence[Mapping[str, Any]] | None = None,
    ) -> Report:
        """Raporu kurar (depoya yazmaz, yayınlamaz).

        ``top_alarms`` verilirse (alarm özeti) depo sorgulanmaz: alarm listesi ve istatistikler (``alarm_stats``)
        bu listeden türetilir, rapor kimliği alarm kimliklerini de içerir. Verilmezse pencere istatistikleri
        ``store.stats`` ile, alarmlar depodan skora göre seçilir (en fazla ``TOP_ALARMS_LIMIT``).
        ``narrative=False`` LLM'i çağırmaz; anlatı şablondan üretilir.
        """
        kind = (kind or "").strip()
        if not kind:
            raise ValueError("Rapor türü (kind) boş olamaz")
        start, end = to_aware(window_start), to_aware(window_end)
        if start > end:
            raise ValueError(f"Rapor penceresi geçersiz: başlangıç ({start.isoformat()}) bitişten sonra")

        stats: dict[str, Any]
        if top_alarms is None:
            raw_stats = self.store.stats(start, end)
            stats = dict(raw_stats) if isinstance(raw_stats, Mapping) else {}
            alarms = self._top_alarms_from_store(start, end)
            report_id = report_id_for(kind, start, end)
        else:
            alarms = rank_alarms(top_alarms)
            stats = alarm_stats(alarms, start, end)
            report_id = report_id_for(kind, start, end, [a["alarm_id"] for a in alarms])

        if narrative:
            text, model = self._narrative(kind, start, end, stats, alarms)
        else:
            text, model = build_template_narrative(kind, start, end, stats, alarms), TEMPLATE_MODEL

        report = Report(
            report_id=report_id,
            kind=kind,
            window_start=start,
            window_end=end,
            generated_at=utcnow(),
            stats=stats,
            narrative=text,
            top_alarms=alarms,
            model=model,
        )
        log.info(
            "Rapor kuruldu: %s (%s) pencere=%s–%s haber=%d alarm=%d model=%s",
            report.report_id,
            kind,
            format_tr(start),
            format_tr(end),
            _as_int(stats.get("total")),
            len(alarms),
            model,
        )
        return report

    # --- yardımcılar ---
    def _top_alarms_from_store(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Pencereye haber tarihine (``published_at``) göre düşen alarmlar; ``store.stats`` ile aynı eksen.

        Liste yükseltilme anına göre seçilseydi geç kazınan ya da yeniden skorlanan eski bir haber listede görünür
        ama istatistikteki "Alarm: N" sayısına girmezdi. Alarm en erken haber yayımlandığında yükseltilebildiğinden
        ``raised_at >= start`` ön filtresi aday kaçırmaz; yayın tarihi olmayan alarmda yükseltilme anı esas alınır.
        """
        docs = self.store.recent_alarms(since=start, size=_RECENT_ALARMS_FETCH)
        in_window = [doc for doc in docs if _in_window(doc, start, end)]
        return rank_alarms(in_window)[:TOP_ALARMS_LIMIT]

    def _narrative(
        self,
        kind: str,
        start: datetime,
        end: datetime,
        stats: Mapping[str, Any],
        alarms: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        if self._offline:
            log.info("Sezgisel LLM modunda rapor anlatısı şablonla üretiliyor (%s)", kind)
            return build_template_narrative(kind, start, end, stats, alarms), TEMPLATE_MODEL
        system = build_report_system_prompt(kind)
        user = build_report_user_prompt(kind, start, end, stats, alarms)
        try:
            text = normalize_ws(self.llm.generate_text(system, user))
        except LLMError as exc:
            log.warning("Rapor anlatısı LLM ile üretilemedi (%s); şablon anlatı kullanılacak: %s", kind, exc)
            return build_template_narrative(kind, start, end, stats, alarms), TEMPLATE_MODEL
        if not text:
            log.warning("LLM boş rapor anlatısı döndürdü (%s); şablon anlatı kullanılacak", kind)
            return build_template_narrative(kind, start, end, stats, alarms), TEMPLATE_MODEL
        return text, self.llm.model_name


def _in_window(doc: Mapping[str, Any], start: datetime, end: datetime) -> bool:
    ts = parse_datetime(doc.get("published_at")) or parse_datetime(doc.get("raised_at"))
    return ts is None or start <= ts <= end


def _alarm_rank_key(alarm: Mapping[str, Any]) -> tuple[int, float, str]:
    raised = parse_datetime(alarm.get("raised_at") or alarm.get("published_at"))
    ts = raised.timestamp() if raised is not None else 0.0
    return (-_as_int(alarm.get("alarm_score")), -ts, str(alarm.get("alarm_id") or ""))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
