"""Raporlama ve RAG katmanının Türkçe istemleri ile bunları kurarken kullanılan biçimleme yardımcıları.

Üç istem ailesi vardır:

- **Rapor anlatısı** (``build_report_system_prompt`` / ``build_report_user_prompt``): pencere istatistikleri ve
  en yüksek alarm skorlu haberler → yöneticiler için Türkçe anlatı.
- **Sorgu yeniden yazma** (``QUERY_REWRITE_SYSTEM_PROMPT`` / ``build_query_rewrite_prompt``): kullanıcı sorusu →
  ``{"search_terms": [...], "entities": [...]}`` JSON'u.
- **RAG yanıtı** (``RAG_SYSTEM_PROMPT`` / ``build_context_block`` / ``build_rag_user_prompt``): en yeniden en
  eskiye numaralanmış haber bağlamı → ``[n]`` atıflı Türkçe yanıt.

Tarihler Türkiye saatiyle (sabit UTC+3) ``gg.aa.yyyy SS:DD`` biçiminde yazılır; depodan gelen belgelerde tarihler
ISO dizesi olduğundan ``parse_datetime`` hem dize hem ``datetime`` kabul eder.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from ..textutil import excerpt, normalize_ws

TR_TZ = timezone(timedelta(hours=3), "TRT")  # Türkiye 2016'dan beri sabit UTC+3 kullanır
UNKNOWN_DATE = "bilinmiyor"

#: Model, bağlamdaki haberler soruyu yanıtlamaya yetmiyorsa tam olarak bu cümleyle başlamalıdır.
INSUFFICIENT_EVIDENCE_TEXT = "Elimdeki haberlerde bu konuda yeterli bilgi yok."

#: Bağlam bloğunda haber başına içerik üst sınırı (karakter).
CONTEXT_CONTENT_CHARS = 1200
#: Rapor istemine / şablon anlatıya alınan alarm gerekçesinin uzunluk sınırı.
REASON_LINE_LIMIT = 220
#: Rapor istemindeki alarm satırı üst sınırı (alarm özetlerinde tampon bu sayıdan büyük olabilir).
PROMPT_ALARM_LINES = 25

KIND_LABELS: dict[str, str] = {
    "periodic": "Periyodik rapor",
    "alarm_digest": "Alarm özeti",
    "adhoc": "İsteğe bağlı rapor",
    "daily": "Günlük rapor",
}

_LLM_SUMMARY_SPLIT_RE = re.compile(r"\n\s*LLM Özeti:", re.IGNORECASE)


# ---------------------------------------------------------------------------------------------------------
# Tarih ve metin yardımcıları
# ---------------------------------------------------------------------------------------------------------


def parse_datetime(value: Any) -> datetime | None:
    """ISO dizesi / ``datetime`` / epoch saniyesini zaman dilimli ``datetime``'a çevirir; çözülemezse ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def to_aware(value: datetime) -> datetime:
    """Zaman dilimi olmayan ``datetime``'ı UTC kabul eder."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def format_tr(value: Any, default: str = UNKNOWN_DATE) -> str:
    """``02.10.2026 22:20`` (Türkiye saati); tarih yoksa ``default``."""
    parsed = parse_datetime(value)
    if parsed is None:
        return default
    return parsed.astimezone(TR_TZ).strftime("%d.%m.%Y %H:%M")


def format_tr_date(value: Any, default: str = UNKNOWN_DATE) -> str:
    """``02.10.2026`` (Türkiye saati)."""
    parsed = parse_datetime(value)
    if parsed is None:
        return default
    return parsed.astimezone(TR_TZ).strftime("%d.%m.%Y")


def flat_text(value: Any) -> str:
    """Boşlukları normalize edip tek satıra indirger (``None`` → boş dize)."""
    return normalize_ws(str(value or "")).replace("\n", " ").strip()


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind or "Rapor")


def one_line_reason(alarm: Mapping[str, Any], limit: int = REASON_LINE_LIMIT) -> str:
    """Alarm gerekçesinin tek satırlık hali: ``apply_verdict``'in eklediği "LLM Özeti" kısmı atılır.

    Gerekçe boşsa LLM özeti, o da yoksa sabit bir metin döner.
    """
    reason = str(alarm.get("alarm_reason") or "")
    head = _LLM_SUMMARY_SPLIT_RE.split(reason, maxsplit=1)[0]
    text = flat_text(head) or flat_text(alarm.get("llm_summary"))
    return excerpt(text, limit) if text else "gerekçe belirtilmedi"


def format_distribution(mapping: Mapping[str, Any] | None, limit: int = 8) -> str:
    """``{"hurriyet": 4, "12punto": 2}`` → ``"hurriyet: 4, 12punto: 2"`` (sayıya göre azalan); boşsa ``"-"``."""
    if not mapping:
        return "-"
    items: list[tuple[str, int]] = []
    for key, value in mapping.items():
        try:
            items.append((str(key), int(value or 0)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{key}: {count}" for key, count in items[:limit]) or "-"


def peak_hour(by_hour: Sequence[Mapping[str, Any]] | None) -> dict[str, Any] | None:
    """``stats["by_hour"]`` içindeki en yoğun saati (``{ts, count, alarms}``) döndürür; veri yoksa ``None``."""
    best: Mapping[str, Any] | None = None
    best_count = 0
    for bucket in by_hour or []:
        try:
            count = int(bucket.get("count") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if count > best_count:
            best, best_count = bucket, count
    return dict(best) if best is not None else None


def format_alarm_lines(top_alarms: Sequence[Mapping[str, Any]], limit: int = PROMPT_ALARM_LINES) -> list[str]:
    """``1. [85/100] Başlık (kaynak, tarih) — tek satır gerekçe`` biçiminde numaralı satırlar."""
    lines: list[str] = []
    for index, alarm in enumerate(list(top_alarms)[:limit], 1):
        title = flat_text(alarm.get("title")) or "(başlıksız)"
        try:
            score = int(alarm.get("alarm_score") or 0)
        except (TypeError, ValueError):
            score = 0
        source = flat_text(alarm.get("source")) or "-"
        date = format_tr(alarm.get("published_at") or alarm.get("raised_at"))
        lines.append(f"{index}. [{score}/100] {title} ({source}, {date}) — {one_line_reason(alarm)}")
    return lines


def alarm_ratio_text(total: int, alarms: int) -> str:
    return f"%{alarms * 100 / total:.0f}" if total > 0 else "%0"


# ---------------------------------------------------------------------------------------------------------
# Rapor anlatısı istemleri
# ---------------------------------------------------------------------------------------------------------


def build_report_system_prompt(kind: str) -> str:
    """Analist rolü + bölüm yapısı + 'veri uydurma' kuralları; alarm özetinde vurgu alarm listesine kayar."""
    if kind == "alarm_digest":
        focus = (
            "Bu bir ALARM ÖZETİDİR: verilen alarm listesi ana içeriktir. Alarmları aciliyet (skor) sırasına "
            "göre anlat; ortak konuları, tekrar eden kişi/kurum adlarını ve birbirine bağlı gelişmeleri vurgula."
        )
    else:
        focus = (
            "Bu bir DÖNEM RAPORUDUR: genel tabloyu (haber hacmi, alarm oranı, kaynak/anahtar kelime/kategori "
            "dağılımı, en yoğun saat) ve en yüksek skorlu alarmları anlat."
        )
    return (
        "Sen Türkiye gündemini izleyen bir haber izleme sisteminin kıdemli analistisin. Sana bir zaman "
        "penceresine ait istatistikler ve en yüksek alarm skorlu haberler verilecek; bunlardan yöneticiler için "
        "Türkçe bir rapor anlatısı yazacaksın.\n"
        f"{focus}\n"
        "\n"
        "KURALLAR:\n"
        "- Yalnızca verilen verilere dayan; sayı, isim veya olay uydurma. Pencerede veri yoksa bunu açıkça yaz.\n"
        "- Şu dört bölümü, her biri kendi satırında ve başlığı iki nokta ile biten şekilde yaz: "
        "Yönetici özeti, Öne çıkan gelişmeler, Dağılım, İzlenmesi gerekenler.\n"
        "- 'Yönetici özeti' 2-3 cümle olsun. 'Öne çıkan gelişmeler' alarmları skora göre azalan sırada, her "
        "biri gerekçesiyle birlikte '- ' ile başlayan tek bir madde olsun.\n"
        "- Anahtar kelimenin yan anlamda geçtiği (örneğin 'bakan' fiili, 'fon' müziği) alarmlar varsa belirt.\n"
        "- Düz metin yaz; markdown başlığı (#), tablo veya kod bloğu kullanma. En fazla 350 kelime.\n"
        "- Tarihleri gg.aa.yyyy biçiminde ve Türkiye saatiyle yaz."
    )


def build_report_user_prompt(
    kind: str,
    window_start: datetime,
    window_end: datetime,
    stats: Mapping[str, Any],
    top_alarms: Sequence[Mapping[str, Any]],
) -> str:
    """İstatistikleri ve alarm listesini modele etiketli satırlar halinde sunar."""
    total = _as_int(stats.get("total"))
    alarms = _as_int(stats.get("alarms"))
    avg = stats.get("avg_alarm_score")
    avg_text = f"{float(avg):.1f}" if isinstance(avg, int | float) and alarms > 0 else "-"
    lines = [
        f"Rapor türü: {kind_label(kind)} ({kind})",
        f"Pencere: {format_tr(window_start)} – {format_tr(window_end)} (Türkiye saati)",
        f"Toplam haber: {total} | Alarm: {alarms} ({alarm_ratio_text(total, alarms)}) | "
        f"Ortalama alarm skoru: {avg_text}",
        f"Kaynak dağılımı: {format_distribution(stats.get('by_source'))}",
        f"Anahtar kelime dağılımı: {format_distribution(stats.get('by_keyword'))}",
        f"Kategori dağılımı: {format_distribution(stats.get('by_category'))}",
    ]
    peak = peak_hour(stats.get("by_hour"))
    if peak is not None:
        lines.append(
            f"En yoğun saat: {format_tr(peak.get('ts'))} ({_as_int(peak.get('count'))} haber, "
            f"{_as_int(peak.get('alarms'))} alarm)"
        )
    if kind == "alarm_digest":
        lines.append(f"Özetlenen alarm sayısı: {len(top_alarms)}")
    alarm_lines = format_alarm_lines(top_alarms)
    lines.append("En yüksek skorlu alarmlar:" if alarm_lines else "En yüksek skorlu alarmlar: yok")
    lines.extend(alarm_lines)
    lines.append("")
    lines.append("Bu verilerden kurallara uygun Türkçe rapor anlatısını yaz.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------
# Sorgu yeniden yazma istemi
# ---------------------------------------------------------------------------------------------------------

QUERY_REWRITE_SYSTEM_PROMPT = (
    "Sen bir Türkçe haber arama asistanısın. Kullanıcının sorusunu, haber arşivinde arama yapmak için arama "
    "terimlerine ve özel adlara çevireceksin.\n"
    'YALNIZCA şu biçimde geçerli bir JSON nesnesi döndür: {"search_terms": [...], "entities": [...]}\n'
    "- search_terms: Sorunun konusunu bulmaya yarayacak 2-6 Türkçe anahtar sözcük veya ifade; yalın hâlde "
    "(ek almamış: 'tartışma', 'kurultay', 'açıklama'). 'son durum', 'ne oldu', 'nedir' gibi soru kalıplarını "
    "ve 'ile', 've', 'arasındaki' gibi bağlaçları EKLEME.\n"
    "- entities: Sorudaki kişi, kurum, parti, şirket ve yer adları; tam ve doğru yazımla (örn. 'Özgür Özel', "
    "'Kemal Kılıçdaroğlu', 'CHP'). Soruda geçmeyen ad uydurma; yalnızca sorudaki kişinin yaygın olarak "
    "bilinen kurumunu/partisini ekleyebilirsin.\n"
    "- Markdown, kod bloğu, açıklama veya ek metin yazma."
)


def build_query_rewrite_prompt(question: str, today: datetime | None = None) -> str:
    return f"Bugünün tarihi: {format_tr_date(today or datetime.now(UTC))}\nSoru: {flat_text(question)}\n\nJSON:"


# ---------------------------------------------------------------------------------------------------------
# RAG yanıt istemi
# ---------------------------------------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "Sen Türkiye gündemini izleyen bir haber analistisin. Kullanıcının sorusunu YALNIZCA sana verilen numaralı "
    "haberlere dayanarak Türkçe yanıtlayacaksın.\n"
    "\n"
    "KURALLAR:\n"
    "- Haberler en yeniden en eskiye doğru numaralanmıştır; [1] en yeni haberdir. Yanıta EN GÜNCEL gelişmeyle "
    "başla: ilk cümlede son gelişmenin ne olduğunu ve tarihini (gg.aa.yyyy) açıkça belirt; sonra gerekiyorsa "
    "geriye doğru kronolojiyi kısaca özetle.\n"
    "- Soruyla ilgisiz haberleri yok say: 'en güncel gelişme' soruyla İLGİLİ haberler arasındaki en yeni "
    "olandır; listede [1] olmak zorunda değildir.\n"
    "- Her iddianın sonuna dayandığı haberin numarasını köşeli parantezle ekle: [1], [2] gibi; birden fazla "
    "habere dayanıyorsa [1][3].\n"
    "- Haberlerde olmayan bilgi ekleme, tahmin yürütme, genel bilgine başvurma; haberler arasındaki "
    "çelişkileri belirt.\n"
    "- Verilen haberler soruyla ilgisizse ya da soruyu yanıtlamaya yetmiyorsa yanıta tam olarak şu cümleyle "
    f"başla: '{INSUFFICIENT_EVIDENCE_TEXT}' ve varsa kısmen ilgili haberi tek cümleyle, atıfla belirt.\n"
    "- Kısa ve net ol (en fazla 200 kelime); düz metin yaz, markdown başlığı veya tablo kullanma."
)


def build_context_block(
    index: int, doc: Mapping[str, Any], *, max_content_chars: int = CONTEXT_CONTENT_CHARS
) -> str:
    """``[n] (kaynak, tarih) Başlık — alt başlık — içerik — LLM özeti`` (boş parçalar atlanır)."""
    source = flat_text(doc.get("source")) or "-"
    date = format_tr(doc.get("published_at") or doc.get("@timestamp") or doc.get("scraped_at"))
    title = flat_text(doc.get("title")) or "(başlıksız)"
    parts = [title]
    subtitle = flat_text(doc.get("subtitle"))
    if subtitle and subtitle != title:
        parts.append(subtitle)
    content = flat_text(doc.get("content"))
    if content:
        parts.append(excerpt(content, max(80, int(max_content_chars))))
    summary = flat_text(doc.get("llm_summary"))
    if summary:
        parts.append(f"LLM özeti: {excerpt(summary, 400)}")
    return f"[{index}] ({source}, {date}) " + " — ".join(parts)


def build_rag_user_prompt(
    question: str,
    context_blocks: Sequence[str],
    *,
    since_days: int | None,
    today: datetime | None = None,
) -> str:
    """Soru + en yeniden en eskiye numaralanmış haber bağlamı + yanıt talimatı."""
    window = f"son {since_days} gün" if since_days and since_days > 0 else "tüm arşiv"
    body = "\n\n".join(context_blocks) if context_blocks else "(haber bulunamadı)"
    return (
        f"Bugünün tarihi: {format_tr_date(today or datetime.now(UTC))}\n"
        f"Soru: {flat_text(question)}\n"
        "\n"
        f"Haberler ({window}, {len(context_blocks)} haber; en yeniden en eskiye):\n"
        "\n"
        f"{body}\n"
        "\n"
        "Yukarıdaki haberlere dayanarak, kurallara uygun biçimde Türkçe yanıt ver."
    )


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
