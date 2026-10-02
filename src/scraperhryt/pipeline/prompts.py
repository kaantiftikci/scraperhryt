"""LLM skorlama için Türkçe istemler (prompt) ve alarm rubriği.

Sistem istemi analist rolünü, 0-100 alarm rubriğini, sabit konu listesini ve katı JSON şemasını açıklar.
Kullanıcı istemi tek bir haberi etiketli bölümler halinde taşır; ``parse_user_prompt`` aynı bölümleri geri
çözebilir (HeuristicLLM ve testler bunu kullanır).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from ..models import NewsRecord
from ..textutil import normalize_ws

# Modelin döndürmesi gereken JSON anahtarları (sıra sabittir).
VERDICT_KEYS: tuple[str, ...] = ("alarm_score", "is_alarm", "reason", "summary", "topics", "entities")

# Sabit konu listesi; model bunların dışında konu üretmemelidir.
TOPICS: tuple[str, ...] = (
    "siyaset",
    "ekonomi",
    "hukuk",
    "güvenlik",
    "dış politika",
    "finans/fon",
    "sosyal",
    "diğer",
)

TRUNCATION_MARKER = "[...kısaltıldı]"

# Kullanıcı istemindeki bölüm etiketleri.
LABEL_SOURCE = "Kaynak"
LABEL_TITLE = "Başlık"
LABEL_SUBTITLE = "Alt başlık"
LABEL_DATE = "Haber tarihi"
LABEL_KEYWORDS = "Eşleşen anahtar kelimeler"
LABEL_CONTENT = "Haber metni"
CONTENT_OPEN = "<<<HABER"
CONTENT_CLOSE = "HABER>>>"
NO_KEYWORDS_TEXT = "yok (filtreden geçmeden doğrudan değerlendirmeye alındı)"
UNKNOWN_TEXT = "bilinmiyor"

_ISTANBUL_TZ = timezone(timedelta(hours=3), name="+03:00")


def build_system_prompt(alarm_threshold: int) -> str:
    """Analist rolü + rubrik + katı JSON talimatı. ``alarm_threshold`` politikayla tutarlı olsun diye gömülür."""
    topics = ", ".join(TOPICS)
    return (
        "Sen Türkiye gündemini izleyen deneyimli bir haber analistisin. Görevin, sana verilen tek bir haberi "
        "kamu yönetimi, siyaset, hukuk, güvenlik ve finans açısından değerlendirip bir ALARM SKORU üretmek.\n"
        "\n"
        "ALARM RUBRİĞİ (alarm_score, 0-100 arası tam sayı):\n"
        "- 80-100 KRİTİK: Cumhurbaşkanı veya bakan düzeyinde geniş etkili kararlar/kararnameler; fon, banka veya "
        "finansal suç soruşturmaları (kara para, dolandırıcılık, TMSF/MASAK/SPK işlemleri); kamu görevlilerinin "
        "tutuklanması/gözaltına alınması; piyasaları doğrudan etkileyecek politika değişiklikleri; büyük güvenlik "
        "krizleri.\n"
        "- 60-79 ÖNEMLİ: Bakanlık düzeyinde açıklama ve düzenlemeler, önemli yasa teklifleri, üst düzey "
        "görevden alma/atama, kapsamlı operasyonlar, ekonomik göstergelerde dikkat çeken gelişmeler.\n"
        "- 30-59 DİKKATE DEĞER: Siyasi tartışmalar, parti içi gelişmeler, yerel yönetim kararları, "
        "soruşturma iddiaları, sınırlı etkili düzenlemeler.\n"
        "- 0-29 RUTİN/İLGİSİZ: Spor, magazin, yaşam, tanıtım/PR, hava durumu, geçmiş olayların tekrarı, "
        "anahtar kelimenin yalnızca yan anlamda geçtiği haberler.\n"
        "\n"
        "ANAHTAR KELİME BAĞLAMI: Haber, 'bakan', 'cumhurbaşkanı', 'fon' gibi anahtar kelimelerle filtrelenmiş "
        "olabilir. Kelimenin bağlamını MUTLAKA kontrol et: 'denize bakan oda' ifadesindeki 'bakan' bir fiildir "
        "(bakmak), bir bakan (minister) değildir; 'fon' müzikteki fon müziği veya arka plan anlamında da "
        "kullanılabilir. Yanlış bağlamdaki eşleşmeler skoru YÜKSELTMEZ ve reason alanında bunu açıkça belirt.\n"
        "\n"
        f"Alarm eşiği {alarm_threshold} puandır: is_alarm alanı alarm_score >= {alarm_threshold} ise true, "
        "aksi halde false olmalıdır.\n"
        "\n"
        "ÇIKTI KURALLARI:\n"
        "- YALNIZCA geçerli bir JSON nesnesi döndür. Markdown, kod bloğu, açıklama veya ek metin YAZMA.\n"
        "- Tam olarak şu anahtarları kullan: alarm_score, is_alarm, reason, summary, topics, entities.\n"
        "- alarm_score: 0-100 arası tam sayı.\n"
        "- is_alarm: true/false (boolean).\n"
        "- reason: Bu skoru neden verdiğini 1-3 cümlede Türkçe açıkla; eşleşen anahtar kelimelerin bağlamına "
        "(gerçek bakan mı, fiil mi; gerçek fon mu, yan anlam mı) mutlaka değin.\n"
        "- summary: Haberin 2-3 cümlelik tarafsız Türkçe özeti (kim, ne, ne zaman, sonuç).\n"
        f"- topics: Yalnızca şu listeden 1-3 konu: {topics}.\n"
        "- entities: Haberde adı geçen kişi, kurum, parti ve şirketlerin listesi (özel adlar, en fazla 10).\n"
        "- Tüm metin alanları Türkçe olmalı.\n"
        "\n"
        "ÖRNEK ÇIKTI:\n"
        '{"alarm_score": 85, "is_alarm": true, "reason": "İçişleri Bakanı\'nın doğrudan talimatıyla başlatılan '
        "ve 12 belediye yetkilisinin tutuklanmasıyla sonuçlanan soruşturma; 'bakan' kelimesi gerçek bir bakanı "
        'ifade ediyor.", "summary": "İçişleri Bakanlığı koordinasyonunda yürütülen operasyonda 12 belediye '
        "yetkilisi tutuklandı. Soruşturma ihale yolsuzluğu iddialarına dayanıyor. Bakan, sürecin genişleyeceğini "
        'açıkladı.", "topics": ["hukuk", "siyaset"], "entities": ["İçişleri Bakanlığı", "Ali Yerlikaya"]}'
    )


STRICT_JSON_REMINDER = (
    "\n\nUYARI: Önceki yanıtın geçerli JSON değildi. Bu kez SADECE tek bir JSON nesnesi döndür; "
    "markdown kod bloğu, açıklama veya başka metin ekleme. Anahtarlar tam olarak şunlar olmalı: "
    + ", ".join(VERDICT_KEYS)
    + ". alarm_score tam sayı, is_alarm boolean, topics ve entities dize listesi olmalıdır."
)


def truncate_content(text: str, limit: int) -> str:
    """Metni ``limit`` karaktere kısaltır; kesilirse kelime sınırından keser ve işaret ekler."""
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind(" ")
    if cut > limit // 2:
        head = head[:cut]
    return head.rstrip() + "\n" + TRUNCATION_MARKER


def format_published_at(dt: datetime | None) -> str:
    """Haber tarihini Türkiye saatinde, insan ve model için okunur biçimde yazar."""
    if dt is None:
        return UNKNOWN_TEXT
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ISTANBUL_TZ)
    local = dt.astimezone(_ISTANBUL_TZ)
    return local.strftime("%d.%m.%Y %H:%M (%z)")


def build_user_prompt(
    record: NewsRecord,
    *,
    max_content_chars: int,
    matched_keywords: list[str] | None = None,
) -> str:
    """Tek bir haberi etiketli bölümler halinde modele sunar. İçerik ``max_content_chars`` ile sınırlanır."""
    keywords = list(matched_keywords if matched_keywords is not None else record.matched_keywords)
    keyword_text = ", ".join(keywords) if keywords else NO_KEYWORDS_TEXT
    content = truncate_content(normalize_ws(record.content), max_content_chars) or "(haber metni boş)"
    subtitle = normalize_ws(record.subtitle).replace("\n", " ") or "-"
    title = normalize_ws(record.title).replace("\n", " ") or "-"
    return (
        "Aşağıdaki haberi rubriğe göre değerlendir ve yalnızca JSON döndür.\n"
        "\n"
        f"{LABEL_SOURCE}: {record.source}\n"
        f"{LABEL_TITLE}: {title}\n"
        f"{LABEL_SUBTITLE}: {subtitle}\n"
        f"{LABEL_DATE}: {format_published_at(record.published_at)}\n"
        f"{LABEL_KEYWORDS}: {keyword_text}\n"
        f"{LABEL_CONTENT}:\n"
        f"{CONTENT_OPEN}\n"
        f"{content}\n"
        f"{CONTENT_CLOSE}\n"
    )


_LINE_RE = re.compile(r"^(?P<label>[^:\n]{1,40}):[ \t]*(?P<value>.*)$", re.MULTILINE)


def parse_user_prompt(user: str) -> dict[str, str]:
    """``build_user_prompt`` çıktısını bölümlerine ayırır.

    Dönen sözlük anahtarları: source, title, subtitle, date, keywords, content (eksikler boş dize).
    """
    text = user or ""
    content = ""
    start = text.find(CONTENT_OPEN)
    end = text.rfind(CONTENT_CLOSE)
    if start != -1 and end != -1 and end > start:
        content = text[start + len(CONTENT_OPEN) : end].strip()
        head = text[:start]
    else:
        head = text
    fields = {"source": "", "title": "", "subtitle": "", "date": "", "keywords": "", "content": content}
    label_map = {
        LABEL_SOURCE: "source",
        LABEL_TITLE: "title",
        LABEL_SUBTITLE: "subtitle",
        LABEL_DATE: "date",
        LABEL_KEYWORDS: "keywords",
    }
    for m in _LINE_RE.finditer(head):
        key = label_map.get(m.group("label").strip())
        if key:
            value = m.group("value").strip()
            fields[key] = "" if value in ("-", UNKNOWN_TEXT, NO_KEYWORDS_TEXT) else value
    return fields
