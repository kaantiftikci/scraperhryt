"""Merkezi yapılandırma. Tüm servisler bu Settings nesnesini kullanır.

Değerler ortam değişkenlerinden (veya proje kökündeki .env dosyasından) okunur.
Örnek: RABBITMQ_URL, ELASTICSEARCH_URL, OLLAMA_BASE_URL, OLLAMA_MODEL, KEYWORDS, ALARM_THRESHOLD ...
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 scraperhryt/0.1"
)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Genel ----
    log_level: str = "INFO"
    environment: str = "dev"

    # ---- RabbitMQ ----
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F"
    rabbitmq_exchange: str = "news.topic"
    rabbitmq_dlx: str = "news.dlx"
    rabbitmq_prefetch: int = 8
    rabbitmq_max_attempts: int = 5
    rabbitmq_retry_delay_ms: int = 15_000  # kuyruk argümanıdır: değiştirince q.*.retry kuyruklarını silip yeniden oluşturun
    rabbitmq_heartbeat: int = 60

    # ---- Elasticsearch ----
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str = ""
    es_index_articles: str = "news-articles"
    es_index_alarms: str = "news-alarms"
    es_index_reports: str = "news-reports"
    es_request_timeout: float = 30.0

    # ---- Ollama / LLM ----
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = ""  # boş ise sadece BM25 arama kullanılır (ör: nomic-embed-text)
    ollama_timeout: float = 180.0
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 8192
    ollama_max_content_chars: int = 6000  # LLM'e gönderilen haber metninin üst sınırı
    embedding_dims: int = 768  # ollama_embedding_model ayarlıysa dense_vector boyutu (nomic-embed-text=768)

    # ---- Anahtar kelime filtresi ----
    # Virgülle ayrılmış liste. Varsayılan eşleşme modu Türkçe ek-toleranslı kök eşleşmesidir
    # ("bakan" → bakanı, bakanlık, bakanlar...). "=kelime" tam eşleşme, "re:..." regex.
    keywords: str = "bakan,cumhurbaşkanı,fon"
    alarm_threshold: int = Field(default=60, ge=0, le=100)
    llm_score_all: bool = False  # True ise anahtar kelime içermeyen haberler de LLM'e gönderilir
    keyword_aliases_path: str = "config/keyword_aliases.json"  # kanonik anahtar kelime → eş anlamlı/varlık listesi
    # Kaynak/kategori bazlı eşikler, JSON: {"source:12punto": 70, "category:Spor": 90}; yoksa alarm_threshold
    alarm_thresholds_json: str = ""

    # ---- Ön sınıflandırıcı (embedding tabanlı ilgililik filtresi; LLM'den önce yanlış pozitifleri eler) ----
    preclassifier_enabled: bool = False
    # relevance = cos(metin, ilgili merkez) - cos(metin, ilgisiz merkez); nomic-embed-text ile tipik aralık ±0.05,
    # bu yüzden varsayılan 0.0 (ilgisiz merkeze daha yakınsa ele). Yükseltmek daha agresif eler.
    preclassifier_threshold: float = 0.0
    preclassifier_prototypes_path: str = "config/preclassifier_prototypes.json"

    # ---- LLM skorlama kalitesi ----
    llm_samples: int = 1  # >1 ise aynı haber N kez skorlanır, medyan alınır (öz-tutarlılık)
    llm_sample_temperature: float = 0.4  # çoklu örneklemede kullanılan sıcaklık
    llm_disagreement_threshold: int = 25  # örnekler arası skor farkı bunu aşarsa needs_review=True
    llm_fewshot_examples: int = 0  # prompta altın setten eklenecek örnek sayısı
    golden_set_path: str = "config/golden_set.jsonl"  # etiketli kalibrasyon örnekleri
    rescore_batch_size: int = 50

    # ---- Scraper ----
    sources: str = "hurriyet,12punto"
    scrape_interval_seconds: int = 300
    request_timeout: float = 25.0
    request_delay_seconds: float = 0.5  # aynı siteye ardışık istekler arası bekleme
    user_agent: str = DEFAULT_USER_AGENT
    state_db_path: str = "data/state.sqlite3"
    backfill_days: int = 0  # >0 ise 12punto arşiv aramasıyla geriye dönük tarama yapılır
    max_articles_per_run: int = 400
    hurriyet_gundem_rss: str = "https://www.hurriyet.com.tr/rss/gundem"
    hurriyet_gundem_listing: str = "https://www.hurriyet.com.tr/gundem/"
    punto_base_url: str = "https://12punto.com.tr"
    # 12punto'nun /rss/<kategori> beslemesi ve /<kategori> listesi olan tüm bölümler. Karışık /rss yalnızca son
    # 20 haberi verdiğinden listede olmayan bir bölüm, tarama aralığı uzadığında sessizce kaçar.
    hurriyet_deep_pages: int = 0  # >0 ise Playwright ile Hürriyet gündem listesinde JS sayfalama/"daha fazla" ile bu kadar sayfa derin taranır
    deep_crawl_timeout: float = 60.0
    punto_categories: str = (
        "gundem,siyaset,dunya,ekonomi,yasam,spor,bilim-teknoloji,kulis,medya,adalet-hukuk,"
        "yerel-haberler,kultur-sanat,saglik,egitim,cevre,turkiye,kamu-gundemi,is-dunyasi,secim,"
        "otomotiv,seyahat,gurme,trend-bilgi-kapsulu"
    )

    # ---- Alarm katmanı ----
    alarm_webhook_url: str = ""  # genel JSON webhook (Slack/Discord/Mattermost uyumlu "text" alanı da gönderilir)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    alarm_log_path: str = "data/alarms.jsonl"
    # Olay kümeleme / tekrar alarm bastırma
    alarm_dedup_enabled: bool = True
    alarm_dedup_window_hours: int = 24
    alarm_dedup_similarity: float = 0.82  # embedding kosinüs benzerliği eşiği
    alarm_dedup_title_jaccard: float = 0.6  # embedding yoksa başlık kelime Jaccard eşiği
    alarm_notify_duplicates: bool = False  # tekrar (duplicate_of dolu) alarmlar bildirim kanallarına gitmesin

    # ---- Raporlama / API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    report_interval_minutes: int = 60
    report_window_hours: int = 24
    report_digest_every: int = 10      # q.alarms'tan bu kadar alarm birikince alarm özeti raporu üret
    report_digest_minutes: int = 30    # ... veya en son özetten bu kadar dakika geçince
    rag_top_k: int = 12
    rag_recency_days: int = 14
    rag_hybrid: bool = True  # embedding modeli ayarlıysa BM25 + kNN (RRF) birleşik arama
    rag_eval_path: str = "config/rag_eval.jsonl"
    # Ani artış (burst) tespiti: aynı konu/varlık için pencere içinde en az N alarm → burst raporu
    burst_window_minutes: int = 60
    burst_min_articles: int = 5
    # Prometheus metrikleri
    metrics_enabled: bool = True
    metrics_port: int = 9108

    # ---- Türetilmiş yardımcılar ----
    @property
    def keyword_list(self) -> list[str]:
        return _split_csv(self.keywords)

    @property
    def source_list(self) -> list[str]:
        return _split_csv(self.sources)

    @property
    def punto_category_list(self) -> list[str]:
        return _split_csv(self.punto_categories)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
