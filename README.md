# scraperhryt — Hürriyet Gündem + 12punto haber izleme ve alarm boru hattı

`scraperhryt`, **Hürriyet Gündem** ve **12punto** sitelerindeki haberleri sürekli toplayan, anahtar kelime
(`bakan`, `cumhurbaşkanı`, `fon` …) içerenleri **Ollama** üzerinde çalışan yerel bir LLM'e yorumlatıp
**alarm skoru** üreten, sonuçları **RabbitMQ** ve **Elasticsearch**'e yazan, alarmları bir **alarm katmanına**
ileten, üzerine bir **raporlama katmanı** ve "son durum ne?" tarzı soruları son haberlerden yanıtlayan bir
**RAG soru-cevap** servisi kuran Python 3.11 projesidir.

Tüm katmanlar **tek bir nesneyi** (`NewsRecord`) taşır: içerik URL'si, başlık, alt başlık, haber tarihi, haber
içeriği, alarm skoru ve alarm sebebi. Alarm verilen kayıtlarda `alarm_reason` gerekçe + LLM özetini içerir;
alarma gitmeyen kayıtlarda bu alanlar **boş kalır** ama nesne yine de hem RabbitMQ'ya hem Elasticsearch'e yazılır.

- Mimari ayrıntıları ve raporlama/RAG tasarımı: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- İşletme el kitabı (başlat/durdur, sağlık, ölü mektup, model değiştirme): [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- Mühendislik sözleşmesi (modül sınırları): [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md)
- Gereksinim izleme matrisi (istek → kod yolu → kanıtlayan test): [`docs/REQUIREMENTS_TRACE.md`](docs/REQUIREMENTS_TRACE.md)

---

## 1. Sistem ne yapar? (İstekten katmanlara)

| # | İstek | Nasıl karşılanıyor | Modül → Kuyruk |
|---|-------|--------------------|----------------|
| 1 | hurriyet.com gündem kategorisindeki ve 12punto sitesindeki tüm içerikleri çek | Hürriyet Gündem RSS (+ liste sayfası) ve 12punto.com.tr RSS / kategori RSS / liste sayfaları / arşiv araması düzenli aralıklarla taranır; her haber `NewsRecord` olarak yayınlanır | `scrapers/` → `q.articles.raw` |
| 2 | içinde *bakan, cumhurbaşkanı, fon* gibi anahtar kelimeler geçen içerikler RabbitMQ'daki bir kuyruğa obje olarak gitsin | Türkçe ek-toleranslı anahtar kelime eşleyici (`bakan` → bakanı, bakanlık, bakanlar …) eşleşen haberleri `matched_keywords` ile zenginleştirip kuyruğa yazar | `pipeline/keyword_filter.py` → `q.articles.keyword` |
| 3 | Ollama üzerinden çalışan bir LLM ile yorumlatıp RabbitMQ'ya gönder | Türkçe analist istemiyle Ollama (`/api/chat`, JSON modu) 0-100 skor, gerekçe, özet, konu ve varlık üretir; `alarm_score >= ALARM_THRESHOLD` ise alarm | `pipeline/scorer.py`, `pipeline/llm.py` → `q.articles.scored` |
| 4 | RabbitMQ'dan alarm katmanına alarm gitsin; alarmsa RabbitMQ + Elastic'e, değilse yine RabbitMQ + Elastic'e kaydedilsin | Alarm katmanı **her** kaydı `news-articles` indeksine yazar; alarm olanları ayrıca `news-alarms`'a, `q.alarms` kuyruğuna ve kanallara (log/webhook/Telegram) iletir | `pipeline/alarm.py` → ES + `q.alarms` |
| 5 | alarmdan sonra raporlama katmanına gelsin; bu katmanın mimarisini geliştir | Alarm özetleri (`alarm_digest`), periyodik raporlar ve isteğe bağlı raporlar Elasticsearch toplulaştırmaları + LLM anlatısıyla üretilir; `news-reports` indeksine ve `q.reports` kuyruğuna yazılır; pano ve API ile sunulur | `reporting/` → ES + `q.reports` |
| 6 | "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?" diyince son gelen içeriklerden yanıt versin | RAG: sorgu yeniden yazma → Türkçe BM25 + yenilik ağırlığı (isteğe bağlı vektör arama) → en yeni haberler önce → kaynak atıflı Türkçe yanıt | `reporting/rag.py`, `scraperhryt ask`, `POST /ask` |

---

## 2. Önemli bulgular (siteler hakkında)

- **12punto.com park edilmiş bir alan adıdır.** Gerçek site **`https://12punto.com.tr`**'dir; `PUNTO_BASE_URL`
  varsayılanı budur. RSS'teki bağlantılar `http://` ile gelir; kazıyıcı bunları `https://` olarak normalize eder.
- **Her iki sitede de liste sayfalama JavaScript ile yapılır.** Hürriyet'te `/gundem/?page=N` hep aynı sayfayı
  döndürür; 12punto'da "Sonraki Haberler" düğmesi JS'dir. Bu yüzden keşif şu kaynaklara dayanır:
  - Hürriyet Gündem RSS (`https://www.hurriyet.com.tr/rss/gundem`): son **100** haber, **tam metin** (`<text>`)
    ve spot (`<abstract>`) içerir → sayfa alınamasa bile kayıt üretilebilir;
  - Hürriyet `/gundem/` liste sayfası (30 bağlantı);
  - 12punto `/rss` (20 karışık öğe) + her kategori için `/rss/<kategori>` (`PUNTO_CATEGORIES`);
  - 12punto kategori liste sayfaları (`/<kategori>`, 25 bağlantı);
  - 12punto **tarih aralıklı arşiv araması** (`/Arama/Ara?search=&StartDate=…&EndDate=…`) — yalnızca geriye dönük
    tarama (`--backfill-days N` / `BACKFILL_DAYS`) için, gün gün.
- **Hürriyet `robots.txt` `/api/` ve `/arama/` yollarını yasaklar.** Kazıyıcı bu yollara **asla** istek atmaz; bu
  nedenle Hürriyet için arşiv/arama taraması yoktur (`--backfill-days` Hürriyet'te yok sayılır).
- Haber sayfaları JSON-LD `NewsArticle` ile ayrıştırılır (Hürriyet: sözlük; 12punto: listenin ilk öğesi). Hürriyet
  gövdesine sızan "Haberlerimizi Google'da takip edin … tercih edilen kaynak olarak ekleyin" kalıbı temizlenir.
- Tarihler saat dilimli saklanır (naif tarihler Europe/Istanbul, +03:00 kabul edilir); ES'te `@timestamp` haber
  tarihidir (yoksa kazınma zamanı).

---

## 3. Hızlı başlangıç (Docker Compose + host üzerinde Ollama)

Önkoşullar: Docker 24+ ve Compose v2; ana makinede [Ollama](https://ollama.com).

```bash
# 1) Modeli (ve isteğe bağlı embedding modelini) ana makinedeki Ollama'ya indirin
ollama pull qwen2.5:7b
ollama pull nomic-embed-text      # isteğe bağlı: RAG'de vektör (kNN) arama için

# 2) Ayar dosyası (varsayılanlar compose'a göredir: OLLAMA_BASE_URL=http://host.docker.internal:11434,
#    RABBITMQ_URL ve ELASTICSEARCH_URL compose servis adlarını kullanır; KEYWORDS ve ALARM_THRESHOLD'u düzenleyin)
cp .env.example .env
# Uygulama konteynerleri ana makinedeki Ollama'ya docker-compose.yml'deki
#   extra_hosts: ["host.docker.internal:host-gateway"]
# eşlemesiyle ulaşır (Linux dahil). Embedding kullanacaksanız:
#   OLLAMA_EMBEDDING_MODEL=nomic-embed-text, EMBEDDING_DIMS=768

# 3) Yığını başlatın: rabbitmq (5672, yönetim 15672) + elasticsearch (9200) → tek seferlik `setup` servisi
#    (topoloji + indeksler + model denetimi) → scraper, filter, scorer, alarm, reporter, api
docker compose up -d --build

# 4) Durum
docker compose ps
docker compose logs -f scorer alarm
```

Elasticsearch konteyneri 1 GB heap ile başlar; Linux'ta `vm.max_map_count` en az 262144 olmalıdır
(`sudo sysctl -w vm.max_map_count=262144`), aksi halde konteyner başlangıçta çıkar.

- Pano: <http://localhost:8000/> — API belgeleri: <http://localhost:8000/docs>
- RabbitMQ yönetim arayüzü: <http://localhost:15672> (guest / guest)
- Elasticsearch: <http://localhost:9200>

İsteğe bağlı profiller:

```bash
docker compose --profile kibana up -d    # Kibana (5601) — Elasticsearch verisini görselleştirmek için
docker compose --profile ollama up -d    # Ollama'yı konteynerde çalıştırmak için; .env'de OLLAMA_BASE_URL=http://ollama:11434
                                         # yapın. `ollama-pull` servisi OLLAMA_MODEL (ve varsa OLLAMA_EMBEDDING_MODEL)'i indirir.
```

`Makefile` sık kullanılan kısayolları içerir: `make up`, `make down`, `make logs`, `make ps`, `make setup`,
`make check`, `make test`, `make lint`, `make ask Q="soru"`.

---

## 4. Docker olmadan çalıştırma

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                 # ayarlar proje kökündeki .env dosyasından okunur; komutları kökte çalıştırın
# .env içindeki "yerel çalıştırma" satırlarını açın (compose servis adları yerine localhost):
#   RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2F
#   ELASTICSEARCH_URL=http://localhost:9200
#   OLLAMA_BASE_URL=http://localhost:11434

# RabbitMQ ve Elasticsearch'ü ayrıca başlatın (ör. docker compose up -d rabbitmq elasticsearch)
# Ollama ana makinede: ollama serve && ollama pull qwen2.5:7b

scraperhryt check                     # RabbitMQ / Elasticsearch / Ollama bağlantı ve model kontrolü
scraperhryt setup                     # exchange, kuyruklar, retry/ölü mektup kuyrukları ve ES indekslerini oluşturur (idempotent)
scraperhryt run-all                   # tüm katmanları tek süreçte, iş parçacıklarıyla çalıştırır
```

Katmanları ayrı süreçlerde çalıştırmak (üretimde önerilen; her biri bağımsız ölçeklenir):

```bash
scraperhryt scrape        # kazıyıcı (SCRAPE_INTERVAL_SECONDS aralıklarla)
scraperhryt filter        # anahtar kelime filtresi
scraperhryt score         # LLM skorlayıcı
scraperhryt alarm         # alarm katmanı
scraperhryt report        # raporlama katmanı (alarm özetleri + periyodik raporlar)
scraperhryt api           # FastAPI + pano
```

Altyapısız deneme (RabbitMQ, Elasticsearch ve Ollama **olmadan**; siteler gerçekten kazınır):

```bash
scraperhryt run-all --once --in-memory --fake-llm
# aynı çalıştırmanın sonunda bellek içi depoya soru da sorulabilir:
scraperhryt run-all --once --in-memory --fake-llm --ask "Fon soruşturmasında son durum ne?"
```

`--in-memory` bellek içi broker ve depo kullanır (veri süreç bitince kaybolur), `--fake-llm` Ollama yerine
deterministik `HeuristicLLM`'i (anahtar kelime/risk terimi sayımına dayalı skor) kullanır, `--once` tek tur çalışıp çıkar.
`--ask` boru hattı bittikten sonra aynı süreçteki depo üzerinde RAG sorusunu yanıtlar (bellek içi depo başka bir
süreçten erişilemediği için `scraperhryt ask` burada kullanılamaz). Küçük denemelerde `MAX_ARTICLES_PER_RUN=40` gibi
bir bütçe verin; bütçe kaynaklar arasında eşit paylaşılır (20 Hürriyet + 20 12punto).

---

## 5. Komutlar

| Komut | Açıklama |
|-------|----------|
| `scraperhryt [--log-level SEVIYE] [--version] <komut>` | Genel seçenekler: `--log-level` `LOG_LEVEL`'i geçersiz kılar (DEBUG/INFO/WARNING/ERROR), `--version` sürümü yazdırır. |
| `scraperhryt setup [--ollama-optional] [--ollama-wait SN] [--timeout SN]` | Veri dizinlerini, RabbitMQ topolojisini (`news.topic`, `news.dlx`, ana/retry/ölü mektup kuyrukları) ve Elasticsearch indekslerini oluşturur, Ollama'da `OLLAMA_MODEL`'in yüklü olduğunu denetler. Ollama erişilemez ya da model yüklü değilse **HATA verip çıkış kodu 1 ile biter** (compose yığınında uygulama servisleri başlamaz); `--ollama-optional` ya da `OLLAMA_OPTIONAL=1` ile Ollama sorunu yalnızca uyarıdır ve çıkış kodu 0 olur (`--fake-llm` kullanımı için). `--ollama-wait SN` (`OLLAMA_WAIT_SECONDS`) Ollama + model hazır olana dek en çok SN saniye bekler (varsayılan 0); `--timeout` Ollama sondası zaman aşımı (varsayılan 5 s). Tekrar çalıştırmak güvenlidir; compose yığınında `setup` servisi olarak otomatik çalışır. |
| `scraperhryt check [--timeout SN] [--ollama-optional]` | RabbitMQ, Elasticsearch ve Ollama erişilebilirliğini (gecikmelerle) ve `OLLAMA_MODEL`'in yüklü olup olmadığını denetler. `--timeout` servis başına zaman aşımı (varsayılan 5 s); `--ollama-optional` ile Ollama sorunu çıkış kodunu bozmaz. |
| `scraperhryt scrape [--once] [--source X] [--backfill-days N] [--interval SN]` | Kazıyıcı. `--once` tek tur; `--source hurriyet` veya `--source 12punto` tek kaynak; `--backfill-days N` 12punto arşivini N gün geriye tarar; `--interval` turlar arası bekleme (`SCRAPE_INTERVAL_SECONDS`). |
| `scraperhryt filter` | `q.articles.raw` tüketicisi: anahtar kelime filtresi. |
| `scraperhryt score [--fake-llm]` | `q.articles.keyword` tüketicisi: Ollama ile skorlama; `--fake-llm` Ollama yerine sezgisel değerlendirici. |
| `scraperhryt alarm` | `q.articles.scored` tüketicisi: ES'e yazma, alarm üretme, kanallar, `q.alarms`. |
| `scraperhryt report [--once] [--fake-llm]` | `q.alarms` tüketicisi (alarm özetleri) + periyodik rapor üretici. `--once` tek periyodik rapor üretir, yazdırır ve çıkar. |
| `scraperhryt api [--host ADRES] [--port PORT] [--fake-llm]` | HTTP API ve pano (varsayılan `API_HOST:API_PORT`). |
| `scraperhryt ask "soru" [--since-days N] [--top-k N] [--json] [--fake-llm]` | RAG soru-cevap: son haberlerden kaynak atıflı Türkçe yanıt üretir. `--since-days` yalnızca son N gün (`RAG_RECENCY_DAYS`); `--top-k` bağlama alınacak haber sayısı (`RAG_TOP_K`); `--json` `Answer` nesnesini JSON yazdırır. |
| `scraperhryt run-all [--once] [--in-memory] [--fake-llm] [--llm-fallback] [--no-api] [--backfill-days N] [--idle-timeout SN] [--ask "soru"]` | Tüm katmanlar tek süreçte (her tüketici iş parçacığı kendi broker bağlantısını kullanır). `--once`: kazıyıcı tek tur çalışır, kuyruklar boşalınca özet yazdırıp çıkar (API başlatılmaz); RabbitMQ modunda kuyruklar `--idle-timeout` saniye (varsayılan 30) boş kalınca çıkar. `--no-api` sürekli modda API'yi başlatmaz. `--ask "soru"`: `--once` ile boru hattı bitince aynı süreçteki depo üzerinde RAG sorusunu yanıtlayıp yazdırır (bellek içi depoda tek yol; `--fake-llm` ile yanıt haberlerden derlenen özetleyici geri dönüştür). `--llm-fallback`: Ollama başlangıçta erişilemez ya da model yüklü değilse bu çalıştırma **boyunca** Ollama yerine sezgisel değerlendirici (`HeuristicLLM`) kullanılır; skorlar yaklaşık olur ve Ollama sonradan ayağa kalksa da kullanılmaz. Varsayılan (bayrak yok): Ollama istemcisi korunur, LLM gerektiren mesajlar Ollama hazır olana dek gecikmeli yeniden denenir. Başlangıçta Elasticsearch erişilemezse (`--in-memory` değilse) komut beklemeden çıkış kodu 1 ile biter. SIGINT/SIGTERM ile düzgün kapanır. |

---

## 6. Ortam değişkenleri

Tümü `.env` dosyasından ya da ortamdan okunur (önek yok, büyük/küçük harf duyarsız). `.env.example` tam listeyi içerir.

### Genel

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `LOG_LEVEL` | `INFO` | Günlük seviyesi (`DEBUG`, `INFO`, `WARNING` …). Günlükler stderr'e yazılır. |
| `ENVIRONMENT` | `dev` | Ortam etiketi (`dev`, `prod` …); yalnızca bilgilendirme amaçlı. |

### RabbitMQ

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/%2F` | AMQP bağlantı URL'si. `memory://` verilirse bellek içi broker kullanılır. |
| `RABBITMQ_EXCHANGE` | `news.topic` | Ana topic exchange. |
| `RABBITMQ_DLX` | `news.dlx` | Ölü mektup exchange'i (fanout) → `q.dead_letter`. |
| `RABBITMQ_PREFETCH` | `8` | Tüketici başına aynı anda alınan onaylanmamış mesaj sayısı (skorlayıcı LLM yavaş olduğu için 1 kullanır). |
| `RABBITMQ_MAX_ATTEMPTS` | `5` | Bir mesajın toplam deneme sayısı; aşılınca ölü mektup kuyruğuna gider. |
| `RABBITMQ_RETRY_DELAY_MS` | `15000` | Başarısız mesajın `<kuyruk>.retry` kuyruğunda bekleme süresi (TTL). |
| `RABBITMQ_HEARTBEAT` | `60` | AMQP heartbeat (saniye). |

### Elasticsearch

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch adresi. |
| `ELASTICSEARCH_API_KEY` | *(boş)* | API anahtarı (güvenlik açık kurulumlarda). |
| `ES_INDEX_ARTICLES` | `news-articles` | Tüm haberlerin indeksi. |
| `ES_INDEX_ALARMS` | `news-alarms` | Alarm olaylarının indeksi. |
| `ES_INDEX_REPORTS` | `news-reports` | Raporların indeksi. |
| `ES_REQUEST_TIMEOUT` | `30.0` | ES istek zaman aşımı (saniye). |

### Ollama / LLM

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP adresi. |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Skorlama, rapor anlatısı ve RAG yanıtı için model. |
| `OLLAMA_EMBEDDING_MODEL` | *(boş)* | Boşsa yalnızca BM25 arama; `nomic-embed-text` gibi bir model verilirse ES'e `dense_vector` eklenir ve RAG'de kNN kullanılır. |
| `OLLAMA_TIMEOUT` | `180.0` | LLM isteği zaman aşımı (saniye). |
| `OLLAMA_TEMPERATURE` | `0.1` | Örnekleme sıcaklığı (düşük = tutarlı JSON). |
| `OLLAMA_NUM_CTX` | `8192` | Bağlam penceresi (token). |
| `OLLAMA_MAX_CONTENT_CHARS` | `6000` | LLM'e gönderilen haber metninin üst sınırı (karakter); fazlası kelime sınırından kesilir. |
| `EMBEDDING_DIMS` | `768` | `OLLAMA_EMBEDDING_MODEL` ayarlıysa vektör boyutu (nomic-embed-text = 768). |

### Anahtar kelime filtresi ve alarm politikası

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `KEYWORDS` | `bakan,cumhurbaşkanı,fon` | Virgülle ayrılmış liste. Varsayılan mod Türkçe ek-toleranslı kök eşleşmesi; `=kelime` tam kelime, `~parça` alt dize, `re:desen` regex. |
| `ALARM_THRESHOLD` | `60` | `alarm_score >= ALARM_THRESHOLD` → alarm (0-100). |
| `LLM_SCORE_ALL` | `false` | `true` ise anahtar kelime içermeyen haberler de LLM'e gönderilir (maliyetli). |

### Kazıyıcı

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `SOURCES` | `hurriyet,12punto` | Etkin kaynaklar. |
| `SCRAPE_INTERVAL_SECONDS` | `300` | Turlar arası bekleme. |
| `REQUEST_TIMEOUT` | `25.0` | HTTP okuma zaman aşımı (saniye). |
| `REQUEST_DELAY_SECONDS` | `0.5` | Aynı siteye ardışık istekler arası bekleme (kibarlık). |
| `USER_AGENT` | `Mozilla/5.0 … scraperhryt/0.1` | HTTP User-Agent; proje adını içerir. |
| `STATE_DB_PATH` | `data/state.sqlite3` | Görülen haberlerin SQLite kaydı (yinelenen yayını önler, güncellenen haberi yakalar). |
| `BACKFILL_DAYS` | `0` | >0 ise 12punto arşiv aramasıyla bu kadar gün geriye dönük tarama. |
| `MAX_ARTICLES_PER_RUN` | `400` | Tur başına toplam sayfa çekme bütçesi. Kaynaklar arasında adil paylaşılır (her kaynak en fazla `ceil(kalan bütçe / kalan kaynak)`; kullanılmayan pay sonraki kaynağa devreder). |
| `HURRIYET_GUNDEM_RSS` | `https://www.hurriyet.com.tr/rss/gundem` | Hürriyet Gündem RSS adresi. |
| `HURRIYET_GUNDEM_LISTING` | `https://www.hurriyet.com.tr/gundem/` | Hürriyet Gündem liste sayfası. |
| `PUNTO_BASE_URL` | `https://12punto.com.tr` | 12punto gerçek alan adı. |
| `PUNTO_CATEGORIES` | `gundem,siyaset,dunya,ekonomi,yasam,spor,bilim-teknoloji,kulis,medya,adalet-hukuk,yerel-haberler,kultur-sanat,saglik,egitim,cevre,turkiye,kamu-gundemi,is-dunyasi,secim,otomotiv,seyahat,gurme,trend-bilgi-kapsulu` | Taranan 12punto kategorileri (RSS + liste). |

### Alarm kanalları

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `ALARM_WEBHOOK_URL` | *(boş)* | Genel JSON webhook; gövde `{"text": "...", "event": {...}}` (Slack/Discord/Mattermost uyumlu `text` alanı). |
| `TELEGRAM_BOT_TOKEN` | *(boş)* | Telegram bot token'ı (`TELEGRAM_CHAT_ID` ile birlikte gerekir). |
| `TELEGRAM_CHAT_ID` | *(boş)* | Alarmların gönderileceği sohbet. |
| `ALARM_LOG_PATH` | `data/alarms.jsonl` | Her alarmın bir JSON satırı olarak eklendiği dosya (her zaman etkin). |

### Raporlama / API / RAG

| Değişken | Varsayılan | Anlamı |
|----------|------------|--------|
| `API_HOST` | `0.0.0.0` | API dinleme adresi. |
| `API_PORT` | `8000` | API portu. |
| `REPORT_INTERVAL_MINUTES` | `60` | Periyodik rapor sıklığı. |
| `REPORT_WINDOW_HOURS` | `24` | Periyodik raporun kapsadığı pencere. |
| `REPORT_DIGEST_EVERY` | `10` | `q.alarms`'tan bu kadar alarm birikince alarm özeti raporu. |
| `REPORT_DIGEST_MINUTES` | `30` | … ya da son özetten bu kadar dakika geçince (birikmiş alarm varsa). |
| `RAG_TOP_K` | `12` | Soru-cevapta bağlama alınan haber sayısı. |
| `RAG_RECENCY_DAYS` | `14` | Soru-cevapta geriye bakılan varsayılan gün sayısı. |

---

## 7. Kuyruklar ve routing key'ler

Exchange: `news.topic` (topic, kalıcı). Ölü mektup exchange'i: `news.dlx` (fanout) → `q.dead_letter`.

| Kuyruk | Binding (routing key) | Yazan | Okuyan | İçerik |
|--------|-----------------------|-------|--------|--------|
| `q.articles.raw` | `article.raw` | kazıyıcı | anahtar kelime filtresi | Tüm haberler (`NewsRecord`, `stage=raw`) |
| `q.articles.keyword` | `article.keyword` | anahtar kelime filtresi | LLM skorlayıcı | Anahtar kelime eşleşen haberler (`stage=keyword`, `matched_keywords` dolu) |
| `q.articles.scored` | `article.scored` | LLM skorlayıcı **ve** filtre (eşleşmeyenler) | alarm katmanı | Skorlanmış/skorlanmamış **tüm** haberler, aynı nesne (`stage=scored`) |
| `q.alarms` | `alarm.raised` | alarm katmanı | raporlama katmanı | `AlarmEvent` (içinde tam `record`) |
| `q.reports` | `report.#` (`report.generated`, `report.alarm_digest`) | raporlama katmanı | dış tüketiciler (e-posta, BI, Slack …) | `Report` |
| `q.dead_letter` | `#` (`news.dlx` üzerinden) | broker | operatör | Bozuk ya da `RABBITMQ_MAX_ATTEMPTS` kez başarısız mesajlar |

Her ana kuyruğun `<kuyruk>.retry` eşi vardır (`x-message-ttl = RABBITMQ_RETRY_DELAY_MS`); TTL dolunca mesaj
aynı routing key ile `news.topic`'e geri düşer. Deneme sayısı `x-attempts`, son hata `x-error`, kaynak kuyruk
`x-origin-queue` başlıklarında taşınır. Kazıyıcı ayrıca `x-source` (kaynak adı) ve `x-change` (`new`/`updated`) ekler.

---

## 8. Birleşik haber nesnesi (`NewsRecord`)

Kuyruklardaki ve `news-articles` indeksindeki JSON aynıdır (ES belgesinde ek olarak `@timestamp` ve
`content_length` bulunur). `id`, kanonik URL'nin SHA-256 özetinin ilk 32 karakteridir; `content_hash`
başlık+alt başlık+içeriğin SHA-1 özetidir (güncellenen haberleri yakalamak için).

### Alarm verilen kayıt (alarm katmanından sonra)

```json
{
  "schema_version": 1,
  "id": "9b7cdb83ef4a882d0e8b1764cda0ea77",
  "source": "hurriyet",
  "content_url": "https://www.hurriyet.com.tr/gundem/bakan-acikladi-yatirim-fonlarina-yeni-denetim-geliyor-42000001",
  "title": "Bakan açıkladı: Yatırım fonlarına yeni denetim düzenlemesi geliyor",
  "subtitle": "Hazine ve Maliye Bakanlığı, fon yönetim şirketleri için ek raporlama yükümlülüğü getiren taslağı görüşe açtı.",
  "published_at": "2026-10-02T19:20:00+03:00",
  "updated_at": "2026-10-02T19:45:00+03:00",
  "content": "Hazine ve Maliye Bakanı, yatırım fonlarının denetimini sıkılaştıran düzenlemenin önümüzdeki hafta Resmi Gazete'de yayımlanacağını açıkladı.\n\nTaslağa göre fon yönetim şirketleri portföy hareketlerini günlük olarak SPK'ya raporlayacak. Bakan, düzenlemenin küçük yatırımcıyı korumayı amaçladığını söyledi.",
  "category": "gundem",
  "author": "Hürriyet",
  "image_url": "https://image.hurimg.com/i/hurriyet/75/0x0/ornek-gorsel.jpg",
  "tags": [
    "ekonomi",
    "yatırım fonu"
  ],
  "language": "tr",
  "scraped_at": "2026-10-02T16:31:05Z",
  "content_hash": "387be1f58743bd543f1fb79f9126dbfd9ba3c5cb",
  "stage": "alarm",
  "matched_keywords": [
    "bakan",
    "fon"
  ],
  "alarm_score": 85,
  "is_alarm": true,
  "alarm_reason": "Bakan düzeyinde, yatırım fonlarının tamamını etkileyen ve Resmi Gazete'de yayımlanacak bir düzenleme; 'bakan' gerçek bir bakanı, 'fon' gerçek yatırım fonlarını ifade ediyor.\n\nLLM Özeti: Hazine ve Maliye Bakanı, fon yönetim şirketlerine günlük raporlama yükümlülüğü getiren denetim düzenlemesini duyurdu. Düzenleme önümüzdeki hafta Resmi Gazete'de yayımlanacak ve küçük yatırımcıyı korumayı amaçlıyor.",
  "llm_summary": "Hazine ve Maliye Bakanı, fon yönetim şirketlerine günlük raporlama yükümlülüğü getiren denetim düzenlemesini duyurdu. Düzenleme önümüzdeki hafta Resmi Gazete'de yayımlanacak ve küçük yatırımcıyı korumayı amaçlıyor.",
  "llm": {
    "model": "qwen2.5:7b",
    "alarm_score": 85,
    "is_alarm": true,
    "reason": "Bakan düzeyinde, yatırım fonlarının tamamını etkileyen ve Resmi Gazete'de yayımlanacak bir düzenleme; 'bakan' gerçek bir bakanı, 'fon' gerçek yatırım fonlarını ifade ediyor.",
    "summary": "Hazine ve Maliye Bakanı, fon yönetim şirketlerine günlük raporlama yükümlülüğü getiren denetim düzenlemesini duyurdu. Düzenleme önümüzdeki hafta Resmi Gazete'de yayımlanacak ve küçük yatırımcıyı korumayı amaçlıyor.",
    "topics": [
      "ekonomi",
      "finans/fon",
      "siyaset"
    ],
    "entities": [
      "Hazine ve Maliye Bakanlığı",
      "SPK",
      "Resmi Gazete"
    ],
    "scored_at": "2026-10-02T16:31:42Z",
    "latency_ms": 8430,
    "attempts": 1,
    "raw": "{\"alarm_score\": 85, \"is_alarm\": true, \"reason\": \"Bakan düzeyinde, yatırım fonlarının tamamını etkileyen ve Resmi Gazete'de yayımlanacak bir düzenleme; 'bakan' gerçek bir bakanı, 'fon' gerçek yatırım fonlarını ifade ediyor.\", \"summary\": \"Hazine ve Maliye Bakanı, fon yönetim şirketlerine günlük raporlama yükümlülüğü getiren denetim düzenlemesini duyurdu. Düzenleme önümüzdeki hafta Resmi Gazete'de yayımlanacak ve küçük yatırımcıyı korumayı amaçlıyor.\", \"topics\": [\"ekonomi\", \"finans/fon\", \"siyaset\"], \"entities\": [\"Hazine ve Maliye Bakanlığı\", \"SPK\", \"Resmi Gazete\"]}"
  },
  "alarm_id": "39795092e8937e504501",
  "alarmed_at": "2026-10-02T16:31:43Z",
  "processed_at": "2026-10-02T16:31:42Z"
}
```

### Alarm verilmeyen kayıt (LLM skorladı, eşiğin altında kaldı)

```json
{
  "schema_version": 1,
  "id": "7894d0e56ffc71e53f8941e381c62ac8",
  "source": "12punto",
  "content_url": "https://12punto.com.tr/gundem/bakan-il-ziyaretinde-ciftcilerle-bir-araya-geldi-123456",
  "title": "Bakan il ziyaretinde çiftçilerle bir araya geldi",
  "subtitle": "Tarım ve Orman Bakanı, hasat dönemi öncesi üreticilerin taleplerini dinledi.",
  "published_at": "2026-10-02T18:05:00+03:00",
  "updated_at": null,
  "content": "Tarım ve Orman Bakanı, il ziyareti kapsamında çiftçilerle bir araya geldi. Üreticiler mazot ve gübre maliyetlerini gündeme getirdi. Bakan, destek ödemelerinin takvime uygun süreceğini söyledi.",
  "category": "gundem",
  "author": "12punto",
  "image_url": "",
  "tags": [
    "tarım"
  ],
  "language": "tr",
  "scraped_at": "2026-10-02T16:32:10Z",
  "content_hash": "9f999ae9263e4e63c2277e7bd56a64698cf01fc0",
  "stage": "alarm",
  "matched_keywords": [
    "bakan"
  ],
  "alarm_score": 35,
  "is_alarm": false,
  "alarm_reason": "",
  "llm_summary": "",
  "llm": {
    "model": "qwen2.5:7b",
    "alarm_score": 35,
    "is_alarm": false,
    "reason": "Rutin bir bakan ziyareti; 'bakan' gerçek bir bakanı ifade ediyor ancak geniş etkili bir karar ya da soruşturma yok.",
    "summary": "Tarım ve Orman Bakanı il ziyaretinde çiftçilerle görüştü; üreticiler girdi maliyetlerini dile getirdi, Bakan destek ödemelerinin süreceğini söyledi.",
    "topics": [
      "siyaset",
      "ekonomi"
    ],
    "entities": [
      "Tarım ve Orman Bakanlığı"
    ],
    "scored_at": "2026-10-02T16:32:51Z",
    "latency_ms": 6120,
    "attempts": 1,
    "raw": "{\"alarm_score\": 35, \"is_alarm\": false, \"reason\": \"Rutin bir bakan ziyareti; 'bakan' gerçek bir bakanı ifade ediyor ancak geniş etkili bir karar ya da soruşturma yok.\", \"summary\": \"Tarım ve Orman Bakanı il ziyaretinde çiftçilerle görüştü; üreticiler girdi maliyetlerini dile getirdi, Bakan destek ödemelerinin süreceğini söyledi.\", \"topics\": [\"siyaset\", \"ekonomi\"], \"entities\": [\"Tarım ve Orman Bakanlığı\"]}"
  },
  "alarm_id": "",
  "alarmed_at": null,
  "processed_at": "2026-10-02T16:32:51Z"
}
```

Anahtar kelime hiç eşleşmeyen (LLM'e gitmeyen) haberlerde ise `matched_keywords` boş liste, `alarm_score` 0,
`llm` `null`, `alarm_reason` ve `llm_summary` yine boş dizedir; kayıt aynı şemayla `q.articles.scored` ve
`news-articles`'a yazılır.

`llm` nesnesi modelin **ham** kararıdır (izlenebilirlik için saklanır); alarm kararı ise deterministik politikadır
(aşağıda). Model `is_alarm` için ne derse desin geçerli olan `alarm_score >= ALARM_THRESHOLD` karşılaştırmasıdır.

---

## 9. Alarm politikası

`NewsRecord.apply_verdict(verdict, threshold)`:

- `alarm_score = verdict.alarm_score` (0-100).
- `is_alarm = alarm_score >= ALARM_THRESHOLD` (varsayılan 60).
- Alarmsa: `llm_summary = verdict.summary`; `alarm_reason = verdict.reason + "\n\nLLM Özeti: " + llm_summary`
  (gerekçe boşsa "LLM alarm eşiğini aşan skor üretti.").
- Değilse: `alarm_reason = ""`, `llm_summary = ""`.

LLM rubriği (`pipeline/prompts.py`): **80-100 kritik** (cumhurbaşkanı/bakan düzeyinde geniş etkili kararlar,
fon/finansal suç soruşturmaları, kamu görevlisi tutuklamaları, piyasayı etkileyen politika), **60-79 önemli**,
**30-59 dikkate değer**, **0-29 rutin/ilgisiz** (spor, magazin, PR, anahtar kelimenin yan anlamda geçmesi: "denize
*bakan* oda", "*fon* müziği"). Model yalnızca JSON döndürür:
`{"alarm_score": int, "is_alarm": bool, "reason": str, "summary": str, "topics": [str], "entities": [str]}`.

Alarm olan kayıtlar için alarm katmanı bir `AlarmEvent` üretir (`alarm_id = sha1(id + ":" + content_hash)[:20]`),
kanallara bildirir (`log` her zaman; `webhook`/`telegram` yapılandırılmışsa), `alarm.raised` ile `q.alarms`'a
yayınlar ve `news-alarms`'a yazar. Aynı alarm yeniden teslim edilirse (ağ kesintisi, retry) ikinci kez
bildirilmez/yayınlanmaz.

---

## 10. Elasticsearch indeksleri

| İndeks | Belge | Kimlik | Not |
|--------|-------|--------|-----|
| `news-articles` | `NewsRecord.to_es_document()` — tüm alanlar + `@timestamp` (haber tarihi) + `content_length` | `id` | Alarm olsun olmasın her haber; `is_alarm`, `alarm_score`, `matched_keywords`, `source`, `category` keyword/sayı alanları; `OLLAMA_EMBEDDING_MODEL` ayarlıysa `embedding` (`dense_vector`, cosine) |
| `news-alarms` | `AlarmEvent.to_es_document()` (tam `record` hariç; `content` ve `category` eklenir) + `@timestamp` (`raised_at`) | `alarm_id` | Alarm olayları, bildirilen kanallar (`channels_notified`), `acknowledged` |
| `news-reports` | `Report.to_es_document()` + `@timestamp` (`generated_at`) | `report_id` | `kind`, pencere, `stats`, `narrative`, `top_alarms` |

Metin alanları (`title`, `subtitle`, `content`, `alarm_reason`, `llm_summary`, `narrative`) özel `tr_text`
analizörüyle indekslenir: `standard` tokenizer + `apostrophe` + Türkçe `lowercase` + Türkçe `stop` + Türkçe
`stemmer`. Böylece "Kılıçdaroğlu'nun" sorgusu "Kılıçdaroğlu" ile eşleşir. Yazma işlemleri `id` ile upsert'tür;
aynı haber ikinci kez işlenirse belge üzerine yazılır (yinelenme olmaz).

---

## 11. Raporlama katmanı (özet)

Raporlama katmanı `q.alarms` akışını ve Elasticsearch'teki tüm haberleri girdi alır; üç tür rapor üretir:

| Tür (`kind`) | Tetikleyici | Pencere |
|--------------|-------------|---------|
| `alarm_digest` | `REPORT_DIGEST_EVERY` alarm birikince **veya** son özetten `REPORT_DIGEST_MINUTES` geçince | Birikmiş alarmların aralığı |
| `periodic` | Her `REPORT_INTERVAL_MINUTES` | Son `REPORT_WINDOW_HOURS` saat |
| `adhoc` | `POST /reports/generate` | İstekte verilen aralık |

Her rapor: ES toplulaştırmaları (`total`, `alarms`, `by_source`, `by_keyword`, `by_category`,
`avg_alarm_score`, `by_hour`, `top_alarms`) + en yüksek skorlu alarmlar + LLM'in Türkçe anlatısı (Ollama
erişilemezse şablon tabanlı anlatı). Raporlar `news-reports`'a yazılır ve `report.generated` /
`report.alarm_digest` ile `q.reports`'a yayınlanır; `GET /reports` ve pano üzerinden okunur. Ayrıntılı mimari
(bileşenler, veri saklama, Kibana entegrasyonu): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#5-raporlama-katmanı-mimarisi).

---

## 12. Soru sorma: "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"

Komut satırından:

```bash
scraperhryt ask "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"
scraperhryt ask "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?" --since-days 7 --json
make ask Q="Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"
# altyapısız: kazı + boru hattı + soru aynı süreçte
scraperhryt run-all --once --in-memory --fake-llm --ask "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"
```

HTTP API'den (`since_days` ve `top_k` isteğe bağlıdır; varsayılanlar `RAG_RECENCY_DAYS` ve `RAG_TOP_K`):

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?", "since_days": 14, "top_k": 12}'
```

Yanıt (`Answer` modeli) — `answer` metni **en yeni gelişmeden başlayarak** anlatır ve her iddiayı `[n]` ile
`sources` listesindeki habere bağlar; yeterli haber yoksa bunu açıkça söyler:

```json
{
  "question": "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?",
  "answer": "Son durum: 2 Ekim itibarıyla Özgür Özel, Kemal Kılıçdaroğlu'nun açıklamalarına yanıt vererek … [1]. Bir gün önce Kılıçdaroğlu … demişti [2]. Daha önce … [3].",
  "sources": [
    {
      "id": "3f9c0c2a7c1e4b0e9a1d2c3b4a5f6e7d",
      "title": "Özel'den Kılıçdaroğlu'na yanıt",
      "content_url": "https://www.hurriyet.com.tr/gundem/ozelden-kilicdarogluna-yanit-42000123",
      "source": "hurriyet",
      "published_at": "2026-10-02T18:40:00+03:00",
      "score": 0.0325,
      "snippet": "CHP Genel Başkanı Özgür Özel, …"
    }
  ],
  "model": "qwen2.5:7b",
  "retrieved_count": 12,
  "generated_at": "2026-10-02T19:05:12.481212+00:00",
  "search_terms": ["Özgür Özel", "Kemal Kılıçdaroğlu", "CHP", "tartışma"]
}
```

Aynı soru panodaki (`GET /`) "Sor" kutusundan da sorulabilir. RAG tasarımı (sorgu yeniden yazma, hibrit arama,
bağlam kurma, sınırlamalar) için `docs/ARCHITECTURE.md` §6'ya bakın.

---

## 13. Testler

```bash
# Çevrimdışı testler (fixtures + bellek içi broker/depo + FakeOllama); ağ gerektirmez
python -m pytest -m "not live" -q

# Canlı testler: gerçek sitelere istek atar (RSS, liste ve birer haber sayfası); kibar bekleme uygulanır
python -m pytest -m live tests/test_live_scrapers.py -q

# Lint
ruff check src tests
```

Test dosyaları: `tests/test_scrapers.py` (fixture'larla ayrıştırma), `tests/test_pipeline.py` (filtre, LLM
istemcisi, skorlama), `tests/test_store.py`, `tests/test_alarm.py`, `tests/test_reporting.py` (rapor, RAG, API),
`tests/test_live_scrapers.py` (`@pytest.mark.live`).

---

## 14. Sorun giderme

| Belirti | Neden / ne olur | Ne yapmalı |
|---------|-----------------|------------|
| Skorlayıcı logunda `LLM erişilemiyor, mesaj yeniden denenecek` | Ollama kapalı, zaman aşımı veya 5xx (`LLMUnavailable`). Mesaj `q.articles.keyword.retry` kuyruğuna gider, `RABBITMQ_RETRY_DELAY_MS` sonra geri gelir; `RABBITMQ_MAX_ATTEMPTS` (5) denemeden sonra `q.dead_letter`'a düşer. | `ollama serve` / `scraperhryt check`; Ollama ayağa kalkınca retry kuyruğundakiler kendiliğinden işlenir; ölü mektuba düşenleri RUNBOOK'taki gibi yeniden oynatın. |
| `Model 'qwen2.5:7b' Ollama'da yüklü görünmüyor` | Model indirilmemiş; Ollama 404 döndürür → `LLMUnavailable` gibi davranır. | `ollama pull qwen2.5:7b`. |
| `LLM çıktısı çözümlenemedi (deneme 1/3)` | Model JSON yerine metin üretti (`LLMBadOutput`). Süreç içinde katı JSON uyarısıyla 2 kez daha denenir; yine olmazsa broker retry'ına düşer. | Sıcaklığı düşük tutun (`OLLAMA_TEMPERATURE=0.1`); daha büyük/uyumlu bir model deneyin. |
| Alarm katmanı `Elasticsearch indeksleri hazırlanamadı … tekrar denenecek` | ES kapalı. Başlangıçta üstel geri çekilmeyle bekler; çalışırken yazma hataları `Retry` üretir → retry kuyruğu → ölü mektup. | `curl localhost:9200/_cluster/health`; ES gelince kuyruklar boşalır. API arama uçları ES yokken 5xx döner. |
| `RabbitMQ bağlantısı kurulamadı (deneme N)` | Broker kapalı. Tüm servisler üstel geri çekilmeyle yeniden bağlanır; kazıyıcı turu başarısız olursa bir sonraki aralıkta tekrarlanır. | `docker compose up -d rabbitmq`; yönetim arayüzünde kuyrukların göründüğünü doğrulayın. |
| `q.dead_letter` doluyor | Bozuk mesaj (`Reject`, anında) ya da tükenen denemeler. Mesaj orijinal routing key'ini ve `x-death` başlığını korur. | Yönetim arayüzü → Queues → `q.dead_letter` → *Get messages* (Ack mode: *Nack message requeue true*) ile inceleyin; kök nedeni giderip RUNBOOK'taki yeniden oynatma adımlarını uygulayın. |
| Haberler ikinci turda gelmiyor | Normal: `SeenStore` (`STATE_DB_PATH`) görülen haberleri 6 saat boyunca yeniden çekmez; içerik özeti değişmeyen haber yeniden yayınlanmaz. | Her şeyi yeniden işlemek için servisleri durdurup `data/state.sqlite3` dosyasını silin. |
| Anahtar kelime yanlış pozitif ("denize bakan oda") | Filtre kök eşleşmesi yapar; bağlam kararı LLM'e bırakılmıştır (düşük skor → alarm yok, ama kayıt yine ES'te). | Gerekirse `=bakan` (tam kelime) ya da `re:` desenleriyle `KEYWORDS`'ü daraltın. |
| HTTP 429 / 5xx uyarıları kazıyıcıda | Site yavaş ya da oran sınırı. `HttpClient` 3 denemeye kadar üstel bekler; haber başına hata sayılır, tur devam eder. | `REQUEST_DELAY_SECONDS`'ı artırın, `MAX_ARTICLES_PER_RUN`'ı düşürün. |
| `elasticsearch` konteyneri hemen çıkıyor (`max virtual memory areas vm.max_map_count [65530] is too low`) | Linux çekirdek sınırı ES için düşük. | `sudo sysctl -w vm.max_map_count=262144` (kalıcı için `/etc/sysctl.conf`), sonra `docker compose up -d elasticsearch`. |
| Yerel (venv) çalıştırmada `rabbitmq`/`elasticsearch` adları çözümlenemiyor | `.env.example` varsayılanları compose servis adlarını kullanır. | `.env`'deki "yerel çalıştırma" satırlarını (localhost) açın. |

---

## 15. Yasal / etik not

- Yalnızca herkese açık sayfalar ve sitelerin kendi yayınladığı RSS beslemeleri kullanılır; oturum açma, API ya da
  yasaklı yollar kullanılmaz. **Hürriyet `robots.txt`'de yasaklanan `/api/` ve `/arama/` yollarına hiç istek atılmaz.**
- İstekler kibar hızda atılır: aynı siteye ardışık istekler arasında en az `REQUEST_DELAY_SECONDS` (0.5 s) beklenir,
  tur başına `MAX_ARTICLES_PER_RUN` bütçesi vardır, 429/5xx yanıtlarında üstel geri çekilme uygulanır ve `User-Agent`
  projeyi tanıtır. Bu değerleri sitelere yük bindirecek şekilde düşürmeyin.
- Toplanan içerik yalnızca izleme/alarm/raporlama amacıyla saklanır; haber metinleri ilgili yayıncıların telif hakkı
  altındadır. Yeniden yayınlamayın; kullanım koşullarına uyun ve gerekiyorsa yayıncıdan izin alın.
- Kişisel verilere (haberde adı geçen kişiler) ilişkin yükümlülükler (KVKK) kullanıcıya aittir; saklama süresini
  sınırlamak için `docs/ARCHITECTURE.md`'deki veri saklama önerilerine bakın.

## Doğruluk geliştirmeleri

| Mekanizma | Ayar / Komut | Ne yapar |
|---|---|---|
| Eş anlamlı anahtar kelimeler | `KEYWORD_ALIASES_PATH=config/keyword_aliases.json` | "SPK", "Erdoğan", "Hazine ve Maliye Bakanlığı" gibi ifadeler kanonik anahtar kelimeye (`fon`, `cumhurbaşkanı`, `bakan`) katlanır |
| Ön sınıflandırıcı | `PRECLASSIFIER_ENABLED=true`, `PRECLASSIFIER_THRESHOLD`, `config/preclassifier_prototypes.json` | Kural katmanı "bakan" fiil kullanımını ("pencereden bakan adam") eler; embedding katmanı (OLLAMA_EMBEDDING_MODEL gerekir) metni ilgili/ilgisiz örnek merkezleriyle karşılaştırır, `relevance` eşiğin altındaysa LLM'e gitmeden skor 0 ile depolanır (`prefilter_reason`) |
| Kaynak/kategori bazlı eşik | `ALARM_THRESHOLDS_JSON={"source:12punto":70,"category:spor":90,"keyword:fon":50}` | Öncelik: kategori > anahtar kelime > kaynak > `ALARM_THRESHOLD`; uygulanan eşik `alarm_threshold_used` alanında |
| Öz-tutarlılık (çoklu örnekleme) | `LLM_SAMPLES=3`, `LLM_SAMPLE_TEMPERATURE`, `LLM_DISAGREEMENT_THRESHOLD` | Aynı haber N kez puanlanır, medyan alınır; örnekler arası fark eşiği aşarsa `needs_review=true`, `confidence` düşer |
| Olay kümeleme / tekrar bastırma | `ALARM_DEDUP_*`, `ALARM_NOTIFY_DUPLICATES` | Pencere içindeki benzer alarm (embedding kosinüsü veya başlık Jaccard) aynı `event_id` altında toplanır; tekrar alarm depolanır ve `q.alarms`'a gider ama bildirim kanallarına gitmez (`x-duplicate` başlığı) |
| Geri bildirim döngüsü | `POST /alarms/{id}/feedback`, `GET /feedback/stats`, `scraperhryt feedback ID --label ...` | Doğru/yanlış pozitif etiketleri `news-feedback` indeksinde; kesinlik tahmini ve kalibrasyon girdisi |
| Kalibrasyon | `scraperhryt calibrate [--from-feedback] [--fake-llm]`, `GOLDEN_SET_PATH=config/golden_set.jsonl` | Altın set üzerinde her eşik için kesinlik/duyarlılık/F1, önerilen `ALARM_THRESHOLD` |
| Yeniden skorlama | `scraperhryt rescore --since-days 7 [--all] [--limit N] [--dry-run]` | Model/prompt/eşik değişince depodaki kayıtlar `q.articles.keyword`'e geri yazılır, olağan yoldan yeniden değerlendirilir |
| Zaman çizelgesi | `scraperhryt ask`, `POST /ask` → `timeline` | RAG cevabında kaynaklar kronolojik (eski → yeni) listelenir |
| Ölü mektup geri oynatma | `scraperhryt replay-dead-letters [--dry-run] [--limit N] [--to KUYRUK]` | `q.dead_letter` mesajları köken kuyruğuna (`x-origin-queue` / `x-death`) geri yazılır |

Önerilen üretim profili: `OLLAMA_MODEL=qwen2.5:7b` (veya daha büyük), `LLM_SAMPLES=3`, `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`, `ALARM_DEDUP_ENABLED=true`, düzenli `scraperhryt calibrate --from-feedback` ile eşik gözden geçirme.

## Soru-cevap: tam metin özet ve sayı doğrulama

- **Haberin tamamı okunur:** Soruya göre her haberin tüm metni taranır. Modele yalnızca başlık ya da ilk paragraf değil, soruyla en ilgili cümleler verilir (ilk iki cümle bağlam olarak her zaman eklenir).
- **Model cevap veremezse:** Yedek özet de haberin kendi metninden üretilir. Konunun kendisi olan haberlerde alt başlık ve ilk cümleler öne çıkar. Adın yalnızca metnin içinde geçtiği haberlerde ise o adı içeren cümleler seçilir. Her habere önce bir cümle düşer, böylece farklı gelişmeler kapsanır. Skorlama aşamasındaki LLM özetleri kullanılmaz.
- **İlgililik:** Sorudaki özel adlar ve kısaltmalar (TFF, MHK) zorunludur. Unvanlı adlar ("Bakan Fidan") ada indirgenir. Cümle başındaki ad ("Erdoğan ne dedi?") haberlerdeki kullanımına bakılarak özel ad olarak tanınır. "fiyat", "zam", "kriz" gibi genel sözcükler tek başına bir haberi ilgili saymaz. Hafif bir Türkçe kök bulucu çekimleri eşler ("tutuklamalarında" ile "tutuklandı").
- **Sayı doğrulama:** Cevaptaki ve alarm özetlerindeki her sayı, büyüklüğüyle birlikte (bin, milyon, milyar) kaynak haberde aranır. Kaynakta karşılığı olmayan sayıyı içeren cümle atılır; örneğin haberde "20 milyar" yazarken model "20 milyon" yazarsa. Yazıyla yazılmış sayılar ("beşinci dalga") tanınır. Tarih ve saatler bu denetimin dışındadır.

## Arayüz

Tek sayfa: `http://localhost:8000/` (Docker'da `API_PORT` ile değiştirilebilir; eski `/ara` adresi buraya yönlenir).

- **Durum şeridi:** son tarama, sonraki taramaya kalan süre, çekilemeyen (bekleyen) haberler ve uyarılar. Tarama sürerken canlı ilerleme çubuğu kaynağı, aşamayı ve yüzdeyi gösterir; turlar arasında çubuk sonraki taramaya kalan süreyi doldurur.
- **Sekmeler:** Ara (filtreli tam metin arama), Soru sor (kısa özet; kaynaklar ve zaman çizelgesi açılır bölümde), Alarmlar (Doğru / Yanlış / Belirsiz geri bildirimi), Raporlar.
- Haberler ve alarmlar kare kartlarda gösterilir: geniş ekranda satırda üç, tablette iki, telefonda bir kart. Açık ve koyu tema desteklenir; ikon ve emoji kullanılmaz.
- Sayısal özetler arayüzde değil API'de: `GET /stats?hours=24` (kaynak, anahtar kelime, kategori ve saatlik dağılım).
