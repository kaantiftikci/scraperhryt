# Haber Radarı kurulumu

Sistem üç parçadan oluşur:

- **Docker Compose:** RabbitMQ, Elasticsearch, scraper, filter, scorer, alarm, reporter ve api servisleri.
- **Aynı bilgisayarda, Docker dışında:** Ollama ile bge-m3 (embedding) ve llama.cpp ile bge-reranker-v2-m3 (reranker).
- **Uzak sunucuda:** LLM (phi-4).

Kurulum macOS, Windows ve Linux'ta aynıdır; farklı olan yerler belirtilmiştir.

## Gerekenler

- Docker Desktop (macOS, Windows) ya da Docker Engine ve Docker Compose (Linux)
- Git
- [Ollama](https://ollama.com/download)
- llama.cpp
- Uzak LLM sunucusunun adresi (ekipten ayrıca alınır, depoya yazılmaz)

## 1. Projeyi indir

```bash
git clone https://github.com/kaantiftikci/scraperhryt.git
cd scraperhryt
git checkout claude/news-pipeline
cp .env.example .env
```

Sonraki tüm `docker compose` komutları bu `scraperhryt` klasöründe çalıştırılır.

## 2. Embedding modelini kur (bge-m3)

```bash
ollama pull bge-m3
```

Ollama arka planda çalışır. **Linux'ta** Docker'dan erişilebilmesi için Ollama'yı şöyle başlatın:

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

## 3. Reranker'ı başlat (bge-reranker-v2-m3)

llama.cpp kurulumu:

| Sistem | Komut |
|---|---|
| macOS | `brew install llama.cpp` |
| Windows | `winget install llama.cpp` |
| Linux | [llama.cpp sürümlerinden](https://github.com/ggml-org/llama.cpp/releases) indirin |

Reranker'ı ayrı bir terminalde başlatın ve terminali açık bırakın:

```bash
llama-server --hf-repo gpustack/bge-reranker-v2-m3-GGUF --hf-file bge-reranker-v2-m3-Q8_0.gguf --reranking --port 8012 -c 8192 -b 2048 -ub 2048
```

İlk çalıştırmada model (~600 MB) indirilir. **Linux'ta** komutun sonuna `--host 0.0.0.0` ekleyin.

## 4. `.env` dosyasını düzenle

`.env` içinde aşağıdaki satırları bulup değiştirin; diğer ayarlar varsayılan değerleriyle kalabilir.

```
OLLAMA_BASE_URL=https://<llm-sunucusu>/llm/
OLLAMA_MODEL=jacob-ebey/phi4-tools:latest
OLLAMA_VERIFY_TLS=false
OLLAMA_NUM_CTX=10000
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_EMBEDDING_BASE_URL=http://host.docker.internal:11434
EMBEDDING_DIMS=1024
RERANKER_URL=http://host.docker.internal:8012
RERANKER_API=llamacpp
API_PORT=8080
```

- `<llm-sunucusu>` yerine size verilen adresi yazın.
- `OLLAMA_VERIFY_TLS=false` yalnızca LLM sunucusu kendi imzaladığı bir sertifika kullanıyorsa gerekir. Sertifika dosyası varsa bu satır yerine `OLLAMA_CA_BUNDLE=<dosya yolu>` kullanın.
- LLM sunucusu aynı anda iki isteği kaldırmıyorsa `SCORER_REPLICAS=1` yapın.

## 5. Sistemi başlat

```bash
docker compose up -d --build
docker compose logs setup
```

`setup` çıktısındaki tabloda **RabbitMQ topolojisi**, **Elasticsearch indeksleri**, **Ollama** ve **Reranker** satırları `OK` olmalı ve son satırda `Kurulum tamam.` yazmalı. Bir satırda `HATA` varsa aşağıdaki sorun giderme tablosuna bakın.

## 6. Arayüzü aç

Tarayıcıda **http://localhost:8080** adresini açın. İlk tarama hemen başlar; haberler birkaç dakika içinde gelmeye başlar, sonra her 5 dakikada bir yeni tarama yapılır.

## Günlük kullanım

| İş | Komut |
|---|---|
| Servislerin durumu | `docker compose ps` |
| Bir servisin logları | `docker compose logs -f scorer` |
| Güncelleme | `git pull` ve ardından `docker compose up -d --build` |
| Durdurma (veriler kalır) | `docker compose down` |
| Tamamen sıfırlama (tüm haberler silinir) | `docker compose down -v` |

Bilgisayar yeniden başladığında Docker'ı, Ollama'yı ve reranker'ı (3. adımdaki komut) yeniden açmanız yeterlidir; servisler kendiliğinden kalkar.

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop'ı açıp tamamen başlamasını bekleyin. |
| `no configuration file provided` | Komutu `scraperhryt` klasörünün içinde çalıştırın (`cd scraperhryt`). |
| `setup` çıktısında Ollama `HATA` | `.env` içindeki `OLLAMA_BASE_URL` adresini ve LLM sunucusuna ağ erişimini kontrol edin. Embedding hatasıysa Ollama'nın açık olduğundan ve `bge-m3` modelinin indirildiğinden emin olun (Linux'ta `OLLAMA_HOST=0.0.0.0`). |
| `setup` çıktısında Reranker `UYARI` | 3. adımdaki `llama-server` komutunun çalıştığı terminal açık mı kontrol edin. Reranker kapalıyken de soru-cevap çalışır, yalnızca isabeti düşer. |
| Embedding boyutu uyuşmuyor hatası (768 / 1024) | İndeksler eski boyutla kurulmuş; aşağıdaki "Haberleri sıfırlama" adımlarını uygulayın. |
| Eski haberlerde anlamsal arama sonuç vermiyor | Eski haberlerin vektörlerini üretin: `docker compose run --rm api scraperhryt embed-backfill --since-days 0` |
| Sayfa açılmıyor | `docker compose ps` ile `api` servisinin çalıştığını ve `.env` içinde `API_PORT=8080` olduğunu kontrol edin. |

## Haberleri sıfırlama

İndeksler yanlış ayarla kurulduysa ya da baştan başlamak istiyorsanız haberler ve alarmlar silinip tarama yeniden yapılır. Raporlar ve ayarlar korunur.

```bash
docker compose stop scraper
curl -X DELETE localhost:9200/news-articles
curl -X DELETE localhost:9200/news-alarms
docker compose run --rm --no-deps api rm -f /app/data/state.sqlite3
docker compose up -d --force-recreate setup
docker compose up -d
```
