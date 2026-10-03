# scraperhryt — İşletme El Kitabı (Runbook)

Kısa ve pratik: başlatma/durdurma, sağlık kontrolleri, ölü mektupları yeniden oynatma, anahtar kelime ve model
değişikliği, yeniden indeksleme. Mimari için [`ARCHITECTURE.md`](ARCHITECTURE.md), ayar listesi için
[`../README.md`](../README.md#6-ortam-değişkenleri).

---

## 1. Başlatma / durdurma

### Docker Compose

```bash
docker compose up -d --build                       # rabbitmq + elasticsearch → setup (tek seferlik) → scraper, filter,
                                                   # scorer, alarm, reporter, api (healthcheck'ler ve setup beklenir)
make up                                            # aynı şey (Makefile kısayolu); make down / make logs / make ps
docker compose --profile kibana up -d              # isteğe bağlı Kibana
docker compose ps                                  # durum
docker compose logs -f --tail=100 scorer alarm     # canlı günlük

docker compose stop scorer                         # tek servisi durdur (kuyruk birikir, kayıp olmaz)
docker compose restart filter                      # yeniden başlat (ör. KEYWORDS değişti)
docker compose down                                # tümünü durdur (volume'lar kalır)
docker compose down -v                             # DİKKAT: RabbitMQ ve ES verisi de silinir
```

Servisler SIGTERM'de mevcut mesajı bitirip onaylar; `docker compose stop` varsayılan 10 s bekler. LLM isteği
`OLLAMA_TIMEOUT` (180 s) sürebildiği için skorlayıcıyı durdururken `docker compose stop -t 200 scorer` kullanın;
aksi halde mesaj onaylanmaz ve yeniden teslim edilir (idempotent olduğu için güvenlidir, yalnızca iş tekrarlanır).

### Docker olmadan

```bash
scraperhryt check && scraperhryt setup
scraperhryt run-all                   # tek süreç; Ctrl+C ile düzgün kapanır
# ya da ayrı terminallerde:
scraperhryt scrape | scraperhryt filter | scraperhryt score | scraperhryt alarm | scraperhryt report | scraperhryt api
```

Başlatma sırası büyük ölçüde önemsizdir: her servis topolojiyi ve indeksleri idempotent olarak oluşturur ve
RabbitMQ yoksa tüm servisler üstel geri çekilmeyle (1 s → 30 s, 8 deneme) bağlanmayı bekler. Elasticsearch için
ise yalnızca bağımsız `alarm` servisi başlangıçta üstel geri çekilmeyle (1 s → 30 s) bekler; `report`, `api`,
`ask` ve `run-all` komutları başlarken Elasticsearch'e erişemezse (indeks
hazırlığı `Unavailable` üretir) `Elasticsearch deposu hazırlanamadı` / `Depo hazırlanamadı` loglayıp **beklemeden
çıkış kodu 1 ile biter**. Bu yüzden Elasticsearch'ü (ve `scraperhryt setup`'ı) bu komutlardan önce başlatın;
compose'da `depends_on` + `restart: unless-stopped` bunu sağlar (ES kısa süreliğine erişilemezse konteyner
ES gelene dek yeniden başlatılır), compose dışında `scraperhryt check` ile ES'in hazır olduğunu doğrulayın.

---

## 2. Sağlık kontrolleri

```bash
scraperhryt check                                   # RabbitMQ + Elasticsearch + Ollama + model yüklü mü

curl -s http://localhost:8000/health                # API ve depo/LLM özeti
curl -s http://localhost:9200/_cluster/health?pretty # ES: status green/yellow (tek düğümde yellow normaldir)
curl -s http://localhost:11434/api/tags             # Ollama ve yüklü modeller
curl -s -u guest:guest http://localhost:15672/api/queues | python -m json.tool | grep -E '"name"|"messages"|"consumers"'

docker compose exec rabbitmq rabbitmqctl list_queues name messages consumers   # kuyruk derinliği ve tüketici sayısı
```

Beklenen tablo (sağlıklı sistem):

| Gözlem | Normal | Sorun işareti |
|--------|--------|---------------|
| `q.articles.raw` / `q.articles.keyword` / `q.articles.scored` / `q.alarms` | `consumers >= 1`, derinlik kısa sürede 0'a iner | `consumers = 0` (servis düşmüş) ya da derinlik sürekli artıyor (tüketici yavaş/takılı) |
| `*.retry` kuyrukları | genellikle 0 | sürekli dolu → bağımlılık (Ollama/ES) erişilemiyor; `docker compose logs scorer alarm` |
| `q.dead_letter` | 0 | > 0 → §3 |
| `q.reports` | tüketici yoksa birikir (normal) | — ; dış tüketici bağlayın ya da TTL/limit koyun |
| Kazıyıcı günlüğü | her turda `Kazıma turu … yayınlanan=N` | arka arkaya `yayınlanan=0` ve `hata>0` → site/ağ sorunu |
| `data/alarms.jsonl` | alarm başına bir satır | — |

---

## 3. Ölü mektupları (q.dead_letter) inceleme ve yeniden oynatma

**Neden düşer?** (a) Bozuk/şemaya uymayan mesaj → `Reject` (yeniden denenmez); (b) `RABBITMQ_MAX_ATTEMPTS` (5)
deneme boyunca `Retry`/istisna (çoğunlukla Ollama ya da ES erişilemiyor). Mesaj **orijinal routing key**'ini
(`article.keyword`, `article.scored` …) ve RabbitMQ'nun `x-death` başlığını korur; retry'dan gelenlerde ayrıca
`x-attempts`, `x-error`, `x-origin-queue` bulunur.

**İnceleme (yönetim arayüzü):** <http://localhost:15672> → *Queues* → `q.dead_letter` → *Get messages* →
*Ack mode: Nack message requeue true* (mesajı tüketmeden bakar) → *Payload* ve *Properties/headers* alanları.
`x-error` son hatayı, `x-death[0].queue` kaynak kuyruğu, `x-death[0].routing-keys` routing key'i gösterir.

**Kök nedeni giderin** (Ollama'yı başlatın, modeli indirin, ES'i ayağa kaldırın) — aksi halde mesajlar 5 denemeden
sonra yine ölü mektuba düşer.

**Yeniden oynatma — yol A (Shovel, arayüzden):**

```bash
docker compose exec rabbitmq rabbitmq-plugins enable rabbitmq_shovel rabbitmq_shovel_management
```

Ardından *Queues* → `q.dead_letter` → *Move messages* → *Destination queue*: hedef ana kuyruk (ör.
`q.articles.keyword`). Not: Shovel başlıkları olduğu gibi taşır; `x-attempts` yüksek kaldığı için mesajın **tek**
denemesi olur. Taze deneme bütçesi için yol B'yi kullanın.

**Yeniden oynatma — yol B (başlıkları temizleyerek, orijinal routing key ile):**

```bash
python - <<'PY'
import pika
from scraperhryt.config import get_settings

s = get_settings()
conn = pika.BlockingConnection(pika.URLParameters(s.rabbitmq_url))
ch = conn.channel()
ch.confirm_delivery()
strip = {"x-attempts", "x-error", "x-origin-queue", "x-death", "x-first-death-exchange",
         "x-first-death-queue", "x-first-death-reason", "x-last-death-exchange", "x-last-death-queue",
         "x-last-death-reason"}
moved = 0
while True:
    method, props, body = ch.basic_get("q.dead_letter", auto_ack=False)
    if method is None:
        break
    headers = {k: v for k, v in (props.headers or {}).items() if k not in strip}
    routing_key = method.routing_key or ((props.headers or {}).get("x-death") or [{}])[0].get("routing-keys", [""])[0]
    if not routing_key:
        ch.basic_nack(method.delivery_tag, requeue=True)   # routing key yoksa dokunma, elle incele
        break
    ch.basic_publish(
        exchange=s.rabbitmq_exchange,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2, headers=headers),
    )
    ch.basic_ack(method.delivery_tag)
    moved += 1
print(f"{moved} mesaj yeniden yayınlandı")
conn.close()
PY
```

`Reject` ile düşen (bozuk) mesajları yeniden oynatmak anlamsızdır; gövdeyi inceleyip nedenini (şema değişikliği,
kesik JSON) giderdikten sonra *Get messages → Ack mode: Ack message requeue false* ile ya da `rabbitmqctl purge_queue
q.dead_letter` ile temizleyin.

---

## 4. Anahtar kelimeleri değiştirme (KEYWORDS)

1. `.env` içinde `KEYWORDS` değerini güncelleyin. Sözdizimi: virgülle ayrılmış liste; varsayılan mod Türkçe ek
   toleranslı kök eşleşmesi (`bakan` → bakanı, bakanlık, bakanlar; `cumhurbaşkanı` → cumhurbaşkanlığı);
   `=kelime` tam kelime, `~parça` alt dize, `re:desen` regex. Aksansız yazım da eşleşir (`bakan` ↔ `bakan`,
   `cumhurbaşkanı` ↔ `cumhurbaskani`).

   ```bash
   KEYWORDS=bakan,cumhurbaşkanı,fon,=TMSF,re:kayy[ıi]m
   ```

2. Yalnızca filtreyi yeniden başlatın: `docker compose restart filter` (ayarlar başlangıçta okunur). Skorlayıcı
   istemi de anahtar kelimeleri bağlam olarak kullandığı için gerekirse `scorer`'ı da yeniden başlatın.
3. Etkisi **yeni gelen** haberlerde görülür; ES'teki geçmiş kayıtların `matched_keywords` değeri değişmez.
   Geçmişi yeni listeyle yeniden sınıflandırmak için: servisleri durdurun, `data/state.sqlite3` dosyasını silin
   (`SeenStore`), kazıyıcıyı başlatın — RSS/listede hâlâ bulunan haberler yeniden yayınlanıp baştan işlenir; daha
   eski haberler için 12punto'da `scraperhryt scrape --once --backfill-days 7` kullanılabilir (Hürriyet arşivi
   robots.txt gereği taranmaz).

Eşiği değiştirmek için `ALARM_THRESHOLD` (0-100) → `scorer` ve `alarm` servislerini yeniden başlatın; eşik hem
istemde modele bildirilir hem de `apply_verdict` politikasında uygulanır.

---

## 5. Modeli değiştirme

```bash
ollama pull qwen2.5:14b                 # (ya da llama3.1:8b, gemma2:9b … JSON üretebilen bir sohbet modeli)
# .env:  OLLAMA_MODEL=qwen2.5:14b
docker compose restart scorer reporter api
scraperhryt check                       # modelin Ollama'da göründüğünü doğrular
```

- Model adını Ollama'daki etiketiyle yazın (`qwen2.5:14b`); etiketsiz ad `:latest` sayılır.
- Daha büyük model → `OLLAMA_TIMEOUT`'u artırın; 7B altı modellerde JSON uyumu düşer (günlükte `LLM çıktısı
  çözümlenemedi` artarsa `OLLAMA_TEMPERATURE=0` deneyin).
- Her kaydın `llm.model` alanı hangi modelle skorlandığını saklar; model değişikliği sonrası ES'te `llm.model` ile
  karşılaştırma yapılabilir.
- **Embedding modeli** değişiyorsa (`OLLAMA_EMBEDDING_MODEL`, `EMBEDDING_DIMS`) `news-articles` indeksindeki
  `dense_vector` boyutu sabit olduğu için **yeniden indeksleme** gerekir (§6). Embedding'i ilk kez açarken de aynı
  durum geçerlidir (`ensure_indices` var olan indeksin eşlemesini değiştirmez).

---

## 6. Yeniden indeksleme

Ne zaman: eşleme değişikliği (yeni alan/analizör), embedding açma/kapama ya da boyut değişikliği, bozuk indeks.

```bash
export ES=http://localhost:9200

# 1) Yazan servisleri durdurun
docker compose stop alarm reporter

# 2) Yeni indeksi yeni adla oluşturun: .env → ES_INDEX_ARTICLES=news-articles-v2, sonra
scraperhryt setup                                       # yeni eşlemeyle news-articles-v2 oluşur (yalnızca yoksa)

# 3) Veriyi kopyalayın (embedding alanı eski indekste yoksa/boyutu farklıysa hariç tutun)
curl -s -X POST "$ES/_reindex?wait_for_completion=true" -H 'Content-Type: application/json' -d '{
  "source": {"index": "news-articles", "_source": {"excludes": ["embedding"]}},
  "dest":   {"index": "news-articles-v2"}
}'

# 4) Servisleri yeni ada göre başlatın; eski indeksi doğruladıktan sonra silin
docker compose up -d alarm reporter api
curl -s "$ES/_cat/indices/news-*?v"
curl -s -X DELETE "$ES/news-articles"
```

- Vektörler yeniden üretilmez; embedding açıldıktan sonra kazınan haberler vektörlü olur. Geçmişi vektörlemek
  için haberlerin boru hattından yeniden geçmesi gerekir: `data/state.sqlite3`'ü silip kazıyıcıyı çalıştırın
  (RSS/listede olanlar) ve 12punto için `--backfill-days` kullanın.
- `news-alarms` ve `news-reports` için aynı adımlar `ES_INDEX_ALARMS` / `ES_INDEX_REPORTS` ile uygulanır.
- Alias kullanmak isterseniz (`ES_INDEX_ARTICLES=news-articles-write`) alias'ı ve ilk indeksi `setup`'tan önce
  elle oluşturun; depo adı doğrudan kullanır.
- Yalnızca "temiz başlangıç" isteniyorsa: servisleri durdurun, `curl -X DELETE "$ES/news-articles,news-alarms,news-reports"`,
  `scraperhryt setup`, `data/state.sqlite3`'ü silin, servisleri başlatın.
