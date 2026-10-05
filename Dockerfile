# syntax=docker/dockerfile:1
# scraperhryt uygulama imajı: tüm alt komutlar (scrape, filter, score, alarm, report, api, run-all) aynı imajdan çalışır.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Istanbul

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# 1) Bağımlılıkları ayrı katmanda kur (kaynak ya da README değişince yeniden indirilmesin): yalnızca
#    pyproject.toml kopyalanır; paket meta verisinin istediği README.md bu katmanda boş bir yer tutucudur.
COPY pyproject.toml ./
RUN mkdir -p src/scraperhryt \
    && touch src/scraperhryt/__init__.py README.md \
    && pip install . \
    && rm -rf src build README.md

# 2) Gerçek kaynak kodu, README ve betikler.
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
# Eş anlamlılar, ön sınıflandırıcı örnekleri, altın set ve RAG değerlendirme seti (KEYWORD_ALIASES_PATH vb.).
COPY config ./config
RUN pip install --no-deps . \
    && chmod +x scripts/*.sh \
    && mkdir -p /app/data \
    && chown -R app:app /app

USER app
VOLUME ["/app/data"]
EXPOSE 8000

CMD ["scraperhryt", "--help"]
