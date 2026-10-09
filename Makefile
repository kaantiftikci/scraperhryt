# scraperhryt — geliştirme ve işletme kısayolları
#
#   make install            venv içine paketi (dev bağımlılıklarıyla) kur
#   make test | make lint   çevrimdışı testler / ruff
#   make up | make down     docker compose yığınını başlat / durdur
#   make up-ollama          Ollama'yı da konteynerde çalıştır (önce modeli indirir, sonra yığını başlatır)
#   make ask Q="soru"       RAG soru-cevap

PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
COMPOSE ?= docker compose
Q       ?= Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?

.PHONY: help install test test-live lint check up up-ollama down logs setup scrape-once run-all-inmemory ask ps

help:
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/^([a-zA-Z_-]+):.*## /  \1\t/'

install: ## paketi düzenlenebilir modda dev bağımlılıklarıyla kur
	$(PIP) install -e ".[dev]"

test: ## çevrimdışı testler (canlı site testleri hariç)
	$(PYTHON) -m pytest -q -m "not live"

test-live: ## gerçek sitelere giden canlı kazıyıcı testleri
	$(PYTHON) -m pytest -q -m live

lint: ## ruff
	$(PYTHON) -m ruff check src tests

check: ## RabbitMQ / Elasticsearch / Ollama erişilebilirlik raporu
	scraperhryt check

up: ## docker compose yığınını (yeniden derleyerek) arka planda başlat
	$(COMPOSE) up -d --build

up-ollama: ## Ollama konteynerde: önce modeli indir (ollama-pull bitene dek bekler), sonra yığını başlat
	$(COMPOSE) --profile ollama run --rm ollama-pull
	$(COMPOSE) --profile ollama up -d --build

down: ## docker compose yığınını durdur
	$(COMPOSE) down

logs: ## tüm servislerin günlüklerini izle
	$(COMPOSE) logs -f --tail=200

setup: ## compose içinde tek seferlik kurulum (topoloji + indeksler + Ollama denetimi)
	$(COMPOSE) run --rm setup

ps: ## compose servis durumları
	$(COMPOSE) ps

scrape-once: ## yerelde tek kazıma turu (RabbitMQ gerekir)
	scraperhryt scrape --once

run-all-inmemory: ## tüm boru hattını bellek içi broker/depo ve sezgisel LLM ile tek tur çalıştır
	scraperhryt run-all --once --in-memory --fake-llm

ask: ## RAG soru-cevap: make ask Q="..."
	scraperhryt ask "$(Q)"
