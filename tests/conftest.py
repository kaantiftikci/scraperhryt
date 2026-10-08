"""Test izolasyonu: hiçbir test gerçek RabbitMQ / Elasticsearch / Ollama'ya ulaşmamalı.

Varsayılan ayarlar localhost'u gösterir; geliştirici makinesinde bu servisler çalışıyorsa bir test
yanlışlıkla gerçek kuyruğa mesaj yazabilir. Bu yüzden her test için ortam değişkenlerini erişilemez
uç noktalara çeker ve Settings önbelleğini temizleriz. ``live`` işaretli testler yalnızca haber sitelerine
(HTTP) çıkar; onlar da broker/depo için aynı korumayı kullanır.
"""

from __future__ import annotations

import os

import pytest

_ISOLATED_ENV = {
    "RABBITMQ_URL": "amqp://guest:guest@127.0.0.1:1/%2F",  # port 1: bağlantı anında reddedilir
    "ELASTICSEARCH_URL": "http://127.0.0.1:1",
    "OLLAMA_BASE_URL": "http://127.0.0.1:1",
    "OLLAMA_EMBEDDING_MODEL": "",
    "ALARM_WEBHOOK_URL": "",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "KEYWORD_ALIASES_PATH": "",  # testler eş anlamlı dosyasını açıkça seçer
    "ALARM_THRESHOLDS_JSON": "",
}


@pytest.fixture(autouse=True)
def _isolate_external_services(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for key, value in _ISOLATED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("ALARM_LOG_PATH", str(tmp_path / "alarms.jsonl"))
    from scraperhryt.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    # Toplama aşamasında (fixture'lar devreye girmeden) içe aktarılan modüller de güvenli değerleri görsün.
    for key, value in _ISOLATED_ENV.items():
        os.environ.setdefault(key, value)
