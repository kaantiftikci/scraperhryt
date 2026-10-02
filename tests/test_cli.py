"""CLI ve dağıtım dosyası testleri.

Tümü çevrimdışıdır: alt komutlar sahte modüller/sondalarla çalıştırılır, run-all bellek içi broker + depo ve
sezgisel LLM ile tek tur koşturulur, compose/Makefile/.env.example yapısal olarak doğrulanır.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import sys
import threading
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scraperhryt import cli, logging_setup
from scraperhryt.broker import InMemoryBroker, Message, Queue, RoutingKey
from scraperhryt.models import Answer, Citation, NewsRecord

ROOT = Path(__file__).resolve().parents[1]
SUBCOMMANDS = ("setup", "check", "scrape", "filter", "score", "alarm", "report", "api", "ask", "run-all")
APP_SERVICES = ("setup", "scraper", "filter", "scorer", "alarm", "reporter", "api")

INTEGRATION_MODULES = (
    "scraperhryt.alarm_sinks",
    "scraperhryt.store",
    "scraperhryt.pipeline.alarm",
    "scraperhryt.pipeline.keyword_filter",
    "scraperhryt.pipeline.scorer",
    "scraperhryt.scrapers.runner",
    "scraperhryt.reporting.builder",
    "scraperhryt.reporting.service",
)


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


INTEGRATION_READY = all(_importable(name) for name in INTEGRATION_MODULES)


def patch_module_attr(monkeypatch: pytest.MonkeyPatch, module_name: str, attr: str, value: Any) -> None:
    """Modül varsa niteliğini değiştirir; henüz yazılmamışsa sys.modules'e sahte bir modül koyar."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = types.ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(module, attr, value, raising=False)


class FakeStore:
    """`ask` için yeterli en küçük depo."""

    def __init__(self) -> None:
        self.ensured = False

    def ensure_indices(self) -> None:
        self.ensured = True


# ---------------------------------------------------------------------------------------------------------
# Yardım / argüman ayrıştırma
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_every_subcommand_has_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: scraperhryt {command}")
    assert "-h, --help" in out


def test_top_level_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in SUBCOMMANDS:
        assert command in out
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("scraperhryt ")


def test_missing_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_overrides_settings() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--log-level", "debug", "scrape", "--source", "12punto", "--backfill-days", "3", "--interval", "42"])
    settings = cli.build_settings(args)
    assert settings.log_level == "debug"
    assert settings.source_list == ["12punto"]
    assert settings.backfill_days == 3
    assert settings.scrape_interval_seconds == 42
    api_args = parser.parse_args(["api", "--host", "127.0.0.1", "--port", "9999"])
    api_settings = cli.build_settings(api_args)
    assert (api_settings.api_host, api_settings.api_port) == ("127.0.0.1", 9999)


def test_invalid_env_value_is_reported(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ALARM_THRESHOLD", "yüksek")
    assert cli.main(["check"]) == 2
    assert "Ayar hatası" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------
# Biçimlendirme yardımcıları
# ---------------------------------------------------------------------------------------------------------


def test_render_table_aligns_columns() -> None:
    text = cli.render_table(("Ad", "Durum"), [("RabbitMQ", "OK"), ("Elasticsearch", "HATA")])
    lines = text.splitlines()
    assert lines[0] == "Ad             Durum"
    assert set(lines[1]) == {"-", " "}
    assert lines[2].startswith("RabbitMQ       OK")
    assert lines[3].startswith("Elasticsearch  HATA")


def test_fmt_dt_uses_istanbul_time() -> None:
    assert cli.fmt_dt(datetime(2026, 10, 2, 19, 41, tzinfo=UTC)) == "02.10.2026 22:41"
    assert cli.fmt_dt("2026-10-02T19:41:00Z") == "02.10.2026 22:41"
    assert cli.fmt_dt("2026-10-02T22:41:00+03:00") == "02.10.2026 22:41"
    assert cli.fmt_dt(None) == "-"
    assert cli.fmt_dt("bozuk") == "bozuk"


def test_redact_url_hides_password() -> None:
    assert cli.redact_url("amqp://guest:gizli@rabbitmq:5672/%2F") == "amqp://guest:***@rabbitmq:5672/%2F"
    assert cli.redact_url("http://elasticsearch:9200") == "http://elasticsearch:9200"


def test_logging_configure_is_idempotent() -> None:
    root = logging.getLogger()
    logging_setup.configure("debug")
    logging_setup.configure("warning")
    marked = [h for h in root.handlers if getattr(h, "_scraperhryt_handler", False)]
    assert len(marked) == 1
    assert root.level == logging.WARNING
    assert logging.getLogger("pika").level == logging.WARNING
    assert logging_setup.parse_level("INFO") == logging.INFO
    assert logging_setup.parse_level("saçma") == logging.INFO
    assert logging_setup.parse_level(10) == logging.DEBUG


# ---------------------------------------------------------------------------------------------------------
# check / sondalar
# ---------------------------------------------------------------------------------------------------------


def _patch_probes(monkeypatch: pytest.MonkeyPatch, rabbit: bool, es: bool, ollama: bool) -> None:
    monkeypatch.setattr(
        cli, "probe_rabbitmq", lambda settings, timeout=5.0: cli.ServiceStatus("RabbitMQ", rabbit, "rabbit-detay", 3.2)
    )
    monkeypatch.setattr(
        cli, "probe_elasticsearch", lambda settings, timeout=5.0: cli.ServiceStatus("Elasticsearch", es, "es-detay", 8.0)
    )
    monkeypatch.setattr(
        cli,
        "probe_ollama",
        lambda settings, *, required=True: cli.ServiceStatus("Ollama", ollama, "ollama-detay", 1.0, required),
    )


def test_check_reports_and_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_probes(monkeypatch, rabbit=True, es=False, ollama=True)
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "Elasticsearch" in out and "HATA" in out and "es-detay" in out and "3 ms" in out
    assert "Zorunlu servis(ler) erişilemez: Elasticsearch" in out

    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=True)
    assert cli.main(["check"]) == 0
    assert "Tüm zorunlu servisler erişilebilir." in capsys.readouterr().out


def test_check_ollama_optional_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=False)
    assert cli.main(["check"]) == 1
    capsys.readouterr()
    assert cli.main(["check", "--ollama-optional"]) == 0
    out = capsys.readouterr().out
    assert "UYARI" in out and "HATA" not in out


def test_probes_fail_fast_on_closed_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RABBITMQ_URL", "amqp://guest:guest@127.0.0.1:1/%2F")
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    settings = cli.build_settings(cli.build_parser().parse_args(["check"]))
    started = time.monotonic()
    rabbit = cli.probe_rabbitmq(settings, timeout=2.0)
    es = cli.probe_elasticsearch(settings, timeout=2.0)
    ollama = cli.probe_ollama(settings, required=False)
    assert time.monotonic() - started < 20
    assert not rabbit.ok and "bağlanılamadı" in rabbit.detail and rabbit.label == "HATA"
    assert not es.ok and "erişilemiyor" in es.detail
    assert not ollama.ok and "ollama serve" in ollama.detail and ollama.label == "UYARI"


# ---------------------------------------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------------------------------------


def _fake_answer(question: str) -> Answer:
    return Answer(
        question=question,
        answer="Son duruma göre Özgür Özel ile Kemal Kılıçdaroğlu arasındaki gerilim sürüyor [1].",
        sources=[
            Citation(
                id="abc",
                title="Özel'den Kılıçdaroğlu'na yanıt",
                content_url="https://www.hurriyet.com.tr/gundem/ozel-kilicdaroglu-1",
                source="hurriyet",
                published_at=datetime(2026, 10, 2, 19, 41, tzinfo=UTC),
                score=1.5,
            )
        ],
        model="fake-model",
        retrieved_count=1,
        search_terms=["özgür özel", "kılıçdaroğlu"],
    )


def _install_fake_qa(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class FakeQAEngine:
        def __init__(self, settings: Any, store: Any, llm: Any) -> None:
            captured["store"] = store
            captured["llm"] = llm

        def ask(self, question: str, *, since_days: int | None = None, top_k: int | None = None, sources: Any = None) -> Answer:
            captured.update(question=question, since_days=since_days, top_k=top_k)
            return _fake_answer(question)

    patch_module_attr(monkeypatch, "scraperhryt.reporting.rag", "QAEngine", FakeQAEngine)
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", lambda settings, *, in_memory=False: FakeStore())
    return captured


def test_ask_prints_answer_with_sources(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    captured = _install_fake_qa(monkeypatch)
    question = "Özgür Özel ile Kemal Kılıçdaroğlu arasındaki son durum ne?"
    assert cli.main(["ask", question, "--since-days", "7", "--top-k", "5", "--fake-llm"]) == 0
    out = capsys.readouterr().out
    assert f"Soru: {question}" in out
    assert "gerilim sürüyor [1]" in out
    assert "Kaynaklar:" in out and "[1] Özel'den Kılıçdaroğlu'na yanıt (hurriyet | 02.10.2026 22:41)" in out
    assert "https://www.hurriyet.com.tr/gundem/ozel-kilicdaroglu-1" in out
    assert captured["question"] == question
    assert captured["since_days"] == 7 and captured["top_k"] == 5
    assert isinstance(captured["store"], FakeStore) and captured["store"].ensured
    assert captured["llm"].model_name == "heuristic"


def test_ask_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _install_fake_qa(monkeypatch)
    assert cli.main(["ask", "son durum?", "--json", "--fake-llm"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["question"] == "son durum?"
    assert payload["sources"][0]["source"] == "hurriyet"
    assert payload["retrieved_count"] == 1


def test_ask_reports_engine_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class BrokenQAEngine:
        def __init__(self, settings: Any, store: Any, llm: Any) -> None:
            pass

        def ask(self, question: str, **kwargs: Any) -> Answer:
            raise RuntimeError("LLM yanıt vermedi")

    patch_module_attr(monkeypatch, "scraperhryt.reporting.rag", "QAEngine", BrokenQAEngine)
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", lambda settings, *, in_memory=False: FakeStore())
    assert cli.main(["ask", "soru", "--fake-llm"]) == 1
    assert "Soru yanıtlanamadı" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------
# İş parçacığı denetimi
# ---------------------------------------------------------------------------------------------------------


def test_supervisor_records_crash_and_stops_everyone() -> None:
    stop = threading.Event()
    supervisor = cli.Supervisor(stop)

    def crash() -> None:
        raise RuntimeError("patladı")

    supervisor.spawn("waiter", lambda: stop.wait(10))
    supervisor.spawn("crasher", crash)
    supervisor.wait()
    supervisor.join(timeout=5)
    assert stop.is_set()
    assert supervisor.failures == ["crasher"]
    assert not supervisor.ok
    assert not any(t.is_alive() for t in supervisor.threads)


def test_consume_loop_tracks_processed_and_idle() -> None:
    broker = InMemoryBroker()
    broker.declare_topology()
    stop = threading.Event()
    counters = cli.PipelineCounters()
    seen: list[str] = []

    def handler(msg: Message) -> None:
        seen.append(msg.body["id"])

    thread = threading.Thread(
        target=cli.consume_loop, args=(broker, str(Queue.ARTICLES_RAW), handler, stop, counters), daemon=True
    )
    thread.start()
    broker.publish(RoutingKey.ARTICLE_RAW, {"id": "a"})
    broker.publish(RoutingKey.ARTICLE_RAW, {"id": "b"})
    deadline = time.monotonic() + 5
    while counters.snapshot().get(str(Queue.ARTICLES_RAW), 0) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sorted(seen) == ["a", "b"]
    time.sleep(0.2)
    assert counters.is_idle(0.1)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------------------------------------
# run-all --once --in-memory --fake-llm (uçtan uca, çevrimdışı)
# ---------------------------------------------------------------------------------------------------------


class FakeSource:
    """Üç haber üreten sahte kaynak: ikisinde 'bakan' geçer, üçüncüsü spor haberidir."""

    name = "fake"

    def __init__(self) -> None:
        now = datetime(2026, 10, 2, 19, 0, tzinfo=UTC)
        self.records = [
            NewsRecord.new(
                source="hurriyet",
                content_url="https://www.hurriyet.com.tr/gundem/bakan-sorusturma-1",
                title="Bakan hakkında soruşturma başlatıldı",
                subtitle="Gözaltı kararı verildi",
                content="Bakanlık bünyesindeki ihale soruşturması kapsamında bakan gözaltına alındı ve tutuklandı.",
                published_at=now,
                category="gündem",
            ),
            NewsRecord.new(
                source="12punto",
                content_url="https://12punto.com.tr/siyaset/cumhurbaskani-kararname-2",
                title="Cumhurbaşkanı kararnamesiyle yeni bakan atandı",
                content="Resmi Gazete'de yayımlanan kararname ile bakanlığa yeni atama yapıldı; kabine değişti.",
                published_at=now,
                category="siyaset",
            ),
            NewsRecord.new(
                source="hurriyet",
                content_url="https://www.hurriyet.com.tr/gundem/derbi-mac-3",
                title="Derbi maçında üç gol",
                content="Futbol maçı galibiyetle bitti; teknik direktör transfer için açıklama yaptı.",
                published_at=now,
                category="spor",
            ),
        ]

    def discover(self, client: Any, *, backfill_days: int = 0, limit: int | None = None) -> list[Any]:
        from scraperhryt.scrapers.base import DiscoveredLink

        return [DiscoveredLink(url=r.content_url, title_hint=r.title, published_hint=r.published_at, origin="rss") for r in self.records]

    def fetch_article(self, client: Any, link: Any) -> NewsRecord | None:
        return next((r for r in self.records if r.content_url == link.canonical), None)


@pytest.mark.timeout(120)
@pytest.mark.xfail(not INTEGRATION_READY, reason="integration pending", strict=False)
def test_run_all_once_in_memory_fake_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # proje kökündeki olası .env okunmasın
    monkeypatch.setenv("ALARM_LOG_PATH", str(tmp_path / "alarms.jsonl"))
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("KEYWORDS", "bakan,cumhurbaşkanı,fon")
    monkeypatch.setenv("ALARM_THRESHOLD", "30")
    monkeypatch.setenv("LLM_SCORE_ALL", "false")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "")
    monkeypatch.setenv("ALARM_WEBHOOK_URL", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    patch_module_attr(monkeypatch, "scraperhryt.scrapers.runner", "build_sources", lambda settings: [FakeSource()])

    assert cli.main(["run-all", "--once", "--in-memory", "--fake-llm"]) == 0
    out = capsys.readouterr().out

    assert "run-all özeti" in out
    assert "Alarm sayısı: 2" in out
    assert re.search(r"^q\.articles\.raw\s+3$", out, re.M)
    assert re.search(r"^q\.articles\.keyword\s+2$", out, re.M)
    assert re.search(r"^q\.articles\.scored\s+3$", out, re.M)
    assert re.search(r"^q\.alarms\s+2$", out, re.M)
    assert "Ölü mektup (q.dead_letter): 0" in out
    assert "En yüksek 2 alarm:" in out
    assert "bakan-sorusturma-1" in out and "cumhurbaskani-kararname-2" in out
    assert "derbi-mac-3" not in out
    alarm_log = tmp_path / "alarms.jsonl"
    assert alarm_log.exists() and len(alarm_log.read_text(encoding="utf-8").strip().splitlines()) == 2


# ---------------------------------------------------------------------------------------------------------
# Dağıtım dosyaları
# ---------------------------------------------------------------------------------------------------------


def test_docker_compose_structure() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in ("rabbitmq", "elasticsearch", "kibana", "ollama", "ollama-pull", *APP_SERVICES):
        assert name in services, name
    assert services["rabbitmq"]["image"] == "rabbitmq:3.13-management"
    assert services["rabbitmq"]["healthcheck"]["test"] == ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
    es = services["elasticsearch"]
    assert es["image"] == "docker.elastic.co/elasticsearch/elasticsearch:8.15.3"
    assert es["environment"]["discovery.type"] == "single-node"
    assert es["environment"]["xpack.security.enabled"] == "false"
    assert es["ulimits"]["memlock"] == {"soft": -1, "hard": -1}
    assert "yellow" in es["healthcheck"]["test"][1]
    assert services["kibana"]["profiles"] == ["kibana"]
    assert services["ollama"]["profiles"] == ["ollama"] and services["ollama-pull"]["profiles"] == ["ollama"]
    assert services["ollama-pull"]["entrypoint"][-1] == "/ollama-pull.sh"
    for name in APP_SERVICES:
        svc = services[name]
        assert svc["build"] == "." and svc["env_file"] == ".env", name
        assert svc["environment"]["RABBITMQ_URL"] == "amqp://guest:guest@rabbitmq:5672/%2F"
        assert svc["environment"]["ELASTICSEARCH_URL"] == "http://elasticsearch:9200"
        assert svc["depends_on"]["rabbitmq"] == {"condition": "service_healthy"}
        assert svc["depends_on"]["elasticsearch"] == {"condition": "service_healthy"}
        assert "host.docker.internal:host-gateway" in svc["extra_hosts"]
        assert "data:/app/data" in svc["volumes"]
        assert svc["command"][0] == "scraperhryt"
    assert services["setup"]["restart"] == "no"
    assert services["setup"]["command"] == ["scraperhryt", "setup"]
    for name in APP_SERVICES[1:]:
        assert services[name]["restart"] == "unless-stopped", name
        assert services[name]["depends_on"]["setup"] == {"condition": "service_completed_successfully"}
    assert services["api"]["ports"] == ["${API_PORT:-8000}:8000"]
    assert set(compose["volumes"]) >= {"data", "rabbitmq", "esdata", "ollama"}


def test_env_example_covers_settings() -> None:
    from scraperhryt.config import Settings

    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assigned = {line.split("=", 1)[0] for line in text.splitlines() if line and not line.startswith("#") and "=" in line}
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= assigned, sorted(expected - assigned)
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in text
    assert "# OLLAMA_BASE_URL=http://ollama:11434" in text
    assert "# RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2F" in text
    assert "# ELASTICSEARCH_URL=http://localhost:9200" in text


def test_dockerfile_and_makefile() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# syntax=docker/dockerfile:1\n")
    assert "FROM python:3.11-slim" in dockerfile
    assert "USER app" in dockerfile
    assert 'CMD ["scraperhryt", "--help"]' in dockerfile
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("install", "test", "lint", "up", "down", "logs", "setup", "scrape-once", "run-all-inmemory", "ask", "ps"):
        assert re.search(rf"^{re.escape(target)}:", makefile, re.M), target
    assert 'scraperhryt ask "$(Q)"' in makefile
    for script in ("wait-for.sh", "ollama-pull.sh"):
        path = ROOT / "scripts" / script
        assert path.read_text(encoding="utf-8").startswith("#!")
        assert path.stat().st_mode & 0o111, f"{script} çalıştırılabilir olmalı"
