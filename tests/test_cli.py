"""CLI ve dağıtım dosyası testleri.

Tümü çevrimdışıdır: alt komutlar sahte modüller/sondalarla çalıştırılır, run-all bellek içi broker + depo ve
sezgisel LLM ile tek tur koşturulur, compose/Makefile/.env.example yapısal olarak doğrulanır.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scraperhryt import cli, logging_setup
from scraperhryt.broker import InMemoryBroker, Message, Queue, RoutingKey, Unavailable
from scraperhryt.models import Answer, Citation, NewsRecord

ROOT = Path(__file__).resolve().parents[1]
SUBCOMMANDS = ("setup", "check", "scrape", "filter", "score", "alarm", "report", "api", "ask", "run-all")
APP_SERVICES = ("setup", "scraper", "filter", "scorer", "alarm", "reporter", "api")
CLOSED_PORT_URL = "http://127.0.0.1:1"  # bağlantı hemen reddedilir (Ollama erişilemez senaryosu)


@pytest.fixture(autouse=True)
def _no_ollama_optional_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geliştirici ortamındaki OLLAMA_OPTIONAL değişkeni `setup`/`check` varsayılanını etkilemesin."""
    monkeypatch.delenv(cli.OLLAMA_OPTIONAL_ENV, raising=False)

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


def _patch_probes(monkeypatch: pytest.MonkeyPatch, rabbit: bool, es: bool, ollama: bool) -> dict[str, list[float]]:
    """Sondaları sahteler; her servis için sondaya geçirilen ``timeout`` değerlerini kaydeder."""
    timeouts: dict[str, list[float]] = {"RabbitMQ": [], "Elasticsearch": [], "Ollama": []}

    def fake_rabbit(settings: Any, timeout: float = 5.0) -> cli.ServiceStatus:
        timeouts["RabbitMQ"].append(timeout)
        return cli.ServiceStatus("RabbitMQ", rabbit, "rabbit-detay", 3.2)

    def fake_es(settings: Any, timeout: float = 5.0) -> cli.ServiceStatus:
        timeouts["Elasticsearch"].append(timeout)
        return cli.ServiceStatus("Elasticsearch", es, "es-detay", 8.0)

    def fake_ollama(settings: Any, *, required: bool = True, timeout: float = 5.0) -> cli.ServiceStatus:
        timeouts["Ollama"].append(timeout)
        return cli.ServiceStatus("Ollama", ollama, "ollama-detay", 1.0, required)

    monkeypatch.setattr(cli, "probe_rabbitmq", fake_rabbit)
    monkeypatch.setattr(cli, "probe_elasticsearch", fake_es)
    monkeypatch.setattr(cli, "probe_ollama", fake_ollama)
    return timeouts


def test_check_reports_and_exit_code(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_probes(monkeypatch, rabbit=True, es=False, ollama=True)
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "Elasticsearch" in out and "HATA" in out and "es-detay" in out and "3 ms" in out
    assert "Zorunlu servis(ler) erişilemez: Elasticsearch" in out

    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=True)
    assert cli.main(["check"]) == 0
    assert "Tüm zorunlu servisler erişilebilir." in capsys.readouterr().out


def test_check_timeout_applies_to_every_probe_including_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts = _patch_probes(monkeypatch, rabbit=True, es=True, ollama=True)
    assert cli.main(["check", "--timeout", "3"]) == 0
    assert timeouts == {"RabbitMQ": [3.0], "Elasticsearch": [3.0], "Ollama": [3.0]}


def test_check_ollama_optional_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=False)
    assert cli.main(["check"]) == 1
    capsys.readouterr()
    assert cli.main(["check", "--ollama-optional"]) == 0
    out = capsys.readouterr().out
    assert "UYARI" in out and "HATA" not in out


def _install_fake_setup_infra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeBroker:
        def declare_topology(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "data" / "state.sqlite3"))
    monkeypatch.setenv("ALARM_LOG_PATH", str(tmp_path / "data" / "alarms.jsonl"))
    monkeypatch.setattr(cli, "make_broker", lambda settings, *, in_memory=False: FakeBroker())
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", lambda settings, *, in_memory=False: FakeStore())


def test_setup_fails_when_ollama_is_not_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Ollama'sız başlayan compose yığını her anahtar kelime eşleşmesini ölü mektuba düşürür; setup bunu HATA sayar."""
    _install_fake_setup_infra(monkeypatch, tmp_path)
    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=False)
    assert cli.main(["setup"]) == 1
    out = capsys.readouterr().out
    assert "HATA" in out and "OLLAMA_HOST=0.0.0.0 ollama serve" in out and "Kurulum tamamlanamadı" in out


def test_setup_ollama_optional_flag_and_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _install_fake_setup_infra(monkeypatch, tmp_path)
    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=False)
    assert cli.main(["setup", "--ollama-optional"]) == 0
    out = capsys.readouterr().out
    assert "UYARI" in out and "HATA" not in out and "Kurulum tamam." in out

    monkeypatch.setenv(cli.OLLAMA_OPTIONAL_ENV, "1")  # docker compose: .env → konteyner ortamı
    assert cli.main(["setup"]) == 0
    assert "Kurulum tamam." in capsys.readouterr().out
    assert cli.main(["check"]) == 0

    _patch_probes(monkeypatch, rabbit=True, es=True, ollama=True)
    monkeypatch.setenv(cli.OLLAMA_OPTIONAL_ENV, "0")
    assert cli.main(["setup"]) == 0
    assert "Kurulum tamam." in capsys.readouterr().out
    assert cli.env_flag(cli.OLLAMA_OPTIONAL_ENV) is False
    monkeypatch.setenv(cli.OLLAMA_OPTIONAL_ENV, " TRUE ")
    assert cli.env_flag(cli.OLLAMA_OPTIONAL_ENV) is True


def _patch_ollama_probe_sequence(monkeypatch: pytest.MonkeyPatch, outcomes: list[bool]) -> dict[str, list[float]]:
    """``probe_ollama``'yı sırayla verilen sonuçları döndürecek biçimde sahteler (liste bitince son değer sürer);
    ``cli.time`` sahte bir saattir: ``sleep`` beklemez, yalnızca ``monotonic``'i ilerletir ve kaydedilir.
    Dönen sözlük: sondaya geçen timeout'lar ve istenen uykular."""
    calls: dict[str, list[float]] = {"timeout": [], "sleep": []}
    remaining = list(outcomes)
    clock = {"now": 1000.0}

    def fake_sleep(seconds: float) -> None:
        calls["sleep"].append(seconds)
        clock["now"] += seconds

    def fake_probe(settings: Any, *, required: bool = True, timeout: float = 5.0) -> cli.ServiceStatus:
        ok = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        calls["timeout"].append(timeout)
        detail = "model 'qwen2.5:7b' yüklü" if ok else "sunucu çalışıyor ama 'qwen2.5:7b' yüklü değil"
        return cli.ServiceStatus("Ollama", ok, detail, 1.0, required)

    monkeypatch.setattr(cli, "probe_ollama", fake_probe)
    fake_time = types.SimpleNamespace(monotonic=lambda: clock["now"], perf_counter=time.perf_counter, sleep=fake_sleep)
    monkeypatch.setattr(cli, "time", fake_time)
    return calls


def test_setup_waits_for_ollama_model_when_wait_is_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`docker compose --profile ollama` ilk açılışı: model indirilirken setup HATA vermek yerine (sınırlı) bekler."""
    _install_fake_setup_infra(monkeypatch, tmp_path)
    calls = _patch_ollama_probe_sequence(monkeypatch, [False, False, True])
    assert cli.main(["setup", "--ollama-wait", "30", "--timeout", "2"]) == 0
    assert "Kurulum tamam." in capsys.readouterr().out
    assert calls["timeout"] == [2.0, 2.0, 2.0]
    assert calls["sleep"] == [cli.OLLAMA_WAIT_POLL_SECONDS] * 2

    # varsayılan: beklemez, tek sonda, HATA (yerel Ollama kapalıysa dakikalarca oyalanmasın)
    calls = _patch_ollama_probe_sequence(monkeypatch, [False])
    assert cli.main(["setup"]) == 1
    assert "Kurulum tamamlanamadı" in capsys.readouterr().out
    assert len(calls["timeout"]) == 1 and calls["sleep"] == []

    # süre dolunca HATA; ayrıntıda beklenen süre görünür
    calls = _patch_ollama_probe_sequence(monkeypatch, [False])
    assert cli.main(["setup", "--ollama-wait", "12"]) == 1
    out = capsys.readouterr().out
    assert "HATA" in out and "12 sn beklendi" in out
    assert calls["sleep"] == [5.0, 5.0, 2.0]

    # ortam değişkeni (compose .env): OLLAMA_WAIT_SECONDS
    monkeypatch.setenv(cli.OLLAMA_WAIT_ENV, "30")
    calls = _patch_ollama_probe_sequence(monkeypatch, [False, True])
    assert cli.main(["setup"]) == 0
    assert "Kurulum tamam." in capsys.readouterr().out
    assert calls["sleep"] == [5.0]
    monkeypatch.setenv(cli.OLLAMA_WAIT_ENV, "saçma")  # sayı değil → beklemez
    calls = _patch_ollama_probe_sequence(monkeypatch, [False])
    assert cli.main(["setup"]) == 1
    assert calls["sleep"] == []


def test_ollama_probe_and_build_llm_use_short_timeout_not_ollama_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paketleri düşüren (reddetmeyen) ana makine: bağlantı açılır ama yanıt gelmez. Sonda ve servis başlangıcı
    OLLAMA_TIMEOUT (180 s) yerine kısa sonda zaman aşımıyla döner; dönen istemci tam zaman aşımını korur."""
    from scraperhryt.pipeline.llm import OllamaClient

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)  # accept edilmez: istek gönderilir, yanıt asla gelmez
    try:
        monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.getsockname()[1]}")
        monkeypatch.setenv("OLLAMA_TIMEOUT", "180")
        monkeypatch.setattr(cli, "PROBE_TIMEOUT_SECONDS", 0.5)
        settings = cli.build_settings(cli.build_parser().parse_args(["check"]))

        started = time.monotonic()
        status = cli.probe_ollama(settings, timeout=0.5)
        assert time.monotonic() - started < 5
        assert not status.ok and "erişilemiyor" in status.detail
        assert settings.ollama_timeout == 180.0  # sonda ayarları kopyalar, LLM çağrı zaman aşımına dokunmaz

        started = time.monotonic()
        llm = cli.build_llm(settings, fake=False, fallback=False)
        try:
            assert time.monotonic() - started < 5
            assert isinstance(llm, OllamaClient) and llm.settings.ollama_timeout == 180.0
        finally:
            cli.close_llm(llm)
    finally:
        server.close()


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
# LLM kurucu / api
# ---------------------------------------------------------------------------------------------------------


def test_build_llm_keeps_ollama_client_unless_fallback_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama başlangıçta hazır değilse istemci korunur (skorlayıcı Retry ile kendini toparlar); sezgisel
    değerlendiriciye yalnızca açıkça istenince (--llm-fallback) düşülür."""
    from scraperhryt.pipeline.llm import HeuristicLLM, OllamaClient

    monkeypatch.setenv("OLLAMA_BASE_URL", CLOSED_PORT_URL)
    settings = cli.build_settings(cli.build_parser().parse_args(["run-all"]))
    llm = cli.build_llm(settings, fake=False, fallback=False)
    try:
        assert isinstance(llm, OllamaClient)
    finally:
        cli.close_llm(llm)
    assert isinstance(cli.build_llm(settings, fake=False, fallback=True), HeuristicLLM)
    assert isinstance(cli.build_llm(settings, fake=True, fallback=False), HeuristicLLM)


def test_run_all_llm_fallback_is_opt_in() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["run-all"]).llm_fallback is False
    assert parser.parse_args(["run-all", "--llm-fallback"]).llm_fallback is True


def test_shutdown_timeout_covers_worst_case_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from scraperhryt.pipeline.llm import HeuristicLLM, OllamaClient
    from scraperhryt.pipeline.scorer import MAX_LLM_ATTEMPTS

    monkeypatch.setenv("OLLAMA_TIMEOUT", "100")
    settings = cli.build_settings(cli.build_parser().parse_args(["run-all"]))
    client = OllamaClient(settings)
    try:
        assert cli.shutdown_timeout_for(settings, client) == MAX_LLM_ATTEMPTS * 100.0
    finally:
        client.close()
    assert cli.shutdown_timeout_for(settings, HeuristicLLM(settings)) == cli.JOIN_TIMEOUT_SECONDS


def test_api_declares_topology_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeBroker:
        def declare_topology(self) -> None:
            events.append("declare")

        def close(self) -> None:
            events.append("close")

    def fake_create_app(settings: Any, store: Any, llm: Any, broker: Any = None) -> str:
        assert isinstance(broker, FakeBroker)
        events.append("create_app")
        return "app"

    def fake_run(app: Any, **kwargs: Any) -> None:
        events.append("serve")

    monkeypatch.setattr(cli, "make_broker", lambda settings, *, in_memory=False: FakeBroker())
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", lambda settings, *, in_memory=False: FakeStore())
    patch_module_attr(monkeypatch, "scraperhryt.reporting.api", "create_app", fake_create_app)
    patch_module_attr(monkeypatch, "uvicorn", "run", fake_run)
    assert cli.main(["api", "--fake-llm"]) == 0
    assert events == ["declare", "create_app", "serve", "close"]


def test_api_starts_even_if_topology_declaration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class BrokenBroker:
        def declare_topology(self) -> None:
            raise ConnectionError("RabbitMQ'ya bağlanılamadı")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "make_broker", lambda settings, *, in_memory=False: BrokenBroker())
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", lambda settings, *, in_memory=False: FakeStore())
    patch_module_attr(monkeypatch, "scraperhryt.reporting.api", "create_app", lambda *a, **k: "app")
    patch_module_attr(monkeypatch, "uvicorn", "run", lambda app, **kwargs: events.append("serve"))
    assert cli.main(["api", "--fake-llm"]) == 0
    assert events == ["serve", "close"]


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


def test_consume_loop_calls_on_idle_and_survives_its_errors() -> None:
    broker = InMemoryBroker()
    broker.declare_topology()
    stop = threading.Event()
    counters = cli.PipelineCounters()
    idle_calls = 0

    def on_idle() -> None:
        nonlocal idle_calls
        idle_calls += 1
        if idle_calls == 1:
            raise RuntimeError("özet üretilemedi")  # döngüyü durdurmamalı

    thread = threading.Thread(
        target=cli.consume_loop,
        args=(broker, str(Queue.ALARMS), lambda msg: None, stop, counters),
        kwargs={"on_idle": on_idle},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while idle_calls < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert idle_calls >= 2


def test_tracked_handler_counts_only_successes() -> None:
    counters = cli.PipelineCounters()

    def flaky(msg: Message) -> None:
        if msg.body["id"] == "bad":
            raise RuntimeError("hata")

    tracked = cli.tracked_handler("q.x", flaky, counters)
    tracked(Message(body={"id": "ok"}, routing_key="x"))
    with pytest.raises(RuntimeError):
        tracked(Message(body={"id": "bad"}, routing_key="x"))
    assert counters.snapshot() == {"q.x": 1}
    assert counters.in_flight == 0


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


class BlockingBroker:
    """RabbitMQ modunu taklit eder: ``InMemoryBroker`` DEĞİLDİR (run-all her iş parçacığına ayrı örnek verir), tek
    bir paylaşılan bellek içi kuyruğa delege eder ve pika gibi ``stop_event`` set edilene kadar bloklar.
    ``consume``/``close`` çağıran iş parçacıklarının adlarını kaydeder (broker sahipliği denetimi için)."""

    def __init__(self, inner: InMemoryBroker) -> None:
        self.inner = inner
        self.consumed_by: list[str] = []
        self.closed_by: list[str] = []

    def declare_topology(self) -> None:
        self.inner.declare_topology()

    def publish(self, routing_key: str, body: dict[str, Any], headers: dict[str, Any] | None = None) -> None:
        self.inner.publish(routing_key, body, headers)

    def consume(
        self,
        queue: str,
        handler: Callable[[Message], None],
        *,
        prefetch: int | None = None,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ) -> int:
        self.consumed_by.append(threading.current_thread().name)
        stop_event = stop_event or threading.Event()
        processed = 0
        while not stop_event.is_set():
            processed += self.inner.consume(queue, handler, prefetch=prefetch, stop_event=stop_event)
            if stop_event.wait(0.05):
                break
        return processed

    def close(self) -> None:
        self.closed_by.append(threading.current_thread().name)


def _prepare_rabbitmq_mode_run_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[InMemoryBroker, list[BlockingBroker], list[Any]]:
    """run-all'ı RabbitMQ modunda (ayrı ``BlockingBroker`` örnekleri + ``InMemoryStore``) çevrimdışı koşturmak için
    ortamı kurar; (paylaşılan kuyruk, üretilen broker'lar, üretilen depolar) döndürür."""
    from scraperhryt.store import InMemoryStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALARM_LOG_PATH", str(tmp_path / "alarms.jsonl"))
    monkeypatch.setenv("STATE_DB_PATH", str(tmp_path / "state.sqlite3"))
    monkeypatch.setenv("KEYWORDS", "bakan,cumhurbaşkanı,fon")
    monkeypatch.setenv("ALARM_THRESHOLD", "30")
    monkeypatch.setenv("REPORT_DIGEST_EVERY", "10")
    monkeypatch.setenv("REPORT_DIGEST_MINUTES", "30")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "")
    monkeypatch.setenv("ALARM_WEBHOOK_URL", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    patch_module_attr(monkeypatch, "scraperhryt.scrapers.runner", "build_sources", lambda settings: [FakeSource()])

    shared = InMemoryBroker()
    brokers: list[BlockingBroker] = []
    stores: list[InMemoryStore] = []

    def fake_make_broker(settings: Any, *, in_memory: bool = False) -> BlockingBroker:
        assert not in_memory
        broker = BlockingBroker(shared)
        brokers.append(broker)
        return broker

    def fake_make_store(settings: Any, *, in_memory: bool = False) -> InMemoryStore:
        store = InMemoryStore()
        stores.append(store)
        return store

    monkeypatch.setattr(cli, "make_broker", fake_make_broker)
    patch_module_attr(monkeypatch, "scraperhryt.store", "make_store", fake_make_store)
    return shared, brokers, stores


@pytest.mark.timeout(120)
def test_run_all_once_rabbitmq_mode_digests_alarms_and_closes_brokers_in_owner_threads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """RabbitMQ modunda (ayrı broker örnekleri, bloklayan consume): rapor tüketicisi çıkışta tamponu özetler
    (REPORT_DIGEST_EVERY dolmasa da alarmlar kaybolmaz), her tüketici broker'ını kendi iş parçacığında kapatır."""
    shared, brokers, stores = _prepare_rabbitmq_mode_run_all(monkeypatch, tmp_path)

    assert cli.main(["run-all", "--once", "--fake-llm", "--no-api", "--idle-timeout", "1"]) == 0
    out = capsys.readouterr().out
    assert "run-all özeti (tek tur)" in out
    assert "Alarm sayısı: 2" in out
    assert re.search(r"^q\.alarms\s+2$", out, re.M)

    digests = stores[0].list_reports(kind="alarm_digest")
    assert len(digests) == 1, "çıkışta tampondaki alarmlar tek bir alarm_digest raporuna girmeli"
    assert len(digests[0]["top_alarms"]) == 2
    digest_messages = [m for m in shared.published if m.routing_key == RoutingKey.REPORT_ALARM_DIGEST]
    assert len(digest_messages) == 1
    assert shared.size(str(Queue.ALARMS)) == 0

    # setup + filter + scorer + alarm + reporter + periodic + tek tur kazıyıcı = 7 broker; hepsi kapatıldı
    assert len(brokers) == 7
    assert all(b.closed_by for b in brokers), [b.closed_by for b in brokers]
    consumers = [b for b in brokers if b.consumed_by]
    assert sorted(b.consumed_by[0] for b in consumers) == ["alarm", "filter", "reporter", "scorer"]
    for broker in consumers:
        assert broker.closed_by == [broker.consumed_by[0]], "broker yalnızca onu tüketen iş parçacığında kapatılmalı"
    assert not any(t.is_alive() for t in threading.enumerate() if t.name in ("filter", "scorer", "alarm", "reporter"))


@pytest.mark.timeout(120)
def test_run_all_shares_stop_event_with_scorer_and_alarm_services(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """run-all tüketicileri ``consume_loop`` ile sürer (``.run()`` çağrılmaz); skorlayıcı ve alarm katmanı yine de
    run-all'ın durdurma olayını kullanmalı ki Ollama/ES kesintisindeki beklemeler SIGINT/SIGTERM'de hemen bitsin."""
    from scraperhryt.pipeline import alarm as alarm_module
    from scraperhryt.pipeline import scorer as scorer_module

    _prepare_rabbitmq_mode_run_all(monkeypatch, tmp_path)
    created: dict[str, Any] = {}

    class CapturingScorer(scorer_module.ScoringService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created["scorer"] = self

    class CapturingAlarm(alarm_module.AlarmService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created["alarm"] = self

    monkeypatch.setattr(scorer_module, "ScoringService", CapturingScorer)
    monkeypatch.setattr(alarm_module, "AlarmService", CapturingAlarm)

    assert cli.main(["run-all", "--once", "--fake-llm", "--no-api", "--idle-timeout", "1"]) == 0
    assert "Alarm sayısı: 2" in capsys.readouterr().out
    scorer, alarm = created["scorer"], created["alarm"]
    stop = scorer.stop_event
    assert alarm._stop_event is stop, "iki servis de run-all'ın paylaşılan durdurma olayını kullanmalı"
    assert stop.is_set(), "run-all çıkışta olayı set eder; kurucudaki özel Event hiç set edilmezdi"

    # Kesinti senaryosu: Ollama hazır değil / ES erişilemiyor; durdurma olayı set edilince beklemeler hemen biter.
    stop.clear()
    monkeypatch.setattr(scorer, "llm_ready", lambda: False)

    def es_down() -> None:
        raise Unavailable("Elasticsearch erişilemiyor")

    monkeypatch.setattr(alarm.store, "ensure_indices", es_down)
    threading.Timer(0.3, stop.set).start()
    started = time.monotonic()
    assert scorer.wait_for_llm(max_wait=60) is False
    assert alarm._prepare_indices(alarm._stop_event) is False
    assert time.monotonic() - started < 5


# ---------------------------------------------------------------------------------------------------------
# scripts/ollama-pull.sh
# ---------------------------------------------------------------------------------------------------------


def _run_ollama_pull(tmp_path: Path, *, model: str, embedding_model: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Sahte bir `ollama` CLI ile betiği çalıştırır; (süreç sonucu, ollama'ya yapılan çağrılar) döndürür."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_log = tmp_path / "calls.log"
    fake = bin_dir / "ollama"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls_log}"\n'
        'case "$1" in\n'
        "  list)\n"
        '    printf "NAME\\tID\\tSIZE\\tMODIFIED\\n"\n'
        '    printf "qwen2.5:7b\\tabc\\t4.7 GB\\t2 days ago\\n"\n'
        '    printf "nomic-embed-text:latest\\tdef\\t274 MB\\t2 days ago\\n"\n'
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "OLLAMA_MODEL": model,
        "OLLAMA_EMBEDDING_MODEL": embedding_model,
        "WAIT_SECONDS": "4",
    }
    result = subprocess.run(
        ["sh", str(ROOT / "scripts" / "ollama-pull.sh")], capture_output=True, text=True, env=env, timeout=60, check=False
    )
    calls = calls_log.read_text(encoding="utf-8").splitlines() if calls_log.exists() else []
    return result, calls


def test_ollama_pull_recognises_tagless_installed_models(tmp_path: Path) -> None:
    result, calls = _run_ollama_pull(tmp_path, model="qwen2.5:7b", embedding_model="nomic-embed-text")
    assert result.returncode == 0, result.stderr
    assert "qwen2.5:7b zaten yüklü" in result.stdout
    assert "nomic-embed-text zaten yüklü (nomic-embed-text:latest)" in result.stdout
    assert not any(call.startswith("pull") for call in calls), calls


def test_ollama_pull_downloads_missing_models(tmp_path: Path) -> None:
    result, calls = _run_ollama_pull(tmp_path, model="llama3.1", embedding_model="nomic-embed-text")
    assert result.returncode == 0, result.stderr
    assert "indiriliyor: llama3.1" in result.stdout
    assert [call for call in calls if call.startswith("pull")] == ["pull llama3.1"]


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
    # LLM çağıran servisler: SIGTERM→SIGKILL arası süren çağrının bitmesine yetecek kadar uzun olmalı
    for name in ("scorer", "reporter", "api"):
        assert services[name]["stop_grace_period"] == "${LLM_STOP_GRACE:-200s}", name
    for name in ("scraper", "filter", "alarm"):
        assert "stop_grace_period" not in services[name], name


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
    assert "OLLAMA_OPTIONAL=0" in text and "OLLAMA_HOST=0.0.0.0 ollama serve" in text
    assert "OLLAMA_WAIT_SECONDS=0" in text and "LLM_STOP_GRACE=200s" in text
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "OLLAMA_HOST=0.0.0.0 ollama serve" in compose


def test_dockerfile_and_makefile() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# syntax=docker/dockerfile:1\n")
    assert "FROM python:3.11-slim" in dockerfile
    assert "USER app" in dockerfile
    assert 'CMD ["scraperhryt", "--help"]' in dockerfile
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("install", "test", "lint", "up", "up-ollama", "down", "logs", "setup", "scrape-once", "run-all-inmemory", "ask", "ps"):
        assert re.search(rf"^{re.escape(target)}:", makefile, re.M), target
    # Ollama profili: model indirme (ollama-pull) bitmeden yığın başlatılmamalı
    up_ollama = makefile.split("up-ollama:", 1)[1].split("\n\n", 1)[0]
    assert up_ollama.index("run --rm ollama-pull") < up_ollama.index("--profile ollama up -d")
    assert 'scraperhryt ask "$(Q)"' in makefile
    for script in ("wait-for.sh", "ollama-pull.sh"):
        path = ROOT / "scripts" / script
        assert path.read_text(encoding="utf-8").startswith("#!")
        assert path.stat().st_mode & 0o111, f"{script} çalıştırılabilir olmalı"
