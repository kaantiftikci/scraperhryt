"""Günlükleme yapılandırması: tüm servisler için tek, kompakt ve iş parçacığı adını içeren biçim.

Çıktı stderr'e yazılır; böylece ``scraperhryt ask --json`` gibi komutların stdout'u makine tarafından
okunabilir kalır. Gürültülü kütüphane günlükleri (pika, httpx, elastic_transport ...) WARNING'e çekilir.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Varsayılan seviyede fazla konuşkan olan üçüncü parti günlükçüler.
NOISY_LOGGERS = (
    "pika",
    "urllib3",
    "httpx",
    "httpcore",
    "elastic_transport",
    "elasticsearch",
    "charset_normalizer",
    "uvicorn.access",
)

#: Bağlantı kopmalarında kendi ERROR+traceback'ini basan pika iç günlükçüleri; broker katmanı zaten uyarı verir.
SILENT_LOGGERS = (
    "pika.adapters.utils.io_services_utils",
    "pika.adapters.utils.connection_workflow",
    "pika.adapters.base_connection",
    "pika.adapters.blocking_connection",
)

_HANDLER_MARK = "_scraperhryt_handler"


def parse_level(level: str | int) -> int:
    """``"debug"``, ``"INFO"`` ya da sayısal seviyeyi ``logging`` sabitine çevirir; bilinmeyen değer → INFO."""
    if isinstance(level, int):
        return level
    name = (level or "").strip().upper()
    resolved = logging.getLevelName(name)
    if isinstance(resolved, int):
        return resolved
    logging.getLogger(__name__).warning("Bilinmeyen günlük seviyesi %r; INFO kullanılıyor", level)
    return logging.INFO


def configure(level: str | int = "INFO", *, stream: TextIO | None = None) -> None:
    """Kök günlükçüyü yapılandırır. Tekrar çağrılabilir: yalnızca bu modülün eklediği işleyici değiştirilir."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)

    resolved = parse_level(level)
    root.setLevel(resolved)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(resolved, logging.WARNING))
    for name in SILENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)
