"""Ölü mektup kuyruğundaki (q.dead_letter) mesajları köken kuyruklarına geri oynatır.

Köken: ``x-origin-queue`` başlığı → ``x-death`` kaydındaki kuyruk (``.retry`` eki atılır) → ``--to`` ile verilen kuyruk.
``dry_run`` modunda mesajlar okunur, listelenir ve kuyruğa geri bırakılır (nack requeue=True).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from .broker import QUEUE_BINDINGS, Queue, RabbitMQBroker, retry_queue_name

log = logging.getLogger(__name__)

_DROP_HEADERS = {"x-attempts", "x-death", "x-error", "x-first-death-exchange", "x-first-death-queue", "x-first-death-reason", "x-last-death-exchange", "x-last-death-queue", "x-last-death-reason"}


@dataclass
class ReplayStats:
    seen: int = 0
    replayed: int = 0
    skipped: int = 0
    dry_run: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def origin_queue(headers: dict[str, Any] | None, fallback: str | None = None) -> str | None:
    headers = headers or {}
    origin = headers.get("x-origin-queue")
    if isinstance(origin, bytes):
        origin = origin.decode("utf-8", "replace")
    if origin:
        return str(origin).removesuffix(".retry")
    deaths = headers.get("x-death") or []
    for death in deaths:
        q = death.get("queue") if isinstance(death, dict) else None
        if isinstance(q, bytes):
            q = q.decode("utf-8", "replace")
        if q:
            return str(q).removesuffix(".retry")
    return fallback


def replay_dead_letters(
    broker: RabbitMQBroker, *, limit: int | None = None, dry_run: bool = False, target_queue: str | None = None
) -> ReplayStats:
    import pika

    stats = ReplayStats(dry_run=dry_run)
    valid = {str(q) for q in QUEUE_BINDINGS} | {retry_queue_name(str(q)) for q in QUEUE_BINDINGS}
    if target_queue and target_queue not in valid:
        raise ValueError(f"Bilinmeyen hedef kuyruk: {target_queue} (geçerli: {', '.join(sorted(valid))})")
    ch = broker._channel()
    held: list[int] = []
    while limit is None or stats.seen < limit:
        method, properties, body = ch.basic_get(queue=str(Queue.DEAD_LETTER), auto_ack=False)
        if method is None:
            break
        stats.seen += 1
        headers = dict(getattr(properties, "headers", None) or {})
        target = origin_queue(headers, target_queue)
        try:
            payload = json.loads(body)
            message_id = str(payload.get("id") or payload.get("alarm_id") or payload.get("report_id") or "")
            title = str(payload.get("title", ""))[:60]
        except Exception:
            message_id, title = "", "<json değil>"
        item = {"message_id": message_id, "title": title, "target": target, "error": str(headers.get("x-error", ""))[:120]}
        stats.items.append(item)
        if dry_run:
            held.append(method.delivery_tag)
            continue
        if not target:
            stats.skipped += 1
            log.warning("Köken kuyruk bilinmiyor, mesaj ölü mektupta bırakıldı: %s", item)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            held.append(-1)
            break
        clean = {k: v for k, v in headers.items() if k not in _DROP_HEADERS}
        clean["x-replayed"] = True
        props = pika.BasicProperties(
            content_type="application/json", delivery_mode=2, headers=clean, message_id=message_id or None
        )
        ch.basic_publish(exchange="", routing_key=target, body=body, properties=props)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        stats.replayed += 1
        log.info("Ölü mektup geri oynatıldı → %s: %s %s", target, message_id, title)
    for tag in held:
        if tag >= 0:
            ch.basic_nack(delivery_tag=tag, requeue=True)
    return stats


def format_replay(stats: ReplayStats) -> str:
    lines = [f"Ölü mektup kuyruğu: {stats.seen} mesaj görüldü, {stats.replayed} geri oynatıldı, {stats.skipped} atlandı{' (dry-run)' if stats.dry_run else ''}"]
    for it in stats.items:
        lines.append(f"  {it['message_id'] or '-':<34} → {it['target'] or '?':<22} {it['title']}  {it['error']}")
    return "\n".join(lines)
