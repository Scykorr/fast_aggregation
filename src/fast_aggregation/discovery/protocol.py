"""Короткий UDP-пакет присутствия. Полные ресурсы забираются по HTTP."""

from __future__ import annotations

import json

from fast_aggregation.constants import DISCOVERY_MAGIC, PROTOCOL_VERSION
from fast_aggregation.models import Announce


def encode_announce(announce: Announce) -> bytes:
    payload = {
        "v": PROTOCOL_VERSION,
        "magic": DISCOVERY_MAGIC,
        "kind": "announce",
        **announce.to_dict(),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_announce(packet: bytes) -> Announce | None:
    try:
        text = packet.decode("utf-8")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("magic") != DISCOVERY_MAGIC or data.get("v") != PROTOCOL_VERSION:
        return None
    if data.get("kind") != "announce":
        return None
    try:
        http_port = int(data["http_port"])
        rpc_port = int(data["rpc_port"])
        node_id = str(data["node_id"]).strip()
        host = str(data["advertise_host"]).strip()
        name = str(data["name"]).strip() or node_id
    except (KeyError, TypeError, ValueError):
        return None
    if not node_id or not host:
        return None
    if not _valid_port(http_port) or not _valid_port(rpc_port):
        return None
    return Announce(
        node_id=node_id,
        name=name,
        advertise_host=host,
        http_port=http_port,
        rpc_port=rpc_port,
        os_name=str(data.get("os_name") or ""),
        arch=str(data.get("arch") or ""),
        preferred_backend=str(data.get("preferred_backend") or "auto"),
        software=str(data.get("software") or ""),
    )


def _valid_port(port: int) -> bool:
    return 1 <= port <= 65535
