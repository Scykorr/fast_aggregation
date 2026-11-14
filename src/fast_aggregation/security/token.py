"""Общий секрет кластера и короткая подпись UDP-объявлений."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def new_token() -> str:
    return secrets.token_urlsafe(24)


def new_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def load_or_create(path: Path, factory) -> str:
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = factory()
    path.write_text(value, encoding="utf-8")
    return value


def sign(token: str, body: bytes) -> str:
    return hmac.new(token.encode("utf-8"), body, hashlib.sha256).hexdigest()


def canonical_announce(payload: dict) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "mac"}
    return json.dumps(unsigned, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8")


def attach_mac(payload: dict, token: str) -> dict:
    signed = dict(payload)
    signed.pop("mac", None)
    signed["mac"] = sign(token, canonical_announce(signed))
    return signed


def mac_ok(payload: dict, token: str) -> bool:
    found = str(payload.get("mac") or "")
    if not found:
        return False
    expected = sign(token, canonical_announce(payload))
    return hmac.compare_digest(found, expected)


def join_uri(host: str, port: int, code: str, fingerprint: str) -> str:
    return f"fastagg://{host}:{port}?code={code}&fp={fingerprint}"


def parse_join_uri(value: str) -> tuple[str, int, str, str]:
    text = value.strip()
    if text.startswith("fastagg://"):
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        host = parsed.hostname or ""
        port = parsed.port or 0
        code = (query.get("code") or [""])[0]
        fingerprint = (query.get("fp") or [""])[0]
        if not host or not port or not code or not fingerprint:
            raise ValueError("В ссылке сопряжения нужны адрес, код и отпечаток")
        return host, port, code, fingerprint
    raise ValueError("Ожидалась ссылка fastagg://host:port?code=...&fp=...")


def membership_path(data_dir: Path) -> Path:
    return data_dir / "membership.json"


def save_membership(data_dir: Path, payload: dict) -> None:
    path = membership_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_membership(data_dir: Path) -> dict | None:
    path = membership_path(data_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
