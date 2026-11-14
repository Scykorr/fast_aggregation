"""Первое подключение к координатору с проверкой отпечатка сертификата."""

from __future__ import annotations

import socket
import ssl
from pathlib import Path
from urllib.parse import urlparse

import httpx

from fast_aggregation.errors import FastAggError
from fast_aggregation.security.certs import fingerprint_pem


def pinned_post(url: str, fingerprint: str, payload: dict, ca_path: Path) -> dict:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host or not port:
        raise FastAggError("В адресе координатора нет host:port")
    pem = _peer_pem(host, port)
    got = fingerprint_pem(pem)
    if _norm(got) != _norm(fingerprint):
        raise FastAggError(
            f"Отпечаток координатора {got} не совпал с ожидаемым {fingerprint}. Подключение остановлено."
        )
    ca_path.parent.mkdir(parents=True, exist_ok=True)
    ca_path.write_bytes(pem)
    try:
        response = httpx.post(url, json=payload, timeout=15, verify=str(ca_path))
    except httpx.HTTPError as exc:
        raise FastAggError(f"Координатор не ответил: {exc}") from exc
    if response.status_code >= 400:
        raise FastAggError(f"Сопряжение отклонено: {response.status_code} {response.text}")
    data = response.json()
    if not isinstance(data, dict) or not data.get("token"):
        raise FastAggError("Координатор не выдал токен")
    return data


def _peer_pem(host: str, port: int) -> bytes:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=5) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
    except OSError as exc:
        raise FastAggError(f"Нет TLS на {host}:{port}: {exc}") from exc
    if not der:
        raise FastAggError("Координатор не показал сертификат")
    # DER в PEM без повторного кодирования через cryptography: достаточно загрузить и сравнить отпечаток.
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cert = x509.load_der_x509_certificate(der)
    return cert.public_bytes(serialization.Encoding.PEM)


def _norm(value: str) -> str:
    return value.replace(":", "").strip().lower()
