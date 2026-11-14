"""HTTP-клиент к узлам. В тестах подменяется фейком."""

from __future__ import annotations

from typing import Protocol

import httpx

from fast_aggregation.errors import FastAggError


class NodeClient(Protocol):
    def fetch(self, host: str, port: int) -> dict: ...

    def start_rpc(self, host: str, port: int, payload: dict) -> dict: ...

    def stop_rpc(self, host: str, port: int) -> dict: ...


class HttpNodeClient:
    def fetch(self, host: str, port: int) -> dict:
        return _request("GET", f"http://{host}:{port}/v1/node", timeout=5)

    def start_rpc(self, host: str, port: int, payload: dict) -> dict:
        return _request("POST", f"http://{host}:{port}/v1/rpc/start", json=payload, timeout=600)

    def stop_rpc(self, host: str, port: int) -> dict:
        return _request("POST", f"http://{host}:{port}/v1/rpc/stop", timeout=60)


def _request(method: str, url: str, timeout: float, json: dict | None = None) -> dict:
    try:
        response = httpx.request(method, url, json=json, timeout=timeout)
    except httpx.HTTPError as exc:
        raise FastAggError(f"Нет связи с узлом {url}: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip()
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("detail"):
                detail = str(body["detail"])
        except ValueError:
            pass
        raise FastAggError(f"Узел {url} ответил {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        raise FastAggError(f"Узел {url} вернул не JSON") from exc
    if not isinstance(data, dict):
        raise FastAggError(f"Узел {url} вернул неожиданный ответ")
    return data
