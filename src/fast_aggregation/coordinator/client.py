"""HTTP-клиент к узлам. В тестах подменяется фейком."""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from fast_aggregation.errors import FastAggError
from fast_aggregation.netmeasure import megabits_per_second


class NodeClient(Protocol):
    def fetch(self, host: str, port: int) -> dict: ...

    def start_rpc(self, host: str, port: int, payload: dict) -> dict: ...

    def stop_rpc(self, host: str, port: int) -> dict: ...


class HttpNodeClient:
    def __init__(self, token: str = "", scheme: str = "http", verify: bool | str = True) -> None:
        self.token = token
        self.scheme = scheme
        self.verify = verify

    def fetch(self, host: str, port: int) -> dict:
        return self._request("GET", self._url(host, port, "/v1/node"), timeout=5)

    def start_rpc(self, host: str, port: int, payload: dict) -> dict:
        return self._request("POST", self._url(host, port, "/v1/rpc/start"), json=payload, timeout=600)

    def stop_rpc(self, host: str, port: int) -> dict:
        return self._request("POST", self._url(host, port, "/v1/rpc/stop"), timeout=60)

    def start_aux(self, host: str, port: int, payload: dict) -> dict:
        return self._request("POST", self._url(host, port, "/v1/aux/start"), json=payload, timeout=60)

    def measure(self, host: str, port: int) -> dict[str, float]:
        started = time.perf_counter()
        httpx.get(self._url(host, port, "/health"), timeout=3, verify=self.verify)
        rtt_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        response = httpx.get(
            self._url(host, port, "/v1/bench?bytes=524288"),
            headers=self._headers(),
            timeout=10,
            verify=self.verify,
        )
        response.raise_for_status()
        elapsed = max(time.perf_counter() - started, 1e-6)
        return {
            "rtt_ms": round(rtt_ms, 2),
            "bandwidth_mbps": round(megabits_per_second(len(response.content), elapsed), 2),
        }

    def _url(self, host: str, port: int, path: str) -> str:
        return f"{self.scheme}://{host}:{port}{path}"

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, url: str, timeout: float, json: dict | None = None) -> dict:
        return _request(method, url, timeout, json=json, headers=self._headers(), verify=self.verify)


def _request(
    method: str,
    url: str,
    timeout: float,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
    verify: bool | str = True,
) -> dict:
    try:
        response = httpx.request(method, url, json=json, headers=headers, timeout=timeout, verify=verify)
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
