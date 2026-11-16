"""HTTP узла: снимок ресурсов и управление локальным RPC."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from fast_aggregation.constants import SOFTWARE
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.service import NodeService


class RpcStartBody(BaseModel):
    release: str = Field(min_length=1)
    backend: str | None = None
    cache: bool = True


class PairOfferBody(BaseModel):
    coordinator: str = Field(min_length=1)
    code: str = Field(min_length=6, max_length=8)
    fingerprint: str = Field(min_length=1)


class PairJoinBody(BaseModel):
    address: str = Field(min_length=1)
    code: str = Field(min_length=6, max_length=8)


_PAIR_PAGE = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>fastagg узел</title>
<style>
body { font: 15px/1.45 system-ui, sans-serif; background:#10140f; color:#e7f0e2; margin:24px; }
button { background:#c6f27a; color:#17200f; border:0; border-radius:8px; padding:8px 12px; font-weight:700; }
input { background:#0e130d; color:#e7f0e2; border:1px solid #2d3a2a; border-radius:8px; padding:8px; width:100%; }
.card { border:1px solid #2d3a2a; border-radius:12px; padding:12px; margin:10px 0; }
</style></head><body>
<h1>Подключиться к координатору</h1>
<div id="pending"></div>
<div id="list"></div>
<form id="manual">
<label>Адрес координатора<br><input id="host" placeholder="10.0.0.25:47080"></label>
<label>Код<br><input id="code" maxlength="8"></label>
<button type="submit">Сопрячь</button>
</form>
<p id="msg"></p>
<script>
async function load() {
  const response = await fetch("/v1/coordinators");
  const data = await response.json();
  const pending = data.pending;
  document.querySelector("#pending").innerHTML = pending
    ? `<div class="card">Координатор прислал код <b>${pending.code}</b> с адреса ${pending.coordinator}. Сверьте его с экраном координатора. <button id="accept" type="button">Принять</button> <button id="reject" type="button">Отклонить</button></div>`
    : "";
  const accept = document.querySelector("#accept");
  const reject = document.querySelector("#reject");
  if (accept) accept.onclick = async () => { await fetch("/v1/pair/accept", {method:"POST"}); load(); };
  if (reject) reject.onclick = async () => { await fetch("/v1/pair/reject", {method:"POST"}); load(); };
  document.querySelector("#list").innerHTML = (data.coordinators || []).map((item) =>
    `<div class="card">${item.name} · ${item.host}:${item.port}<br>${item.summary || ""}<br><button type="button" data-host="${item.host}" data-port="${item.port}">Выбрать</button></div>`
  ).join("") || "<p>Координаторы в эфире пока не видны. Адрес можно ввести вручную.</p>";
}
document.querySelector("#list").addEventListener("click", (event) => {
  const button = event.target;
  if (!button.dataset || !button.dataset.host) return;
  document.querySelector("#host").value = button.dataset.host + ":" + button.dataset.port;
});
document.querySelector("#manual").addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/v1/pair/join", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({address: document.querySelector("#host").value, code: document.querySelector("#code").value})});
  const data = await response.json();
  document.querySelector("#msg").textContent = response.ok ? "Сопряжено" : (data.detail || "Не вышло");
});
load();
</script></body></html>
"""


def create_node_app(
    service: NodeService,
    discovery: DiscoveryService | None = None,
    token: str | None = None,
    session=None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if discovery is not None:
            discovery.start()
        try:
            yield
        finally:
            service.stop_rpc()
            if discovery is not None:
                discovery.stop()

    app = FastAPI(title="fastagg node", lifespan=lifespan)

    auth = session.auth if session is not None else {"token": token or ""}

    def guard(request: Request) -> None:
        expected = auth.get("token") or ""
        if not expected:
            return
        if request.headers.get("authorization") != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Нужен токен кластера")

    def _local(request: Request) -> bool:
        host = request.client.host if request.client else ""
        return host in {"127.0.0.1", "::1", "testclient"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "software": SOFTWARE, "paired": "yes" if auth.get("token") else "no"}

    @app.get("/hello")
    def hello() -> dict:
        from fast_aggregation.node.pairing import hardware_summary

        service.refresh_resources()
        body = service.describe()
        body["paired"] = bool(auth.get("token"))
        body["role"] = "node"
        body["summary"] = hardware_summary(service.resources)
        return body

    @app.get("/pair", response_class=HTMLResponse)
    def pair_page(request: Request) -> str:
        if not _local(request):
            raise HTTPException(status_code=403, detail="Страница сопряжения только с этого компьютера")
        return _PAIR_PAGE

    @app.get("/v1/coordinators")
    def coordinators(request: Request) -> dict:
        if not _local(request):
            raise HTTPException(status_code=403, detail="Список координаторов только с этого компьютера")
        found = list(session.coordinators.values()) if session is not None else []
        return {"coordinators": found, "pending": session.pending if session is not None else None}

    @app.post("/v1/pair/offer")
    def pair_offer(body: PairOfferBody) -> dict:
        if session is None:
            raise HTTPException(status_code=400, detail="Узел не принимает предложения")
        try:
            return session.offer(body.model_dump())
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/pair/accept")
    def pair_accept(request: Request) -> dict:
        if not _local(request):
            raise HTTPException(status_code=403, detail="Принять сопряжение можно только на этом компьютере")
        if session is None:
            raise HTTPException(status_code=400, detail="Узел не принимает предложения")
        try:
            return session.accept()
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/pair/join")
    def pair_join(body: PairJoinBody, request: Request) -> dict:
        if not _local(request):
            raise HTTPException(status_code=403, detail="Сопряжение с этой страницы только на самом узле")
        if session is None:
            raise HTTPException(status_code=400, detail="Узел не принимает предложения")
        host, port = _split_host(body.address)
        known = session.coordinators.get(f"{host}:{port}") or {}
        fingerprint = str(known.get("fingerprint") or "")
        if not fingerprint:
            fingerprint = _fingerprint_of(host, port)
        try:
            result = session.offer(
                {"coordinator": f"https://{host}:{port}", "code": body.code, "fingerprint": fingerprint}
            )
            if result.get("status") == "paired":
                return result
            return session.accept()
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/pair/reject")
    def pair_reject(request: Request) -> dict:
        if not _local(request):
            raise HTTPException(status_code=403, detail="Отклонить можно только на этом компьютере")
        if session is None:
            return {"status": "idle"}
        return session.reject()

    @app.get("/v1/node")
    def node(request: Request) -> dict:
        guard(request)
        service.refresh_resources()
        return service.describe()

    @app.get("/v1/bench")
    def bench(request: Request, nbytes: int = 262144) -> Response:
        guard(request)
        raw = request.query_params.get("bytes")
        if raw is not None:
            nbytes = int(raw)
        size = max(1, min(int(nbytes), 8 * 1024 * 1024))
        return Response(content=b"\0" * size, media_type="application/octet-stream")

    @app.post("/v1/rpc/start")
    def start_rpc(body: RpcStartBody, request: Request) -> dict:
        guard(request)
        try:
            return service.start_rpc(body.release, body.backend, body.cache)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rpc/stop")
    def stop_rpc(request: Request) -> dict:
        guard(request)
        return service.stop_rpc()

    return app


def _split_host(address: str) -> tuple[str, int]:
    text = address.strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0]
    if ":" not in text:
        raise HTTPException(status_code=400, detail="Нужен адрес вида 10.0.0.25:47080")
    host, raw_port = text.rsplit(":", 1)
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Порт в адресе не число") from exc
    return host, port


def _fingerprint_of(host: str, port: int) -> str:
    from fast_aggregation.security.certs import fingerprint_pem
    from fast_aggregation.security.pin import _peer_pem

    try:
        pem = _peer_pem(host, port)
    except FastAggError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return fingerprint_pem(pem)
