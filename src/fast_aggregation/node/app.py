"""HTTP узла: снимок ресурсов и управление локальным RPC."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from fast_aggregation.constants import SOFTWARE
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.service import NodeService


class RpcStartBody(BaseModel):
    release: str = Field(min_length=1)
    backend: str | None = None
    cache: bool = True


def create_node_app(
    service: NodeService,
    discovery: DiscoveryService | None = None,
    token: str | None = None,
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

    def guard(request: Request) -> None:
        if not token:
            return
        if request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Нужен токен кластера")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "software": SOFTWARE}

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
