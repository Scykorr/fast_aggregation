"""HTTP узла: снимок ресурсов и управление локальным RPC."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.service import NodeService


class RpcStartBody(BaseModel):
    release: str = Field(min_length=1)
    backend: str | None = None
    cache: bool = True


def create_node_app(service: NodeService, discovery: DiscoveryService | None = None) -> FastAPI:
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

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/node")
    def node() -> dict:
        service.refresh_resources()
        return service.describe()

    @app.post("/v1/rpc/start")
    def start_rpc(body: RpcStartBody) -> dict:
        try:
            return service.start_rpc(body.release, body.backend, body.cache)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rpc/stop")
    def stop_rpc() -> dict:
        return service.stop_rpc()

    return app
