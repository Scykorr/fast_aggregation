"""HTTP координатора: дашборд, реестр и прокси OpenAI API к llama-server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from fast_aggregation.constants import DEFAULT_VRAM_RESERVE_MB
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.app import RpcStartBody

_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


class ManualNodeBody(BaseModel):
    address: str = Field(min_length=1)


class StartBody(BaseModel):
    model_path: str | None = None
    hf_repo: str | None = None
    ctx_size: int | None = None
    parallel: int = 1
    allow_cpu: bool = False
    reserve_mb: int = DEFAULT_VRAM_RESERVE_MB


class PlanQuery(BaseModel):
    allow_cpu: bool = False
    reserve_mb: int = DEFAULT_VRAM_RESERVE_MB
    model_path: str | None = None
    hf_repo: str | None = None
    ctx_size: int | None = None
    parallel: int = 1


def create_coordinator_app(
    service: CoordinatorService,
    discovery: DiscoveryService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        service.start_background()
        if discovery is not None:
            discovery.start()
        try:
            yield
        finally:
            if discovery is not None:
                discovery.stop()
            service.shutdown()

    app = FastAPI(title="fastagg coordinator", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _INDEX.read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "phase": service.phase}

    @app.get("/api/state")
    def state() -> dict:
        return service.state()

    @app.post("/api/plan")
    def plan(body: PlanQuery) -> dict:
        try:
            placement = service.preview_plan(allow_cpu=body.allow_cpu, reserve_mb=body.reserve_mb)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "plan": placement.to_dict(),
            "command": service.command_preview(
                placement,
                model_path=body.model_path,
                hf_repo=body.hf_repo,
                ctx_size=body.ctx_size,
                parallel=body.parallel,
            ),
        }

    @app.post("/api/nodes")
    def add_node(body: ManualNodeBody) -> dict:
        try:
            return service.add_manual(body.address)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/nodes/{node_id}")
    def delete_node(node_id: str) -> dict[str, bool]:
        try:
            service.remove_node(node_id)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/cluster/start")
    def start_cluster(body: StartBody) -> dict:
        try:
            return service.start_cluster(
                model_path=_blank(body.model_path),
                hf_repo=_blank(body.hf_repo),
                ctx_size=body.ctx_size,
                parallel=body.parallel,
                allow_cpu=body.allow_cpu,
                reserve_mb=body.reserve_mb,
            )
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cluster/stop")
    def stop_cluster() -> dict:
        return service.stop_cluster()

    if service.node is not None:
        node = service.node

        @app.get("/v1/node")
        def local_node() -> dict:
            node.refresh_resources()
            described = node.describe()
            service.apply_payload(described, manual=False)
            return described

        @app.post("/v1/rpc/start")
        def local_rpc_start(body: RpcStartBody) -> dict:
            try:
                result = node.start_rpc(body.release, body.backend, body.cache)
            except FastAggError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            service.publish_local()
            return result

        @app.post("/v1/rpc/stop")
        def local_rpc_stop() -> dict:
            result = node.stop_rpc()
            service.publish_local()
            return result

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_openai(path: str, request: Request) -> StreamingResponse:
        if service.phase != "running":
            raise HTTPException(
                status_code=503,
                detail="Кластер не запущен. Откройте дашборд и запустите модель.",
            )
        url = f"http://127.0.0.1:{service.engine_port}/v1/{path}"
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        client = httpx.AsyncClient(timeout=None)
        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    url,
                    content=body,
                    headers=headers,
                    params=request.query_params,
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"llama-server недоступен: {exc}") from exc

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        media = upstream.headers.get("content-type")
        return StreamingResponse(chunks(), status_code=upstream.status_code, media_type=media)

    return app


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
