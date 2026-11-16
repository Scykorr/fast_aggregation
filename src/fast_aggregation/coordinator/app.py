"""HTTP координатора: дашборд, реестр и прокси OpenAI API к llama-server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
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


class PairBody(BaseModel):
    code: str = Field(min_length=1)
    node_id: str = ""
    name: str = ""
    advertise_host: str = ""
    http_port: int = 0
    rpc_port: int = 0
    cert_pem: str = ""


class LimitsBody(BaseModel):
    excluded: bool = False
    max_temp_c: float | None = None
    min_battery_percent: float | None = None
    memory_reserve_mb: int | None = None
    memory_cap_mb: int | None = None


class ModelQuery(BaseModel):
    path: str
    ctx_size: int = 4096
    allow_cpu: bool = False
    reserve_mb: int = DEFAULT_VRAM_RESERVE_MB


class ScanBody(BaseModel):
    cidr: str = Field(min_length=1)
    port: int = 47000


class HfInspectBody(BaseModel):
    url: str = Field(min_length=1)
    token: str = ""


class HfTokenBody(BaseModel):
    token: str = ""


class HfPullBody(BaseModel):
    files: list[dict]


class OfferBody(BaseModel):
    host: str = Field(min_length=1)
    port: int = 47000


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
    def state(request: Request) -> dict:
        host = request.client.host if request.client else ""
        return service.state(reveal_pair=host in {"127.0.0.1", "::1"})

    @app.post("/api/pair")
    def pair(body: PairBody) -> dict:
        if not service.pair_code_ok(body.code):
            raise HTTPException(status_code=401, detail="Неверный код сопряжения")
        if body.cert_pem:
            _remember_node_cert(service, body.node_id or body.name or "node", body.cert_pem)
        if body.node_id and body.advertise_host and body.http_port:
            service.registry.upsert_details(
                body.node_id,
                name=body.name or body.node_id,
                host=body.advertise_host,
                http_port=body.http_port,
                rpc_port=body.rpc_port,
                manual=True,
            )
        return {"token": service.token, "fingerprint": service.fingerprint}

    @app.get("/api/pair/qr.svg")
    def pair_qr(request: Request, host: str = "") -> Response:
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="QR сопряжения доступен только с этого компьютера")
        view = service.state(reveal_pair=True).get("pairing")
        if not view:
            raise HTTPException(status_code=404, detail="Сопряжение выключено")
        import segno

        from fast_aggregation.security.token import join_uri

        links = view.get("links") or []
        chosen = host if any(item.get("host") == host for item in links) else ""
        if not chosen:
            chosen = links[0]["host"] if links else service.public_host
        uri = join_uri(chosen, service.public_port, view["code"], view["fingerprint"])
        svg = segno.make(uri).svg_inline(scale=4)
        return Response(content=svg, media_type="image/svg+xml")

    @app.post("/api/pair/rotate")
    def rotate_pair(request: Request) -> dict:
        host = request.client.host if request.client else ""
        if host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="Новый код можно выпустить только на этом компьютере")
        return service.rotate_pairing_code() or {}

    @app.get("/api/capacity")
    def capacity(ctx: int = 4096, allow_cpu: bool = False, reserve_mb: int = DEFAULT_VRAM_RESERVE_MB) -> dict:
        return service.capacity_report(allow_cpu=allow_cpu, reserve_mb=reserve_mb, ctx=max(0, ctx))

    @app.get("/api/models")
    def models(ctx: int = 4096, allow_cpu: bool = False, reserve_mb: int = DEFAULT_VRAM_RESERVE_MB) -> dict:
        return {
            "directory": str(service.models_dir) if service.models_dir else None,
            "models": service.list_local_models(ctx, allow_cpu, reserve_mb),
        }

    @app.get("/api/models/hf/token")
    def hf_token_state() -> dict:
        from fast_aggregation.models_pull import load_token

        return {"saved": bool(load_token(service.data_dir))}

    @app.post("/api/models/hf/token")
    def hf_token_save(body: HfTokenBody) -> dict:
        from fast_aggregation.models_pull import save_token

        save_token(service.data_dir, body.token)
        return {"saved": bool(body.token.strip())}

    @app.post("/api/models/hf/inspect")
    def hf_inspect(body: HfInspectBody) -> dict:
        from fast_aggregation.models_pull import save_token

        if body.token.strip():
            save_token(service.data_dir, body.token)
        try:
            return service.model_puller().inspect(body.url)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/models/pull")
    def hf_pull(body: HfPullBody) -> dict:
        from fast_aggregation.models_pull import files_from_payload

        try:
            files = files_from_payload(body.files)
            return service.model_puller().start(files)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/models/pull")
    def hf_pull_state() -> dict:
        puller = getattr(service, "_model_puller", None)
        if puller is None:
            return {"status": "idle", "directory": str(service.models_dir) if service.models_dir else None}
        return puller.view()

    @app.post("/api/models/pull/pause")
    def hf_pause() -> dict:
        return service.model_puller().pause()

    @app.post("/api/models/pull/resume")
    def hf_resume() -> dict:
        return service.model_puller().resume()

    @app.post("/api/models/pull/cancel")
    def hf_cancel() -> dict:
        return service.model_puller().cancel()

    @app.post("/api/models/preview")
    def preview_model(body: ModelQuery) -> dict:
        return service.describe_model(body.path, ctx=body.ctx_size, allow_cpu=body.allow_cpu, reserve_mb=body.reserve_mb)

    @app.post("/api/nodes/{node_id}/limits")
    def limits(node_id: str, body: LimitsBody) -> dict:
        try:
            saved = service.set_limits(node_id, body.model_dump())
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return saved

    @app.get("/api/engine")
    def engine() -> dict:
        return service.engine_snapshot()

    @app.get("/runtime/{asset_name}")
    def runtime_asset(asset_name: str, request: Request) -> FileResponse:
        if service.token:
            if request.headers.get("authorization") != f"Bearer {service.token}":
                raise HTTPException(status_code=401, detail="Нужен токен кластера")
        from fast_aggregation.runtime.downloads import find_staged_archive

        found = find_staged_archive(service.data_dir / "runtimes", asset_name)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail="Сборка не подготовлена на координаторе. Импортируйте архив локально.",
            )
        return FileResponse(found, filename=asset_name)

    @app.post("/api/scan")
    def scan(body: ScanBody) -> dict:
        try:
            return service.begin_scan(body.cidr, body.port)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/scan")
    def scan_state() -> dict:
        return service.scan_view()

    @app.post("/api/nodes/offer")
    def offer_node(body: OfferBody) -> dict:
        try:
            return service.push_offer(body.host, body.port)
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
                detail="Кластер не запущен. На вкладке «Модели» выберите локальный GGUF и нажмите «Запустить».",
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


def _remember_node_cert(service: CoordinatorService, node_id: str, pem: str) -> None:
    if "BEGIN CERTIFICATE" not in pem:
        raise HTTPException(status_code=400, detail="Сертификат узла должен быть в PEM")
    bundle = getattr(service, "trusted_bundle", None)
    if bundle is None:
        return
    path = service.data_dir / "trusted" / f"{node_id}.pem"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pem, encoding="utf-8")
    existing = Path(bundle).read_text(encoding="utf-8") if Path(bundle).is_file() else ""
    if pem.strip() not in existing:
        with Path(bundle).open("a", encoding="utf-8") as handle:
            handle.write("\n" + pem.strip() + "\n")


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
