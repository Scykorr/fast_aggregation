"""Оркестрация: опрос узлов, план памяти, запуск RPC и llama-server."""

from __future__ import annotations

import platform
import threading
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import httpx

from fast_aggregation.cluster.planner import plan_placement
from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.constants import DEFAULT_LLAMA_RELEASE, ENGINE_PORT
from fast_aggregation.coordinator.client import HttpNodeClient, NodeClient
from fast_aggregation.errors import FastAggError, RuntimeError_
from fast_aggregation.models import PlacementPlan, resources_from_dict
from fast_aggregation.node.service import NodeService
from fast_aggregation.resources.probe import normalize_arch
from fast_aggregation.runtime.commands import llama_server_argv, preview_command
from fast_aggregation.runtime.downloads import ensure_runtime, executable
from fast_aggregation.runtime.processes import ManagedProcess
from fast_aggregation.runtime.releases import engine_backend

ProcessFactory = Callable[[list[str], Path | None, Path], ManagedProcess]
HealthCheck = Callable[[], bool]


class CoordinatorService:
    def __init__(
        self,
        *,
        registry: NodeRegistry,
        data_dir: Path,
        release: str = DEFAULT_LLAMA_RELEASE,
        engine_port: int = ENGINE_PORT,
        node: NodeService | None = None,
        client: NodeClient | None = None,
        background: bool = True,
        process_factory: ProcessFactory | None = None,
        health_check: HealthCheck | None = None,
        runtime_fetcher: Callable[[str, str, str, str], Path] | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        self.registry = registry
        self.data_dir = data_dir
        self.release = release
        self.engine_port = engine_port
        self.node = node
        self.client = client or HttpNodeClient()
        self.background = background
        self._process_factory = process_factory or _default_process
        self._health_check = health_check
        self._runtime_fetcher = runtime_fetcher
        self.poll_interval = poll_interval
        self.phase = "idle"
        self.error: str | None = None
        self.plan: PlacementPlan | None = None
        self.engine: ManagedProcess | None = None
        self._started_remote: list[tuple[str, int, str]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        self.publish_local()
        if not self.background or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="fa-poll", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        try:
            self.stop_cluster()
        except FastAggError:
            pass
        if self.node is not None:
            self.node.stop_rpc()

    def publish_local(self) -> None:
        if self.node is None:
            return
        described = self.node.describe()
        self.apply_payload(described, manual=False)

    def apply_payload(self, payload: dict, *, manual: bool) -> None:
        resources = None
        if isinstance(payload.get("resources"), dict):
            resources = resources_from_dict(payload["resources"])
        rpc = payload.get("rpc") if isinstance(payload.get("rpc"), dict) else {}
        node_id = str(payload.get("node_id") or "")
        if not node_id:
            raise FastAggError("Ответ узла не содержит node_id")
        self.registry.upsert_details(
            node_id,
            name=str(payload.get("name") or node_id),
            host=str(payload.get("advertise_host") or payload.get("host") or ""),
            http_port=int(payload.get("http_port") or 0),
            rpc_port=int(payload.get("rpc_port") or 0),
            resources=resources,
            preferred_backend=str(payload.get("preferred_backend") or "auto"),
            rpc_running=bool(rpc.get("running")),
            rpc_backend=rpc.get("backend"),
            rpc_error=rpc.get("error") or "",
            manual=manual,
        )

    def add_manual(self, address: str) -> dict:
        host, port = parse_address(address)
        payload = self.client.fetch(host, port)
        payload["advertise_host"] = host
        payload["http_port"] = port
        self.apply_payload(payload, manual=True)
        node_id = str(payload.get("node_id"))
        tracked = self.registry.get(node_id)
        if tracked is None:
            raise FastAggError("Узел не попал в реестр")
        return tracked.to_public(time.time(), self.registry.ttl)

    def remove_node(self, node_id: str) -> None:
        if self.node is not None and node_id == self.node.settings.node_id:
            raise FastAggError("Локальный узел координатора остаётся в списке, пока координатор запущен")
        if not self.registry.remove(node_id):
            raise FastAggError("Узел не найден")

    def preview_plan(self, *, allow_cpu: bool, reserve_mb: int) -> PlacementPlan:
        return plan_placement(
            self.registry.cluster_nodes(),
            allow_cpu=allow_cpu,
            reserve_mb=reserve_mb,
        )

    def command_preview(
        self,
        plan: PlacementPlan,
        *,
        model_path: str | None,
        hf_repo: str | None,
        ctx_size: int | None,
        parallel: int,
    ) -> str | None:
        if not model_path and not hf_repo:
            return None
        binary = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        try:
            argv = llama_server_argv(
                Path(binary),
                plan,
                model_path=model_path,
                hf_repo=hf_repo,
                port=self.engine_port,
                ctx_size=ctx_size,
                parallel=parallel,
                local_node_id=self.node.settings.node_id if self.node else None,
            )
        except ValueError:
            return None
        return preview_command(argv)

    def start_cluster(
        self,
        *,
        model_path: str | None,
        hf_repo: str | None,
        ctx_size: int | None,
        parallel: int,
        allow_cpu: bool,
        reserve_mb: int,
    ) -> dict:
        with self._lock:
            if self.phase in {"starting", "stopping"}:
                raise FastAggError("Кластер уже меняет состояние")
            if self.phase == "running":
                raise FastAggError("Кластер уже запущен. Сначала остановите его.")
            self.phase = "starting"
            self.error = None
            self._started_remote = []
            try:
                self._validate_model(model_path, hf_repo)
                if self.node is not None:
                    self.node.refresh_resources()
                    self.publish_local()
                plan = self.preview_plan(allow_cpu=allow_cpu, reserve_mb=reserve_mb)
                self.plan = plan
                self._start_workers(plan)
                binary = self._ensure_engine_binary()
                argv = llama_server_argv(
                    binary,
                    plan,
                    model_path=model_path,
                    hf_repo=hf_repo,
                    port=self.engine_port,
                    ctx_size=ctx_size,
                    parallel=parallel,
                    local_node_id=self.node.settings.node_id if self.node else None,
                )
                log_path = self.data_dir / "logs" / "llama-server.log"
                process = self._process_factory(argv, binary.parent, log_path)
                process.start()
                self.engine = process
                if not self._wait_until_healthy():
                    tail = "\n".join(process.tail(40))
                    code = process.exit_code()
                    raise RuntimeError_(
                        "llama-server не ответил на /health. "
                        + (f"Код выхода: {code}. " if code not in (None, 0) else "")
                        + tail
                    )
                self.phase = "running"
                return self.state()
            except Exception as exc:
                self.error = str(exc)
                self.phase = "error"
                self._stop_processes()
                if isinstance(exc, FastAggError):
                    raise
                raise RuntimeError_(str(exc)) from exc

    def stop_cluster(self) -> dict:
        with self._lock:
            self.phase = "stopping"
            self._stop_processes()
            self.phase = "idle"
            self.plan = None
            self.error = None
            return self.state()

    def state(self) -> dict:
        engine = None
        if self.engine is not None:
            engine = {
                "running": self.engine.running(),
                "pid": self.engine.pid(),
                "port": self.engine_port,
                "args": list(self.engine.args),
                "log_tail": self.engine.tail(80),
            }
        plan = self.plan.to_dict() if self.plan else None
        return {
            "phase": self.phase,
            "error": self.error,
            "participate": self.node is not None,
            "release": self.release,
            "engine_port": self.engine_port,
            "local_node_id": self.node.settings.node_id if self.node else None,
            "plan": plan,
            "nodes": self.registry.public_nodes(),
            "engine": engine,
        }

    def _start_workers(self, plan: PlacementPlan) -> None:
        launched: set[str] = set()
        for device in plan.devices:
            if device.node_id in launched:
                continue
            launched.add(device.node_id)
            if self.node is not None and device.node_id == self.node.settings.node_id:
                self.node.start_rpc(self.release, device.backend)
                self.publish_local()
                continue
            tracked = self.registry.get(device.node_id)
            if tracked is None or not tracked.host or not tracked.http_port:
                raise FastAggError(f"Узел {device.node_name} не имеет HTTP-адреса")
            self.client.start_rpc(
                tracked.host,
                tracked.http_port,
                {"release": self.release, "backend": device.backend, "cache": True},
            )
            self._started_remote.append((tracked.host, tracked.http_port, device.node_id))
            self.registry.upsert_details(
                device.node_id,
                rpc_running=True,
                rpc_backend=device.backend,
                rpc_error="",
                seen=False,
            )

    def _ensure_engine_binary(self) -> Path:
        system = self.node.resources.os_name if self.node else platform.system()
        arch = self.node.resources.arch if self.node else normalize_arch(platform.machine())
        backend = engine_backend(system)
        if self._runtime_fetcher is not None:
            folder = self._runtime_fetcher(self.release, system, arch, backend)
        else:
            folder = ensure_runtime(
                self.release,
                system,
                arch,
                backend,
                self.data_dir / "runtimes",
                role="server",
            )
        return executable(folder, system, "server")

    def _wait_until_healthy(self) -> bool:
        if self._health_check is not None:
            return self._health_check()
        deadline = time.monotonic() + 180
        url = f"http://127.0.0.1:{self.engine_port}/health"
        while time.monotonic() < deadline:
            if self.engine is not None and not self.engine.running() and self.engine.exit_code() not in (None, 0):
                return False
            try:
                response = httpx.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.4)
        return False

    def _stop_processes(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        if self.node is not None:
            self.node.stop_rpc()
            self.publish_local()
        for host, port, node_id in reversed(self._started_remote):
            try:
                self.client.stop_rpc(host, port)
                self.registry.upsert_details(node_id, rpc_running=False, rpc_backend="", seen=False)
            except FastAggError as exc:
                self.registry.mark_error(node_id, str(exc))
        self._started_remote = []

    def _validate_model(self, model_path: str | None, hf_repo: str | None) -> None:
        if bool(model_path) == bool(hf_repo):
            raise FastAggError("Укажите либо путь к GGUF на координаторе, либо репозиторий Hugging Face")
        if model_path and not Path(model_path).is_file():
            raise FastAggError(
                f"Файл модели не найден на координаторе: {model_path}. "
                "GGUF должен лежать на машине координатора, воркеры получают веса по RPC."
            )

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception:
                continue

    def _poll_once(self) -> None:
        if self.node is not None:
            self.node.refresh_resources()
            self.publish_local()
        for node in self.registry.cluster_nodes():
            if self.node is not None and node.node_id == self.node.settings.node_id:
                continue
            if not node.host or not node.http_port:
                continue
            try:
                payload = self.client.fetch(node.host, node.http_port)
            except FastAggError as exc:
                self.registry.mark_error(node.node_id, str(exc))
                continue
            manual = node.source == "manual"
            payload["advertise_host"] = node.host
            payload["http_port"] = node.http_port
            self.apply_payload(payload, manual=manual)


def parse_address(value: str) -> tuple[str, int]:
    text = value.strip()
    if not text:
        raise FastAggError("Пустой адрес узла")
    if "://" in text:
        parsed = urllib.parse.urlparse(text)
        host = parsed.hostname
        port = parsed.port
    elif text.count(":") == 1:
        host, _, port_text = text.partition(":")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise FastAggError("Порт в адресе должен быть числом") from exc
    else:
        raise FastAggError("Адрес нужен в виде host:port")
    if not host or port is None or not (1 <= port <= 65535):
        raise FastAggError("Адрес нужен в виде host:port")
    return host, port


def _default_process(argv: list[str], cwd: Path | None, log_path: Path) -> ManagedProcess:
    return ManagedProcess(argv, cwd=cwd, log_path=log_path)
