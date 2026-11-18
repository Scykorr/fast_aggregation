"""Состояние одного устройства: снимок ресурсов и процесс RPC."""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fast_aggregation.version import full_version
from fast_aggregation.cluster.planner import choose_backend
from fast_aggregation.constants import SOFTWARE
from fast_aggregation.errors import FastAggError, RuntimeError_
from fast_aggregation.models import Announce, MachineResources
from fast_aggregation.resources.probe import apply_vram_override, probe_machine
from fast_aggregation.runtime.commands import rpc_server_argv
from fast_aggregation.runtime.downloads import ensure_runtime, executable
from fast_aggregation.runtime.ports import wait_for_port
from fast_aggregation.runtime.processes import ManagedProcess

Prober = Callable[[], MachineResources]
RuntimeFetcher = Callable[[str, str], Path]
ProcessFactory = Callable[[list[str], Path | None, Path], ManagedProcess]
PortReady = Callable[[int], bool]


@dataclass
class NodeSettings:
    node_id: str
    name: str
    http_port: int
    rpc_port: int
    advertise_host: str
    data_dir: Path
    preferred_backend: str = "auto"
    vram_override_mb: int | None = None
    cuda_build: str | None = None
    release: str = ""
    runtime_mirror: str | None = None
    allow_download: bool = False
    bind_host: str = ""


class NodeService:
    def __init__(
        self,
        settings: NodeSettings,
        *,
        prober: Prober | None = None,
        runtime_fetcher: RuntimeFetcher | None = None,
        process_factory: ProcessFactory | None = None,
        port_ready: PortReady | None = None,
    ) -> None:
        self.settings = settings
        self._prober = prober
        self._runtime_fetcher = runtime_fetcher
        self._process_factory = process_factory or _default_process
        self._port_ready = port_ready or (lambda port: wait_for_port(port, timeout=25))
        self._lock = threading.Lock()
        self._process: ManagedProcess | None = None
        self._deep_probe = prober is None
        self.rpc_backend: str | None = None
        self.rpc_error: str | None = None
        self.cuda_build_used: str | None = None
        self.cuda_reason: str | None = None
        self._aux: dict[str, dict] = {}
        self.resources = self.refresh_resources()

    def refresh_resources(self) -> MachineResources:
        if self._prober is not None:
            resources = self._prober()
        else:
            resources = probe_machine(deep_vulkan=self._deep_probe)
            self._deep_probe = False
        apply_vram_override(resources, self.settings.vram_override_mb)
        self.resources = resources
        return resources

    def announce(self) -> Announce:
        return Announce(
            node_id=self.settings.node_id,
            name=self.settings.name,
            advertise_host=self.settings.advertise_host,
            http_port=self.settings.http_port,
            rpc_port=self.settings.rpc_port,
            os_name=self.resources.os_name,
            arch=self.resources.arch,
            preferred_backend=self.settings.preferred_backend,
            software=SOFTWARE,
            version=full_version(),
        )

    def describe(self) -> dict:
        process = self._process
        running = bool(process and process.running())
        return {
            "node_id": self.settings.node_id,
            "name": self.settings.name,
            "software": SOFTWARE,
            "version": full_version(),
            "advertise_host": self.settings.advertise_host,
            "http_port": self.settings.http_port,
            "rpc_port": self.settings.rpc_port,
            "preferred_backend": self.settings.preferred_backend,
            "resources": self.resources.to_dict(),
            "rpc": {
                "running": running,
                "backend": self.rpc_backend,
                "cuda_build": self.cuda_build_used,
                "cuda_reason": self.cuda_reason,
                "error": self.rpc_error,
                "log_tail": process.tail(40) if process else [],
            },
        }

    def start_rpc(self, release: str, backend: str | None = None, cache: bool = True) -> dict:
        with self._lock:
            self.refresh_resources()
            chosen = choose_backend(
                self.resources,
                backend or self.settings.preferred_backend,
                self.resources.os_name,
            )
            cuda_build = self._resolve_cuda_build(chosen)
            folder = self._fetch_runtime(release, chosen, cuda_build)
            binary = executable(folder, self.resources.os_name, "rpc")
            self._stop_locked()
            log_path = self.settings.data_dir / "logs" / "rpc-server.log"
            host = self.settings.bind_host or "0.0.0.0"
            argv = rpc_server_argv(binary, host=host, port=self.settings.rpc_port, cache=cache)
            process = self._process_factory(argv, binary.parent, log_path)
            process.start()
            self._process = process
            if not self._port_ready(self.settings.rpc_port):
                tail = "\n".join(process.tail(30))
                self.rpc_error = "ggml-rpc-server не открыл порт. " + tail
                code = process.exit_code()
                self._stop_locked()
                suffix = f" Код выхода: {code}." if code not in (None, 0) else ""
                raise RuntimeError_(self.rpc_error + suffix)
            # Порт мог быть занят чужим процессом: наш сервер тогда падает
            # сразу после старта, а проверка порта всё равно успешна.
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline and process.running() and not _rpc_failed(process.tail(40)):
                time.sleep(0.1)
            if not process.running() or _rpc_failed(process.tail(40)):
                tail = "\n".join(process.tail(40))
                code = process.exit_code()
                self.rpc_error = "ggml-rpc-server не удержался на порту. " + tail
                self._stop_locked()
                suffix = f" Код выхода: {code}." if code not in (None, 0) else ""
                raise RuntimeError_(self.rpc_error + suffix)
            self.rpc_backend = chosen
            self.rpc_error = None
            return self.describe()["rpc"]

    def start_aux(self, role: str, model_path: str, port: int, extra_args: list[str]) -> dict:
        folder = self._fetch_runtime(self.settings.release or "b12359", self.settings.preferred_backend or "cpu")
        binary = executable(folder, self.resources.os_name, "server")
        argv = [str(binary), "--host", "0.0.0.0", "--port", str(port), "--model", model_path, *extra_args]
        log_path = self.settings.data_dir / "logs" / f"aux-{role}.log"
        process = self._process_factory(argv, binary.parent, log_path)
        process.start()
        if not self._port_ready(port):
            process.stop()
            raise RuntimeError_(f"llama-server роли {role} не открыл порт {port}")
        self._aux[role] = {"process": process, "port": port, "model": model_path, "running": True}
        return {"role": role, "port": port, "running": True, "model": model_path}

    def stop_aux(self, role: str) -> dict:
        item = self._aux.pop(role, None)
        if item and item.get("process") is not None:
            item["process"].stop()
        return {"role": role, "running": False}

    def aux_status(self) -> dict:
        return {
            role: {"port": item["port"], "model": item["model"], "running": bool(item.get("process") and item["process"].running())}
            for role, item in self._aux.items()
        }

    def stop_rpc(self) -> dict:
        with self._lock:
            self._stop_locked()
            self.rpc_backend = None
            return {"running": False, "backend": None, "error": self.rpc_error}

    def _stop_locked(self) -> None:
        if self._process is not None:
            self._process.stop()
            self._process = None

    def _resolve_cuda_build(self, backend: str) -> str | None:
        if backend != "cuda":
            return None
        from fast_aggregation.runtime.cuda_select import select_windows_cuda_build

        preference = self.settings.cuda_build or "auto"
        if self.resources.os_name.lower() != "windows" or self.resources.arch != "x64":
            return None if preference in {"", "auto"} else preference
        driver = next((gpu.driver for gpu in self.resources.gpus if gpu.vendor == "nvidia" and gpu.driver), None)
        choice = select_windows_cuda_build(preference, driver)
        self.cuda_build_used = choice.build
        self.cuda_reason = choice.reason
        return choice.build

    def _fetch_runtime(self, release: str, backend: str, cuda_build: str | None = None) -> Path:
        if self._runtime_fetcher is not None:
            return self._runtime_fetcher(release, backend)
        try:
            return ensure_runtime(
                release,
                self.resources.os_name,
                self.resources.arch,
                backend,
                self.settings.data_dir / "runtimes",
                role="rpc",
                cuda_build=cuda_build,
                allow_download=self.settings.allow_download,
                mirror=self.settings.runtime_mirror,
            )
        except FastAggError:
            raise
        except OSError as exc:
            raise RuntimeError_(f"Не удалось подготовить сборку llama.cpp: {exc}") from exc


def _default_process(argv: list[str], cwd: Path | None, log_path: Path) -> ManagedProcess:
    return ManagedProcess(argv, cwd=cwd, log_path=log_path)


def default_node_name() -> str:
    return platform.node() or "node"


def _rpc_failed(lines: list[str]) -> bool:
    text = "\n".join(lines)
    markers = (
        "Failed to create server socket",
        "No devices found",
        "Failed to find RPC backend",
        "Failed to obtain RPC backend",
    )
    return any(marker in text for marker in markers)
