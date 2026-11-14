from pathlib import Path

import pytest

from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.service import NodeService, NodeSettings
from tests.helpers import gpu, machine
from tests.test_api import _factory, _node


class _Proc:
    def __init__(self, args: list[str], cwd: Path | None, log_path: Path) -> None:
        self.args = list(args)
        self.cwd = cwd
        self.log_path = log_path
        self.started = False
        self._running = False

    def start(self) -> None:
        self.started = True
        self._running = True

    def running(self) -> bool:
        return self._running

    def stop(self, timeout: float = 8) -> None:
        del timeout
        self._running = False

    def tail(self, limit: int = 80) -> list[str]:
        del limit
        return ["health ok"]

    def exit_code(self) -> int | None:
        return None

    def pid(self) -> int | None:
        return 42


def _seed(registry: NodeRegistry) -> None:
    specs = [
        ("b", "bee", "192.168.1.20", 8192),
        ("a", "ant", "192.168.1.10", 24576),
    ]
    for node_id, name, host, memory in specs:
        registry.upsert_details(
            node_id,
            name=name,
            host=host,
            http_port=47000,
            rpc_port=50052,
            resources=machine(gpus=[gpu(name=name, free=memory, total=memory)]),
            preferred_backend="cuda",
        )


def test_start_launches_workers_then_engine_with_rpc_first(tmp_path: Path):
    registry = NodeRegistry(ttl=60)
    _seed(registry)
    calls: list[tuple] = []

    class Client:
        def fetch(self, host: str, port: int) -> dict:
            raise AssertionError((host, port))

        def start_rpc(self, host: str, port: int, payload: dict) -> dict:
            calls.append(("start", host, port, payload["backend"], payload["release"]))
            return {"running": True, "backend": payload["backend"]}

        def stop_rpc(self, host: str, port: int) -> dict:
            calls.append(("stop", host, port))
            return {"running": False}

    spawned: list[_Proc] = []

    def factory(argv: list[str], cwd: Path | None, log_path: Path) -> _Proc:
        process = _Proc(argv, cwd, log_path)
        spawned.append(process)
        return process

    def fetcher(release: str, system: str, arch: str, backend: str) -> Path:
        assert release == "b12359"
        assert backend == "cpu"
        folder = tmp_path / "engine"
        folder.mkdir(exist_ok=True)
        name = "llama-server.exe" if system.lower() == "windows" else "llama-server"
        (folder / name).write_text("srv", encoding="utf-8")
        return folder

    service = CoordinatorService(
        registry=registry,
        data_dir=tmp_path,
        release="b12359",
        node=None,
        client=Client(),
        background=False,
        process_factory=factory,  # type: ignore[arg-type]
        health_check=lambda: True,
        runtime_fetcher=fetcher,
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    state = service.start_cluster(
        model_path=str(model),
        hf_repo=None,
        ctx_size=2048,
        parallel=1,
        allow_cpu=False,
        reserve_mb=0,
    )
    assert state["phase"] == "running"
    assert [call[1] for call in calls] == ["192.168.1.10", "192.168.1.20"]
    assert {call[3] for call in calls} == {"cuda"}
    argv = spawned[0].args
    rpc_at = argv.index("--rpc")
    assert rpc_at < argv.index("--device")
    assert argv[rpc_at + 1] == "192.168.1.10:50052,192.168.1.20:50052"
    assert argv[argv.index("--tensor-split") + 1] == "24576,8192"
    service.stop_cluster()
    assert service.phase == "idle"
    assert ("stop", "192.168.1.20", 47000) in calls
    assert ("stop", "192.168.1.10", 47000) in calls


def test_missing_model_does_not_start_workers(tmp_path: Path):
    registry = NodeRegistry(ttl=60)
    _seed(registry)
    started: list[str] = []

    class Client:
        def fetch(self, host: str, port: int) -> dict:
            del host, port
            return {}

        def start_rpc(self, host: str, port: int, payload: dict) -> dict:
            del port, payload
            started.append(host)
            return {"running": True}

        def stop_rpc(self, host: str, port: int) -> dict:
            del host, port
            return {"running": False}

    service = CoordinatorService(
        registry=registry,
        data_dir=tmp_path,
        node=None,
        client=Client(),
        background=False,
        health_check=lambda: True,
    )
    with pytest.raises(FastAggError):
        service.start_cluster(
            model_path=str(tmp_path / "missing.gguf"),
            hf_repo=None,
            ctx_size=None,
            parallel=1,
            allow_cpu=False,
            reserve_mb=512,
        )
    assert started == []
    assert service.phase == "error"


def test_participating_coordinator_uses_local_rpc_and_loopback(tmp_path: Path):
    resources = machine(os_name="Windows", gpus=[gpu(free=4096, total=4096)])
    node = _node(tmp_path, resources)
    remote_calls: list[str] = []

    class Client:
        def fetch(self, host: str, port: int) -> dict:
            del host, port
            return {}

        def start_rpc(self, host: str, port: int, payload: dict) -> dict:
            del port, payload
            remote_calls.append(host)
            return {"running": True}

        def stop_rpc(self, host: str, port: int) -> dict:
            del host, port
            return {"running": False}

    spawned: list[_Proc] = []

    def factory(argv: list[str], cwd: Path | None, log_path: Path) -> _Proc:
        process = _Proc(argv, cwd, log_path)
        spawned.append(process)
        return process

    def fetcher(release: str, system: str, arch: str, backend: str) -> Path:
        del release, arch
        assert backend == "cpu"
        folder = tmp_path / "engine"
        folder.mkdir(exist_ok=True)
        name = "llama-server.exe" if system.lower() == "windows" else "llama-server"
        (folder / name).write_text("srv", encoding="utf-8")
        return folder

    service = CoordinatorService(
        registry=NodeRegistry(ttl=60),
        data_dir=tmp_path,
        release="b12359",
        node=node,
        client=Client(),
        background=False,
        process_factory=factory,  # type: ignore[arg-type]
        health_check=lambda: True,
        runtime_fetcher=fetcher,
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    state = service.start_cluster(
        model_path=str(model),
        hf_repo=None,
        ctx_size=None,
        parallel=1,
        allow_cpu=False,
        reserve_mb=0,
    )
    assert state["phase"] == "running"
    assert remote_calls == []
    assert node.describe()["rpc"]["running"] is True
    argv = spawned[0].args
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50052"
    service.stop_cluster()


def test_cuda_build_is_passed_to_downloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str | None] = {}

    def fake_ensure(tag, system, arch, backend, cache, role="rpc", cuda_build=None, downloader=None):
        del tag, system, arch, backend, cache, role, downloader
        captured["cuda_build"] = cuda_build
        folder = tmp_path / "rt"
        folder.mkdir()
        (folder / "ggml-rpc-server.exe").write_text("rpc", encoding="utf-8")
        return folder

    monkeypatch.setattr("fast_aggregation.node.service.ensure_runtime", fake_ensure)
    resources = machine(os_name="Windows", gpus=[gpu()])
    service = NodeService(
        NodeSettings(
            node_id="n",
            name="n",
            http_port=1,
            rpc_port=50052,
            advertise_host="127.0.0.1",
            data_dir=tmp_path,
            cuda_build="13.5",
        ),
        prober=lambda: resources,
        process_factory=_factory,
        port_ready=lambda _port: True,
    )
    service.start_rpc("b12359", "cuda")
    assert captured["cuda_build"] == "13.5"
    service.stop_rpc()
