from pathlib import Path

from fastapi.testclient import TestClient

from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.coordinator.app import create_coordinator_app
from fast_aggregation.coordinator.service import CoordinatorService, parse_address
from fast_aggregation.errors import FastAggError
from fast_aggregation.node.app import create_node_app
from fast_aggregation.node.service import NodeService, NodeSettings
from fast_aggregation.runtime.processes import ManagedProcess
from tests.helpers import gpu, machine

import pytest


class _Proc:
    def __init__(self, args: list[str], cwd: Path | None, log_path: Path) -> None:
        self.args = args
        self.cwd = cwd
        self.log_path = log_path
        self._running = False

    def start(self) -> None:
        self._running = True

    def running(self) -> bool:
        return self._running

    def stop(self, timeout: float = 8) -> None:
        del timeout
        self._running = False

    def tail(self, limit: int = 80) -> list[str]:
        del limit
        return ["rpc ready"]

    def exit_code(self) -> int | None:
        return None

    def pid(self) -> int | None:
        return 7


def _factory(argv: list[str], cwd: Path | None, log_path: Path) -> ManagedProcess:
    return _Proc(argv, cwd, log_path)  # type: ignore[return-value]


def _node(tmp_path: Path, resources) -> NodeService:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "ggml-rpc-server.exe").write_text("rpc", encoding="utf-8")
    (binaries / "llama-server.exe").write_text("srv", encoding="utf-8")
    (binaries / "ggml-rpc-server").write_text("rpc", encoding="utf-8")
    (binaries / "llama-server").write_text("srv", encoding="utf-8")

    def fetch(_release: str, _backend: str) -> Path:
        return binaries

    return NodeService(
        NodeSettings(
            node_id="local-1",
            name="coordinator-pc",
            http_port=8080,
            rpc_port=50052,
            advertise_host="192.168.1.10",
            data_dir=tmp_path,
            preferred_backend="auto",
        ),
        prober=lambda: resources,
        runtime_fetcher=fetch,
        process_factory=_factory,
        port_ready=lambda _port: True,
    )


def test_node_http_publishes_resources_and_starts_rpc(tmp_path: Path):
    resources = machine(os_name="Windows", gpus=[gpu()])
    service = _node(tmp_path, resources)
    app = create_node_app(service)
    with TestClient(app) as client:
        described = client.get("/v1/node")
        assert described.status_code == 200
        body = described.json()
        assert body["node_id"] == "local-1"
        assert body["resources"]["gpus"][0]["vram_free_mb"] == 24000
        assert body["resources"]["os_name"] == "Windows"
        started = client.post("/v1/rpc/start", json={"release": "b12359", "backend": "cuda"})
        assert started.status_code == 200, started.text
        payload = started.json()
        assert payload["running"] is True
        assert payload["backend"] == "cuda"
    assert service.describe()["rpc"]["running"] is False


def test_rpc_start_reports_bind_failure_even_if_port_is_open(tmp_path: Path):
    resources = machine(os_name="Windows", gpus=[gpu()])
    service = _node(tmp_path, resources)

    class DeadProc(_Proc):
        def tail(self, limit: int = 80) -> list[str]:
            del limit
            return ["Failed to create server socket"]

    service._process_factory = lambda argv, cwd, log_path: DeadProc(argv, cwd, log_path)  # type: ignore[method-assign]
    with pytest.raises(Exception) as caught:
        service.start_rpc("b12359", "cuda")
    assert "не удержался" in str(caught.value)
    assert service.describe()["rpc"]["running"] is False


def test_rpc_argv_binds_lan(tmp_path: Path):
    resources = machine(os_name="Windows", gpus=[gpu()])
    service = _node(tmp_path, resources)
    service.start_rpc("b12359", "cuda")
    process = service._process
    assert process is not None
    assert process.args[1:5] == ["--host", "0.0.0.0", "--port", "50052"]
    service.stop_rpc()


def test_dashboard_lists_local_node_and_blocks_openai_until_start(tmp_path: Path):
    resources = machine(gpus=[gpu(free=8192, total=8192)])
    node = _node(tmp_path, resources)
    service = CoordinatorService(
        registry=NodeRegistry(ttl=30),
        data_dir=tmp_path,
        node=node,
        background=False,
    )
    app = create_coordinator_app(service)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Узлы" in page.text
        state = client.get("/api/state").json()
        assert state["phase"] == "idle"
        assert state["nodes"][0]["name"] == "coordinator-pc"
        assert state["nodes"][0]["online"] is True
        plan = client.post("/api/plan", json={"reserve_mb": 0, "model_path": "D:/m.gguf"})
        assert plan.status_code == 200, plan.text
        assert plan.json()["plan"]["tensor_split"] == [8192]
        assert "--rpc" in plan.json()["command"]
        blocked = client.post("/v1/chat/completions", json={"messages": []})
        assert blocked.status_code == 503
        local = client.get("/v1/node")
        assert local.status_code == 200
        assert local.json()["node_id"] == "local-1"


def test_manual_node_uses_typed_address(tmp_path: Path):
    remote = machine(gpus=[gpu(name="worker-gpu", free=4096, total=4096)])

    class Client:
        def fetch(self, host: str, port: int) -> dict:
            assert (host, port) == ("10.0.0.8", 47000)
            return {
                "node_id": "remote-1",
                "name": "worker",
                "advertise_host": "10.0.0.5",
                "http_port": 1,
                "rpc_port": 50052,
                "preferred_backend": "vulkan",
                "resources": remote.to_dict(),
                "rpc": {"running": False, "backend": None, "error": None},
            }

        def start_rpc(self, host: str, port: int, payload: dict) -> dict:
            del host, port, payload
            return {"running": True}

        def stop_rpc(self, host: str, port: int) -> dict:
            del host, port
            return {"running": False}

    service = CoordinatorService(
        registry=NodeRegistry(ttl=30),
        data_dir=tmp_path,
        node=None,
        client=Client(),
        background=False,
    )
    app = create_coordinator_app(service)
    with TestClient(app) as client:
        added = client.post("/api/nodes", json={"address": "http://10.0.0.8:47000"})
        assert added.status_code == 200, added.text
        body = added.json()
        assert body["host"] == "10.0.0.8"
        assert body["http_port"] == 47000
        assert body["source"] == "manual"
        assert body["resources"]["gpus"][0]["name"] == "worker-gpu"
        removed = client.delete("/api/nodes/remote-1")
        assert removed.status_code == 200
        assert client.get("/api/state").json()["nodes"] == []


def test_parse_address():
    assert parse_address("192.168.0.2:47000") == ("192.168.0.2", 47000)
    assert parse_address("http://pc.local:8080/v1/node") == ("pc.local", 8080)
    with pytest.raises(FastAggError):
        parse_address("just-a-host")
