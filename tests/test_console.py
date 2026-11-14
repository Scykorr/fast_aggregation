import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fast_aggregation.cluster.catalog import recommend, summarize_capacity
from fast_aggregation.cluster.planner import plan_placement
from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.coordinator.app import create_coordinator_app
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.errors import RuntimeError_
from fast_aggregation.models import NodeLimits
from fast_aggregation.runtime.downloads import ensure_runtime
from tests.helpers import gpu, machine


def _service(tmp_path: Path, resources) -> CoordinatorService:
    registry = NodeRegistry(ttl=60)
    registry.upsert_details(
        "desk",
        name="desk",
        host="127.0.0.1",
        http_port=47000,
        rpc_port=50052,
        resources=resources,
        preferred_backend="cuda",
    )
    return CoordinatorService(registry=registry, data_dir=tmp_path, node=None, background=False)


def test_limits_exclude_node_and_cap_memory():
    resources = machine(gpus=[gpu(free=8000, total=8000)])
    node_id = "desk"
    from fast_aggregation.models import ClusterNode

    node = ClusterNode(
        node_id=node_id,
        name="desk",
        host="127.0.0.1",
        http_port=1,
        rpc_port=2,
        online=True,
        resources=resources,
        preferred_backend="cuda",
        limits=NodeLimits(excluded=True),
    )
    with pytest.raises(Exception, match="исключ"):
        plan_placement([node], reserve_mb=0)
    node.limits = NodeLimits(memory_cap_mb=1000)
    plan = plan_placement([node], reserve_mb=0)
    assert plan.devices[0].memory_mb == 1000


def test_recommendation_fits_small_model_only():
    resources = machine(gpus=[gpu(free=8000, total=8000)])
    from fast_aggregation.models import ClusterNode

    node = ClusterNode(
        node_id="a",
        name="a",
        host="h",
        http_port=1,
        rpc_port=2,
        online=True,
        resources=resources,
        bandwidth_mbps=1000,
        rtt_ms=1,
    )
    plan = plan_placement([node], reserve_mb=0)
    capacity = summarize_capacity(plan, [node])
    rows = {item["id"]: item for item in recommend(capacity, ctx=4096)}
    assert rows["7b-q4"]["fits"] is True
    assert rows["7b-q4"]["tokens_per_second"] > 0
    assert rows["70b-q8"]["fits"] is False
    assert rows["7b-q4"]["max_context"] >= 4096


def test_offline_runtime_refuses_network_and_uses_local_zip(tmp_path: Path):
    def boom(url: str, dest: Path) -> None:
        raise AssertionError(url)

    with pytest.raises(RuntimeError_, match="интернета"):
        ensure_runtime(
            "b12359",
            "Windows",
            "x64",
            "cpu",
            tmp_path / "empty",
            allow_download=False,
            downloader=boom,
        )
    archive = tmp_path / "stash" / "llama-b12359-bin-win-cpu-x64.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("llama-server.exe", b"srv")
        bundle.writestr("ggml-rpc-server.exe", b"rpc")
    folder = ensure_runtime(
        "b12359",
        "Windows",
        "x64",
        "cpu",
        tmp_path / "stash",
        allow_download=False,
        downloader=boom,
    )
    assert (folder / "llama-server.exe").is_file()


def test_ui_and_catalog_are_local(tmp_path: Path):
    html = Path("src/fast_aggregation/web/index.html").read_text(encoding="utf-8")
    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "googleapis" not in lowered
    assert "cdn." not in lowered
    assert "Новый диалог" in html

    service = _service(tmp_path, machine(gpus=[gpu(free=16000, total=16000)]))
    service.models_dir = tmp_path
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 32)
    app = create_coordinator_app(service)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "fastagg" in page.text
        report = client.get("/api/capacity?ctx=2048").json()
        assert report["usable_mib"] > 0
        assert any(row["id"] == "7b-q4" for row in report["recommendations"])
        preview = client.post("/api/models/preview", json={"path": str(gguf), "ctx_size": 0}).json()
        assert preview["fits"] is True
        saved = client.post("/api/nodes/desk/limits", json={"excluded": True, "max_temp_c": 70})
        assert saved.status_code == 200
        assert saved.json()["excluded"] is True
        staged = service.data_dir / "runtimes" / "llama-b12359-bin-win-cpu-x64.zip.archive"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"archive-bytes")
        runtime = client.get("/runtime/llama-b12359-bin-win-cpu-x64.zip")
        assert runtime.status_code == 200
        assert runtime.content == b"archive-bytes"
