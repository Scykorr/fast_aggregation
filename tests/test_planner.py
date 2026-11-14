import pytest

from fast_aggregation.cluster.planner import choose_backend, plan_placement
from fast_aggregation.errors import BackendError, PlacementError
from fast_aggregation.models import ClusterNode
from tests.helpers import gpu, machine


def _node(
    name: str,
    node_id: str,
    *,
    host: str = "192.168.1.10",
    resources=None,
    online: bool = True,
    preferred: str = "auto",
    rpc_port: int = 50052,
) -> ClusterNode:
    return ClusterNode(
        node_id=node_id,
        name=name,
        host=host,
        http_port=47000,
        rpc_port=rpc_port,
        online=online,
        resources=resources,
        preferred_backend=preferred,
    )


def test_split_is_proportional_to_free_vram_after_reserve():
    small = _node(
        "bee",
        "b",
        host="192.168.1.20",
        resources=machine(gpus=[gpu(name="RTX 4060", total=8192, free=8192)]),
    )
    large = _node(
        "ant",
        "a",
        host="192.168.1.10",
        resources=machine(gpus=[gpu(name="RTX 4090", total=24576, free=24576)]),
    )
    plan = plan_placement([small, large], reserve_mb=0)
    assert [device.node_name for device in plan.devices] == ["ant", "bee"]
    assert plan.tensor_split == [24576, 8192]
    assert plan.devices[0].share == pytest.approx(24576 / (24576 + 8192))
    assert plan.rpc_endpoints == ["192.168.1.10:50052", "192.168.1.20:50052"]
    assert plan.device_flags == ["RPC0", "RPC1"]


def test_reserve_reduces_usable_memory():
    node = _node("solo", "1", resources=machine(gpus=[gpu(free=8192, total=8192)]))
    plan = plan_placement([node], reserve_mb=512)
    assert plan.tensor_split == [7680]


def test_two_gpus_on_one_node_share_one_rpc_endpoint():
    resources = machine(
        gpus=[
            gpu(name="RTX 3090", total=24576, free=24576),
            gpu(name="RTX 3060", total=12288, free=12288),
        ]
    )
    plan = plan_placement([_node("desk", "1", resources=resources)], reserve_mb=0)
    assert plan.tensor_split == [24576, 12288]
    assert plan.rpc_endpoints == ["192.168.1.10:50052"]
    assert plan.device_flags == ["RPC0", "RPC1"]
    assert plan.backends_by_node() == {"1": "cuda"}


def test_cpu_nodes_are_skipped_until_allowed():
    cpu_only = _node("laptop", "cpu", resources=machine(gpus=[], cuda=False, ram=16384))
    with pytest.raises(PlacementError, match="CPU-узлы"):
        plan_placement([cpu_only], allow_cpu=False)
    plan = plan_placement([cpu_only], allow_cpu=True)
    assert plan.devices[0].kind == "cpu"
    assert plan.devices[0].backend == "cpu"
    assert plan.devices[0].memory_mb == 8192


def test_offline_and_unknown_nodes_do_not_receive_layers():
    offline = _node("gone", "g", online=False, resources=machine(gpus=[gpu()]))
    unknown = _node("wait", "w", resources=None)
    ready = _node("ok", "o", resources=machine(gpus=[gpu(free=4096, total=4096)]))
    plan = plan_placement([offline, unknown, ready], reserve_mb=0)
    assert [device.node_id for device in plan.devices] == ["o"]
    assert any("не в сети" in item for item in plan.warnings)
    assert any("нет снимка" in item for item in plan.warnings)


def test_unknown_vram_is_not_guessed():
    node = _node(
        "amd",
        "amd",
        resources=machine(
            gpus=[gpu(name="Radeon", vendor="amd", total=None, free=None, source="win32")],
            cuda=False,
            vulkan=True,
        ),
    )
    with pytest.raises(PlacementError):
        plan_placement([node])


def test_windows_prefers_cuda_before_vulkan_linux_prefers_rocm_before_vulkan():
    windows = machine(os_name="Windows", cuda=True, vulkan=True, rocm=True)
    linux = machine(os_name="Linux", cuda=False, vulkan=True, rocm=True)
    assert choose_backend(windows, "auto") == "cuda"
    assert choose_backend(linux, "auto") == "rocm"
    assert choose_backend(windows, "vulkan") == "vulkan"
    with pytest.raises(BackendError):
        choose_backend(machine(cuda=False, vulkan=False), "cuda")
