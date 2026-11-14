from pathlib import Path

from fast_aggregation.cluster.planner import plan_placement
from fast_aggregation.models import ClusterNode
from fast_aggregation.runtime.commands import llama_server_argv, preview_command, rpc_server_argv
from tests.helpers import gpu, machine


def test_rpc_server_binds_all_interfaces():
    argv = rpc_server_argv(Path("ggml-rpc-server.exe"), port=50052, cache=True)
    assert argv[:5] == ["ggml-rpc-server.exe", "--host", "0.0.0.0", "--port", "50052"]
    assert argv[-1] == "--cache"


def test_llama_server_puts_rpc_before_device_and_rewrites_local_host():
    nodes = [
        ClusterNode(
            node_id="local",
            name="alpha",
            host="192.168.1.10",
            http_port=8080,
            rpc_port=50052,
            online=True,
            resources=machine(gpus=[gpu(name="local", free=8000, total=8000)]),
            preferred_backend="cuda",
        ),
        ClusterNode(
            node_id="remote",
            name="beta",
            host="192.168.1.20",
            http_port=47000,
            rpc_port=50052,
            online=True,
            resources=machine(gpus=[gpu(name="remote", free=16000, total=16000)]),
            preferred_backend="cuda",
        ),
    ]
    plan = plan_placement(nodes, reserve_mb=0)
    argv = llama_server_argv(
        Path(r"C:\fastagg\llama-server.exe"),
        plan,
        model_path=r"D:\models\qwen.gguf",
        hf_repo=None,
        port=18080,
        ctx_size=4096,
        parallel=1,
        local_node_id="local",
    )
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50052,192.168.1.20:50052"
    assert argv.index("--rpc") < argv.index("--device")
    assert argv[argv.index("--device") + 1] == "RPC0,RPC1"
    assert argv[argv.index("--split-mode") + 1] == "layer"
    assert argv[argv.index("--gpu-layers") + 1] == "all"
    assert argv[argv.index("--tensor-split") + 1] == "8000,16000"
    assert argv[argv.index("--model") + 1] == r"D:\models\qwen.gguf"
    rendered = preview_command(argv)
    assert "llama-server.exe" in rendered
    assert "--rpc 127.0.0.1:50052,192.168.1.20:50052" in rendered
