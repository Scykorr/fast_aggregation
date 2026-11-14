"""Аргументы ggml-rpc-server и llama-server.

`--rpc` обязан идти раньше `--device`: иначе llama.cpp ещё не знает
имён RPC0, RPC1 и отвергает список устройств.
"""

from __future__ import annotations

from pathlib import Path

from fast_aggregation.models import PlacementPlan


def rpc_server_argv(
    binary: Path,
    *,
    host: str = "0.0.0.0",
    port: int,
    threads: int | None = None,
    cache: bool = True,
) -> list[str]:
    argv = [str(binary), "--host", host, "--port", str(port)]
    if threads is not None and threads > 0:
        argv.extend(["--threads", str(threads)])
    if cache:
        argv.append("--cache")
    return argv


def llama_server_argv(
    binary: Path,
    plan: PlacementPlan,
    *,
    model_path: str | None,
    hf_repo: str | None,
    host: str = "127.0.0.1",
    port: int,
    ctx_size: int | None = None,
    parallel: int = 1,
    gpu_layers: str = "all",
    local_node_id: str | None = None,
) -> list[str]:
    if not model_path and not hf_repo:
        raise ValueError("Нужен путь к GGUF или репозиторий Hugging Face")
    endpoints = []
    for device in plan.devices:
        host_for_device = device.host
        if local_node_id and device.node_id == local_node_id:
            host_for_device = "127.0.0.1"
        endpoint = f"{host_for_device}:{device.rpc_port}"
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    argv = [
        str(binary),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if model_path:
        argv.extend(["--model", model_path])
    else:
        argv.extend(["--hf-repo", hf_repo or ""])
    if ctx_size and ctx_size > 0:
        argv.extend(["--ctx-size", str(ctx_size)])
    if parallel and parallel > 0:
        argv.extend(["--parallel", str(parallel)])
    argv.extend(
        [
            "--rpc",
            ",".join(endpoints),
            "--split-mode",
            "layer",
            "--gpu-layers",
            str(gpu_layers),
            "--device",
            ",".join(plan.device_flags),
            "--tensor-split",
            ",".join(str(value) for value in plan.tensor_split),
        ]
    )
    return argv


def preview_command(argv: list[str]) -> str:
    return " ".join(_quote(part) for part in argv)


def _quote(part: str) -> str:
    if part == "":
        return '""'
    if any(char.isspace() for char in part) or '"' in part:
        return '"' + part.replace('"', '\\"') + '"'
    return part
