"""Команды fastagg: scan, node, coordinator, doctor."""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys

import uvicorn

from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.constants import (
    COORDINATOR_PORT,
    DEFAULT_LLAMA_RELEASE,
    DISCOVERY_PORT,
    ENGINE_PORT,
    NODE_HTTP_PORT,
    RPC_PORT,
)
from fast_aggregation.coordinator.app import create_coordinator_app
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.netinfo import guess_advertise_host
from fast_aggregation.node.app import create_node_app
from fast_aggregation.node.service import NodeService, NodeSettings, default_node_name
from fast_aggregation.paths import home_dir, load_node_id
from fast_aggregation.resources.probe import normalize_arch, probe_machine
from fast_aggregation.runtime.downloads import ensure_runtime, executable
from fast_aggregation.runtime.releases import engine_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastagg",
        description="Распределённый запуск LLM в локальной сети через llama.cpp RPC.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Показать ресурсы этой машины")
    scan.add_argument("--json", action="store_true")

    node = sub.add_parser("node", help="Узел: публикует ресурсы и поднимает RPC по запросу")
    _common(node)
    node.add_argument("--http-port", type=int, default=NODE_HTTP_PORT)

    coord = sub.add_parser("coordinator", help="Координатор, дашборд и OpenAI-совместимый API")
    _common(coord)
    coord.add_argument("--port", type=int, default=COORDINATOR_PORT)
    coord.add_argument("--engine-port", type=int, default=ENGINE_PORT)
    coord.add_argument("--no-participate", action="store_true", help="Не отдавать ресурсы этой машины в план")

    doctor = sub.add_parser("doctor", help="Проверка окружения и, по желанию, бинарника llama.cpp")
    doctor.add_argument("--fetch", action="store_true", help="Скачать CPU-сборку закреплённого релиза и показать версию")
    doctor.add_argument("--release", default=DEFAULT_LLAMA_RELEASE)
    doctor.add_argument("--data-dir", default="")

    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "scan":
        return _scan(args.json)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "node":
        return _serve_node(args)
    if args.command == "coordinator":
        return _serve_coordinator(args)
    parser.error("Неизвестная команда")
    return 2


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", default="")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--advertise", default="", help="IP, который увидят другие машины. По умолчанию определяется сам")
    parser.add_argument("--rpc-port", type=int, default=RPC_PORT)
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument("--backend", default="auto", help="auto, cuda, vulkan, rocm, sycl, cpu, metal")
    parser.add_argument("--vram-mb", type=int, default=0, help="Подставить VRAM, если датчик её не сообщил")
    parser.add_argument(
        "--cuda-build",
        default="",
        help="Сборка CUDA, если бэкенд cuda: на Windows x64 это 12.4 или 13.5",
    )
    parser.add_argument("--release", default=DEFAULT_LLAMA_RELEASE, help="Тег сборки llama.cpp, один на весь кластер")
    parser.add_argument("--data-dir", default="")


def _scan(as_json: bool) -> int:
    resources = probe_machine(deep_vulkan=True)
    if as_json:
        import json

        print(json.dumps(resources.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"ОС: {resources.os_name} {resources.os_version} ({resources.arch})")
    physical = resources.cpu_physical if resources.cpu_physical is not None else "н/д"
    print(f"CPU: {resources.cpu_model} — {resources.cpu_logical} логических, {physical} физических")
    print(f"ОЗУ: {resources.ram_available_mb} МиБ свободно из {resources.ram_total_mb} МиБ")
    if resources.gpus:
        print("GPU:")
        for gpu in resources.gpus:
            free = "н/д" if gpu.vram_free_mb is None else f"{gpu.vram_free_mb} МиБ"
            total = "н/д" if gpu.vram_total_mb is None else f"{gpu.vram_total_mb} МиБ"
            print(f"  {gpu.name} [{gpu.vendor}] свободно {free}, всего {total}, источник {gpu.source}")
    else:
        print("GPU: не обнаружены")
    print("Бэкенды llama.cpp:")
    for backend in resources.backends:
        mark = "да" if backend.available else "нет"
        print(f"  {backend.name}: {mark} — {backend.detail}")
    return 0


def _serve_node(args: argparse.Namespace) -> int:
    service = _build_node(args, http_port=args.http_port)
    discovery = DiscoveryService(
        service.announce(),
        port=args.discovery_port,
        on_announce=lambda _announce, _addr: None,
    )
    app = create_node_app(service, discovery)
    _print_banner(
        f"Узел {service.settings.name} слушает http://{args.bind}:{args.http_port} , RPC-порт {args.rpc_port}"
    )
    uvicorn.run(app, host=args.bind, port=args.http_port, log_level="info")
    return 0


def _serve_coordinator(args: argparse.Namespace) -> int:
    if args.port == args.engine_port:
        print("Порт дашборда и порт llama-server должны различаться", file=sys.stderr)
        return 2
    data = _data_dir(args)
    registry = NodeRegistry()
    node = None if args.no_participate else _build_node(args, http_port=args.port)
    service = CoordinatorService(
        registry=registry,
        data_dir=data,
        release=args.release,
        engine_port=args.engine_port,
        node=node,
    )
    discovery = DiscoveryService(
        node.announce() if node is not None else None,
        port=args.discovery_port,
        on_announce=lambda announce, _addr: registry.upsert_announce(announce),
    )
    app = create_coordinator_app(service, discovery)
    _print_banner(
        f"Координатор: http://{args.bind}:{args.port}\n"
        f"OpenAI API: http://127.0.0.1:{args.port}/v1\n"
        f"Сборка llama.cpp: {args.release}. RPC без шифрования — только доверенная сеть."
    )
    uvicorn.run(app, host=args.bind, port=args.port, log_level="info")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    _scan(False)
    print(f"Закреплённый тег llama.cpp: {args.release}")
    if not args.fetch:
        print("Скачивание бинарника пропущено. Повторите с --fetch, чтобы проверить архив релиза.")
        return 0
    system = platform.system()
    arch = normalize_arch(platform.machine())
    backend = engine_backend(system)
    try:
        folder = ensure_runtime(
            args.release,
            system,
            arch,
            backend,
            _data_dir(args) / "runtimes",
            role="server",
        )
        binary = executable(folder, system, "server")
    except FastAggError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"llama-server: {binary}")
    completed = subprocess.run(
        [str(binary), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    sys.stdout.write(completed.stdout or "")
    sys.stderr.write(completed.stderr or "")
    help_run = subprocess.run(
        [str(binary), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    help_text = (help_run.stdout or "") + (help_run.stderr or "")
    if "--rpc" not in help_text:
        print("В --help нет --rpc. Эта сборка не годится как координатор.", file=sys.stderr)
        return 1
    print("Флаг --rpc в llama-server есть.")
    try:
        rpc_binary = executable(folder, system, "rpc")
    except FastAggError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"ggml-rpc-server: {rpc_binary}")
    return 0 if completed.returncode == 0 else completed.returncode


def _build_node(args: argparse.Namespace, *, http_port: int) -> NodeService:
    data = _data_dir(args)
    settings = NodeSettings(
        node_id=load_node_id(data / "node_id"),
        name=args.name or default_node_name(),
        http_port=http_port,
        rpc_port=args.rpc_port,
        advertise_host=args.advertise or guess_advertise_host(),
        data_dir=data,
        preferred_backend=args.backend,
        vram_override_mb=args.vram_mb or None,
        cuda_build=args.cuda_build or None,
        release=args.release,
    )
    return NodeService(settings)


def _data_dir(args: argparse.Namespace):
    from pathlib import Path

    if getattr(args, "data_dir", ""):
        return Path(args.data_dir)
    return home_dir()


def _print_banner(text: str) -> None:
    print(text, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
