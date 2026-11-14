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
from fast_aggregation.coordinator.client import HttpNodeClient
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import FastAggError
from fast_aggregation.netinfo import guess_advertise_host
from fast_aggregation.node.app import create_node_app
from fast_aggregation.node.service import NodeService, NodeSettings, default_node_name
from fast_aggregation.paths import home_dir, load_node_id
from fast_aggregation.resources.probe import normalize_arch, probe_machine
from fast_aggregation.security.certs import ensure_cert, local_names
from fast_aggregation.security.token import (
    load_membership,
    load_or_create,
    new_code,
    new_token,
    save_membership,
)
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
    node.add_argument("--open", action="store_true", help="Без токена. Только своя лаборатория")
    node.add_argument("--insecure-http", action="store_true", help="HTTP без TLS")

    coord = sub.add_parser("coordinator", help="Координатор, дашборд и OpenAI-совместимый API")
    _common(coord)
    coord.add_argument("--port", type=int, default=COORDINATOR_PORT)
    coord.add_argument("--engine-port", type=int, default=ENGINE_PORT)
    coord.add_argument("--no-participate", action="store_true", help="Не отдавать ресурсы этой машины в план")
    coord.add_argument("--open", action="store_true", help="Без токена и без TLS. Только своя лаборатория")
    coord.add_argument("--insecure-http", action="store_true")

    join = sub.add_parser("join", help="Сопрячь этот компьютер с координатором")
    join.add_argument("--url", required=True, help="fastagg://... или https://host:port")
    join.add_argument("--code", default="")
    join.add_argument("--fingerprint", default="")
    join.add_argument("--data-dir", default="")
    join.add_argument("--advertise", default="")
    join.add_argument("--http-port", type=int, default=NODE_HTTP_PORT)
    join.add_argument("--rpc-port", type=int, default=RPC_PORT)
    join.add_argument("--name", default="")

    scan_net = sub.add_parser("scan-net", help="Обойти подсеть по TCP в поисках узлов")
    scan_net.add_argument("cidr")
    scan_net.add_argument("--port", type=int, default=NODE_HTTP_PORT)

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
    if args.command == "join":
        return _join(args)
    if args.command == "scan-net":
        return _scan_net(args)
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
    temp = "н/д" if resources.temperature_c is None else f"{resources.temperature_c:.0f} °C"
    battery = "нет" if resources.battery_percent is None else f"{resources.battery_percent:.0f}%"
    plugged = "от сети" if resources.power_plugged else "от батареи" if resources.power_plugged is False else "н/д"
    print(f"Состояние: класс {resources.device_class}, температура {temp}, батарея {battery} ({plugged})")
    if resources.cpu_percent is not None:
        print(f"Загрузка CPU: {resources.cpu_percent:.0f}%")
    return 0


def _serve_node(args: argparse.Namespace) -> int:
    data = _data_dir(args)
    membership = load_membership(data)
    token = str(membership.get("token") or "") if membership else ""
    if not token and not args.open:
        print(
            "Узел ещё не сопряжён. На координаторе возьмите код и выполните fastagg join. "
            "Для лаборатории без токена: --open.",
            file=sys.stderr,
        )
        return 2
    service = _build_node(args, http_port=args.http_port)
    discovery = DiscoveryService(
        service.announce(),
        port=args.discovery_port,
        on_announce=lambda _announce, _addr: None,
        token=token or None,
    )
    app = create_node_app(service, discovery, token=token or None)
    scheme = "http" if args.open or args.insecure_http else "https"
    ssl_kwargs = _ssl_kwargs(data, args.advertise or guess_advertise_host()) if scheme == "https" else {}
    _print_banner(
        f"Узел {service.settings.name} слушает {scheme}://{args.bind}:{args.http_port}, RPC-порт {args.rpc_port}"
    )
    uvicorn.run(app, host=args.bind, port=args.http_port, log_level="info", **ssl_kwargs)
    return 0


def _serve_coordinator(args: argparse.Namespace) -> int:
    if args.port == args.engine_port:
        print("Порт дашборда и порт llama-server должны различаться", file=sys.stderr)
        return 2
    data = _data_dir(args)
    registry = NodeRegistry()
    advertise = args.advertise or guess_advertise_host()
    open_mode = args.open
    token = ""
    code = ""
    fingerprint = ""
    ssl_kwargs: dict = {}
    if not open_mode:
        token = load_or_create(data / "cluster_token", new_token)
        code = load_or_create(data / "pairing_code", new_code)
        _cert, _key, fingerprint = ensure_cert(data / "tls", local_names(advertise))
        if not args.insecure_http:
            ssl_kwargs = {"ssl_certfile": str(data / "tls" / "cert.pem"), "ssl_keyfile": str(data / "tls" / "key.pem")}
    node = None if args.no_participate else _build_node(args, http_port=args.port)
    bundle = data / "trusted-bundle.pem"
    if ssl_kwargs and not bundle.is_file():
        bundle.write_text((data / "tls" / "cert.pem").read_text(encoding="utf-8"), encoding="utf-8")
    client = HttpNodeClient(
        token=token,
        scheme="https" if ssl_kwargs else "http",
        verify=str(bundle) if ssl_kwargs else False,
    )
    service = CoordinatorService(
        registry=registry,
        data_dir=data,
        release=args.release,
        engine_port=args.engine_port,
        node=node,
        client=client,
        token=token,
        pairing_code=code,
        fingerprint=fingerprint,
    )
    service.public_host = advertise
    service.public_port = args.port
    service.trusted_bundle = data / "trusted-bundle.pem"
    discovery = DiscoveryService(
        node.announce() if node is not None else None,
        port=args.discovery_port,
        on_announce=lambda announce, _addr: registry.upsert_announce(announce),
        token=token or None,
    )
    app = create_coordinator_app(service, discovery)
    scheme = "https" if ssl_kwargs else "http"
    banner = (
        f"Координатор: {scheme}://{advertise}:{args.port}\n"
        f"OpenAI API: {scheme}://127.0.0.1:{args.port}/v1\n"
        f"Сборка llama.cpp: {args.release}."
    )
    if code:
        from fast_aggregation.security.token import join_uri

        banner += (
            f"\nКод сопряжения: {code}\n"
            f"Отпечаток: {fingerprint}\n"
            f"Ссылка: {join_uri(advertise, args.port, code, fingerprint)}\n"
            "Управляющий канал зашифрован TLS. Поток llama.cpp RPC между машинами остаётся открытым TCP."
        )
    else:
        banner += "\nРежим --open: без токена и без TLS."
    _print_banner(banner)
    uvicorn.run(app, host=args.bind, port=args.port, log_level="info", **ssl_kwargs)
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


def _join(args: argparse.Namespace) -> int:
    from fast_aggregation.security.pin import pinned_post
    from fast_aggregation.security.token import parse_join_uri

    url = args.url.strip()
    code = args.code.strip()
    fingerprint = args.fingerprint.strip()
    if url.startswith("fastagg://"):
        host, port, code, fingerprint = parse_join_uri(url)
        pair_url = f"https://{host}:{port}/api/pair"
    else:
        pair_url = url.rstrip("/") + "/api/pair"
    if not code or not fingerprint:
        print("Нужны код и отпечаток: --code и --fingerprint или ссылка fastagg://", file=sys.stderr)
        return 2
    data = _data_dir(args)
    advertise = args.advertise or guess_advertise_host()
    cert_path, _key, _fp = ensure_cert(data / "tls", local_names(advertise))
    try:
        result = pinned_post(
            pair_url,
            fingerprint,
            {
                "code": code,
                "node_id": load_node_id(data / "node_id"),
                "name": args.name or default_node_name(),
                "advertise_host": advertise,
                "http_port": args.http_port,
                "rpc_port": args.rpc_port,
                "cert_pem": cert_path.read_text(encoding="utf-8"),
            },
            data / "coordinator.pem",
        )
    except FastAggError as exc:
        print(exc, file=sys.stderr)
        return 1
    save_membership(
        data,
        {
            "token": result["token"],
            "coordinator": pair_url,
            "fingerprint": fingerprint,
        },
    )
    print("Сопряжение сохранено. Запустите: fastagg node")
    return 0


def _scan_net(args: argparse.Namespace) -> int:
    from fast_aggregation.discovery.scan import hosts_in_cidr, scan_tcp

    try:
        hosts = hosts_in_cidr(args.cidr)
    except FastAggError as exc:
        print(exc, file=sys.stderr)
        return 2
    found = scan_tcp(hosts, args.port)
    if not found:
        print(f"На порту {args.port} в {args.cidr} узлов не видно.")
        return 1
    for host in found:
        print(f"{host}:{args.port}")
    return 0


def _ssl_kwargs(data, advertise: str) -> dict[str, str]:
    ensure_cert(data / "tls", local_names(advertise))
    return {
        "ssl_certfile": str(data / "tls" / "cert.pem"),
        "ssl_keyfile": str(data / "tls" / "key.pem"),
    }


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
