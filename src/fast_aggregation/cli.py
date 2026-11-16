"""Команды fastagg: scan, node, coordinator, doctor."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

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
from fast_aggregation.runtime.downloads import ensure_runtime, executable, stage_archive
from fast_aggregation.runtime.releases import engine_backend


def _console_safe() -> None:
    """Не падать, если консоль Windows не умеет кириллицу (cp1252 в CI)."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def _dispatch(argv: list[str] | None = None) -> int:
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
    node.add_argument("--allow-download", action="store_true", help="Разрешить скачивание llama.cpp из интернета")
    node.add_argument(
        "--runtime-from",
        default="",
        help="Базовый URL раздачи сборок координатора, обычно https://host:47080/runtime",
    )

    coord = sub.add_parser("coordinator", help="Координатор, дашборд и OpenAI-совместимый API")
    _common(coord)
    coord.add_argument(
        "--port",
        type=int,
        default=None,
        help="Порт дашборда. По умолчанию 47080 или порт из coordinator.json. Если занят — следующий свободный до 47099",
    )
    coord.add_argument("--engine-port", type=int, default=ENGINE_PORT)
    coord.add_argument("--no-participate", action="store_true", help="Не отдавать ресурсы этой машины в план")
    coord.add_argument("--open", action="store_true", help="Без токена и без TLS. Только своя лаборатория")
    coord.add_argument("--insecure-http", action="store_true")
    coord.add_argument("--allow-download", action="store_true", help="Разрешить скачивание llama.cpp из интернета")
    coord.add_argument("--models-dir", default="", help="Локальная папка с GGUF. Интернет для моделей не нужен")

    join = sub.add_parser("join", help="Сопрячь этот компьютер с координатором")
    join.add_argument("--url", required=True, help="fastagg://... или https://host:port")
    join.add_argument("--code", default="")
    join.add_argument("--fingerprint", default="")
    join.add_argument("--data-dir", default="")
    join.add_argument("--advertise", default="")
    join.add_argument("--http-port", type=int, default=NODE_HTTP_PORT)
    join.add_argument("--rpc-port", type=int, default=RPC_PORT)
    join.add_argument("--name", default="")

    bench = sub.add_parser("bench", help="Короткий замер скорости llama-server, около 200 токенов")
    bench.add_argument("--url", default="http://127.0.0.1:18080", help="Базовый адрес llama-server")
    bench.add_argument("--data-dir", default="")

    pull = sub.add_parser("pull", help="Скачать GGUF с Hugging Face в папку моделей")
    pull.add_argument("url", help="Ссылка на файл .gguf или на репозиторий")
    pull.add_argument("--file", default="", help="Имя файла или подпись квантования, если в репозитории их несколько")
    pull.add_argument("--token", default="", help="Токен Hugging Face для закрытой модели. Сохраняется локально")
    pull.add_argument("--models-dir", default="")
    pull.add_argument("--data-dir", default="")

    imported = sub.add_parser("runtime-import", help="Положить локальный архив llama.cpp в кэш, без интернета")
    imported.add_argument("path")
    imported.add_argument("--tag", default=DEFAULT_LLAMA_RELEASE)
    imported.add_argument("--data-dir", default="")

    scan_net = sub.add_parser("scan-net", help="Обойти подсеть по TCP в поисках узлов")
    scan_net.add_argument("cidr")
    scan_net.add_argument("--port", type=int, default=NODE_HTTP_PORT)

    tray = sub.add_parser("tray", help="Иконка в трее: статус и открытие консоли")
    tray.add_argument("--role", choices=("node", "coordinator"), default="node")
    tray.add_argument("--open", action="store_true")
    tray.add_argument("--insecure-http", action="store_true")

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
    if args.command == "runtime-import":
        return _runtime_import(args)
    if args.command == "pull":
        return _pull(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "tray":
        return _tray(args)
    parser.error("Неизвестная команда")
    return 2


def main(argv: list[str] | None = None) -> int:
    _console_safe()
    try:
        return _dispatch(argv)
    except Exception as exc:
        from fast_aggregation.instance import wait_before_close

        print(f"fastagg остановился: {exc}", file=sys.stderr)
        wait_before_close()
        return 1


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
        help="auto, 12.4 или 13.5. auto смотрит драйвер в nvidia-smi: до 580 берёт 12.4, с 580 — 13.5",
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
    nvidia_driver = next((gpu.driver for gpu in resources.gpus if gpu.vendor == "nvidia" and gpu.driver), None)
    if resources.os_name.lower() == "windows" and resources.arch == "x64" and any(
        gpu.vendor == "nvidia" for gpu in resources.gpus
    ):
        from fast_aggregation.runtime.cuda_select import select_windows_cuda_build

        choice = select_windows_cuda_build("auto", nvidia_driver)
        print(f"CUDA: {choice.reason}")
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


_PAIR_CODE = __import__("re").compile(r"^[A-Za-z0-9]{6,8}$")


def classify_pair_input(text: str) -> str:
    raw = text.strip()
    if not raw:
        return "empty"
    if raw.lower().startswith("fastagg://"):
        return "link"
    if _PAIR_CODE.fullmatch(raw):
        return "code"
    return "invalid"


def want_pair_prompt(args: argparse.Namespace, token: str) -> bool:
    if token or getattr(args, "open", False):
        return False
    if os.environ.get("FA_NO_PAUSE") == "1" or os.environ.get("CI"):
        return False
    if not getattr(args, "interactive", True):
        return False
    return sys.stdin is not None and bool(sys.stdin.isatty())


def _link_from_code(address: str, code: str) -> str:
    from fast_aggregation.security.certs import fingerprint_pem
    from fast_aggregation.security.pin import _peer_pem
    from fast_aggregation.security.token import join_uri

    text = address.strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0]
    if ":" not in text:
        raise FastAggError("Нужен адрес координатора вида 10.32.237.25:47080")
    host, raw_port = text.rsplit(":", 1)
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise FastAggError("Порт координатора должен быть числом") from exc
    fingerprint = fingerprint_pem(_peer_pem(host, port))
    return join_uri(host, port, code.strip().upper(), fingerprint)


def _prompt_for_pairing(args: argparse.Namespace, data: Path) -> str:
    if not want_pair_prompt(args, ""):
        return ""
    print(
        "Узел ещё не сопряжён. Вставьте ссылку fastagg:// или код из 6–8 символов.\n"
        "Пустой ввод — только слушать сеть: сопрячь можно позже кнопкой на координаторе."
    )
    try:
        entered = input("Ссылка или код: ").strip()
    except EOFError:
        return ""
    kind = classify_pair_input(entered)
    if kind == "empty":
        return ""
    if kind == "invalid":
        print("Не разобрал ввод. Ожидалась ссылка fastagg:// или код.", file=sys.stderr)
        return ""
    url = entered
    if kind == "code":
        try:
            address = input("Адрес координатора (host:port): ").strip()
        except EOFError:
            return ""
        try:
            url = _link_from_code(address, entered)
        except FastAggError as exc:
            print(exc, file=sys.stderr)
            return ""
    join_args = argparse.Namespace(
        url=url,
        code="",
        fingerprint="",
        data_dir=str(data),
        advertise=getattr(args, "advertise", ""),
        name=getattr(args, "name", ""),
        http_port=getattr(args, "http_port", NODE_HTTP_PORT),
        rpc_port=getattr(args, "rpc_port", RPC_PORT),
    )
    if _join(join_args) != 0:
        print("Сопряжение не сохранено. Узел запустится и будет доступен для кнопки «Сопрячь».")
        return ""
    saved = load_membership(data) or {}
    print("Сопряжение запомнено. Следующий автозапуск подхватит его сам.")
    return str(saved.get("token") or "")


def _serve_node(args: argparse.Namespace) -> int:
    data = _data_dir(args)
    membership = load_membership(data)
    token = str(membership.get("token") or "") if membership else ""
    if not token:
        token = _prompt_for_pairing(args, data)
        membership = load_membership(data) or membership
    service = _build_node(args, http_port=args.http_port)
    service.settings.allow_download = bool(args.allow_download)
    mirror = args.runtime_from.strip()
    if not mirror and membership:
        mirror = str(membership.get("runtime_mirror") or "")
    service.settings.runtime_mirror = mirror or None
    from fast_aggregation.node.pairing import PairSession, hardware_summary

    session = PairSession(data, joiner=_node_joiner(data, service))
    session.auth["token"] = token
    announce = service.announce()
    announce.paired = bool(token)
    announce.summary = hardware_summary(service.resources)
    announce.role = "node"
    session.announce = announce

    def _remember(found, _addr) -> None:
        session.note_coordinator(found)

    discovery = DiscoveryService(
        announce,
        port=args.discovery_port,
        on_announce=_remember,
        token=token or None,
    )
    app = create_node_app(service, discovery, token=token or None, session=session)
    scheme = "http" if args.open or args.insecure_http else "https"
    ssl_kwargs = _ssl_kwargs(data, args.advertise or guess_advertise_host()) if scheme == "https" else {}
    if token:
        line = f"Узел {service.settings.name} сопряжён и слушает {scheme}://{args.bind}:{args.http_port}"
    else:
        line = (
            f"Узел {service.settings.name} слушает {scheme}://{args.bind}:{args.http_port}, RPC-порт {args.rpc_port}. "
            "Сопряжения ещё нет: координатор может нажать «Сопрячь», либо откройте на этом компьютере страницу узла "
            f"и введите код. Объявление в сети уже идёт."
        )
    _print_banner(line)
    uvicorn.run(app, host=args.bind, port=args.http_port, log_level="info", **ssl_kwargs)
    return 0


def _serve_coordinator(args: argparse.Namespace) -> int:
    from fast_aggregation.instance import (
        InstanceLock,
        attach_message,
        choose_coordinator_port,
        configure_log,
        format_owners,
        health_ok,
        preferred_port,
        read_state,
        wait_before_close,
        write_state,
    )

    data = _data_dir(args)
    log_path = configure_log("coordinator")
    pause = bool(getattr(args, "pause", True))
    own_lock = not getattr(args, "lock_held", False)
    lock = InstanceLock(data / "coordinator.lock")
    if own_lock and not lock.try_acquire():
        return _coordinator_already_running(data, pause, log_path, attach_message, health_ok, read_state, wait_before_close)
    try:
        return _run_coordinator(
            args,
            data,
            log_path,
            pause,
            choose_coordinator_port,
            format_owners,
            preferred_port,
            read_state,
            wait_before_close,
            write_state,
        )
    finally:
        if own_lock:
            lock.release()


def _coordinator_already_running(data, pause, log_path, attach_message, health_ok, read_state, wait_before_close) -> int:
    import webbrowser

    state = read_state(data)
    url = str(state.get("dashboard_url") or "")
    if url and health_ok(url):
        print(attach_message(state))
        try:
            webbrowser.open(url)
        except OSError:
            pass
        wait_before_close(pause)
        return 0
    message = str(state.get("error") or "Координатор уже запускается, но дашборд ещё не отвечает.")
    print(message, file=sys.stderr)
    print(f"Журнал: {log_path}", file=sys.stderr)
    wait_before_close(pause)
    return 1


def _record_error(data, write_state, read_state, message: str) -> None:
    state = read_state(data)
    state["error"] = message
    state["pid"] = os.getpid()
    write_state(data, state)


def _run_coordinator(
    args,
    data,
    log_path,
    pause,
    choose_coordinator_port,
    format_owners,
    preferred_port,
    read_state,
    wait_before_close,
    write_state,
) -> int:
    saved = read_state(data)
    saved_port = saved.get("port")
    try:
        saved_port = int(saved_port) if saved_port else None
    except (TypeError, ValueError):
        saved_port = None
    wanted = preferred_port(args.port, saved_port)
    if wanted == args.engine_port:
        message = "Порт дашборда и порт llama-server должны различаться"
        print(message, file=sys.stderr)
        wait_before_close(pause)
        return 2
    try:
        choice = choose_coordinator_port(wanted, avoid={args.engine_port})
    except FastAggError as exc:
        _record_error(data, write_state, read_state, str(exc))
        print(exc, file=sys.stderr)
        print(f"Журнал: {log_path}", file=sys.stderr)
        wait_before_close(pause)
        return 1

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
    scheme = "https" if ssl_kwargs else "http"
    dashboard_url = f"{scheme}://127.0.0.1:{choice.port}"
    advertise_url = f"{scheme}://{advertise}:{choice.port}"
    note = " ".join(choice.notes)
    write_state(
        data,
        {
            "pid": os.getpid(),
            "port": choice.port,
            "scheme": scheme,
            "dashboard_url": dashboard_url,
            "advertise_url": advertise_url,
            "error": None,
            "note": note,
        },
    )
    node = None if args.no_participate else _build_node(args, http_port=choice.port)
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
    service.public_port = choice.port
    service.trusted_bundle = data / "trusted-bundle.pem"
    service.allow_download = bool(args.allow_download)
    service.models_dir = Path(args.models_dir) if args.models_dir else (data / "models")
    service.models_dir.mkdir(parents=True, exist_ok=True)
    discovery = DiscoveryService(
        node.announce() if node is not None else None,
        port=args.discovery_port,
        on_announce=lambda announce, _addr: registry.upsert_announce(announce),
        token=token or None,
    )
    from fast_aggregation.models import Announce

    discovery.beacons.append(
        Announce(
            node_id="coordinator",
            name=args.name or "coordinator",
            advertise_host=advertise,
            http_port=choice.port,
            rpc_port=args.rpc_port,
            os_name="",
            arch="",
            role="coordinator",
            paired=True,
            summary="координатор",
            fingerprint=fingerprint,
        )
    )
    app = create_coordinator_app(service, discovery)
    lines = []
    if note:
        lines.append(note)
    lines.extend(
        [
            f"Дашборд: {dashboard_url}",
            f"OpenAI API: {dashboard_url}/v1",
            f"Для узлов: {advertise_url}",
            f"Сборка llama.cpp: {args.release}.",
            f"Журнал: {log_path}",
        ]
    )
    if code:
        from fast_aggregation.security.token import join_uri

        lines.extend(
            [
                f"Код сопряжения: {code}",
                f"Отпечаток: {fingerprint}",
                f"Ссылка: {join_uri(advertise, choice.port, code, fingerprint)}",
                "Управляющий канал зашифрован TLS. Поток llama.cpp RPC между машинами остаётся открытым TCP.",
            ]
        )
    else:
        lines.append("Режим --open: без токена и без TLS.")
    _print_banner("\n".join(lines))
    config = uvicorn.Config(app, host=args.bind, port=choice.port, log_level="info", **ssl_kwargs)
    server = uvicorn.Server(config)
    holder = getattr(args, "server_holder", None)
    if isinstance(holder, dict):
        holder["server"] = server
    try:
        server.run()
    except SystemExit as exc:
        message = f"Не удалось открыть порт {choice.port}. {format_owners(choice.port)} Журнал: {log_path}"
        _record_error(data, write_state, read_state, message)
        print(message, file=sys.stderr)
        wait_before_close(pause)
        return exc.code if isinstance(exc.code, int) else 1
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
    origin = pair_url.rsplit("/api/pair", 1)[0]
    if url.startswith("fastagg://"):
        coordinator_port = port
    else:
        from urllib.parse import urlparse

        coordinator_port = urlparse(pair_url).port or COORDINATOR_PORT
    save_membership(
        data,
        {
            "token": result["token"],
            "coordinator": pair_url,
            "coordinator_port": coordinator_port,
            "fingerprint": fingerprint,
            "runtime_mirror": origin + "/runtime",
        },
    )
    print("Сопряжение сохранено. Запустите: fastagg node")
    return 0


def _filled_namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "name": "",
        "bind": "0.0.0.0",
        "advertise": "",
        "rpc_port": RPC_PORT,
        "discovery_port": DISCOVERY_PORT,
        "backend": "auto",
        "vram_mb": 0,
        "cuda_build": "",
        "release": DEFAULT_LLAMA_RELEASE,
        "data_dir": "",
        "open": False,
        "insecure_http": False,
        "allow_download": False,
        "runtime_from": "",
        "port": None,
        "engine_port": ENGINE_PORT,
        "no_participate": False,
        "models_dir": "",
        "http_port": NODE_HTTP_PORT,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _tray(args: argparse.Namespace) -> int:
    import threading

    from fast_aggregation.instance import configure_log, health_ok, log_dir, read_state, tray_status_text
    from fast_aggregation.tray import run_tray

    data = _data_dir(args)
    logs = configure_log("tray" if args.role == "node" else "coordinator")
    holder: dict = {}
    thread_box: dict = {"thread": None}
    child = _filled_namespace(
        open=args.open,
        insecure_http=args.insecure_http,
        pause=False,
        lock_held=False,
        server_holder=holder,
        interactive=False,
    )

    def serve() -> None:
        target = _serve_node if args.role == "node" else _serve_coordinator
        try:
            target(child)
        except Exception as exc:
            from fast_aggregation.instance import write_state

            state = read_state(data)
            state["error"] = str(exc)
            write_state(data, state)

    def start_server() -> None:
        thread = threading.Thread(target=serve, name="fa-tray-server", daemon=True)
        thread_box["thread"] = thread
        thread.start()

    def restart() -> None:
        server = holder.get("server")
        if server is not None:
            server.should_exit = True
        thread = thread_box.get("thread")
        if thread is not None:
            thread.join(timeout=5)
        holder["server"] = None
        start_server()

    def status() -> str:
        if args.role != "coordinator":
            scheme = "http" if args.open or args.insecure_http else "https"
            url = f"{scheme}://127.0.0.1:{child.http_port}"
            return "работает" if health_ok(url) else "запускается"
        state = read_state(data)
        url = str(state.get("dashboard_url") or "")
        thread = thread_box.get("thread")
        alive = bool(thread and thread.is_alive())
        return tray_status_text(state, health_ok=health_ok(url), thread_alive=alive)

    def current_url() -> str | None:
        if args.role != "coordinator":
            return None
        return str(read_state(data).get("dashboard_url") or "") or None

    def stop_server() -> None:
        server = holder.get("server")
        if server is not None:
            server.should_exit = True

    if args.role == "coordinator":
        # Блокировку держит процесс трея, чтобы перезапуск не отпустил её между попытками.
        from fast_aggregation.instance import InstanceLock

        lock = InstanceLock(data / "coordinator.lock")
        if lock.try_acquire():
            child.lock_held = True
            start_server()

            def quit_primary() -> None:
                stop_server()
                lock.release()

            run_tray(
                url_getter=current_url,
                status=status,
                logs_dir=str(logs.parent),
                on_restart=restart,
                on_quit=quit_primary,
                on_copy=lambda: _copy_pairing_links(data),
            )
            lock.release()
            return 0
    else:
        start_server()
    node_page = None
    if args.role == "node":
        scheme = "http" if args.open or args.insecure_http else "https"
        node_page = f"{scheme}://127.0.0.1:{child.http_port}/pair"
    run_tray(
        url_getter=(lambda: node_page) if node_page else (current_url if args.role == "coordinator" else None),
        status=status,
        logs_dir=str(log_dir()),
        on_restart=None,
        on_quit=stop_server if args.role == "node" else None,
        open_label="Подключиться к координатору" if args.role == "node" else "Открыть дашборд",
        on_copy=(lambda: _copy_pairing_links(data)) if args.role == "coordinator" else None,
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    from fast_aggregation.cluster.bench import BenchRun, parse_timings

    holder: dict = {}

    def on_done(parsed: dict) -> dict:
        holder["parsed"] = parsed
        return parsed

    run = BenchRun()
    try:
        run.start(args.url, on_done)
    except FastAggError as exc:
        print(exc, file=sys.stderr)
        return 1
    thread = run._thread
    if thread is not None:
        thread.join(timeout=70)
    view = run.view()
    if view["status"] != "done":
        print(view.get("error") or view["status"], file=sys.stderr)
        return 1
    parsed = holder.get("parsed") or parse_timings({})
    print(f"префилл: {parsed['prefill_tps']} ток/с, токенов {parsed['prompt_tokens']}")
    print(f"генерация: {parsed['generation_tps']} ток/с, токенов {parsed['generated_tokens']}")
    return 0


def _pull(args: argparse.Namespace) -> int:
    import time

    from fast_aggregation.models_pull import ModelPuller, RemoteFile, save_token

    data = _data_dir(args)
    folder = Path(args.models_dir) if args.models_dir else data / "models"
    folder.mkdir(parents=True, exist_ok=True)
    if args.token:
        save_token(data, args.token)
    puller = ModelPuller(folder, data)
    try:
        try:
            inspected = puller.inspect(args.url)
        except FastAggError as exc:
            print(exc, file=sys.stderr)
            return 1
        choices = inspected["choices"]
        wanted = args.file.strip()
        if wanted:
            selected = [
                choice
                for choice in choices
                if choice["label"] == wanted
                or any(item["path"] == wanted or Path(item["path"]).name == wanted for item in choice["files"])
            ]
            if not selected:
                print("Такого файла нет. Доступны:", file=sys.stderr)
                for choice in choices:
                    print(f"  {choice['label']}", file=sys.stderr)
                return 2
        elif len(choices) > 1:
            print("В репозитории несколько GGUF. Повторите с --file и одной из подписей:", file=sys.stderr)
            for choice in choices:
                size = choice.get("size")
                suffix = f"{size / (1024 * 1024):.1f} МиБ" if isinstance(size, int) else "размер неизвестен"
                print(f"  {choice['label']} — {suffix}", file=sys.stderr)
            return 2
        else:
            selected = choices
        files = [
            RemoteFile(item["path"], item["url"], item.get("size"), item.get("sha256"))
            for choice in selected
            for item in choice["files"]
        ]
        try:
            puller.start(files)
        except FastAggError as exc:
            print(exc, file=sys.stderr)
            return 1
        while True:
            view = puller.view()
            total = view.get("bytes_total") or 0
            done = view.get("bytes_done") or 0
            speed = (view.get("speed_bps") or 0) / (1024 * 1024)
            print(
                f"\r{view.get('name') or ''} {done}/{total} байт, {speed:.1f} МиБ/с, {view.get('status')}",
                end="",
                flush=True,
            )
            if view.get("status") in {"done", "error", "cancelled"}:
                print()
                break
            time.sleep(0.4)
        if view.get("status") != "done":
            print(view.get("error") or view.get("status"), file=sys.stderr)
            return 1
        for path in view.get("paths") or []:
            print(path)
        return 0
    finally:
        puller.close()


def _runtime_import(args: argparse.Namespace) -> int:
    source = Path(args.path)
    cache = _data_dir(args) / "runtimes"
    files = [source] if source.is_file() else sorted(source.glob("*"))
    copied = 0
    for item in files:
        if not item.is_file():
            continue
        if item.suffix.lower() not in {".zip", ".gz"} and not item.name.endswith(".tar.gz"):
            continue
        try:
            dest = stage_archive(cache, args.tag, item)
        except FastAggError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(dest)
        copied += 1
    if copied == 0:
        print("Не найдено архивов .zip или .tar.gz", file=sys.stderr)
        return 1
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
    if getattr(args, "data_dir", ""):
        return Path(args.data_dir)
    return home_dir()


def _copy_pairing_links(data: Path) -> None:
    from fast_aggregation.netinfo import local_ipv4_addresses
    from fast_aggregation.security.certs import fingerprint_pem
    from fast_aggregation.security.token import join_uri

    code_path = data / "pairing_code"
    cert_path = data / "tls" / "cert.pem"
    if not code_path.is_file() or not cert_path.is_file():
        return
    from fast_aggregation.instance import read_state

    code = code_path.read_text(encoding="utf-8").strip()
    fingerprint = fingerprint_pem(cert_path.read_bytes())
    port = int(read_state(data).get("port") or COORDINATOR_PORT)
    hosts = [ip for ip in local_ipv4_addresses() if not ip.startswith("127.")] or ["127.0.0.1"]
    _copy_text("\n".join(join_uri(host, port, code, fingerprint) for host in hosts))


def _copy_text(text: str) -> None:
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(0):
            return
        try:
            user32.EmptyClipboard()
            raw = text.encode("utf-16-le") + b"\0\0"
            handle = kernel32.GlobalAlloc(0x0002, len(raw))
            locked = kernel32.GlobalLock(handle)
            ctypes.memmove(locked, raw, len(raw))
            kernel32.GlobalUnlock(handle)
            user32.SetClipboardData(13, handle)
        finally:
            user32.CloseClipboard()
        return
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=False)
    except OSError:
        return


def _node_joiner(data: Path, service):
    def join(pending: dict) -> dict:
        from fast_aggregation.security.certs import local_names
        from fast_aggregation.security.pin import pinned_post

        origin = str(pending["coordinator"]).rstrip("/")
        cert_path, _key, _fp = ensure_cert(data / "tls", local_names(service.settings.advertise_host))
        result = pinned_post(
            origin + "/api/pair",
            pending["fingerprint"],
            {
                "code": pending["code"],
                "node_id": service.settings.node_id,
                "name": service.settings.name,
                "advertise_host": service.settings.advertise_host,
                "http_port": service.settings.http_port,
                "rpc_port": service.settings.rpc_port,
                "cert_pem": cert_path.read_text(encoding="utf-8"),
            },
            data / "coordinator.pem",
        )
        save_membership(
            data,
            {
                "token": result["token"],
                "coordinator": origin + "/api/pair",
                "fingerprint": pending["fingerprint"],
                "runtime_mirror": origin + "/runtime",
            },
        )
        return result

    return join


def _print_banner(text: str) -> None:
    print(text, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
