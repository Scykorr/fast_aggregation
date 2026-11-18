"""Выбор порта и сетевого адаптера, когда VPN занимает привычные номера.

Windows с Hyper-V, WSL и полным туннелем VPN отвечает WSAEADDRINUSE на bind,
хотя netstat не показывает владельца. Такие порты перечислены в
`netsh int ipv4 show excludedportrange protocol=tcp`. Их надо обходить,
а если окно 47080–47099 целиком недоступно — брать другой диапазон
или эфемерный порт, который выдала система.
"""

from __future__ import annotations

import re
import socket
import subprocess
from dataclasses import dataclass

import psutil

from fast_aggregation.errors import FastAggError

_RANGE_LINE = re.compile(r"^\s*(\d+)\s+(\d+)")
VPN_MARKERS = (
    "tap",
    "tun",
    "wintun",
    "wireguard",
    "nordlynx",
    "tailscale",
    "zerotier",
    "openvpn",
    "anyconnect",
    "fortinet",
    "cisco",
    "outline",
    "radmin",
    "hamachi",
    "softether",
    "manta",
    "vpn",
    "wg",
    "ppp",
)
# Несколько окон, затем эфемерный порт. 50000–50059 на этой машине уже исключены.
DASHBOARD_RANGES: tuple[tuple[int, int], ...] = (
    (47080, 47099),
    (47100, 47499),
    (18090, 18189),
    (28080, 28179),
)
RPC_RANGES: tuple[tuple[int, int], ...] = (
    (50052, 50052),
    (50160, 50240),
    (47150, 47249),
)
API_RANGES: tuple[tuple[int, int], ...] = (
    (11435, 11435),
    (11436, 11450),
    (18083, 18089),
)


@dataclass(frozen=True)
class Adapter:
    name: str
    ip: str
    vpn: bool = False


@dataclass(frozen=True)
class ListenChoice:
    port: int
    host: str
    notes: tuple[str, ...]


def parse_excluded_ranges(text: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for line in text.splitlines():
        match = _RANGE_LINE.match(line)
        if not match:
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if 0 < start <= end <= 65535:
            found.append((start, end))
    return found


def port_excluded(port: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= port <= end for start, end in ranges)


def load_excluded_ranges() -> list[tuple[int, int]]:
    """Диапазоны TCP, которые Windows не отдаёт. На других ОС список пуст."""

    try:
        completed = subprocess.run(
            ["netsh", "int", "ipv4", "show", "excludedportrange", "protocol=tcp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_excluded_ranges((completed.stdout or "") + "\n" + (completed.stderr or ""))


def vpn_port_message(port: int) -> str:
    return (
        f"Порт {port} не слушает ни один процесс, но система не даёт его занять (WSAEADDRINUSE). "
        "Так бывает при полном туннеле VPN или когда Hyper-V/WSL исключил диапазон. "
        "Список: netsh int ipv4 show excludedportrange protocol=tcp."
    )


def _bindable(port: int, host: str, probe) -> bool:
    if probe is not None:
        return bool(probe(port, host))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    else:
        return True
    finally:
        sock.close()


def _ephemeral(host: str) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def choose_listening_port(
    preferred: int,
    *,
    host: str = "0.0.0.0",
    ranges: tuple[tuple[int, int], ...] = DASHBOARD_RANGES,
    avoid: set[int] | None = None,
    excluded: list[tuple[int, int]] | None = None,
    probe=None,
    ephemeral=None,
    label: str = "службы",
) -> ListenChoice:
    """Первый порт, который реально открывается bind. Исключённые диапазоны пропускаются."""

    blocked = excluded or []
    skipped = set(avoid or set())
    notes: list[str] = []
    if port_excluded(preferred, blocked):
        notes.append(vpn_port_message(preferred))

    candidates: list[int] = []
    if preferred not in candidates:
        candidates.append(preferred)
    for start, end in ranges:
        for port in range(start, end + 1):
            if port not in candidates:
                candidates.append(port)

    for port in candidates:
        if port in skipped or port_excluded(port, blocked):
            continue
        if _bindable(port, host, probe):
            if port != preferred:
                notes.append(f"Для {label} выбран порт {port} вместо {preferred}.")
            return ListenChoice(port, host, tuple(notes))
        if port == preferred and not notes:
            notes.append(vpn_port_message(port))

    borrow = ephemeral or _ephemeral
    for _ in range(12):
        try:
            port = int(borrow(host))
        except OSError:
            break
        if port in skipped or port_excluded(port, blocked):
            continue
        if _bindable(port, host, probe):
            notes.append(
                f"Для {label} система выдала свободный порт {port}: привычные диапазоны недоступны."
            )
            return ListenChoice(port, host, tuple(notes))
    span = ", ".join(f"{start}–{end}" for start, end in ranges)
    raise FastAggError(
        f"Свободного порта для {label} нет (смотрели {span} и эфемерные). {vpn_port_message(preferred)}"
    )


def is_vpn_name(name: str) -> bool:
    lowered = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    compact_markers = tuple(marker.replace("-", "") for marker in VPN_MARKERS)
    return any(marker in lowered for marker in compact_markers)


def is_private_ipv4(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    octets = [int(part) for part in parts]
    if octets[0] == 10:
        return True
    if octets[0] == 192 and octets[1] == 168:
        return True
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    return False


def pick_lan_adapter(adapters: list[Adapter], explicit: str = "") -> Adapter | None:
    """Частный адрес не-VPN интерфейса. Явный IP или имя имеет приоритет."""

    wanted = explicit.strip()
    if wanted:
        for adapter in adapters:
            if adapter.ip == wanted or adapter.name == wanted:
                return adapter
        if is_private_ipv4(wanted) or wanted.count(".") == 3:
            return Adapter(name=wanted, ip=wanted, vpn=False)
    private = [item for item in adapters if is_private_ipv4(item.ip) and not item.vpn]
    if private:
        return private[0]
    any_lan = [item for item in adapters if is_private_ipv4(item.ip)]
    return any_lan[0] if any_lan else None


def lan_adapters() -> list[Adapter]:
    from fast_aggregation.netinfo import _skip_interface

    found: list[Adapter] = []
    for name, addresses in psutil.net_if_addrs().items():
        if _skip_interface(name):
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            ip = address.address or ""
            if not ip or ip.startswith("127."):
                continue
            found.append(Adapter(name=name, ip=ip, vpn=is_vpn_name(name)))
    return found


def egress_ip(host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, 9))
        return str(sock.getsockname()[0])
    except OSError:
        return ""
    finally:
        sock.close()


@dataclass(frozen=True)
class StartupPlan:
    dashboard_port: int
    dashboard_host: str
    api_port: int
    api_bind: str
    rpc_port: int
    rpc_host: str
    discovery_port: int
    advertise: str
    adapter_name: str
    notes: tuple[str, ...]


def plan_ports(
    *,
    dashboard: int | None,
    api: int | None,
    rpc: int | None,
    discovery: int | None,
    bind: str,
    advertise: str,
    saved: dict,
    excluded: list[tuple[int, int]] | None = None,
    adapters: list[Adapter] | None = None,
    probe=None,
    ephemeral=None,
) -> StartupPlan:
    """Порты и адрес, которые реально открываются. Сохранённые значения используются снова."""

    blocked = list(excluded or [])
    cards = list(adapters or [])
    chosen = pick_lan_adapter(cards, advertise or (bind if bind not in {"", "0.0.0.0"} else ""))
    advertise_ip = advertise or (chosen.ip if chosen else "127.0.0.1")
    adapter_name = chosen.name if chosen else ""
    node_host = bind if bind not in {"", "0.0.0.0"} else (chosen.ip if chosen else "0.0.0.0")
    notes: list[str] = []

    def saved_port(key: str) -> int | None:
        try:
            value = int(saved.get(key) or 0)
        except (TypeError, ValueError):
            return None
        return value or None

    dash_pref = dashboard or saved_port("port") or 47080
    dash = choose_listening_port(
        dash_pref,
        host="0.0.0.0" if bind in {"", "0.0.0.0"} else node_host,
        ranges=DASHBOARD_RANGES,
        excluded=blocked,
        probe=probe,
        ephemeral=ephemeral,
        label="дашборда",
    )
    notes.extend(dash.notes)

    api_pref = api or saved_port("api_port") or 11435
    api_choice = choose_listening_port(
        api_pref,
        host="127.0.0.1",
        ranges=API_RANGES,
        avoid={dash.port},
        excluded=blocked,
        probe=probe,
        ephemeral=ephemeral,
        label="API",
    )
    notes.extend(api_choice.notes)

    rpc_pref = rpc if rpc not in {None, 0} else (saved_port("rpc_port") or 50052)
    rpc_choice = choose_listening_port(
        rpc_pref,
        host=node_host,
        ranges=RPC_RANGES,
        avoid={dash.port, api_choice.port},
        excluded=blocked,
        probe=probe,
        ephemeral=ephemeral,
        label="RPC",
    )
    notes.extend(rpc_choice.notes)

    disc_pref = discovery or saved_port("discovery_port") or 45221
    discovery_port = disc_pref
    if port_excluded(disc_pref, blocked):
        discovery_port = 45222 if not port_excluded(45222, blocked) else 45921
        notes.append(f"UDP-объявления перенесены на порт {discovery_port}.")

    if chosen and chosen.vpn:
        notes.append(f"Адрес {chosen.ip} принадлежит {chosen.name}. Это похоже на VPN, локальная сеть может быть недоступна.")
    elif adapter_name:
        notes.append(f"Сеть узла: {adapter_name} {advertise_ip}.")

    return StartupPlan(
        dashboard_port=dash.port,
        dashboard_host=dash.host,
        api_port=api_choice.port,
        api_bind="127.0.0.1",
        rpc_port=rpc_choice.port,
        rpc_host=node_host if node_host not in {"", "0.0.0.0"} else "0.0.0.0",
        discovery_port=discovery_port,
        advertise=advertise_ip,
        adapter_name=adapter_name,
        notes=tuple(notes),
    )


def route_warning(host: str, adapters: list[Adapter], egress: str) -> str:
    for adapter in adapters:
        if adapter.ip == egress and adapter.vpn:
            return (
                f"Маршрут к {host} выходит через {adapter.name} ({egress}). "
                "Полный туннель VPN может не пускать узлы локальной сети."
            )
    return ""
