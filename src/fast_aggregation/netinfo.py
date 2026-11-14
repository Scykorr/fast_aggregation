"""Локальные IPv4-адреса и широковещательные адреса интерфейсов."""

from __future__ import annotations

import socket
from dataclasses import dataclass

import psutil


SKIP_INTERFACE_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "cni", "flannel", "zt")


@dataclass(frozen=True)
class BroadcastTarget:
    address: str
    interface: str


def guess_advertise_host() -> str:
    """Адрес, с которого исходящий UDP выбрал бы маршрут в локальную сеть.

    Соединение не устанавливается и пакеты не отправляются: ядро только
    выбирает локальный адрес источника.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        host = sock.getsockname()[0]
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    if host.startswith("127."):
        for address in local_ipv4_addresses():
            if not address.startswith("127."):
                return address
    return host


def local_ipv4_addresses() -> list[str]:
    found: list[str] = []
    for name, addresses in psutil.net_if_addrs().items():
        if _skip_interface(name):
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            ip = address.address
            if ip and ip not in found:
                found.append(ip)
    return found


def broadcast_targets() -> list[BroadcastTarget]:
    targets: list[BroadcastTarget] = []
    seen: set[str] = set()
    for name, addresses in psutil.net_if_addrs().items():
        if _skip_interface(name):
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            broadcast = getattr(address, "broadcast", None)
            if not broadcast or broadcast in seen:
                continue
            if broadcast.startswith("127.") or broadcast == "0.0.0.0":
                continue
            seen.add(broadcast)
            targets.append(BroadcastTarget(address=broadcast, interface=name))
    if "255.255.255.255" not in seen:
        targets.append(BroadcastTarget(address="255.255.255.255", interface="*"))
    return targets


def _skip_interface(name: str) -> bool:
    lowered = name.lower()
    return lowered == "lo" or lowered.startswith(SKIP_INTERFACE_PREFIXES)
