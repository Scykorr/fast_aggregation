"""Unicast-обход подсети, когда broadcast по Wi-Fi не доходит.

Клиентская изоляция точки доступа режет и unicast между клиентами.
Тогда помогает только общий L2/L3 без изоляции или ручной адрес.
Скан ограничен сетями не шире /24, чтобы не обходить кампус.
"""

from __future__ import annotations

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor

from fast_aggregation.errors import FastAggError


def hosts_in_cidr(cidr: str) -> list[str]:
    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError as exc:
        raise FastAggError(f"Некорректная подсеть: {cidr}") from exc
    if network.version != 4:
        raise FastAggError("Автоскан пока только для IPv4")
    if network.prefixlen < 24:
        raise FastAggError(
            f"Подсеть {network} шире /24. Укажите конкретную /24, например 192.168.1.0/24. "
            "Обход /16 затопит сеть."
        )
    return [str(host) for host in network.hosts()]


def scan_tcp(hosts: list[str], port: int, *, timeout: float = 0.25, workers: int = 64) -> list[str]:
    """Возвращает адреса, у которых TCP-порт принял соединение."""

    if not hosts:
        return []

    def probe(host: str) -> str | None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            if sock.connect_ex((host, port)) == 0:
                return host
        except OSError:
            return None
        finally:
            sock.close()
        return None

    found: list[str] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(hosts))) as pool:
        for host in pool.map(probe, hosts):
            if host:
                found.append(host)
    return found
