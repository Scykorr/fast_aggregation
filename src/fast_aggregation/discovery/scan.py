"""Unicast-обход подсети, когда broadcast по Wi-Fi не доходит.

Клиентская изоляция точки доступа режет и unicast между клиентами.
Тогда помогает только общий L2/L3 без изоляции или ручной адрес.
Скан ограничен сетями не шире /24, чтобы не обходить кампус.
Проба /hello идёт и по HTTPS с самоподписанным сертификатом, и по HTTP.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

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


_REFUSED = {111, 61, 10061, 10054}


@dataclass
class HostProbe:
    host: str
    state: str
    detail: str = ""
    hello: dict | None = None


def fetch_hello(host: str, port: int, timeout: float = 1.2) -> dict | None:
    """Сначала HTTPS без проверки сертификата, затем обычный HTTP."""

    for scheme in ("https", "http"):
        try:
            response = httpx.get(
                f"{scheme}://{host}:{port}/hello",
                timeout=timeout,
                verify=False,
            )
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("software") == "fast-aggregation":
            payload["scheme"] = scheme
            return payload
    return None


def probe_host(host: str, port: int, *, timeout: float = 0.4, http_timeout: float = 1.2) -> HostProbe:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        err = sock.connect_ex((host, port))
    except OSError as exc:
        return HostProbe(host, "timeout", str(exc))
    finally:
        sock.close()
    if err != 0:
        if err in _REFUSED:
            return HostProbe(host, "refused", "порт закрыт, служба узла не слушает")
        return HostProbe(host, "timeout", f"нет ответа, код {err}")
    hello = fetch_hello(host, port, http_timeout)
    if hello is None:
        return HostProbe(host, "foreign", "порт открыт, но /hello не ответил как fastagg")
    summary = str(hello.get("summary") or hello.get("name") or "")
    return HostProbe(host, "hello", summary, hello)


def scan_hosts(
    hosts: list[str],
    port: int,
    *,
    timeout: float = 0.4,
    http_timeout: float = 1.2,
    workers: int = 48,
    on_progress=None,
) -> list[HostProbe]:
    if not hosts:
        return []
    done = 0
    lock = threading.Lock()

    def one(host: str) -> HostProbe:
        nonlocal done
        result = probe_host(host, port, timeout=timeout, http_timeout=http_timeout)
        with lock:
            done += 1
            current = done
        if on_progress is not None:
            on_progress(current, len(hosts), result)
        return result

    with ThreadPoolExecutor(max_workers=min(workers, len(hosts))) as pool:
        return list(pool.map(one, hosts))


def explain_scan(results: list[HostProbe]) -> str:
    total = len(results)
    hellos = [item for item in results if item.state == "hello"]
    if hellos:
        unpaired = sum(1 for item in hellos if not (item.hello or {}).get("paired", True))
        return f"Проверено {total}. Узлов fastagg: {len(hellos)}, без сопряжения: {unpaired}."
    refused = [item for item in results if item.state == "refused"]
    foreign = [item for item in results if item.state == "foreign"]
    if refused:
        sample = ", ".join(item.host for item in refused[:5])
        return (
            f"Проверено {total}. {len(refused)} компьютеров живы, но порт закрыт ({sample}). "
            "Служба узла fastagg на них не запущена."
        )
    if foreign:
        sample = ", ".join(item.host for item in foreign[:5])
        return (
            f"Проверено {total}. Порт открыт на {sample}, но это не ответ fastagg /hello. "
            "Возможно, порт занят другой программой."
        )
    return (
        f"Проверено {total} адресов, порт узла нигде не открылся. "
        "Если ping до этих машин проходит, сеть часто помечена как «Общедоступная», "
        "а старое правило брандмауэра пускало только частную сеть, либо узел не запущен."
    )
