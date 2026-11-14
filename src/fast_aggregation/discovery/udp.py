"""Слушатель и отправитель UDP-объявлений.

Боевой режим шлёт пакет на широковещательные адреса интерфейсов.
Тесты отключают broadcast и передают явные unicast-адреса, чтобы не
задевать реальную сеть виртуальной машины.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable

from fast_aggregation.constants import ANNOUNCE_INTERVAL_SEC, DISCOVERY_PORT
from fast_aggregation.discovery.protocol import decode_announce, encode_announce
from fast_aggregation.models import Announce
from fast_aggregation.netinfo import broadcast_targets

log = logging.getLogger(__name__)

OnAnnounce = Callable[[Announce, tuple[str, int]], None]


class DiscoveryService:
    def __init__(
        self,
        announce: Announce | None,
        *,
        port: int = DISCOVERY_PORT,
        bind_host: str = "",
        interval: float = ANNOUNCE_INTERVAL_SEC,
        enable_broadcast: bool = True,
        extra_targets: list[tuple[str, int]] | None = None,
        on_announce: OnAnnounce | None = None,
    ) -> None:
        self.announce = announce
        self.port = port
        self.bind_host = bind_host
        self.interval = interval
        self.enable_broadcast = enable_broadcast
        self.extra_targets = list(extra_targets or [])
        self.on_announce = on_announce
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        listener = threading.Thread(target=self._listen_loop, name="fa-discovery-listen", daemon=True)
        listener.start()
        self._threads.append(listener)
        if self.announce is not None:
            sender = threading.Thread(target=self._send_loop, name="fa-discovery-send", daemon=True)
            sender.start()
            self._threads.append(sender)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()

    def send_once(self) -> None:
        if self.announce is None:
            return
        packet = encode_announce(self.announce)
        targets = self._targets()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            for host, port in targets:
                try:
                    sock.sendto(packet, (host, port))
                except OSError as exc:
                    log.debug("discovery send to %s:%s failed: %s", host, port, exc)
        finally:
            sock.close()

    def _targets(self) -> list[tuple[str, int]]:
        targets = list(self.extra_targets)
        if self.enable_broadcast:
            for item in broadcast_targets():
                targets.append((item.address, self.port))
        unique: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for item in targets:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.bind((self.bind_host, self.port))
            sock.settimeout(0.4)
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(8192)
                except TimeoutError:
                    continue
                except OSError as exc:
                    log.debug("discovery recv failed: %s", exc)
                    continue
                announce = decode_announce(data)
                if announce is None:
                    continue
                if self.announce is not None and announce.node_id == self.announce.node_id:
                    continue
                if self.on_announce is not None:
                    try:
                        self.on_announce(announce, addr)
                    except Exception:
                        log.exception("discovery callback failed")
        finally:
            sock.close()

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            self.send_once()
            self._stop.wait(self.interval)
