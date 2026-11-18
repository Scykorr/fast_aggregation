"""Короткий TLS-мост: открытый TCP с одной стороны, TLS с другой.

llama-server говорит с RPC простым TCP. Мост на координаторе принимает
этот TCP на localhost и везёт байты к узлу по TLS. На узле встречный
мост принимает TLS и отдаёт байты локальному ggml-rpc-server.
"""

from __future__ import annotations

import socket
import ssl
import threading
from pathlib import Path


class TlsTunnel:
    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        target_host: str,
        target_port: int,
        *,
        server_cert: tuple[Path, Path] | None = None,
        client_ca: Path | None = None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.server_cert = server_cert
        self.client_ca = client_ca
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.listen_host, self.listen_port))
        sock.listen(8)
        sock.settimeout(0.4)
        self._sock = sock
        self._thread = threading.Thread(target=self._accept_loop, name="fa-tls-tunnel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                client, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._bridge, args=(client,), daemon=True).start()

    def _bridge(self, client: socket.socket) -> None:
        remote: socket.socket | None = None
        try:
            left = self._wrap_listen(client)
            remote = socket.create_connection((self.target_host, self.target_port), timeout=5)
            right = self._wrap_target(remote)
            _pump(left, right)
        except OSError:
            pass
        finally:
            for sock in (client, remote):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _wrap_listen(self, sock: socket.socket) -> socket.socket:
        if self.server_cert is None:
            return sock
        cert, key = self.server_cert
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        return ctx.wrap_socket(sock, server_side=True)

    def _wrap_target(self, sock: socket.socket) -> socket.socket:
        if self.client_ca is None:
            return sock
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(self.client_ca)
        ctx.check_hostname = False
        return ctx.wrap_socket(sock, server_side=False)


def _pump(left: socket.socket, right: socket.socket) -> None:
    left.settimeout(0.5)
    right.settimeout(0.5)
    stop = threading.Event()

    def copy(src: socket.socket, dst: socket.socket) -> None:
        try:
            while not stop.is_set():
                try:
                    chunk = src.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    break
                dst.sendall(chunk)
        except OSError:
            pass
        finally:
            stop.set()
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    first = threading.Thread(target=copy, args=(left, right), daemon=True)
    second = threading.Thread(target=copy, args=(right, left), daemon=True)
    first.start()
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)
