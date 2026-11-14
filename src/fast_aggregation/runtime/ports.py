"""Ожидание TCP-порта, который открывает ggml-rpc-server или llama-server."""

from __future__ import annotations

import socket
import time


def wait_for_port(port: int, timeout: float, host: str = "127.0.0.1") -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        try:
            if sock.connect_ex((host, port)) == 0:
                return True
        except OSError:
            return False
        finally:
            sock.close()
        time.sleep(0.15)
    return False
