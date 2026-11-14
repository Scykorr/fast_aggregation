"""Долгоживущий процесс llama.cpp с кольцевым буфером журнала."""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from pathlib import Path


class ManagedProcess:
    def __init__(self, args: list[str], *, cwd: Path | None, log_path: Path, env: dict[str, str] | None = None) -> None:
        self.args = list(args)
        self.cwd = cwd
        self.log_path = log_path
        self.env = env
        self._proc: subprocess.Popen[str] | None = None
        self._lines: deque[str] = deque(maxlen=400)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.running():
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = os.environ.copy()
        environment.setdefault("GGML_RPC_NO_RDMA", "1")
        if self.env:
            environment.update(self.env)
        self._proc = subprocess.Popen(
            self.args,
            cwd=str(self.cwd) if self.cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=creationflags,
        )
        self._thread = threading.Thread(target=self._pump, name="fa-proc-log", daemon=True)
        self._thread.start()

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def exit_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def pid(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.pid

    def stop(self, timeout: float = 8) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._proc = None

    def tail(self, limit: int = 80) -> list[str]:
        with self._lock:
            rows = list(self._lines)
        if limit <= 0:
            return rows
        return rows[-limit:]

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        with self.log_path.open("a", encoding="utf-8", errors="replace") as handle:
            for line in proc.stdout:
                clean = line.rstrip("\r\n")
                with self._lock:
                    self._lines.append(clean)
                handle.write(line)
            handle.flush()
