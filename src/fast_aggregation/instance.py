"""Один координатор на машину и выбор свободного порта дашборда.

Файл блокировки не даёт второму окну снова занять порт. Если порт
уже слушает чужая программа (на 8080 это часто httpd из EDB Postgres),
берётся следующий свободный порт из окна 47080–47099. Выбранный порт
и точный URL дашборда пишутся в coordinator.json: и консоль, и трей
читают один и тот же файл.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psutil

from fast_aggregation.constants import COORDINATOR_PORT, COORDINATOR_PORT_SPAN
from fast_aggregation.errors import FastAggError
from fast_aggregation.paths import home_dir

log = logging.getLogger(__name__)

_LOG_READY = False


@dataclass(frozen=True)
class PortOwner:
    pid: int | None
    name: str


@dataclass(frozen=True)
class PortChoice:
    port: int
    notes: tuple[str, ...]


def coordinator_port_last() -> int:
    return COORDINATOR_PORT + COORDINATOR_PORT_SPAN - 1


def log_dir() -> Path:
    """Журналы. На Windows это %LOCALAPPDATA%\\fastagg\\logs."""

    override = os.environ.get("FA_LOG_DIR")
    if override:
        return Path(override)
    home = os.environ.get("FA_HOME")
    if home:
        return Path(home) / "logs"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "fastagg" / "logs"
    return home_dir() / "logs"


def configure_log(name: str) -> Path:
    global _LOG_READY
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.log"
    if not _LOG_READY:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _LOG_READY = True
    return path


def state_path(data_dir: Path) -> Path:
    return data_dir / "coordinator.json"


def read_state(data_dir: Path) -> dict:
    path = state_path(data_dir)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_state(data_dir: Path, state: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = state_path(data_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def preferred_port(explicit: int | None, saved_port: int | None) -> int:
    if explicit is not None:
        return explicit
    if saved_port:
        return int(saved_port)
    return COORDINATOR_PORT


def port_free(port: int, host: str = "0.0.0.0") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    else:
        return True
    finally:
        sock.close()


def owners_of(port: int) -> list[PortOwner]:
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, OSError):
        return []
    found: list[PortOwner] = []
    seen: set[int | None] = set()
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or not conn.laddr or conn.laddr.port != port:
            continue
        if conn.pid in seen:
            continue
        seen.add(conn.pid)
        name = "неизвестный процесс"
        if conn.pid:
            try:
                name = psutil.Process(conn.pid).name()
            except (psutil.Error, OSError):
                name = "неизвестный процесс"
        found.append(PortOwner(conn.pid, name))
    return found


def format_owners(port: int) -> str:
    owners = owners_of(port)
    if not owners:
        return f"Порт {port} занят, имя процесса узнать не удалось."
    parts = []
    for owner in owners:
        if owner.pid:
            parts.append(f"{owner.name} (pid {owner.pid})")
        else:
            parts.append(owner.name)
    return f"Порт {port} занят: {', '.join(parts)}."


def candidate_ports(preferred: int, *, first: int, last: int) -> list[int]:
    ordered = [preferred]
    width = last - first + 1
    if first <= preferred <= last:
        ordered.extend(range(preferred + 1, last + 1))
        ordered.extend(range(first, preferred))
    else:
        ordered.extend(range(preferred + 1, preferred + width))
        ordered.extend(range(first, last + 1))
    unique: list[int] = []
    for port in ordered:
        if port not in unique and 0 < port < 65536:
            unique.append(port)
    return unique


def choose_coordinator_port(
    preferred: int,
    *,
    first: int | None = None,
    last: int | None = None,
    avoid: set[int] | None = None,
) -> PortChoice:
    """Первый свободный порт. Занятый preferred описан в notes, затем берётся следующий."""

    start = COORDINATOR_PORT if first is None else first
    end = coordinator_port_last() if last is None else last
    skipped = avoid or set()
    notes: list[str] = []
    saw_busy = False
    for port in candidate_ports(preferred, first=start, last=end):
        if port in skipped:
            continue
        if port_free(port):
            if saw_busy:
                notes.append(f"Дашборд слушает порт {port}.")
            return PortChoice(port, tuple(notes))
        saw_busy = True
        if not notes:
            notes.append(format_owners(port) + " Это не fastagg.")
    busy = format_owners(preferred)
    raise FastAggError(f"Свободного порта для дашборда нет ({start}–{end}). {busy}")


def health_ok(url: str, timeout: float = 1.5) -> bool:
    if not url:
        return False
    target = url.rstrip("/") + "/health"
    context = None
    if target.startswith("https://"):
        import ssl

        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(target, timeout=timeout, context=context) as response:
            body = response.read(300).decode("utf-8", "replace")
            return response.status == 200 and "ok" in body
    except (OSError, ValueError):
        return False


def attach_message(state: dict) -> str:
    url = str(state.get("dashboard_url") or "")
    return (
        f"Координатор уже запущен: {url}\n"
        "Второе окно порт не занимает. Открываю этот дашборд в браузере."
    )


def tray_status_text(state: dict | None, *, health_ok: bool, thread_alive: bool) -> str:
    if health_ok:
        return "работает"
    error = str((state or {}).get("error") or "").strip()
    if error:
        return "ошибка: " + error
    if not thread_alive:
        note = str((state or {}).get("note") or "").strip()
        if note:
            return "ошибка: " + note
        return "ошибка: координатор остановился"
    return "запускается"


def wait_before_close(enabled: bool = True) -> None:
    """Окно ярлыка не закрывается мгновенно, пока человек не прочитает ошибку."""

    if not enabled:
        return
    if os.environ.get("FA_NO_PAUSE") == "1" or os.environ.get("CI"):
        return
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return
    try:
        input("\nНажмите Enter, чтобы закрыть окно...")
    except EOFError:
        return


class InstanceLock:
    """Эксклюзивная блокировка файла. На Windows её держит и именованный мьютекс."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None
        self._mutex = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                mutex = _windows_mutex()
                if mutex is None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    handle.close()
                    return False
                self._mutex = mutex
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._fh = handle
        return True

    def release(self) -> None:
        handle = self._fh
        self._fh = None
        if handle is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        mutex = self._mutex
        self._mutex = None
        if mutex is not None:
            _release_windows_mutex(mutex)


def _windows_mutex():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    created = kernel32.CreateMutexW(None, False, "Local\\fastagg-coordinator")
    if not created:
        return None
    waited = kernel32.WaitForSingleObject(created, 0)
    # WAIT_OBJECT_0 = 0, WAIT_ABANDONED = 128. Оба значат, что мьютекс наш.
    if waited not in (0, 128):
        kernel32.CloseHandle(created)
        return None
    return created


def _release_windows_mutex(handle) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex(handle)
    kernel32.CloseHandle(handle)
