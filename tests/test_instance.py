import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from fast_aggregation.constants import COORDINATOR_PORT
from fast_aggregation.errors import FastAggError
from fast_aggregation.instance import (
    InstanceLock,
    attach_message,
    candidate_ports,
    choose_coordinator_port,
    log_dir,
    preferred_port,
    read_state,
    tray_status_text,
    wait_before_close,
    write_state,
)


def test_default_dashboard_port_is_not_8080():
    assert COORDINATOR_PORT == 47080
    assert preferred_port(None, None) == 47080
    assert preferred_port(None, 47083) == 47083
    assert preferred_port(8080, 47083) == 8080


def test_next_port_comes_after_the_requested_one():
    ports = candidate_ports(47082, first=47080, last=47084)
    assert ports[:3] == [47082, 47083, 47084]
    outside = candidate_ports(8080, first=47080, last=47082)
    assert outside[0] == 8080
    assert 8081 in outside
    assert 47080 in outside


def _bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", port))
    sock.listen(1)
    return sock


def _free_span(count: int) -> int | None:
    for start in (47080, 27180, 28180, 29180, 30180):
        held = []
        try:
            for port in range(start, start + count):
                held.append(_bind(port))
        except OSError:
            for sock in held:
                sock.close()
            continue
        for sock in held:
            sock.close()
        return start
    return None


def test_busy_port_falls_forward_and_names_the_owner():
    start = _free_span(3)
    assert start is not None
    busy = _bind(start)
    try:
        choice = choose_coordinator_port(start, first=start, last=start + 2)
    finally:
        busy.close()
    assert choice.port == start + 1
    assert choice.notes
    assert "занят" in choice.notes[0]
    assert "pid" in choice.notes[0]
    assert "не fastagg" in choice.notes[0]
    assert str(choice.port) in choice.notes[-1]


def test_no_free_port_explains_who_holds_it():
    start = _free_span(2)
    assert start is not None
    held = [_bind(start), _bind(start + 1)]
    try:
        with pytest.raises(FastAggError) as caught:
            choose_coordinator_port(start, first=start, last=start + 1)
    finally:
        for sock in held:
            sock.close()
    assert "Свободного порта" in str(caught.value)
    assert "занят" in str(caught.value)


def test_state_roundtrip_and_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    write_state(tmp_path, {"port": 47081, "dashboard_url": "https://127.0.0.1:47081", "error": None})
    assert read_state(tmp_path)["port"] == 47081
    monkeypatch.setenv("FA_LOG_DIR", str(tmp_path / "logs"))
    assert log_dir() == tmp_path / "logs"
    monkeypatch.delenv("FA_LOG_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.delenv("FA_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert log_dir() == tmp_path / "Local" / "fastagg" / "logs"


def test_tray_status_shows_the_owner_instead_of_starting_forever():
    state = {"error": "Порт 8080 занят: httpd.exe (pid 4000)."}
    assert tray_status_text(state, health_ok=False, thread_alive=False).startswith("ошибка:")
    assert "httpd.exe" in tray_status_text(state, health_ok=False, thread_alive=True)
    assert tray_status_text({}, health_ok=True, thread_alive=True) == "работает"
    assert tray_status_text({}, health_ok=False, thread_alive=True) == "запускается"
    assert "остановился" in tray_status_text({}, health_ok=False, thread_alive=False)


def test_attach_message_points_at_the_existing_dashboard():
    text = attach_message({"dashboard_url": "https://127.0.0.1:47081"})
    assert "уже запущен" in text
    assert "https://127.0.0.1:47081" in text


def test_fatal_window_does_not_wait_without_a_console(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FA_NO_PAUSE", "1")
    wait_before_close()
    monkeypatch.delenv("FA_NO_PAUSE")
    monkeypatch.setenv("CI", "true")
    wait_before_close()


def test_second_process_cannot_take_the_coordinator_lock(tmp_path: Path):
    script = tmp_path / "hold.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from fast_aggregation.instance import InstanceLock\n"
        "lock = InstanceLock(Path(sys.argv[1]))\n"
        "print('acquired' if lock.try_acquire() else 'denied', flush=True)\n"
        "sys.stdin.readline()\n"
        "lock.release()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path / "coordinator.lock")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    try:
        line = proc.stdout.readline() if proc.stdout is not None else b""
        assert line.strip() == b"acquired"
        lock = InstanceLock(tmp_path / "coordinator.lock")
        assert lock.try_acquire() is False
    finally:
        if proc.stdin is not None:
            proc.stdin.write(b"\n")
            proc.stdin.flush()
        proc.wait(timeout=10)
    lock = InstanceLock(tmp_path / "coordinator.lock")
    assert lock.try_acquire() is True
    lock.release()
