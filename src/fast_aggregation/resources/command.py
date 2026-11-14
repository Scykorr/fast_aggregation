"""Запуск внешних утилит инвентаризации. В тестах подменяется."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SubprocessRunner:
    def which(self, name: str) -> bool:
        return shutil.which(name) is not None

    def run(self, args: list[str], timeout: float) -> CommandResult:
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return CommandResult(127, "", "not found")
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            return CommandResult(124, stdout, "timeout")
        return CommandResult(completed.returncode, completed.stdout or "", completed.stderr or "")


class FakeRunner:
    """Ответы по имени исполняемого файла. Для тестов Windows-веток на Linux."""

    def __init__(
        self,
        responses: dict[str, CommandResult] | None = None,
        present: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.present = present or set(self.responses)
        self.calls: list[list[str]] = []

    def which(self, name: str) -> bool:
        return name in self.present

    def run(self, args: list[str], timeout: float) -> CommandResult:
        del timeout
        self.calls.append(list(args))
        key = args[0] if args else ""
        if key not in self.responses:
            return CommandResult(127, "", "not found")
        return self.responses[key]
