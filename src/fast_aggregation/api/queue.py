"""Одна генерация за раз. Чат, Ollama и замер скорости делят эту очередь."""

from __future__ import annotations

import threading


class GenerationQueue:
    def __init__(self, *, limit: int = 1, max_waiting: int = 8) -> None:
        self.limit = max(1, limit)
        self.max_waiting = max(0, max_waiting)
        self._sem = threading.Semaphore(self.limit)
        self._lock = threading.Lock()
        self.active = 0
        self.waiting = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self.active,
                "waiting": self.waiting,
                "limit": self.limit,
                "max_waiting": self.max_waiting,
            }

    def acquire(self) -> bool:
        """False, если очередь уже длиннее max_waiting. Тогда клиенту нужен 429."""

        with self._lock:
            if self.waiting >= self.max_waiting:
                return False
            self.waiting += 1
        self._sem.acquire()
        with self._lock:
            self.waiting -= 1
            self.active += 1
        return True

    def release(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
        self._sem.release()

    @property
    def busy(self) -> bool:
        snap = self.snapshot()
        return snap["active"] > 0 or snap["waiting"] > 0
