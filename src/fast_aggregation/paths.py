"""Каталог состояния пользователя. FA_HOME переопределяет его в тестах."""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    override = os.environ.get("FA_HOME")
    if override:
        return Path(override)
    return Path.home() / ".fast_aggregation"


def load_node_id(path: Path) -> str:
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid

    value = str(uuid.uuid4())
    path.write_text(value, encoding="utf-8")
    return value
