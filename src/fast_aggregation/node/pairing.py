"""Сопряжение, которое инициирует координатор или страница узла.

Предложение хранит код и отпечаток. Узел показывает их человеку.
Автоприём включается файлом auto-pair в каталоге данных.
"""

from __future__ import annotations

from pathlib import Path

from fast_aggregation.errors import FastAggError
from fast_aggregation.models import Announce


class PairSession:
    def __init__(self, data_dir: Path, *, joiner=None) -> None:
        self.data_dir = data_dir
        self.pending: dict | None = None
        self.coordinators: dict[str, dict] = {}
        self.joiner = joiner
        self.announce: Announce | None = None
        self.auth = {"token": ""}

    def auto(self) -> bool:
        return (self.data_dir / "auto-pair").is_file()

    def set_auto(self, enabled: bool) -> None:
        path = self.data_dir / "auto-pair"
        if enabled:
            path.write_text("1", encoding="utf-8")
        elif path.exists():
            path.unlink()

    def note_coordinator(self, announce: Announce) -> None:
        if announce.role != "coordinator":
            return
        self.coordinators[f"{announce.advertise_host}:{announce.http_port}"] = {
            "name": announce.name,
            "host": announce.advertise_host,
            "port": announce.http_port,
            "summary": announce.summary,
            "fingerprint": announce.fingerprint,
        }

    def offer(self, body: dict) -> dict:
        coordinator = str(body.get("coordinator") or "").strip()
        code = str(body.get("code") or "").strip().upper()
        fingerprint = str(body.get("fingerprint") or "").strip()
        if not coordinator or not code or not fingerprint:
            raise FastAggError("В предложении нужны адрес координатора, код и отпечаток")
        self.pending = {"coordinator": coordinator, "code": code, "fingerprint": fingerprint}
        print(
            f"Координатор {coordinator} предлагает сопряжение. Код {code}. "
            "Сверьте его с экраном координатора и примите на странице узла.",
            flush=True,
        )
        if self.auto():
            return self.accept()
        return {"status": "pending", **self.pending}

    def accept(self, code: str | None = None) -> dict:
        if not self.pending:
            raise FastAggError("Нет предложения сопряжения")
        if code and code.strip().upper() != self.pending["code"]:
            raise FastAggError("Код не совпал с предложением координатора")
        if self.joiner is None:
            raise FastAggError("Сопряжение на этом узле не настроено")
        result = self.joiner(self.pending)
        token = str(result.get("token") or "")
        if not token:
            raise FastAggError("Координатор не выдал токен")
        self.auth["token"] = token
        if self.announce is not None:
            self.announce.paired = True
        self.pending = None
        return {"status": "paired"}

    def reject(self) -> dict:
        self.pending = None
        return {"status": "idle"}


def hardware_summary(resources) -> str:
    parts: list[str] = []
    if getattr(resources, "cpu_model", ""):
        parts.append(str(resources.cpu_model))
    for gpu in getattr(resources, "gpus", []) or []:
        label = gpu.name or gpu.vendor or "GPU"
        if gpu.vram_total_mb:
            label += f" {gpu.vram_total_mb} МиБ"
        parts.append(label)
    if getattr(resources, "ram_total_mb", 0):
        parts.append(f"{resources.ram_total_mb} МиБ ОЗУ")
    return ", ".join(parts) or str(getattr(resources, "os_name", "") or "узел")
