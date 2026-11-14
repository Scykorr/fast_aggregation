"""Общие структуры данных узла, ресурсов и плана размещения."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def resources_from_dict(data: dict[str, Any]) -> MachineResources:
    gpus = [
        GpuDevice(
            name=str(item.get("name") or "GPU"),
            vendor=str(item.get("vendor") or "unknown"),
            vram_total_mb=_optional_int(item.get("vram_total_mb")),
            vram_free_mb=_optional_int(item.get("vram_free_mb")),
            driver=item.get("driver"),
            source=str(item.get("source") or ""),
        )
        for item in data.get("gpus") or []
        if isinstance(item, dict)
    ]
    backends = [
        BackendInfo(
            name=str(item.get("name") or ""),
            available=bool(item.get("available")),
            detail=str(item.get("detail") or ""),
        )
        for item in data.get("backends") or []
        if isinstance(item, dict)
    ]
    return MachineResources(
        os_name=str(data.get("os_name") or ""),
        os_version=str(data.get("os_version") or ""),
        arch=str(data.get("arch") or ""),
        cpu_model=str(data.get("cpu_model") or ""),
        cpu_physical=_optional_int(data.get("cpu_physical")),
        cpu_logical=int(data.get("cpu_logical") or 1),
        ram_total_mb=int(data.get("ram_total_mb") or 0),
        ram_available_mb=int(data.get("ram_available_mb") or 0),
        gpus=gpus,
        backends=backends,
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    return value


@dataclass
class GpuDevice:
    name: str
    vendor: str
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    driver: str | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BackendInfo:
    name: str
    available: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MachineResources:
    os_name: str
    os_version: str
    arch: str
    cpu_model: str
    cpu_physical: int | None
    cpu_logical: int
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GpuDevice] = field(default_factory=list)
    backends: list[BackendInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "arch": self.arch,
            "cpu_model": self.cpu_model,
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "backends": [backend.to_dict() for backend in self.backends],
        }

    def backend_names(self) -> set[str]:
        return {backend.name for backend in self.backends if backend.available}


@dataclass
class Announce:
    node_id: str
    name: str
    advertise_host: str
    http_port: int
    rpc_port: int
    os_name: str
    arch: str
    preferred_backend: str = "auto"
    software: str = "fast-aggregation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterNode:
    """Снимок узла, с которым работает планировщик."""

    node_id: str
    name: str
    host: str
    http_port: int
    rpc_port: int
    online: bool
    resources: MachineResources | None
    preferred_backend: str = "auto"
    source: str = "discovery"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "host": self.host,
            "http_port": self.http_port,
            "rpc_port": self.rpc_port,
            "online": self.online,
            "resources": self.resources.to_dict() if self.resources else None,
            "preferred_backend": self.preferred_backend,
            "source": self.source,
        }


@dataclass
class PlannedDevice:
    node_id: str
    node_name: str
    host: str
    rpc_port: int
    http_port: int
    device_index: int
    backend: str
    name: str
    kind: str
    memory_mb: int
    share: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["share"] = round(self.share, 4)
        return data


@dataclass
class PlacementPlan:
    devices: list[PlannedDevice]
    warnings: list[str] = field(default_factory=list)

    @property
    def rpc_endpoints(self) -> list[str]:
        seen: list[str] = []
        for device in self.devices:
            endpoint = f"{device.host}:{device.rpc_port}"
            if endpoint not in seen:
                seen.append(endpoint)
        return seen

    @property
    def tensor_split(self) -> list[int]:
        return [device.memory_mb for device in self.devices]

    @property
    def device_flags(self) -> list[str]:
        return [f"RPC{index}" for index in range(len(self.devices))]

    def backends_by_node(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for device in self.devices:
            mapping[device.node_id] = device.backend
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "rpc_endpoints": self.rpc_endpoints,
            "tensor_split": self.tensor_split,
            "device_flags": self.device_flags,
            "warnings": list(self.warnings),
            "backends_by_node": self.backends_by_node(),
        }
