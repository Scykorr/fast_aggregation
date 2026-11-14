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
            temperature_c=_optional_float(item.get("temperature_c")),
            utilization=_optional_float(item.get("utilization")),
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
        cpu_percent=_optional_float(data.get("cpu_percent")),
        temperature_c=_optional_float(data.get("temperature_c")),
        battery_percent=_optional_float(data.get("battery_percent")),
        power_plugged=_optional_bool(data.get("power_plugged")),
        device_class=str(data.get("device_class") or "unknown"),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


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
    temperature_c: float | None = None
    utilization: float | None = None

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
    cpu_percent: float | None = None
    temperature_c: float | None = None
    battery_percent: float | None = None
    power_plugged: bool | None = None
    device_class: str = "unknown"

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
            "cpu_percent": self.cpu_percent,
            "temperature_c": self.temperature_c,
            "battery_percent": self.battery_percent,
            "power_plugged": self.power_plugged,
            "device_class": self.device_class,
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
class NodeLimits:
    """Ручные ограничения узла. Пустые поля не меняют общие пороги."""

    excluded: bool = False
    max_temp_c: float | None = None
    min_battery_percent: float | None = None
    memory_reserve_mb: int | None = None
    memory_cap_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def limits_from_dict(data: dict[str, Any] | None) -> NodeLimits | None:
    if not data:
        return None
    return NodeLimits(
        excluded=bool(data.get("excluded")),
        max_temp_c=_optional_float(data.get("max_temp_c")),
        min_battery_percent=_optional_float(data.get("min_battery_percent")),
        memory_reserve_mb=_optional_int(data.get("memory_reserve_mb")),
        memory_cap_mb=_optional_int(data.get("memory_cap_mb")),
    )


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
    rtt_ms: float | None = None
    bandwidth_mbps: float | None = None
    limits: NodeLimits | None = None

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
            "rtt_ms": self.rtt_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "limits": self.limits.to_dict() if self.limits else None,
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
    weight: int = 0
    bandwidth_mbps: float | None = None

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
        if self.devices and all(device.weight > 0 for device in self.devices):
            return [device.weight for device in self.devices]
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
