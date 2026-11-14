"""Размещение слоёв модели пропорционально доступной памяти устройств.

llama.cpp в режиме `--split-mode layer` режет модель на непрерывные
блоки слоёв. `--tensor-split` задаёт доли. Мы передаём сами мегабайты:
llama.cpp нормирует их сам. Порядок чисел совпадает с порядком устройств
RPC, а тот совпадает с порядком эндпоинтов `--rpc` и устройств внутри
каждого `ggml-rpc-server`.
"""

from __future__ import annotations

from fast_aggregation.constants import (
    DEFAULT_CPU_RAM_FRACTION,
    DEFAULT_VRAM_RESERVE_MB,
    MIN_DEVICE_MEMORY_MB,
)
from fast_aggregation.errors import BackendError, PlacementError
from fast_aggregation.models import ClusterNode, GpuDevice, MachineResources, PlacementPlan, PlannedDevice

_WINDOWS_ORDER = ("cuda", "vulkan", "rocm", "sycl", "cpu")
_OTHER_ORDER = ("cuda", "rocm", "vulkan", "sycl", "metal", "cpu")


def choose_backend(resources: MachineResources, preference: str, system: str | None = None) -> str:
    available = resources.backend_names()
    wanted = (preference or "auto").strip().lower()
    if wanted and wanted != "auto":
        if wanted not in available:
            raise BackendError(
                f"Бэкенд «{wanted}» недоступен. Доступны: {', '.join(sorted(available)) or 'нет'}"
            )
        return wanted
    order = _WINDOWS_ORDER if (system or resources.os_name) == "Windows" else _OTHER_ORDER
    for name in order:
        if name in available:
            return name
    return "cpu"


def plan_placement(
    nodes: list[ClusterNode],
    *,
    allow_cpu: bool = False,
    reserve_mb: int = DEFAULT_VRAM_RESERVE_MB,
    cpu_ram_fraction: float = DEFAULT_CPU_RAM_FRACTION,
    min_memory_mb: int = MIN_DEVICE_MEMORY_MB,
) -> PlacementPlan:
    warnings: list[str] = []
    devices: list[PlannedDevice] = []
    ordered = sorted(nodes, key=lambda node: (node.name.lower(), node.node_id))
    for node in ordered:
        if not node.online:
            warnings.append(f"Узел {node.name} не в сети и пропущен")
            continue
        if node.resources is None:
            warnings.append(f"У узла {node.name} ещё нет снимка ресурсов")
            continue
        try:
            backend = choose_backend(node.resources, node.preferred_backend, node.resources.os_name)
        except BackendError as exc:
            warnings.append(f"{node.name}: {exc}")
            continue
        slots, slot_warnings = _slots_for_node(
            node,
            backend=backend,
            allow_cpu=allow_cpu,
            reserve_mb=reserve_mb,
            cpu_ram_fraction=cpu_ram_fraction,
            min_memory_mb=min_memory_mb,
        )
        warnings.extend(slot_warnings)
        devices.extend(slots)

    if not devices:
        detail = "; ".join(warnings) if warnings else "в реестре нет узлов"
        raise PlacementError(f"Нечего распределять: {detail}")

    total = sum(device.memory_mb for device in devices)
    for device in devices:
        device.share = device.memory_mb / total
    return PlacementPlan(devices=devices, warnings=warnings)


def _slots_for_node(
    node: ClusterNode,
    *,
    backend: str,
    allow_cpu: bool,
    reserve_mb: int,
    cpu_ram_fraction: float,
    min_memory_mb: int,
) -> tuple[list[PlannedDevice], list[str]]:
    assert node.resources is not None
    warnings: list[str] = []
    if backend == "cpu":
        if not allow_cpu:
            warnings.append(
                f"{node.name}: только CPU. Отметьте «CPU-узлы», чтобы отдать ему слои"
            )
            return [], warnings
        slot = _cpu_slot(node, fraction=cpu_ram_fraction, min_memory_mb=min_memory_mb)
        if slot is None:
            warnings.append(f"{node.name}: слишком мало свободной ОЗУ для CPU-слоя")
            return [], warnings
        return [slot], warnings

    gpus = _gpus_for_backend(node.resources, backend)
    slots: list[PlannedDevice] = []
    for index, gpu in enumerate(gpus):
        memory = _usable_vram(gpu, reserve_mb)
        if memory is None:
            warnings.append(
                f"{node.name}: у «{gpu.name}» неизвестен объём VRAM, устройство не участвует"
            )
            continue
        if memory < min_memory_mb:
            warnings.append(
                f"{node.name}: у «{gpu.name}» свободно {memory} МиБ, это меньше порога {min_memory_mb}"
            )
            continue
        slots.append(
            PlannedDevice(
                node_id=node.node_id,
                node_name=node.name,
                host=node.host,
                rpc_port=node.rpc_port,
                http_port=node.http_port,
                device_index=index,
                backend=backend,
                name=gpu.name,
                kind="gpu",
                memory_mb=memory,
                share=0.0,
            )
        )
    if slots:
        return slots, warnings
    if allow_cpu:
        warnings.append(f"{node.name}: GPU не попали в план, используется ОЗУ")
        slot = _cpu_slot(node, fraction=cpu_ram_fraction, min_memory_mb=min_memory_mb)
        if slot is not None:
            return [slot], warnings
    elif gpus:
        warnings.append(f"{node.name}: ни одно GPU не попало в план")
    else:
        warnings.append(f"{node.name}: бэкенд {backend}, но GPU не обнаружены")
    return [], warnings


def _gpus_for_backend(resources: MachineResources, backend: str) -> list[GpuDevice]:
    if backend == "cuda":
        return [gpu for gpu in resources.gpus if gpu.vendor == "nvidia" and gpu.source == "nvidia-smi"]
    if backend == "rocm":
        return [gpu for gpu in resources.gpus if gpu.vendor == "amd"]
    if backend == "sycl":
        return [gpu for gpu in resources.gpus if gpu.vendor == "intel"]
    return [gpu for gpu in resources.gpus if gpu.vendor != "microsoft"]


def _usable_vram(gpu: GpuDevice, reserve_mb: int) -> int | None:
    base = gpu.vram_free_mb if gpu.vram_free_mb is not None else gpu.vram_total_mb
    if base is None:
        return None
    return max(0, base - max(0, reserve_mb))


def _cpu_slot(node: ClusterNode, *, fraction: float, min_memory_mb: int) -> PlannedDevice | None:
    assert node.resources is not None
    memory = int(node.resources.ram_available_mb * fraction)
    if memory < min_memory_mb:
        return None
    return PlannedDevice(
        node_id=node.node_id,
        node_name=node.name,
        host=node.host,
        rpc_port=node.rpc_port,
        http_port=node.http_port,
        device_index=0,
        backend="cpu",
        name=node.resources.cpu_model or "CPU",
        kind="cpu",
        memory_mb=memory,
        share=0.0,
    )
