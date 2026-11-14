"""Размещение слоёв модели пропорционально доступной памяти устройств.

llama.cpp в режиме `--split-mode layer` режет модель на непрерывные
блоки слоёв. `--tensor-split` задаёт доли. Мы передаём сами мегабайты:
llama.cpp нормирует их сам. Порядок чисел совпадает с порядком устройств
RPC, а тот совпадает с порядком эндпоинтов `--rpc` и устройств внутри
каждого `ggml-rpc-server`.
"""

from __future__ import annotations

from fast_aggregation.cluster.health import compute_factor, evaluate_guard, network_factor
from fast_aggregation.constants import (
    DEFAULT_CPU_RAM_FRACTION,
    DEFAULT_VRAM_RESERVE_MB,
    MIN_DEVICE_MEMORY_MB,
    PHONE_RAM_FRACTION,
    PHONE_RAM_RESERVE_MB,
    SLOW_LINK_MBPS,
    SLOW_LINK_SHARE_CAP,
)
from fast_aggregation.errors import BackendError, PlacementError
from fast_aggregation.models import ClusterNode, GpuDevice, MachineResources, PlacementPlan, PlannedDevice

_WINDOWS_ORDER = ("cuda", "vulkan", "rocm", "sycl", "cpu")
_ANDROID_ORDER = ("opencl", "cpu")
_OTHER_ORDER = ("cuda", "rocm", "vulkan", "sycl", "metal", "opencl", "cpu")


def choose_backend(resources: MachineResources, preference: str, system: str | None = None) -> str:
    available = resources.backend_names()
    wanted = (preference or "auto").strip().lower()
    if wanted and wanted != "auto":
        if wanted not in available:
            raise BackendError(
                f"Бэкенд «{wanted}» недоступен. Доступны: {', '.join(sorted(available)) or 'нет'}"
            )
        return wanted
    os_name = (system or resources.os_name or "").lower()
    if os_name == "android":
        order = _ANDROID_ORDER
    elif os_name == "windows":
        order = _WINDOWS_ORDER
    else:
        order = _OTHER_ORDER
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
    need_mb: float | None = None,
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

    _cap_slow_links(devices, warnings, need_mb)
    total = sum(device.weight for device in devices) or 1
    for device in devices:
        device.share = device.weight / total
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
    guard = evaluate_guard(node.resources)
    if guard.action == "exclude":
        warnings.append(f"{node.name} исключён: {guard.reason}")
        return [], warnings
    if guard.reason:
        warnings.append(f"{node.name}: {guard.reason}")
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
    effective_reserve = reserve_mb
    if node.resources.device_class == "phone":
        effective_reserve = max(reserve_mb, PHONE_RAM_RESERVE_MB)
    for index, gpu in enumerate(gpus):
        memory = _usable_vram(gpu, effective_reserve)
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
            _device(
                node,
                index=index,
                backend=backend,
                name=gpu.name,
                kind="gpu",
                memory_mb=memory,
                guard_factor=guard.factor,
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
    if node.resources.device_class == "phone":
        fraction = min(fraction, PHONE_RAM_FRACTION)
    memory = int(node.resources.ram_available_mb * fraction)
    if node.resources.device_class == "phone":
        memory = max(0, memory - PHONE_RAM_RESERVE_MB // 2)
    if memory < min_memory_mb:
        return None
    guard = evaluate_guard(node.resources)
    return _device(
        node,
        index=0,
        backend="cpu",
        name=node.resources.cpu_model or "CPU",
        kind="cpu",
        memory_mb=memory,
        guard_factor=guard.factor,
    )


def _device(
    node: ClusterNode,
    *,
    index: int,
    backend: str,
    name: str,
    kind: str,
    memory_mb: int,
    guard_factor: float,
) -> PlannedDevice:
    assert node.resources is not None
    factor = guard_factor * network_factor(node.rtt_ms, node.bandwidth_mbps)
    factor *= compute_factor(backend, node.resources)
    # При нейтральных коэффициентах вес совпадает с мегабайтами.
    weight = memory_mb if abs(factor - 1.0) < 1e-9 else max(1, int(memory_mb * factor))
    return PlannedDevice(
        node_id=node.node_id,
        node_name=node.name,
        host=node.host,
        rpc_port=node.rpc_port,
        http_port=node.http_port,
        device_index=index,
        backend=backend,
        name=name,
        kind=kind,
        memory_mb=memory_mb,
        share=0.0,
        weight=weight,
        bandwidth_mbps=node.bandwidth_mbps,
    )


def _cap_slow_links(devices: list[PlannedDevice], warnings: list[str], need_mb: float | None) -> None:
    measured = [device for device in devices if device.bandwidth_mbps is not None]
    if len({device.node_id for device in measured}) < 2:
        return
    best = max(device.bandwidth_mbps or 0 for device in measured)
    if best <= 0:
        return
    slow_ids = {
        device.node_id
        for device in measured
        if (device.bandwidth_mbps or 0) < SLOW_LINK_MBPS or (device.bandwidth_mbps or 0) < best * 0.2
    }
    if not slow_ids or slow_ids >= {device.node_id for device in devices}:
        return
    fast_memory = sum(device.memory_mb for device in devices if device.node_id not in slow_ids)
    if need_mb and fast_memory < need_mb:
        warnings.append("Медленный канал оставлен с полной долей: без него модель не помещается")
        return
    fast_weight = sum(device.weight for device in devices if device.node_id not in slow_ids)
    slow = [device for device in devices if device.node_id in slow_ids]
    slow_weight = sum(device.weight for device in slow)
    cap = fast_weight * SLOW_LINK_SHARE_CAP / max(1e-9, 1 - SLOW_LINK_SHARE_CAP)
    if slow_weight <= cap or fast_weight <= 0:
        return
    scale = cap / slow_weight
    for device in slow:
        device.weight = max(1, int(device.weight * scale))
    warnings.append("Доля слоёв на медленном канале ограничена 10%, чтобы не тормозить конвейер")
