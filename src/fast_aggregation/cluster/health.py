"""Пороги температуры, батареи и загрузки.

Функции чистые: планировщик и цикл наблюдения передают уже снятый снимок.
"""

from __future__ import annotations

from dataclasses import dataclass

from fast_aggregation.constants import (
    BATTERY_LAPTOP_CRITICAL,
    BATTERY_LAPTOP_THROTTLE,
    BATTERY_PHONE_MIN,
    CPU_LOAD_THROTTLE,
    GOOD_LINK_MBPS,
    GOOD_LINK_RTT_MS,
    TEMP_CRITICAL_C,
    TEMP_THROTTLE_C,
)
from fast_aggregation.models import ClusterNode, MachineResources, PlacementPlan


@dataclass(frozen=True)
class Guard:
    """action: allow | throttle | exclude. factor умножается на вес слоёв."""

    action: str
    factor: float
    reason: str = ""


def hottest_c(resources: MachineResources) -> float | None:
    values: list[float] = []
    if resources.temperature_c is not None:
        values.append(resources.temperature_c)
    for gpu in resources.gpus:
        if gpu.temperature_c is not None:
            values.append(gpu.temperature_c)
    return max(values) if values else None


def evaluate_guard(resources: MachineResources) -> Guard:
    """Решает, можно ли отдавать узлу слои и насколько урезать долю."""

    temp = hottest_c(resources)
    if temp is not None and temp >= TEMP_CRITICAL_C:
        return Guard("exclude", 0.0, f"температура {temp:.0f} °C")
    factor = 1.0
    notes: list[str] = []
    if temp is not None and temp >= TEMP_THROTTLE_C:
        factor *= 0.45
        notes.append(f"температура {temp:.0f} °C, доля снижена")
    if resources.cpu_percent is not None and resources.cpu_percent >= CPU_LOAD_THROTTLE:
        factor *= 0.6
        notes.append(f"загрузка CPU {resources.cpu_percent:.0f}%")

    battery = resources.battery_percent
    unplugged = resources.power_plugged is False
    kind = resources.device_class
    if unplugged and battery is not None:
        if kind == "phone" and battery < BATTERY_PHONE_MIN:
            return Guard("exclude", 0.0, f"телефон на батарее {battery:.0f}%")
        if kind == "phone":
            factor *= 0.4
            notes.append("телефон на батарее")
        elif kind == "laptop" and battery < BATTERY_LAPTOP_CRITICAL:
            return Guard("exclude", 0.0, f"батарея ноутбука {battery:.0f}%")
        elif kind == "laptop" and battery < BATTERY_LAPTOP_THROTTLE:
            factor *= 0.55
            notes.append(f"ноутбук на батарее {battery:.0f}%")
        elif kind == "laptop":
            factor *= 0.75
            notes.append("ноутбук от батареи")
    if factor < 1.0:
        return Guard("throttle", max(0.15, factor), "; ".join(notes))
    return Guard("allow", 1.0, "")


def network_factor(rtt_ms: float | None, bandwidth_mbps: float | None) -> float:
    """Неизвестный канал не штрафуется: замер ещё не пришёл.

    Известный канал сравнивается с гигабитной LAN (200 Мбит/с и 8 мс).
    Медленный Wi-Fi даёт вес заметно меньше единицы.
    """

    if rtt_ms is None and bandwidth_mbps is None:
        return 1.0
    bandwidth = 1.0 if bandwidth_mbps is None else min(1.0, max(bandwidth_mbps, 0.0) / GOOD_LINK_MBPS)
    latency = 1.0 if rtt_ms is None else min(1.0, GOOD_LINK_RTT_MS / max(rtt_ms, 0.1))
    return max(0.05, bandwidth * latency)


def compute_factor(backend: str, resources: MachineResources) -> float:
    base = {
        "cuda": 1.0,
        "rocm": 0.9,
        "metal": 0.9,
        "vulkan": 0.75,
        "opencl": 0.55,
        "sycl": 0.6,
        "cpu": 0.25,
    }.get(backend, 0.5)
    if resources.device_class == "phone" and backend != "cpu":
        base *= 0.7
    if backend == "cpu":
        cores = resources.cpu_logical or 4
        base *= min(1.6, max(0.5, cores / 8))
    return base


def shed_reasons(plan: PlacementPlan, nodes: list[ClusterNode]) -> list[tuple[str, str]]:
    """Узлы, которые надо вынуть из уже запущенного кластера."""

    by_id = {node.node_id: node for node in nodes}
    reasons: list[tuple[str, str]] = []
    seen: set[str] = set()
    for device in plan.devices:
        if device.node_id in seen:
            continue
        seen.add(device.node_id)
        node = by_id.get(device.node_id)
        if node is None or not node.online:
            reasons.append((device.node_id, f"{device.node_name} пропал из сети"))
            continue
        if node.resources is None:
            continue
        guard = evaluate_guard(node.resources)
        if guard.action == "exclude":
            reasons.append((device.node_id, f"{device.node_name}: {guard.reason}"))
    return reasons
