"""Какие модели реально поместятся в текущий кластер и с какой скоростью.

Цифры размеров — типичные GGUF, не паспорт конкретной ревизии.
Скорость — оценка сверху по классу устройства и самому медленному каналу,
не замер.
"""

from __future__ import annotations

from dataclasses import dataclass

from fast_aggregation.models import ClusterNode, PlacementPlan

# Накладные расходы RPC и служебных буферов поверх весов и KV.
RPC_OVERHEAD_MIB = 384

# Типичные плотные модели. KV-головы взяты с GQA, как у Llama 3 / Qwen2.
PRESETS: tuple[dict, ...] = (
    {"id": "7b-q4", "label": "7B Q4_K_M", "params_b": 7.0, "quant": "Q4_K_M", "weight_gib": 4.4, "layers": 32, "embd": 4096, "heads": 32, "kv_heads": 8},
    {"id": "7b-q5", "label": "7B Q5_K_M", "params_b": 7.0, "quant": "Q5_K_M", "weight_gib": 5.1, "layers": 32, "embd": 4096, "heads": 32, "kv_heads": 8},
    {"id": "7b-q8", "label": "7B Q8_0", "params_b": 7.0, "quant": "Q8_0", "weight_gib": 7.7, "layers": 32, "embd": 4096, "heads": 32, "kv_heads": 8},
    {"id": "13b-q4", "label": "13B Q4_K_M", "params_b": 13.0, "quant": "Q4_K_M", "weight_gib": 7.9, "layers": 40, "embd": 5120, "heads": 40, "kv_heads": 40},
    {"id": "13b-q5", "label": "13B Q5_K_M", "params_b": 13.0, "quant": "Q5_K_M", "weight_gib": 9.2, "layers": 40, "embd": 5120, "heads": 40, "kv_heads": 40},
    {"id": "13b-q8", "label": "13B Q8_0", "params_b": 13.0, "quant": "Q8_0", "weight_gib": 14.0, "layers": 40, "embd": 5120, "heads": 40, "kv_heads": 40},
    {"id": "32b-q4", "label": "32B Q4_K_M", "params_b": 32.0, "quant": "Q4_K_M", "weight_gib": 19.0, "layers": 64, "embd": 5120, "heads": 40, "kv_heads": 8},
    {"id": "32b-q5", "label": "32B Q5_K_M", "params_b": 32.0, "quant": "Q5_K_M", "weight_gib": 23.0, "layers": 64, "embd": 5120, "heads": 40, "kv_heads": 8},
    {"id": "32b-q8", "label": "32B Q8_0", "params_b": 32.0, "quant": "Q8_0", "weight_gib": 34.0, "layers": 64, "embd": 5120, "heads": 40, "kv_heads": 8},
    {"id": "70b-q4", "label": "70B Q4_K_M", "params_b": 70.0, "quant": "Q4_K_M", "weight_gib": 40.0, "layers": 80, "embd": 8192, "heads": 64, "kv_heads": 8},
    {"id": "70b-q5", "label": "70B Q5_K_M", "params_b": 70.0, "quant": "Q5_K_M", "weight_gib": 47.0, "layers": 80, "embd": 8192, "heads": 64, "kv_heads": 8},
    {"id": "70b-q8", "label": "70B Q8_0", "params_b": 70.0, "quant": "Q8_0", "weight_gib": 75.0, "layers": 80, "embd": 8192, "heads": 64, "kv_heads": 8},
)

_QUANT_SPEED = {"Q4_K_M": 1.0, "Q5_K_M": 0.85, "Q8_0": 0.55}
_REFERENCE_TPS = 40.0  # 7B Q4 на настольной CUDA-карте, ориентир, не обещание


@dataclass(frozen=True)
class Capacity:
    raw_vram_mib: int
    raw_ram_mib: int
    usable_mib: int
    node_count: int
    slowest_mbps: float | None
    compute_class: float
    notes: tuple[str, ...]


def kv_mib(layers: int, embd: int, heads: int, kv_heads: int, ctx: int, parallel: int = 1) -> float:
    if heads <= 0 or ctx <= 0:
        return 0.0
    head_dim = embd / heads
    slots = max(1, parallel)
    return layers * ctx * kv_heads * head_dim * 2 * 2 * slots / (1024 * 1024)


def summarize_capacity(plan: PlacementPlan | None, nodes: list[ClusterNode]) -> Capacity:
    raw_vram = 0
    raw_ram = 0
    notes: list[str] = []
    for node in nodes:
        if node.resources is None:
            continue
        raw_ram += node.resources.ram_total_mb
        for gpu in node.resources.gpus:
            if gpu.vram_total_mb:
                raw_vram += gpu.vram_total_mb
    usable = 0
    bandwidths: list[float] = []
    classes: list[float] = []
    if plan is not None:
        usable = sum(device.memory_mb for device in plan.devices)
        for device in plan.devices:
            if device.bandwidth_mbps is not None:
                bandwidths.append(device.bandwidth_mbps)
            classes.append(_class_for_backend(device.backend))
        notes.append("Пригодная память уже уменьшена на запас, троттлинг и исключённые узлы.")
    else:
        notes.append("План пуст: пригодная память считается нулевой.")
    notes.append(f"К весам и KV в рекомендации добавляется {RPC_OVERHEAD_MIB} МиБ на служебные буферы RPC.")
    return Capacity(
        raw_vram_mib=raw_vram,
        raw_ram_mib=raw_ram,
        usable_mib=usable,
        node_count=len({device.node_id for device in plan.devices}) if plan else 0,
        slowest_mbps=min(bandwidths) if bandwidths else None,
        compute_class=max(classes) if classes else 0.25,
        notes=tuple(notes),
    )


def recommend(capacity: Capacity, *, ctx: int, parallel: int = 1) -> list[dict]:
    rows = []
    for preset in PRESETS:
        weight = preset["weight_gib"] * 1024
        cache = kv_mib(preset["layers"], preset["embd"], preset["heads"], preset["kv_heads"], ctx, parallel)
        need = weight + cache + RPC_OVERHEAD_MIB
        fits = need <= capacity.usable_mib
        max_ctx = _max_context(preset, capacity.usable_mib, parallel) if fits or capacity.usable_mib > weight + RPC_OVERHEAD_MIB else 0
        rows.append(
            {
                "id": preset["id"],
                "label": preset["label"],
                "quant": preset["quant"],
                "params_b": preset["params_b"],
                "weight_mib": round(weight),
                "kv_mib": round(cache),
                "need_mib": round(need),
                "fits": fits,
                "max_context": max_ctx,
                "tokens_per_second": _tokens_per_second(preset, capacity) if fits else None,
            }
        )
    return rows


def _max_context(preset: dict, usable_mib: int, parallel: int) -> int:
    weight = preset["weight_gib"] * 1024
    room = usable_mib - weight - RPC_OVERHEAD_MIB
    if room <= 0:
        return 0
    low, high = 0, 1024 * 1024
    best = 0
    while low <= high:
        mid = (low + high) // 2
        cache = kv_mib(preset["layers"], preset["embd"], preset["heads"], preset["kv_heads"], mid, parallel)
        if cache <= room:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _tokens_per_second(preset: dict, capacity: Capacity) -> float:
    compute = _REFERENCE_TPS * (7.0 / preset["params_b"]) * _QUANT_SPEED[preset["quant"]] * capacity.compute_class
    if capacity.slowest_mbps is None:
        return round(compute, 1)
    bytes_per_token = 2 * preset["embd"] * 2
    network = (capacity.slowest_mbps * 1_000_000 / 8) / max(bytes_per_token, 1)
    return round(min(compute, network), 1)


def _class_for_backend(backend: str) -> float:
    return {"cuda": 1.0, "rocm": 0.9, "metal": 0.9, "vulkan": 0.7, "opencl": 0.5, "sycl": 0.55, "cpu": 0.2}.get(backend, 0.4)
