"""Общие заготовки ресурсов для тестов планировщика и API."""

from __future__ import annotations

from fast_aggregation.models import BackendInfo, GpuDevice, MachineResources


def gpu(
    name: str = "NVIDIA GeForce RTX 4090",
    *,
    vendor: str = "nvidia",
    total: int | None = 24564,
    free: int | None = 24000,
    source: str = "nvidia-smi",
) -> GpuDevice:
    return GpuDevice(
        name=name,
        vendor=vendor,
        vram_total_mb=total,
        vram_free_mb=free,
        driver="560.94",
        source=source,
    )


def machine(
    *,
    os_name: str = "Windows",
    arch: str = "x64",
    gpus: list[GpuDevice] | None = None,
    ram: int = 32768,
    cuda: bool = True,
    vulkan: bool = False,
    rocm: bool = False,
    metal: bool = False,
    cpu_model: str = "Test CPU",
) -> MachineResources:
    backends = [
        BackendInfo("cpu", True, "cpu"),
        BackendInfo("cuda", cuda, "cuda"),
        BackendInfo("vulkan", vulkan, "vulkan"),
        BackendInfo("rocm", rocm, "rocm"),
    ]
    if metal:
        backends.append(BackendInfo("metal", True, "metal"))
    return MachineResources(
        os_name=os_name,
        os_version="test",
        arch=arch,
        cpu_model=cpu_model,
        cpu_physical=8,
        cpu_logical=16,
        ram_total_mb=ram,
        ram_available_mb=ram,
        gpus=list(gpus or []),
        backends=backends,
    )
