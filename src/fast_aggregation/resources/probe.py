"""Сборка снимка ресурсов машины из psutil и внешних утилит."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from fast_aggregation.models import BackendInfo, GpuDevice, MachineResources
from fast_aggregation.resources.command import SubprocessRunner
from fast_aggregation.resources.parsers import (
    apply_vulkan_memory,
    merge_gpu_lists,
    parse_cpu_name_text,
    parse_macos_displays,
    parse_nvidia_smi,
    parse_vulkan_memory,
    parse_vulkan_summary,
    parse_win32_video_controllers,
    read_linux_drm_gpus,
)

NVIDIA_SMI_ARGS = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free,driver_version",
    "--format=csv,noheader,nounits",
]

POWERSHELL_INVENTORY = r"""
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name
$gpus = @(Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion, PNPDeviceID)
[pscustomobject]@{cpu=$cpu; gpus=$gpus} | ConvertTo-Json -Compress -Depth 4
""".strip()


@dataclass
class RawOutputs:
    nvidia_smi: str = ""
    nvidia_smi_ok: bool = False
    nvidia_smi_detail: str = ""
    powershell: str = ""
    vulkan_summary: str = ""
    vulkan_full: str = ""
    vulkaninfo_present: bool = False
    rocm_present: bool = False
    sysfs_gpus: list[GpuDevice] = field(default_factory=list)
    macos_gpus: list[GpuDevice] = field(default_factory=list)


def probe_machine(
    runner: SubprocessRunner | None = None,
    *,
    system: str | None = None,
    arch: str | None = None,
    os_version: str | None = None,
    sysfs_root: Path | None = None,
    deep_vulkan: bool = False,
    vulkan_dll: bool | None = None,
) -> MachineResources:
    system_name = system or platform.system()
    active = runner or SubprocessRunner()
    raw = collect_outputs(
        active,
        system_name,
        deep_vulkan=deep_vulkan,
        sysfs_root=sysfs_root if sysfs_root is not None else Path("/sys/class/drm"),
    )
    cpu_model, physical, logical = read_cpu(active, system_name, raw.powershell)
    total_mb, available_mb = read_memory()
    return assemble_resources(
        raw,
        system=system_name,
        arch=arch or normalize_arch(platform.machine()),
        os_version=os_version if os_version is not None else platform.version(),
        cpu_model=cpu_model,
        cpu_physical=physical,
        cpu_logical=logical,
        ram_total_mb=total_mb,
        ram_available_mb=available_mb,
        vulkan_dll=vulkan_dll if vulkan_dll is not None else _default_vulkan_dll(system_name),
    )


def collect_outputs(
    runner: SubprocessRunner,
    system: str,
    *,
    deep_vulkan: bool,
    sysfs_root: Path | None,
) -> RawOutputs:
    raw = RawOutputs()
    nvidia = runner.run(NVIDIA_SMI_ARGS, timeout=8)
    raw.nvidia_smi_ok = nvidia.ok
    raw.nvidia_smi = nvidia.stdout
    if nvidia.returncode == 127:
        raw.nvidia_smi_detail = "nvidia-smi не найден"
    elif nvidia.ok:
        raw.nvidia_smi_detail = "nvidia-smi ответил"
    else:
        detail = (nvidia.stderr or nvidia.stdout or "").strip().splitlines()
        raw.nvidia_smi_detail = detail[-1] if detail else f"nvidia-smi код {nvidia.returncode}"

    if system == "Windows":
        script = POWERSHELL_INVENTORY.replace("\n", "; ")
        completed = runner.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; " + script,
            ],
            timeout=20,
        )
        if completed.ok:
            raw.powershell = completed.stdout

    if system == "Linux" and sysfs_root is not None:
        raw.sysfs_gpus = read_linux_drm_gpus(sysfs_root)

    if system == "Darwin":
        profile = runner.run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30)
        if profile.ok:
            raw.macos_gpus = parse_macos_displays(profile.stdout)

    raw.rocm_present = runner.which("rocm-smi")
    raw.vulkaninfo_present = runner.which("vulkaninfo")
    if raw.vulkaninfo_present:
        summary = runner.run(["vulkaninfo", "--summary"], timeout=12)
        if summary.ok or summary.stdout:
            raw.vulkan_summary = summary.stdout
        if deep_vulkan:
            full = runner.run(["vulkaninfo"], timeout=25)
            if full.stdout:
                raw.vulkan_full = full.stdout
    return raw


def assemble_resources(
    raw: RawOutputs,
    *,
    system: str,
    arch: str,
    os_version: str,
    cpu_model: str,
    cpu_physical: int | None,
    cpu_logical: int,
    ram_total_mb: int,
    ram_available_mb: int,
    vulkan_dll: bool,
) -> MachineResources:
    nvidia_gpus = parse_nvidia_smi(raw.nvidia_smi) if raw.nvidia_smi_ok else []
    win32_gpus = parse_win32_video_controllers(raw.powershell) if system == "Windows" else []
    vulkan_gpus = parse_vulkan_summary(raw.vulkan_summary)
    gpus = merge_gpu_lists(nvidia_gpus, raw.macos_gpus, raw.sysfs_gpus, win32_gpus, vulkan_gpus)
    if raw.vulkan_full:
        apply_vulkan_memory(gpus, parse_vulkan_memory(raw.vulkan_full))
    backends = _backends(
        system=system,
        gpus=gpus,
        nvidia_smi_ok=raw.nvidia_smi_ok,
        nvidia_detail=raw.nvidia_smi_detail,
        vulkan_dll=vulkan_dll,
        vulkaninfo_present=raw.vulkaninfo_present,
        rocm_present=raw.rocm_present,
    )
    return MachineResources(
        os_name=system,
        os_version=os_version,
        arch=normalize_arch(arch),
        cpu_model=cpu_model or "unknown",
        cpu_physical=cpu_physical,
        cpu_logical=cpu_logical,
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        gpus=gpus,
        backends=backends,
    )


def apply_vram_override(resources: MachineResources, override_mb: int | None) -> MachineResources:
    if not override_mb or override_mb <= 0:
        return resources
    missing = [gpu for gpu in resources.gpus if gpu.vram_total_mb is None]
    for gpu in missing:
        gpu.vram_total_mb = override_mb
        gpu.vram_free_mb = override_mb
    return resources


def normalize_arch(machine: str) -> str:
    value = machine.lower().strip()
    if value in {"amd64", "x86_64", "x64"}:
        return "x64"
    if value in {"arm64", "aarch64", "aarch64_be"}:
        return "arm64"
    return value or "unknown"


def read_memory() -> tuple[int, int]:
    memory = psutil.virtual_memory()
    return int(memory.total // (1024 * 1024)), int(memory.available // (1024 * 1024))


def read_cpu(
    runner: SubprocessRunner,
    system: str,
    powershell_text: str,
) -> tuple[str, int | None, int]:
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False)
    model = ""
    if system == "Linux":
        model = _linux_cpu_model()
    elif system == "Windows":
        model = parse_cpu_name_text(powershell_text) or ""
    elif system == "Darwin":
        completed = runner.run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        if completed.ok:
            model = completed.stdout.strip()
    if not model:
        model = platform.processor() or platform.machine() or "unknown"
    return model, physical, logical


def _linux_cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    parsed = parse_cpu_name_text(text)
    return parsed or ""


def _backends(
    *,
    system: str,
    gpus: list[GpuDevice],
    nvidia_smi_ok: bool,
    nvidia_detail: str,
    vulkan_dll: bool,
    vulkaninfo_present: bool,
    rocm_present: bool,
) -> list[BackendInfo]:
    has_gpu = bool(gpus)
    vulkan_loader = vulkan_dll or vulkaninfo_present
    vulkan_detail: list[str] = []
    if vulkaninfo_present:
        vulkan_detail.append("vulkaninfo в PATH")
    if vulkan_dll:
        vulkan_detail.append("загрузчик Vulkan найден")
    if not vulkan_loader:
        vulkan_detail.append("загрузчик Vulkan не найден")
    elif not has_gpu:
        vulkan_detail.append("GPU не обнаружен, бэкенд не выбирается")
    cuda_ok = nvidia_smi_ok and any(gpu.vendor == "nvidia" for gpu in gpus)
    rocm_ok = rocm_present and any(gpu.vendor == "amd" for gpu in gpus)
    backends = [
        BackendInfo("cpu", True, "CPU-бэкенд llama.cpp доступен всегда"),
        BackendInfo("cuda", cuda_ok, nvidia_detail or "nvidia-smi недоступен"),
        BackendInfo("vulkan", vulkan_loader and has_gpu, "; ".join(vulkan_detail)),
        BackendInfo(
            "rocm",
            rocm_ok,
            "rocm-smi в PATH и есть GPU AMD" if rocm_ok else "ROCm не подтверждён",
        ),
    ]
    if system == "Darwin":
        backends.append(BackendInfo("metal", True, "сборка llama.cpp для macOS включает Metal"))
    if system == "Windows":
        backends.append(
            BackendInfo(
                "sycl",
                any(gpu.vendor == "intel" for gpu in gpus),
                "Intel GPU виден в инвентаре; проверка oneAPI не выполняется",
            )
        )
    return backends


def _default_vulkan_dll(system: str) -> bool:
    if system == "Windows":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        return Path(root, "System32", "vulkan-1.dll").is_file()
    if system == "Linux":
        candidates = (
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
            "/usr/lib64/libvulkan.so.1",
            "/usr/lib/libvulkan.so.1",
        )
        return any(Path(path).is_file() for path in candidates)
    return False
