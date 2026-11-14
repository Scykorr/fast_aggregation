"""Разбор вывода nvidia-smi, Win32, vulkaninfo и sysfs.

Функции чистые: тесты подставляют зафиксированный текст и не зависят
от наличия GPU или Windows.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from fast_aggregation.models import GpuDevice

_VENDOR_BY_PCI = {
    "10de": "nvidia",
    "1002": "amd",
    "8086": "intel",
    "1414": "microsoft",
}

_SKIP_NAME_RE = re.compile(
    r"microsoft basic display|microsoft remote display|remote desktop|"
    r"indirect display|virtual display adapter|orayidd|parsec virtual",
    re.IGNORECASE,
)

_GPU_BLOCK_RE = re.compile(r"^GPU(\d+):\s*$", re.MULTILINE)
_DEVICE_HEAP_SPLIT = re.compile(r"(?=GPU\d+:)")


def parse_nvidia_smi(text: str) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    if not text or not text.strip():
        return devices
    reader = csv.reader(text.splitlines())
    for row in reader:
        cells = [cell.strip() for cell in row]
        if len(cells) < 2 or not cells[0] or cells[0].lower().startswith("failed"):
            continue
        name = cells[0]
        total = _to_int(cells[1]) if len(cells) > 1 else None
        free = _to_int(cells[2]) if len(cells) > 2 else None
        driver = cells[3] if len(cells) > 3 and cells[3] else None
        if total is None:
            continue
        devices.append(
            GpuDevice(
                name=name,
                vendor="nvidia",
                vram_total_mb=total,
                vram_free_mb=free,
                driver=driver,
                source="nvidia-smi",
            )
        )
    return devices


def parse_win32_video_controllers(text: str) -> list[GpuDevice]:
    """JSON от `Get-CimInstance Win32_VideoController | ConvertTo-Json`."""

    payload = _load_json_payload(text)
    if payload is None:
        return []
    if isinstance(payload, dict) and "gpus" in payload:
        payload = payload["gpus"]
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        return []
    devices: list[GpuDevice] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or item.get("name") or "").strip()
        if not name or _SKIP_NAME_RE.search(name):
            continue
        pnp = str(item.get("PNPDeviceID") or item.get("PNPDeviceId") or "")
        vendor = vendor_from_pnp(pnp) or vendor_from_name(name)
        if vendor == "microsoft":
            continue
        total = win32_adapter_ram_mb(item.get("AdapterRAM"))
        driver = item.get("DriverVersion") or item.get("driver")
        devices.append(
            GpuDevice(
                name=name,
                vendor=vendor,
                vram_total_mb=total,
                vram_free_mb=None,
                driver=str(driver) if driver else None,
                source="win32",
            )
        )
    return devices


def parse_cpu_name_text(text: str) -> str | None:
    payload = _load_json_payload(text)
    if isinstance(payload, dict):
        cpu = payload.get("cpu") if "cpu" in payload else payload
        if isinstance(cpu, dict):
            name = cpu.get("Name") or cpu.get("name")
            if name:
                return str(name).strip()
        if isinstance(cpu, list) and cpu and isinstance(cpu[0], dict):
            name = cpu[0].get("Name") or cpu[0].get("name")
            if name:
                return str(name).strip()
    for line in text.splitlines():
        if ":" in line and line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    stripped = text.strip()
    return stripped or None


def parse_vulkan_summary(text: str) -> list[GpuDevice]:
    if not text or "deviceName" not in text and "GPU" not in text:
        return []
    matches = list(_GPU_BLOCK_RE.finditer(text))
    if not matches:
        return []
    devices: list[GpuDevice] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        name = _field(block, "deviceName")
        if not name:
            continue
        vendor_id = _field(block, "vendorID") or ""
        vendor = vendor_from_hex(vendor_id) or vendor_from_name(name)
        driver = _field(block, "driverVersion") or _field(block, "driverName")
        devices.append(
            GpuDevice(
                name=name,
                vendor=vendor,
                driver=driver,
                source="vulkaninfo",
            )
        )
    return devices


def parse_vulkan_memory(text: str) -> dict[str, int]:
    """Имя устройства -> суммарный device-local heap в МиБ.

    Полный вывод vulkaninfo большой, поэтому парсер ищет куски по GPU
    и складывает кучи с флагом DEVICE_LOCAL.
    """

    if not text:
        return {}
    found: dict[str, int] = {}
    parts = _DEVICE_HEAP_SPLIT.split(text)
    for part in parts:
        name_match = re.search(r"deviceName\s*=\s*(.+)", part)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        total_bytes = 0
        for heap in re.finditer(
            r"memoryHeaps\[\d+\]:\s*size\s*=\s*(\d+)\s*\([^)]*\).*?flags:(.*?)(?=memoryHeaps\[|\n\n|\Z)",
            part,
            re.IGNORECASE | re.DOTALL,
        ):
            flags = heap.group(2)
            if "DEVICE_LOCAL" not in flags.upper():
                continue
            total_bytes += int(heap.group(1))
        if total_bytes > 0:
            found[name] = max(1, total_bytes // (1024 * 1024))
    return found


def parse_macos_displays(text: str) -> list[GpuDevice]:
    payload = _load_json_payload(text)
    if not isinstance(payload, dict):
        return []
    rows = payload.get("SPDisplaysDataType") or []
    if isinstance(rows, dict):
        rows = [rows]
    devices: list[GpuDevice] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("sppci_model") or row.get("spdisplays_device-name") or "").strip()
        if not name:
            continue
        vram_text = str(row.get("_spdisplays_vram") or row.get("spdisplays_vram") or "")
        total = _parse_size_mb(vram_text)
        devices.append(
            GpuDevice(
                name=name,
                vendor=vendor_from_name(name),
                vram_total_mb=total,
                vram_free_mb=total,
                source="system_profiler",
            )
        )
    return devices


def read_linux_drm_gpus(sysfs_root: Path) -> list[GpuDevice]:
    if not sysfs_root.is_dir():
        return []
    devices: list[GpuDevice] = []
    for card in sorted(sysfs_root.glob("card*")):
        device_dir = card / "device"
        total_path = device_dir / "mem_info_vram_total"
        if not total_path.is_file():
            continue
        try:
            total_bytes = int(total_path.read_text().strip() or "0")
        except (OSError, ValueError):
            continue
        if total_bytes <= 0:
            continue
        used_bytes = 0
        used_path = device_dir / "mem_info_vram_used"
        if used_path.is_file():
            try:
                used_bytes = int(used_path.read_text().strip() or "0")
            except (OSError, ValueError):
                used_bytes = 0
        vendor = "unknown"
        vendor_path = device_dir / "vendor"
        if vendor_path.is_file():
            vendor = vendor_from_hex(vendor_path.read_text().strip()) or "unknown"
        name = card.name
        for candidate in (device_dir / "product_name", device_dir / "product"):
            if candidate.is_file():
                product = candidate.read_text(errors="replace").strip("\x00 \n")
                if product:
                    name = product
                    break
        total_mb = max(1, total_bytes // (1024 * 1024))
        free_mb = max(0, (total_bytes - used_bytes) // (1024 * 1024))
        devices.append(
            GpuDevice(
                name=name,
                vendor=vendor,
                vram_total_mb=total_mb,
                vram_free_mb=free_mb,
                source="sysfs",
            )
        )
    return devices


def vendor_from_pnp(pnp: str) -> str | None:
    match = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp)
    if not match:
        return None
    return _VENDOR_BY_PCI.get(match.group(1).lower())


def vendor_from_hex(value: str) -> str | None:
    match = re.search(r"([0-9A-Fa-f]{4})\s*$", value.strip())
    if not match:
        return None
    return _VENDOR_BY_PCI.get(match.group(1).lower())


def vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered or "advanced micro" in lowered:
        return "amd"
    if "intel" in lowered or re.search(r"\barc\b", lowered):
        return "intel"
    if "apple" in lowered:
        return "apple"
    return "unknown"


def win32_adapter_ram_mb(value: object) -> int | None:
    """AdapterRAM в Win32 — uint32 и насыщается около 4 ГиБ.

    Значения у верхней границы uint32 не используем как объём VRAM:
    для больших карт это потолок API, а не настоящий размер.
    """

    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        number += 2**32
    if number <= 0 or number >= 0xFFF00000:
        return None
    mb = number // (1024 * 1024)
    return mb or None


def merge_gpu_lists(*groups: list[GpuDevice]) -> list[GpuDevice]:
    """Склеивает источники. Более ранний список выигрывает при совпадении.

    NVIDIA из nvidia-smi не дублируется записью Win32 с тем же вендором.
    Vulkan-запись дополняет уже известное устройство, если у того нет имени,
    и добавляется, только если такого имени ещё нет.
    """

    merged: list[GpuDevice] = []
    for group in groups:
        for device in group:
            if _already_known(merged, device):
                continue
            merged.append(device)
    return merged


def apply_vulkan_memory(devices: list[GpuDevice], memory_by_name: dict[str, int]) -> None:
    if not memory_by_name:
        return
    normalized = {key.strip().lower(): value for key, value in memory_by_name.items()}
    for device in devices:
        if device.vram_total_mb is not None:
            continue
        total = normalized.get(device.name.strip().lower())
        if total is None:
            continue
        device.vram_total_mb = total
        if device.vram_free_mb is None:
            device.vram_free_mb = total


def _already_known(merged: list[GpuDevice], device: GpuDevice) -> bool:
    for existing in merged:
        if existing.name.strip().lower() == device.name.strip().lower():
            return True
        if (
            device.vendor == "nvidia"
            and existing.vendor == "nvidia"
            and existing.source == "nvidia-smi"
            and device.source != "nvidia-smi"
        ):
            return True
    return False


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+)$", block, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _load_json_payload(text: str) -> object | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _parse_size_mb(value: str) -> int | None:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(GiB|GB|ГБ|MiB|MB|МБ)", value, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    if unit.startswith("g") or unit.startswith("г"):
        return int(number * 1024)
    return int(number)


def _to_int(value: str) -> int | None:
    cleaned = value.strip().strip('"')
    if not cleaned or cleaned in {"[N/A]", "N/A"}:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None
