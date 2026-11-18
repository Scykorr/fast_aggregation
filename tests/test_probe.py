from pathlib import Path

from fast_aggregation.resources.command import CommandResult, FakeRunner
from fast_aggregation.resources.parsers import read_linux_drm_gpus
from fast_aggregation.resources.probe import (
    apply_vram_override,
    assemble_resources,
    collect_outputs,
    probe_machine,
)
from tests.test_parsers import NVIDIA, VULKAN, WIN32


def test_live_linux_reports_cpu_and_ram():
    resources = probe_machine()
    assert resources.os_name == "Linux"
    assert resources.arch in {"x64", "arm64"}
    assert resources.ram_total_mb > 256
    assert 0 < resources.ram_available_mb <= resources.ram_total_mb
    assert resources.cpu_logical >= 1
    assert resources.cpu_model
    cpu = next(item for item in resources.backends if item.name == "cpu")
    assert cpu.available is True


def test_windows_inventory_uses_powershell_and_not_linux_sysfs(tmp_path: Path):
    runner = FakeRunner(
        {
            "nvidia-smi": CommandResult(0, NVIDIA, ""),
            "powershell.exe": CommandResult(0, WIN32, ""),
            "vulkaninfo": CommandResult(0, VULKAN, ""),
        }
    )
    raw = collect_outputs(runner, "Windows", deep_vulkan=False, sysfs_root=tmp_path)
    commands = [call[0] for call in runner.calls]
    assert "powershell.exe" in commands
    assert "nvidia-smi" in commands
    assert raw.sysfs_gpus == []
    resources = assemble_resources(
        raw,
        system="Windows",
        arch="AMD64",
        os_version="10.0.26100",
        cpu_model="AMD Ryzen 7 7700",
        cpu_physical=8,
        cpu_logical=16,
        ram_total_mb=65536,
        ram_available_mb=40000,
        vulkan_dll=True,
    )
    assert resources.arch == "x64"
    assert resources.os_name == "Windows"
    assert any(gpu.source == "nvidia-smi" and gpu.vram_free_mb == 23000 for gpu in resources.gpus)
    assert any(gpu.vendor == "amd" for gpu in resources.gpus)
    by_name = {item.name: item.available for item in resources.backends}
    assert by_name["cuda"] is True
    assert by_name["vulkan"] is True
    assert by_name["cpu"] is True
    assert by_name["sycl"] is False


def test_linux_collect_does_not_call_powershell(tmp_path: Path):
    runner = FakeRunner({"nvidia-smi": CommandResult(127, "", "not found")})
    collect_outputs(runner, "Linux", deep_vulkan=False, sysfs_root=tmp_path)
    assert all(call[0] != "powershell.exe" for call in runner.calls)
    assert all(call[0] != "system_profiler" for call in runner.calls)


def test_linux_sysfs_vram(tmp_path: Path):
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "product_name").write_text("Radeon RX 7900 XTX\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text(str(20 * 1024 * 1024 * 1024), encoding="utf-8")
    (device / "mem_info_vram_used").write_text(str(1 * 1024 * 1024 * 1024), encoding="utf-8")
    gpus = read_linux_drm_gpus(tmp_path)
    assert gpus[0].vendor == "amd"
    assert gpus[0].name == "Radeon RX 7900 XTX"
    assert gpus[0].vram_total_mb == 20480
    assert gpus[0].vram_free_mb == 19456


def test_vulkan_loader_without_gpu_does_not_count_as_backend():
    from fast_aggregation.resources.probe import RawOutputs

    resources = assemble_resources(
        RawOutputs(vulkaninfo_present=True, nvidia_smi_detail="nvidia-smi не найден"),
        system="Linux",
        arch="x64",
        os_version="test",
        cpu_model="cpu",
        cpu_physical=2,
        cpu_logical=2,
        ram_total_mb=8000,
        ram_available_mb=4000,
        vulkan_dll=True,
    )
    by_name = {item.name: item.available for item in resources.backends}
    assert by_name["vulkan"] is False
    assert by_name["cpu"] is True


def test_vram_override_fills_only_unknown_cards():
    from tests.helpers import gpu, machine

    resources = machine(gpus=[gpu(total=None, free=None, source="vulkaninfo"), gpu()])
    apply_vram_override(resources, 12288)
    assert resources.gpus[0].vram_total_mb == 12288
    assert resources.gpus[1].vram_total_mb == 24564


def test_darwin_profiler_is_requested():
    payload = '{"SPDisplaysDataType":[{"sppci_model":"Apple M4","_spdisplays_vram":"24 GB"}]}'
    runner = FakeRunner(
        {
            "nvidia-smi": CommandResult(127, "", "not found"),
            "system_profiler": CommandResult(0, payload, ""),
        }
    )
    raw = collect_outputs(runner, "Darwin", deep_vulkan=False, sysfs_root=None)
    resources = assemble_resources(
        raw,
        system="Darwin",
        arch="arm64",
        os_version="15.0",
        cpu_model="Apple M4",
        cpu_physical=10,
        cpu_logical=10,
        ram_total_mb=24576,
        ram_available_mb=12000,
        vulkan_dll=False,
    )
    assert resources.gpus[0].name == "Apple M4"
    assert resources.gpus[0].vram_total_mb == 24576
    assert any(item.name == "metal" and item.available for item in resources.backends)
