from fast_aggregation.resources.parsers import (
    merge_gpu_lists,
    parse_macos_displays,
    parse_nvidia_smi,
    parse_vulkan_memory,
    parse_vulkan_summary,
    parse_win32_video_controllers,
    win32_adapter_ram_mb,
)

NVIDIA = """\
NVIDIA GeForce RTX 4090, 24564, 23000, 560.94
NVIDIA GeForce RTX 3060, 12288, 11000, 560.94
"""

WIN32 = """
{"cpu":{"Name":"AMD Ryzen 7 7700"},"gpus":[
  {"Name":"NVIDIA GeForce RTX 4090","AdapterRAM":4294967295,"DriverVersion":"32.0.15.6094","PNPDeviceID":"PCI\\\\VEN_10DE&DEV_2684"},
  {"Name":"AMD Radeon RX 7600","AdapterRAM":2147483648,"DriverVersion":"32.0.1","PNPDeviceID":"PCI\\\\VEN_1002&DEV_7480"},
  {"Name":"Microsoft Basic Display Adapter","AdapterRAM":0,"DriverVersion":"10.0","PNPDeviceID":"PCI\\\\VEN_1414&DEV_008C"}
]}
"""

VULKAN = """
GPU0:
	deviceName         = NVIDIA GeForce RTX 4090
	vendorID           = 0x10de
	driverVersion      = 560.94
GPU1:
	deviceName         = AMD Radeon RX 7600
	vendorID           = 0x1002
	driverVersion      = 24.1
"""

VULKAN_MEMORY = """
GPU0:
	deviceName = NVIDIA GeForce RTX 4090
	memoryHeaps[0]:
		size   = 25769803776 (0x600000000) (24.00 GiB)
		flags: count = 1
			MEMORY_HEAP_DEVICE_LOCAL_BIT
	memoryHeaps[1]:
		size   = 1073741824 (0x40000000)
		flags: count = 0
			none
GPU1:
	deviceName = AMD Radeon RX 7600
	memoryHeaps[0]:
		size   = 8589934592 (0x200000000)
		flags:
			MEMORY_HEAP_DEVICE_LOCAL_BIT
"""


def test_nvidia_smi_parses_several_gpus():
    devices = parse_nvidia_smi(NVIDIA)
    assert [item.name for item in devices] == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3060",
    ]
    assert devices[0].vram_total_mb == 24564
    assert devices[0].vram_free_mb == 23000
    assert devices[0].vendor == "nvidia"
    assert devices[0].source == "nvidia-smi"


def test_nvidia_smi_ignores_empty_and_failure_banner():
    text = "Failed to query\n\n , , , \n"
    assert parse_nvidia_smi(text) == []
    assert parse_nvidia_smi("") == []


def test_win32_vendors_and_capped_vram():
    devices = parse_win32_video_controllers(WIN32)
    names = [item.name for item in devices]
    assert "Microsoft Basic Display Adapter" not in names
    assert names == ["NVIDIA GeForce RTX 4090", "AMD Radeon RX 7600"]
    nvidia, amd = devices
    assert nvidia.vendor == "nvidia"
    assert nvidia.vram_total_mb is None
    assert amd.vendor == "amd"
    assert amd.vram_total_mb == 2048
    assert amd.source == "win32"


def test_win32_single_object_and_negative_ram():
    text = '{"Name":"Intel Arc A770","AdapterRAM":-2147483648,"PNPDeviceID":"PCI\\\\VEN_8086&DEV_56A0"}'
    devices = parse_win32_video_controllers(text)
    assert len(devices) == 1
    assert devices[0].vendor == "intel"
    assert devices[0].vram_total_mb == 2048


def test_adapter_ram_saturation():
    assert win32_adapter_ram_mb(4294967295) is None
    assert win32_adapter_ram_mb(0) is None
    assert win32_adapter_ram_mb(None) is None
    assert win32_adapter_ram_mb(1024 * 1024 * 512) == 512


def test_vulkan_summary_and_memory_heaps():
    devices = parse_vulkan_summary(VULKAN)
    assert [item.vendor for item in devices] == ["nvidia", "amd"]
    memory = parse_vulkan_memory(VULKAN_MEMORY)
    assert memory["NVIDIA GeForce RTX 4090"] == 24576
    assert memory["AMD Radeon RX 7600"] == 8192


def test_merge_prefers_nvidia_smi_over_win32_duplicate():
    accurate = parse_nvidia_smi(NVIDIA)
    generic = parse_win32_video_controllers(WIN32)
    vulkan = parse_vulkan_summary(VULKAN)
    merged = merge_gpu_lists(accurate, generic, vulkan)
    names = [item.name for item in merged]
    assert names.count("NVIDIA GeForce RTX 4090") == 1
    rtx = next(item for item in merged if "4090" in item.name)
    assert rtx.source == "nvidia-smi"
    assert rtx.vram_free_mb == 23000
    assert any(item.vendor == "amd" for item in merged)


def test_macos_display_json():
    text = """
    {"SPDisplaysDataType":[{"sppci_model":"Apple M2","_spdisplays_vram":"16 GB"}]}
    """
    devices = parse_macos_displays(text)
    assert devices[0].name == "Apple M2"
    assert devices[0].vendor == "apple"
    assert devices[0].vram_total_mb == 16384
    assert devices[0].source == "system_profiler"
