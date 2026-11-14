import pytest

from fast_aggregation.errors import RuntimeError_
from fast_aggregation.runtime.releases import engine_backend, resolve_asset


def test_windows_asset_names_match_llama_cpp_b12359():
    cuda = resolve_asset("b12359", "Windows", "x64", "cuda")
    assert cuda.name == "llama-b12359-bin-win-cuda-12.4-x64.zip"
    assert cuda.url.endswith("/b12359/" + cuda.name)
    vulkan = resolve_asset("b12359", "Windows", "amd64", "vulkan")
    assert vulkan.name == "llama-b12359-bin-win-vulkan-x64.zip"
    cpu = resolve_asset("b12359", "Windows", "x64", "cpu")
    assert cpu.name == "llama-b12359-bin-win-cpu-x64.zip"
    arm = resolve_asset("b12359", "Windows", "arm64", "cuda")
    assert arm.name == "llama-b12359-bin-win-cuda-13.5-arm64.zip"
    rocm = resolve_asset("b12359", "Windows", "x64", "rocm")
    assert rocm.name == "llama-b12359-bin-win-rocm-10.0-x64.zip"


def test_linux_and_macos_asset_names():
    cpu = resolve_asset("b12359", "Linux", "x86_64", "cpu")
    assert cpu.name == "llama-b12359-bin-ubuntu-x64.tar.gz"
    vulkan = resolve_asset("b12359", "Linux", "x64", "vulkan")
    assert vulkan.name == "llama-b12359-bin-ubuntu-vulkan-x64.tar.gz"
    cuda = resolve_asset("b12359", "Linux", "arm64", "cuda")
    assert cuda.name == "llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz"
    mac = resolve_asset("b12359", "Darwin", "arm64", "metal")
    assert mac.name == "llama-b12359-bin-macos-arm64.tar.gz"
    assert engine_backend("Windows") == "cpu"
    assert engine_backend("Darwin") == "metal"


def test_unsupported_combinations():
    with pytest.raises(RuntimeError_):
        resolve_asset("b12359", "Windows", "x64", "metal")
    with pytest.raises(RuntimeError_):
        resolve_asset("b12359", "Windows", "x64", "cuda", cuda_build="11.8")
    with pytest.raises(RuntimeError_):
        resolve_asset("b12359", "Linux", "ppc64", "cpu")
