import zipfile
from pathlib import Path

import pytest

from fast_aggregation.errors import RuntimeError_
from fast_aggregation.runtime.downloads import ensure_runtime, executable, extract_archive


def _zip_with_binaries(path: Path, prefix: str = "") -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(prefix + "ggml-rpc-server.exe", b"rpc")
        bundle.writestr(prefix + "llama-server.exe", b"srv")


def test_extract_finds_nested_windows_binaries(tmp_path: Path):
    archive = tmp_path / "llama.zip"
    _zip_with_binaries(archive, prefix="build/")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert executable(dest, "Windows", "rpc").read_bytes() == b"rpc"
    assert executable(dest, "Windows", "server").name == "llama-server.exe"


def test_extract_rejects_zip_slip(tmp_path: Path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../evil.txt", b"nope")
    with pytest.raises(RuntimeError_):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path.parent / "evil.txt").exists()


def test_ensure_runtime_uses_downloader_once(tmp_path: Path):
    calls: list[str] = []

    def download(url: str, dest: Path) -> None:
        calls.append(url)
        _zip_with_binaries(dest)

    folder = ensure_runtime(
        "b12359",
        "Windows",
        "x64",
        "vulkan",
        tmp_path,
        role="rpc",
        downloader=download,
        allow_download=True,
    )
    again = ensure_runtime(
        "b12359",
        "Windows",
        "x64",
        "vulkan",
        tmp_path,
        role="server",
        downloader=download,
        allow_download=True,
    )
    assert folder == again
    assert len(calls) == 1
    assert calls[0].endswith("llama-b12359-bin-win-vulkan-x64.zip")
    assert executable(folder, "Windows", "rpc").is_file()
