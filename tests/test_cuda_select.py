from pathlib import Path
import zipfile

from fast_aggregation.runtime.cuda_select import select_windows_cuda_build
from fast_aggregation.runtime.downloads import install_cudart_dlls


def test_driver_picks_12_4_or_13_5():
    assert select_windows_cuda_build("auto", "560.94").build == "12.4"
    assert select_windows_cuda_build("auto", "551.61").build == "12.4"
    assert select_windows_cuda_build("auto", "572.16").build == "12.4"
    palit = select_windows_cuda_build("auto", "581.15")
    assert palit.build == "13.5"
    assert "13.5" in palit.reason
    fresh = select_windows_cuda_build("auto", "620.10")
    assert fresh.build == "13.5"
    manual = select_windows_cuda_build("12.4", "620.10")
    assert manual.build == "12.4"
    assert "13.5" in manual.reason
    old = select_windows_cuda_build("auto", "536.25")
    assert old.build == "12.4"
    assert "551.61" in old.reason
    assert select_windows_cuda_build("13.5", None).build == "13.5"


def test_cudart_dlls_land_next_to_the_server(tmp_path: Path):
    archive = tmp_path / "cudart.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("bin/cudart64_12.dll", b"dll")
        bundle.writestr("readme.txt", b"no")
    target = tmp_path / "bin"
    copied = install_cudart_dlls(target, archive)
    assert copied == 1
    assert (target / "cudart64_12.dll").read_bytes() == b"dll"
    assert not (target / "readme.txt").exists()
