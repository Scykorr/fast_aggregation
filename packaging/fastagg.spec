# -*- mode: python ; coding: utf-8 -*-
"""Сборка одного exe. Запуск из корня репозитория: pyinstaller packaging/fastagg.spec"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).resolve().parent
src = root / "src"
sys.path.insert(0, str(src))
from fast_aggregation.version import stamp_build_info, write_windows_version_file

stamp_build_info()
version_info = root / "packaging" / "file_version_info.txt"
write_windows_version_file(version_info)

datas = collect_data_files("fast_aggregation")
hidden = []
for package in ("fast_aggregation", "uvicorn", "fastapi", "starlette", "anyio", "cryptography", "segno"):
    hidden += collect_submodules(package)
hidden += [
    "pystray",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "psutil",
    "httpx",
    "h11",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
]

a = Analysis(
    [str(src / "fast_aggregation" / "__main__.py")],
    pathex=[str(src)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="fastagg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(root / "packaging" / "fastagg.ico"),
    version=str(version_info),
)
