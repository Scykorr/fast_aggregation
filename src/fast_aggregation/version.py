"""Версия продукта. Единственный источник строки — ``fast_aggregation.__version__``.

Номер сборки CI (``GITHUB_RUN_NUMBER``) дописывается как метаданные PEP 440:
``0.2.0+14``. В именах файлов плюс заменён на ``-b``, потому что ``+`` ломает
часть инструментов: ``0.2.0-b14``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fast_aggregation import __version__

_RELEASE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
# major 0..20, minor 0..99, patch 0..99, сборка 0..9999 — меньше лимита Android 2_100_000_000.
_CODE_MAJOR = 100_000_000
_CODE_MINOR = 1_000_000
_CODE_PATCH = 10_000


def product_version() -> str:
    return __version__


def build_number() -> str:
    for key in ("FASTAGG_BUILD", "GITHUB_RUN_NUMBER"):
        value = os.environ.get(key, "").strip()
        if value.isdigit():
            return value
    return _baked_build()


def _baked_build() -> str:
    try:
        from fast_aggregation import _build_info
    except ImportError:
        return ""
    value = str(getattr(_build_info, "BUILD", "") or "").strip()
    return value if value.isdigit() else ""


def full_version() -> str:
    build = build_number()
    return f"{__version__}+{build}" if build else __version__


def file_version() -> str:
    build = build_number()
    return f"{__version__}-b{build}" if build else __version__


def release_version(value: str | None) -> str:
    """Отбрасывает +метаданные и суффикс -bN, чтобы сравнивать выпуск, а не номер CI."""

    text = (value or "").strip()
    if "+" in text:
        text = text.split("+", 1)[0]
    tail = text.rsplit("-b", 1)
    if len(tail) == 2 and tail[1].isdigit():
        text = tail[0]
    return text


def versions_differ(node_version: str | None, coordinator: str | None = None) -> bool:
    left = release_version(node_version)
    right = release_version(coordinator if coordinator is not None else product_version())
    if not left or not right:
        return True
    return left != right


def version_status(node_version: str | None) -> dict:
    node = release_version(node_version)
    coordinator = product_version()
    if not node:
        return {"mismatch": True, "detail": "узел не сообщил версию"}
    if node != coordinator:
        return {"mismatch": True, "detail": f"версия узла {node}, у координатора {coordinator}"}
    return {"mismatch": False, "detail": ""}


def version_tuple() -> tuple[int, int, int, int]:
    match = _RELEASE.match(product_version())
    if match is None:
        raise ValueError(f"версия должна быть major.minor.patch, сейчас {product_version()}")
    major, minor, patch = (int(part) for part in match.groups())
    build = int(build_number() or "0")
    return major, minor, patch, build


def windows_quad() -> str:
    return ".".join(str(part) for part in version_tuple())


def android_version_code(version: str | None = None, build: int | None = None) -> int:
    match = _RELEASE.match(release_version(version or product_version()))
    if match is None:
        raise ValueError("версия должна быть major.minor.patch")
    major, minor, patch = (int(part) for part in match.groups())
    number = int(build_number() or "0") if build is None else int(build)
    if number < 0 or number > 9999:
        raise ValueError("номер сборки должен быть от 0 до 9999")
    return major * _CODE_MAJOR + minor * _CODE_MINOR + patch * _CODE_PATCH + number


def stamp_build_info(path: Path | None = None) -> Path:
    """Записать номер сборки в пакет, чтобы exe помнил его без переменных CI."""

    build = ""
    for key in ("FASTAGG_BUILD", "GITHUB_RUN_NUMBER"):
        value = os.environ.get(key, "").strip()
        if value.isdigit():
            build = value
            break
    target = path or Path(__file__).resolve().parent / "_build_info.py"
    target.write_text(f'BUILD = "{build}"\n', encoding="utf-8")
    return target


def write_windows_version_file(path: Path) -> None:
    major, minor, patch, build = version_tuple()
    shown = full_version()
    path.write_text(
        "\n".join(
            [
                "VSVersionInfo(",
                "  ffi=FixedFileInfo(",
                f"    filevers=({major}, {minor}, {patch}, {build}),",
                f"    prodvers=({major}, {minor}, {patch}, {build}),",
                "    mask=0x3f,",
                "    flags=0x0,",
                "    OS=0x40004,",
                "    fileType=0x1,",
                "    subtype=0x0,",
                "    date=(0, 0)",
                "  ),",
                "  kids=[",
                "    StringFileInfo([",
                "      StringTable(",
                "        '040904B0',",
                "        [",
                "          StringStruct('CompanyName', 'fastagg'),",
                "          StringStruct('FileDescription', 'fastagg'),",
                f"          StringStruct('FileVersion', '{shown}'),",
                "          StringStruct('InternalName', 'fastagg'),",
                "          StringStruct('OriginalFilename', 'fastagg.exe'),",
                "          StringStruct('ProductName', 'fastagg'),",
                f"          StringStruct('ProductVersion', '{shown}')",
                "        ]",
                "      )",
                "    ]),",
                "    VarFileInfo([VarStruct('Translation', [1033, 1200])])",
                "  ]",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
