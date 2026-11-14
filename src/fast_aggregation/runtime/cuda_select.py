"""Какую сборку CUDA брать на Windows по версии драйвера.

По таблице совместимости NVIDIA (CUDA 13.5 release notes):

- CUDA 12.4 на Windows требует драйвер не ниже 551.61.
- Сборки CUDA 13.x требуют драйвер не ниже 580.
- Драйвер новее своей ветки по-прежнему запускает более старые сборки,
  но на ветке 580 и новее для свежих карт надёжнее сборка 13.5:
  окно minor-совместимости CUDA 12.x формально заканчивается на 580.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WINDOWS_CUDA_12_4_MIN = (551, 61)
WINDOWS_CUDA_13_MIN = (580, 0)


@dataclass(frozen=True)
class CudaSelection:
    build: str
    reason: str


def parse_driver_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)", text or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def select_windows_cuda_build(preference: str | None, driver_text: str | None) -> CudaSelection:
    """preference: auto, 12.4 или 13.5. Пустая строка значит auto."""

    wanted = (preference or "auto").strip()
    version = parse_driver_version(driver_text or "")
    shown = _format(version)
    if wanted in {"12.4", "13.5"}:
        note = _fit_note(wanted, version)
        return CudaSelection(wanted, f"CUDA {wanted} выбрана вручную. {note}".strip())
    if version is None:
        return CudaSelection("12.4", "Версия драйвера неизвестна, берём CUDA 12.4. Можно задать --cuda-build 13.5.")
    if version >= WINDOWS_CUDA_13_MIN:
        return CudaSelection(
            "13.5",
            f"Драйвер {shown} не ниже 580, берём CUDA 13.5. На такой ветке сборка 12.4 часто уже не нужна.",
        )
    if version >= WINDOWS_CUDA_12_4_MIN:
        return CudaSelection(
            "12.4",
            f"Драйвер {shown} подходит для CUDA 12.4 и ниже порога 580 для CUDA 13.5.",
        )
    return CudaSelection(
        "12.4",
        f"Драйвер {shown} старше минимума CUDA 12.4 (551.61). Обновите драйвер NVIDIA или используйте Vulkan.",
    )


def cudart_archive_name(build: str) -> str:
    return f"cudart-llama-bin-win-cuda-{build}-x64.zip"


def _fit_note(build: str, version: tuple[int, ...] | None) -> str:
    if version is None:
        return ""
    shown = _format(version)
    if build == "13.5" and version < WINDOWS_CUDA_13_MIN:
        return f"Драйвер {shown} ниже 580, сборка CUDA 13.5 может не стартовать."
    if build == "12.4" and version < WINDOWS_CUDA_12_4_MIN:
        return f"Драйвер {shown} ниже 551.61, сборка CUDA 12.4 может не стартовать."
    if build == "12.4" and version >= WINDOWS_CUDA_13_MIN:
        return f"Драйвер {shown} из ветки 580 или новее. Если 12.4 не стартует, переключитесь на --cuda-build 13.5."
    return ""


def _format(version: tuple[int, ...] | None) -> str:
    if not version:
        return "неизвестен"
    return ".".join(str(part) for part in version)
