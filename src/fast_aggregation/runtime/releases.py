"""Имена файлов официальных сборок llama.cpp для одного тега релиза.

Имена сверены с релизом b12359 (ночная сборка, на которую ссылается
llama.cpp v1.0.0). Другой тег подставляется в тот же шаблон; если ggml
переименует артефакты, достаточно сменить тег и поправить таблицу.
"""

from __future__ import annotations

from dataclasses import dataclass

from fast_aggregation.constants import LLAMA_RELEASE_REPO
from fast_aggregation.errors import RuntimeError_

# Варианты CUDA, которые реально лежат в релизе b12359.
_CUDA_BUILD = {
    ("windows", "x64"): "12.4",
    ("windows", "arm64"): "13.5",
    ("linux", "x64"): "12.8",
    ("linux", "arm64"): "13.5",
}


@dataclass(frozen=True)
class ReleaseAsset:
    tag: str
    name: str
    system: str
    arch: str
    backend: str

    @property
    def url(self) -> str:
        return f"https://github.com/{LLAMA_RELEASE_REPO}/releases/download/{self.tag}/{self.name}"

    @property
    def kind(self) -> str:
        if self.name.endswith(".zip"):
            return "zip"
        if self.name.endswith(".tar.gz"):
            return "tar.gz"
        raise RuntimeError_(f"Неизвестный тип архива: {self.name}")


def resolve_asset(
    tag: str,
    system: str,
    arch: str,
    backend: str,
    *,
    cuda_build: str | None = None,
) -> ReleaseAsset:
    os_key = _os_key(system)
    arch_key = _arch_key(arch)
    backend_key = backend.strip().lower()
    if os_key == "darwin":
        if backend_key not in {"metal", "cpu", "auto"}:
            raise RuntimeError_(
                "Для macOS в релизе llama.cpp одна сборка с Metal. "
                f"Отдельного архива для бэкенда «{backend_key}» нет."
            )
        return ReleaseAsset(
            tag=tag,
            name=f"llama-{tag}-bin-macos-{arch_key}.tar.gz",
            system=os_key,
            arch=arch_key,
            backend="metal",
        )
    if backend_key == "metal":
        raise RuntimeError_("Сборка Metal есть только для macOS")

    suffix = _backend_suffix(os_key, arch_key, backend_key, cuda_build)
    if os_key == "windows":
        name = f"llama-{tag}-bin-win-{suffix}-{arch_key}.zip"
    else:
        if suffix == "cpu":
            name = f"llama-{tag}-bin-ubuntu-{arch_key}.tar.gz"
        else:
            name = f"llama-{tag}-bin-ubuntu-{suffix}-{arch_key}.tar.gz"
    return ReleaseAsset(
        tag=tag,
        name=name,
        system=os_key,
        arch=arch_key,
        backend=backend_key,
    )


def engine_backend(system: str) -> str:
    """Бэкенд бинарника llama-server на координаторе.

    Считают воркеры через RPC. Самому серверу достаточно сборки с
    клиентом RPC: на Windows и Linux это CPU-архив того же тега,
    на macOS — единственная Metal-сборка.
    """

    if _os_key(system) == "darwin":
        return "metal"
    return "cpu"


def _backend_suffix(system: str, arch: str, backend: str, cuda_build: str | None) -> str:
    if backend == "cpu":
        if system == "windows":
            return "cpu"
        return "cpu"
    if backend == "cuda":
        build = cuda_build or _CUDA_BUILD.get((system, arch))
        if build is None:
            raise RuntimeError_(f"В каталоге релиза нет CUDA-сборки для {system}/{arch}")
        if system == "windows" and arch == "x64" and build not in {"12.4", "13.5"}:
            raise RuntimeError_("Для Windows x64 ожидается CUDA 12.4 или 13.5")
        if system == "windows" and arch == "arm64" and build != "13.5":
            raise RuntimeError_("Для Windows arm64 в релизе есть только CUDA 13.5")
        if system == "linux" and arch == "x64" and build != "12.8":
            raise RuntimeError_("Для Linux x64 в закреплённом релизе CUDA-сборка 12.8")
        if system == "linux" and arch == "arm64" and build != "13.5":
            raise RuntimeError_("Для Linux arm64 в закреплённом релизе CUDA-сборка 13.5")
        return f"cuda-{build}"
    if backend == "vulkan":
        return "vulkan"
    if backend == "rocm":
        if arch != "x64":
            raise RuntimeError_("В закреплённом релизе ROCm-сборка только для x64")
        return "rocm-10.0"
    if backend == "sycl":
        if system != "windows" or arch != "x64":
            raise RuntimeError_("Отдельная SYCL-сборка в этой таблице есть для Windows x64")
        return "sycl"
    raise RuntimeError_(f"Неизвестный бэкенд «{backend}»")


def _os_key(system: str) -> str:
    mapping = {"windows": "windows", "linux": "linux", "darwin": "darwin"}
    key = system.strip().lower()
    if key not in mapping:
        raise RuntimeError_(f"ОС «{system}» не поддерживается загрузчиком сборок")
    return mapping[key]


def _arch_key(arch: str) -> str:
    value = arch.strip().lower()
    if value in {"x64", "amd64", "x86_64"}:
        return "x64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    raise RuntimeError_(f"Архитектура «{arch}» не поддерживается загрузчиком сборок")
