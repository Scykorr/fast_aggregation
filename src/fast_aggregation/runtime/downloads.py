"""Скачивание и распаковка архива llama.cpp в каталог кэша."""

from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from fast_aggregation.errors import RuntimeError_
from fast_aggregation.runtime.releases import ReleaseAsset, resolve_asset

Downloader = Callable[[str, Path], None]


def default_downloader(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 256):
                    handle.write(chunk)
        temporary.replace(dest)
    except httpx.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError_(f"Не удалось скачать {url}: {exc}") from exc


def ensure_runtime(
    tag: str,
    system: str,
    arch: str,
    backend: str,
    cache_root: Path,
    *,
    role: str = "rpc",
    cuda_build: str | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Возвращает каталог с распакованной сборкой.

    role — `rpc` (нужен ggml-rpc-server) или `server` (нужен llama-server).
    """

    asset = resolve_asset(tag, system, arch, backend, cuda_build=cuda_build)
    folder = cache_root / asset.tag / asset.name
    if _has_binary(folder, system, role):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    archive = cache_root / asset.tag / f"{asset.name}.archive"
    fetch = downloader or default_downloader
    if not archive.is_file() or archive.stat().st_size == 0:
        fetch(asset.url, archive)
    extract_archive(archive, folder)
    _mark_executables(folder, system)
    if not _has_binary(folder, system, role):
        raise RuntimeError_(
            f"В архиве {asset.name} нет бинарника для роли «{role}». Каталог: {folder}"
        )
    return folder


def executable(folder: Path, system: str, role: str) -> Path:
    """role: rpc | server."""

    base = "ggml-rpc-server" if role == "rpc" else "llama-server"
    if _os(system) == "windows":
        base += ".exe"
    matches = [path for path in folder.rglob(base) if path.is_file()]
    if not matches:
        raise RuntimeError_(f"В {folder} не найден {base}")
    matches.sort(key=lambda path: len(path.parts))
    return matches[0]


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, dest)
        return
    _extract_tar(archive, dest)


def _extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                if name:
                    _safe_target(dest, name).mkdir(parents=True, exist_ok=True)
                continue
            target = _safe_target(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info, "r") as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        if hasattr(tarfile, "data_filter"):
            # filter="data" отвергает пути с .. и специальные файлы (Python 3.12+).
            bundle.extractall(dest, filter="data")
            return
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError_(f"Архив содержит ссылку {member.name}, распаковка остановлена")
            _safe_target(dest, member.name)
        bundle.extractall(dest)


def _safe_target(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    root = dest.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError_(f"Архив пытается записать файл вне каталога: {name}")
    return target


def _mark_executables(folder: Path, system: str) -> None:
    if _os(system) == "windows":
        return
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == "" or path.name.startswith("lib"):
            mode = path.stat().st_mode
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _has_binary(folder: Path, system: str, role: str) -> bool:
    try:
        executable(folder, system, role)
    except RuntimeError_:
        return False
    return True


def _os(system: str) -> str:
    return system.strip().lower()
