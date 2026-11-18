"""Оценка, влезет ли GGUF в сумму свободной памяти плана.

Читается только заголовок GGUF (ключи метаданных), не тензоры.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from fast_aggregation.errors import PlacementError
from fast_aggregation.models import PlacementPlan

_GGUF_MAGIC = b"GGUF"
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")


@dataclass(frozen=True)
class ModelNeed:
    weight_mb: float
    kv_mb: float
    total_mb: float
    note: str


def estimate_need(path: str | None, *, ctx_size: int | None, parallel: int) -> ModelNeed | None:
    if not path:
        return None
    file = Path(path)
    weight_mb = file.stat().st_size / (1024 * 1024)
    meta = read_gguf_metadata(file)
    ctx = ctx_size or 0
    kv_mb = 0.0
    note = "KV-кэш посчитан по заголовку GGUF"
    if ctx > 0 and meta.get("block_count") and meta.get("embedding_length") and meta.get("head_count"):
        n_layers = int(meta["block_count"])
        n_embd = int(meta["embedding_length"])
        n_head = int(meta["head_count"])
        n_kv = int(meta.get("head_count_kv") or n_head)
        head_dim = n_embd / n_head if n_head else 0
        # K и V в f16, умножить на число параллельных слотов.
        slots = max(1, parallel)
        kv_mb = n_layers * ctx * n_kv * head_dim * 2 * 2 * slots / (1024 * 1024)
    elif ctx > 0:
        kv_mb = weight_mb * 0.15
        note = "В заголовке нет числа слоёв, KV оценён как 15% весов"
    else:
        note = "Контекст не задан, в оценку входит только размер файла"
    return ModelNeed(weight_mb=weight_mb, kv_mb=kv_mb, total_mb=weight_mb + kv_mb, note=note)


def assert_fits(plan: PlacementPlan, need: ModelNeed | None) -> str | None:
    if need is None:
        return "Размер модели заранее неизвестен (репозиторий Hugging Face). Проверка вместимости пропущена."
    have = sum(device.memory_mb for device in plan.devices)
    if need.total_mb > have:
        raise PlacementError(
            f"Нужно около {need.total_mb:.0f} МиБ (веса {need.weight_mb:.0f} + KV {need.kv_mb:.0f}), "
            f"в плане свободно {have} МиБ. Уменьшите контекст, возьмите квантование плотнее "
            "или добавьте устройство."
        )
    return need.note


def read_gguf_metadata(path: Path) -> dict[str, int | str]:
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != _GGUF_MAGIC:
        return {}
    position = 4
    _version, position = _unpack(_U32, data, position)
    _tensors, position = _unpack(_U64, data, position)
    kv_count, position = _unpack(_U64, data, position)
    found: dict[str, int | str] = {}
    for _ in range(kv_count):
        if position >= len(data):
            break
        key, position = _read_string(data, position)
        if position + 4 > len(data):
            break
        vtype, position = _unpack(_U32, data, position)
        value, position = _read_value(data, position, vtype)
        short = _interesting(key)
        if short and isinstance(value, (int, str)):
            found[short] = value
    return found


def _interesting(key: str) -> str | None:
    if key.endswith(".block_count"):
        return "block_count"
    if key.endswith(".embedding_length"):
        return "embedding_length"
    if key.endswith(".attention.head_count_kv"):
        return "head_count_kv"
    if key.endswith(".attention.head_count"):
        return "head_count"
    return None


def _unpack(fmt: struct.Struct, data: bytes, position: int) -> tuple[int, int]:
    value = fmt.unpack_from(data, position)[0]
    return int(value), position + fmt.size


def _read_string(data: bytes, position: int) -> tuple[str, int]:
    length, position = _unpack(_U64, data, position)
    end = position + length
    return data[position:end].decode("utf-8", "replace"), end


def _read_value(data: bytes, position: int, vtype: int) -> tuple[object, int]:
    if vtype == 8:
        return _read_string(data, position)
    if vtype == 9:
        elem_type, position = _unpack(_U32, data, position)
        count, position = _unpack(_U64, data, position)
        for _ in range(count):
            _value, position = _read_value(data, position, elem_type)
        return None, position
    if vtype in {4, 5, 7}:
        value, position = _unpack(_U32, data, position)
        return value, position
    if vtype in {10, 11}:
        return _unpack(_U64, data, position)
    if vtype in {0, 1}:
        return data[position], position + 1
    if vtype in {2, 3}:
        return int.from_bytes(data[position : position + 2], "little"), position + 2
    if vtype == 6:
        return _F32.unpack_from(data, position)[0], position + 4
    if vtype == 12:
        return _F64.unpack_from(data, position)[0], position + 8
    raise ValueError(f"неизвестный тип GGUF {vtype}")
