"""Короткий замер скорости llama-server и калибровка оценок.

Оценка в каталоге остаётся формулой. Замер — отдельный ответ сервера
с полем timings: префилл и генерация. История лежит в bench.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import httpx

from fast_aggregation.cluster.catalog import PRESETS, _QUANT_SPEED, _tokens_per_second
from fast_aggregation.errors import FastAggError

BENCH_TIMEOUT_SEC = 60
PROMPT = "The quick brown fox jumps over the lazy dog. " * 16
GENERATE_TOKENS = 200
_QUANT_IN_NAME = re.compile(r"Q\d(?:_K)?(?:_[A-Z0-9]+)?", re.IGNORECASE)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode())
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 65536))
            digest.update(handle.read(65536))
    return digest.hexdigest()[:16]


def quant_from_name(name: str) -> str:
    match = _QUANT_IN_NAME.search(name or "")
    if not match:
        return "unknown"
    token = match.group(0).upper().replace("_K_K", "_K")
    if token in _QUANT_SPEED:
        return token
    if token.startswith("Q4"):
        return "Q4_K_M"
    if token.startswith("Q5"):
        return "Q5_K_M"
    if token.startswith("Q8"):
        return "Q8_0"
    return token


def cluster_signature(nodes: list[tuple[str, str]], tensor_split: list[int]) -> str:
    body = "|".join(f"{node_id}:{backend}" for node_id, backend in sorted(nodes))
    split = ",".join(str(part) for part in tensor_split)
    return f"{body}#{split}"


def bench_key(file_fp: str, quant: str, ctx: int, signature: str) -> str:
    return f"{file_fp}|{quant}|{ctx}|{signature}"


def parse_timings(payload: dict) -> dict:
    timings = payload.get("timings") if isinstance(payload, dict) else None
    if not isinstance(timings, dict):
        raise FastAggError("В ответе llama-server нет поля timings.")
    try:
        prefill = float(timings["prompt_per_second"])
        generation = float(timings["predicted_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FastAggError("В timings нет prompt_per_second или predicted_per_second.") from exc
    return {
        "prefill_tps": round(prefill, 2),
        "generation_tps": round(generation, 2),
        "prompt_tokens": int(timings.get("prompt_n") or 0),
        "generated_tokens": int(timings.get("predicted_n") or 0),
    }


def completion_body() -> dict:
    return {
        "prompt": PROMPT,
        "n_predict": GENERATE_TOKENS,
        "temperature": 0,
        "stream": False,
    }


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def save_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows[:30], ensure_ascii=False, indent=2), encoding="utf-8")


def mark_stale(rows: list[dict], signature: str) -> list[dict]:
    marked = []
    for item in rows:
        copy = dict(item)
        copy["stale"] = copy.get("signature") != signature
        marked.append(copy)
    return marked


def fresh_for_signature(rows: list[dict], signature: str) -> list[dict]:
    return [item for item in rows if item.get("signature") == signature and not item.get("stale")]


def find_cached(rows: list[dict], key: str, signature: str) -> dict | None:
    for item in rows:
        if item.get("key") == key and item.get("signature") == signature:
            return item
    return None


def scale_from_measurement(preset: dict, measurement: dict) -> tuple[float, str]:
    weight = max(preset["weight_gib"] * 1024, 1.0)
    measured_weight = max(float(measurement.get("weight_mib") or weight), 1.0)
    measured_layers = max(int(measurement.get("layers") or preset["layers"]), 1)
    quant_measured = measurement.get("quant") or "Q4_K_M"
    quant_ratio = _QUANT_SPEED.get(preset["quant"], 1.0) / _QUANT_SPEED.get(quant_measured, 1.0)
    scaled = float(measurement["generation_tps"]) * (measured_weight / weight) * (measured_layers / preset["layers"]) * quant_ratio
    when = str(measurement.get("measured_at") or "")[:10]
    source = (
        f"замер {measurement.get('model_name') or 'модели'} от {when}, "
        f"{measurement['generation_tps']} ток/с генерации"
    )
    return round(max(scaled, 0.1), 1), source


def annotate_speeds(rows: list[dict], history: list[dict], signature: str) -> list[dict]:
    fresh = fresh_for_signature(history, signature)
    latest = fresh[0] if fresh else None
    for row in rows:
        estimate = row.get("tokens_per_second")
        row["estimate_tps"] = estimate
        row["speed_kind"] = "оценка"
        row["measured_prefill_tps"] = None
        row["measured_generation_tps"] = None
        row["measured_at"] = None
        row["calibrated_tps"] = None
        row["calibration_source"] = None
        if latest and estimate:
            calibrated, source = scale_from_measurement(
                next(preset for preset in PRESETS if preset["id"] == row["id"]),
                latest,
            )
            row["calibrated_tps"] = calibrated
            row["calibration_source"] = source
            row["speed_kind"] = "оценка по замеру"
    return rows


def formula_tps(preset_id: str, capacity) -> float | None:
    preset = next((item for item in PRESETS if item["id"] == preset_id), None)
    if preset is None or capacity is None:
        return None
    return _tokens_per_second(preset, capacity)


class BenchRun:
    def __init__(self, client_factory=None) -> None:
        self.status = "idle"
        self.error: str | None = None
        self.result: dict | None = None
        self._cancel = threading.Event()
        self._client: httpx.Client | None = None
        self._thread: threading.Thread | None = None
        self._client_factory = client_factory or (lambda: httpx.Client(timeout=BENCH_TIMEOUT_SEC))

    def view(self) -> dict:
        return {"status": self.status, "error": self.error, "result": self.result}

    def start(self, base_url: str, on_done) -> None:
        if self.status == "running":
            raise FastAggError("Замер уже идёт.")
        self.status = "running"
        self.error = None
        self.result = None
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(base_url.rstrip("/"), on_done),
            name="fa-bench",
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        client = self._client
        if client is not None:
            client.close()
        if self.status == "running":
            self.status = "cancelled"

    def _run(self, base_url: str, on_done) -> None:
        if self._cancel.is_set():
            self.status = "cancelled"
            return
        client = self._client_factory()
        self._client = client
        try:
            response = client.post(base_url + "/completion", json=completion_body())
            if self._cancel.is_set():
                self.status = "cancelled"
                return
            if response.status_code >= 400:
                raise FastAggError(f"llama-server ответил {response.status_code}.")
            parsed = parse_timings(response.json())
            record = on_done(parsed)
            self.result = record
            self.status = "done"
        except FastAggError as exc:
            if self._cancel.is_set():
                self.status = "cancelled"
                return
            self.status = "error"
            self.error = str(exc)
        except (httpx.HTTPError, ValueError) as exc:
            if self._cancel.is_set():
                self.status = "cancelled"
                return
            self.status = "error"
            self.error = f"Замер не выполнился: {exc}"
        finally:
            self._client = None
            client.close()
