import json
from pathlib import Path

import httpx
import pytest

from fast_aggregation.cluster.bench import (
    BenchRun,
    annotate_speeds,
    bench_key,
    cluster_signature,
    mark_stale,
    parse_timings,
    quant_from_name,
    scale_from_measurement,
)
from fast_aggregation.cluster.catalog import PRESETS, recommend, summarize_capacity
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.errors import FastAggError
from fast_aggregation.models import PlannedDevice, PlacementPlan


def test_timings_and_cache_key_and_stale_nodes():
    parsed = parse_timings(
        {"timings": {"prompt_per_second": 120.5, "predicted_per_second": 41.2, "prompt_n": 130, "predicted_n": 200}}
    )
    assert parsed["prefill_tps"] == 120.5
    assert parsed["generation_tps"] == 41.2
    with pytest.raises(FastAggError):
        parse_timings({"choices": []})
    assert quant_from_name("model-Q5_K_M.gguf") == "Q5_K_M"
    signature = cluster_signature([("b", "cpu"), ("a", "cuda")], [1, 2])
    other = cluster_signature([("a", "cuda")], [1])
    key = bench_key("abc", "Q4_K_M", 4096, signature)
    assert key != bench_key("abc", "Q4_K_M", 4096, other)
    rows = mark_stale([{"signature": signature, "key": key}, {"signature": other}], signature)
    assert rows[0]["stale"] is False
    assert rows[1]["stale"] is True


def test_calibration_scales_with_weight_and_labels_the_source():
    preset = next(item for item in PRESETS if item["id"] == "70b-q4")
    measured = {
        "model_name": "tiny-Q4_K_M.gguf",
        "generation_tps": 40,
        "weight_mib": 7 * 1024,
        "layers": 32,
        "quant": "Q4_K_M",
        "measured_at": "2026-11-16T12:00:00+00:00",
    }
    scaled, source = scale_from_measurement(preset, measured)
    assert scaled < 40
    assert "tiny-Q4_K_M.gguf" in source
    assert "2026-11-16" in source
    plan = PlacementPlan(
        devices=[
            PlannedDevice(
                node_id="n",
                node_name="pc",
                host="127.0.0.1",
                rpc_port=50052,
                http_port=47000,
                device_index=0,
                backend="cuda",
                name="gpu",
                kind="gpu",
                memory_mb=200000,
                share=1,
                weight=1,
            )
        ]
    )
    capacity = summarize_capacity(plan, [])
    rows = annotate_speeds(recommend(capacity, ctx=4096), [{**measured, "signature": "same", "stale": False}], "same")
    row = next(item for item in rows if item["id"] == "70b-q4")
    assert row["estimate_tps"] == row["tokens_per_second"]
    assert row["speed_kind"] == "оценка по замеру"
    assert row["calibrated_tps"] == scaled
    assert row["calibration_source"] == source


def test_bench_run_reads_timings_and_chat_blocks_a_manual_start(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["n_predict"] == 200
        assert request.url.path == "/completion"
        return httpx.Response(
            200,
            json={"timings": {"prompt_per_second": 90, "predicted_per_second": 30, "prompt_n": 128, "predicted_n": 200}},
        )

    saved = {}

    def on_done(parsed: dict) -> dict:
        saved.update(parsed)
        return parsed

    run = BenchRun(client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    run.start("http://llama", on_done)
    assert run._thread is not None
    run._thread.join(timeout=5)
    assert run.view()["status"] == "done"
    assert saved["prefill_tps"] == 90
    service = CoordinatorService(
        registry=__import__("fast_aggregation.cluster.registry", fromlist=["NodeRegistry"]).NodeRegistry(),
        data_dir=tmp_path,
        background=False,
    )
    service.phase = "running"
    service._chat_active = 1
    with pytest.raises(FastAggError, match="чате"):
        service.start_bench()


def test_bench_command_is_registered():
    from fast_aggregation.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["bench", "--help"])
    assert caught.value.code == 0
