import threading

from fastapi.testclient import TestClient

from fast_aggregation.api.compat import clamp_ctx, create_compat_app, strip_think
from fast_aggregation.api.queue import GenerationQueue
from fast_aggregation.cluster.roles import AuxCandidate, place_auxiliary, reserve_mb


class _Bridge:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def llm_ready(self) -> bool:
        return True

    def cluster_ctx(self) -> int:
        return 4096

    def primary_name(self) -> str:
        return "model-Q4_K_M.gguf"

    def tags(self) -> list[dict]:
        return [{"name": "fastagg", "size": 0, "digest": "fastagg", "modified_at": ""}]

    def ps(self) -> dict:
        return {"models": [{"name": "fastagg", "context_length": 4096, "effective_ctx": 4096}]}

    def show(self, name):
        return {"model": name or "fastagg", "context_length": 4096, "effective_ctx": 4096}

    def complete(self, prompt, *, system, temperature, num_predict, stop):
        self.started.set()
        self.release.wait(2)
        return ["<think>скрыто</think>Ответ"]

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def rerank(self, query, documents, top_n):
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    def embedding_identity(self):
        return {"gguf": "bge-m3-Q8_0.gguf", "embedding_model_id": "bge-m3-Q8_0", "quant": "Q8_0", "dim": 1024, "normalized": True}

    def health(self):
        return {"ok": True, "roles": {"llm": {"available": True}, "embedding": {"available": False}, "reranker": {"available": False}}}

    def pull_hf(self, name):
        return name


def test_clamp_and_think_strip():
    assert clamp_ctx(8192, 4096) == (4096, True)
    assert clamp_ctx(1024, 4096) == (1024, False)
    assert strip_think("<think>x</think>Ответ", False) == "Ответ"
    assert "<think>" in strip_think("<think>x</think>Ответ", True)


def test_generate_non_stream_and_queue_full():
    bridge = _Bridge()
    gate = GenerationQueue(limit=1, max_waiting=1)
    app = create_compat_app(bridge, queue=gate)
    with TestClient(app) as client:
        holder = {}

        def first():
            holder["response"] = client.post(
                "/api/generate",
                json={"model": "fastagg", "prompt": "hi", "stream": False, "think": False, "options": {"num_ctx": 8192, "num_predict": 32}},
            )

        thread = threading.Thread(target=first)
        thread.start()
        assert bridge.started.wait(2)
        blocked = {}

        def second():
            blocked["response"] = client.post("/api/generate", json={"prompt": "wait", "stream": False})

        waiter = threading.Thread(target=second)
        waiter.start()
        import time

        deadline = time.time() + 2
        while gate.waiting < 1 and time.time() < deadline:
            time.sleep(0.01)
        overflow = client.post("/api/generate", json={"prompt": "overflow", "stream": False})
        assert overflow.status_code == 429
        assert overflow.headers.get("retry-after") == "2"
        bridge.release.set()
        thread.join(timeout=3)
        waiter.join(timeout=3)
        assert holder["response"].status_code == 200
        body = holder["response"].json()
        assert body["response"] == "Ответ"
        assert body["effective_ctx"] == 4096
        assert body["clamped"] is True
        kept = client.post("/api/generate", json={"prompt": "hi", "stream": True})
        assert kept.status_code == 200
        assert kept.headers["content-type"].startswith("application/x-ndjson")
        assert "<think>" in kept.text
        assert "effective_ctx" in kept.text
        too_many = client.post("/v1/embeddings", json={"input": ["x"] * 33})
        assert too_many.status_code == 400


def test_embeddings_shape_and_missing_role():
    bridge = _Bridge()

    def fail(texts):
        from fast_aggregation.api.compat import CompatError

        raise CompatError(503, "Роль embedding не размещена")

    bridge.embed = fail
    app = create_compat_app(bridge)
    with TestClient(app) as client:
        missing = client.post("/v1/embeddings", json={"input": "текст"})
        assert missing.status_code == 503
    bridge.embed = lambda texts: [[0.0] * 1024 for _ in texts]
    app = create_compat_app(bridge)
    with TestClient(app) as client:
        found = client.post("/v1/embeddings", json={"input": ["текст"]})
        assert found.status_code == 200
        payload = found.json()
        assert payload["embedding_model_id"] == "bge-m3-Q8_0"
        assert payload["dim"] == 1024
        assert payload["data"][0]["object"] == "embedding"
        rerank = client.post("/v1/rerank", json={"query": "q", "documents": ["a"], "top_n": 1})
        assert rerank.json()["results"][0]["relevance_score"] == 0.9


def test_aux_placement_avoids_the_main_split_and_reserves_memory():
    heavy = AuxCandidate("gpu", 20000, "cuda", split_weight=8000)
    side = AuxCandidate("side", 4000, "vulkan", split_weight=0)
    picked = place_auxiliary([heavy, side], 1400)
    assert picked is not None and picked.node_id == "side"
    tight = place_auxiliary([heavy, AuxCandidate("tiny", 100, "cpu", 0)], 1400)
    assert tight is not None and tight.node_id == "gpu"
    assert reserve_mb({"embedding": {"node_id": "side", "need_mb": 1400, "status": "up"}})["side"] == 1400


def test_role_failover_moves_then_stops_after_three_tries(tmp_path):
    from fast_aggregation.cluster.registry import NodeRegistry
    from fast_aggregation.coordinator.service import CoordinatorService
    from fast_aggregation.errors import FastAggError
    from tests.helpers import machine

    import time

    registry = NodeRegistry(ttl=12)
    now = time.time()
    registry.upsert_details(
        "dead",
        name="dead",
        host="10.0.0.2",
        http_port=47000,
        resources=machine(ram=8000, gpus=[]),
        seen=True,
        now=now,
    )
    registry.upsert_details(
        "live",
        name="live",
        host="10.0.0.3",
        http_port=47000,
        resources=machine(ram=16000, gpus=[]),
        seen=True,
        now=now,
    )
    service = CoordinatorService(registry=registry, data_dir=tmp_path, node=None, background=False)
    calls: list[str] = []

    def starter(host, port, payload):
        del port, payload
        calls.append(host)
        if host == "10.0.0.3":
            return
        raise FastAggError("порт не открылся")

    service.aux_starter = starter
    assigned = service.assign_role("embedding", "dead")
    assigned["model_path"] = "D:/models/bge-m3-Q8_0.gguf"
    assigned["status"] = "up"
    registry.upsert_details("dead", seen=True, now=now - 30)
    service.reconcile_aux()
    assert calls == ["10.0.0.3"]
    assert service._roles["embedding"]["node_id"] == "live"
    assert service._roles["embedding"]["status"] == "up"
    assert service.integration_view()["embedding"]["replacing"] is False

    import time

    moment = time.time()
    registry.upsert_details(
        "spare",
        name="spare",
        host="10.0.0.4",
        http_port=47000,
        resources=machine(ram=4000, gpus=[]),
        seen=True,
        now=moment,
    )
    registry.upsert_details("live", seen=True, now=moment - 30)
    registry.upsert_details("dead", seen=True, now=moment - 30)

    def fail(host, port, payload):
        del host, port, payload
        raise FastAggError("порт не открылся")

    service.aux_starter = fail
    for _ in range(3):
        service.reconcile_aux()
    view = service.integration_view()["embedding"]
    assert view["available"] is False
    assert view["status"] == "down"
    assert view["replacing"] is False
