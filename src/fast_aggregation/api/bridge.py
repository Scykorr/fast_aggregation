"""Мост совместимого порта к кластеру и ролям."""

from __future__ import annotations

from pathlib import Path

import httpx

from fast_aggregation.api.compat import CompatError
from fast_aggregation.cluster.roles import SUGGESTED_ROLES, place_auxiliary
from fast_aggregation.cluster.roles import AuxCandidate


class ServiceBridge:
    def __init__(self, service) -> None:
        self.service = service

    def llm_ready(self) -> bool:
        return self.service.phase == "running"

    def cluster_ctx(self) -> int:
        return int((self.service._launch or {}).get("ctx_size") or 4096)

    def primary_name(self) -> str:
        path = (self.service._launch or {}).get("model_path") or ""
        return Path(path).name or "fastagg"

    def tags(self) -> list[dict]:
        names: list[str] = []
        for alias in (self.primary_name(), "fastagg", "qwen3:8b"):
            if alias and alias not in names:
                names.append(alias)
        return [{"name": name, "size": 0, "digest": "fastagg", "modified_at": ""} for name in names]

    def ps(self) -> dict:
        ctx = self.cluster_ctx()
        if not self.llm_ready():
            return {"models": []}
        return {"models": [{"name": self.primary_name(), "context_length": ctx, "effective_ctx": ctx}]}

    def show(self, name: str | None) -> dict:
        ctx = self.cluster_ctx()
        return {
            "model": name or self.primary_name(),
            "parameters": f"num_ctx {ctx}",
            "context_length": ctx,
            "effective_ctx": ctx,
            "details": {"family": "fastagg", "parameter_count": ""},
        }

    def complete(self, prompt: str, *, system: str, temperature, num_predict: int, stop) -> list[str]:
        text = f"{system}\n{prompt}".strip() if system else prompt
        payload: dict = {
            "prompt": text,
            "n_predict": num_predict,
            "temperature": 0 if temperature is None else temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop if isinstance(stop, list) else [stop]
        try:
            response = httpx.post(
                f"http://127.0.0.1:{self.service.engine_port}/completion",
                json=payload,
                timeout=300,
            )
        except httpx.HTTPError as exc:
            raise CompatError(503, f"llama-server недоступен: {exc}") from exc
        if response.status_code >= 400:
            raise CompatError(503, f"llama-server ответил {response.status_code}")
        return [str(response.json().get("content") or "")]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _proxy_role(self.service, "embedding", {"input": texts}, "embeddings")

    def rerank(self, query: str, documents: list[str], top_n) -> dict:
        payload = {"query": query, "documents": documents}
        if top_n:
            payload["top_n"] = int(top_n)
        return _proxy_role(self.service, "rerank", payload, "rerank")

    def embedding_identity(self) -> dict:
        spec = (getattr(self.service, "_roles", {}) or {}).get("embedding") or SUGGESTED_ROLES["embedding"]
        return {
            "gguf": spec.get("gguf"),
            "embedding_model_id": spec.get("embedding_model_id"),
            "quant": spec.get("quant"),
            "dim": spec.get("dim"),
            "normalized": spec.get("normalized", True),
        }

    def health(self) -> dict:
        view = self.service.integration_view()
        return {
            "ok": bool(view["llm"]["available"]),
            "roles": {
                "llm": view["llm"],
                "embedding": view["embedding"],
                "reranker": view["reranker"],
            },
            "failover_budget_sec": view["failover_budget_sec"],
            "recommendation": "При 503 RAG может временно вернуться к локальному Ollama (профиль compose ollama).",
        }

    def pull_hf(self, name: str) -> str:
        return f"Ссылка {name} не качается этим портом автоматически. Выполните fastagg pull на координаторе."

    def assign_role(self, role: str, node_id: str, model_path: str = "") -> dict:
        return self.service.assign_role(role, node_id or None, model_path)


def _proxy_role(service, role: str, payload: dict, kind: str):
    spec = (getattr(service, "_roles", {}) or {}).get(role)
    if not spec or spec.get("status") != "up":
        raise CompatError(503, f"Роль {role} не размещена или узел недоступен")
    url = f"http://{spec['host']}:{spec['port']}/v1/{'embeddings' if kind == 'embeddings' else 'rerank'}"
    try:
        response = httpx.post(url, json=payload, timeout=30)
    except httpx.HTTPError as exc:
        raise CompatError(503, f"Узел роли {role} не ответил: {exc}") from exc
    if response.status_code >= 400:
        raise CompatError(503, f"Узел роли {role} ответил {response.status_code}")
    data = response.json()
    if kind == "embeddings":
        return [item["embedding"] for item in data.get("data") or []]
    return data


def candidates_for(service) -> list[AuxCandidate]:
    weights: dict[str, int] = {}
    if service.plan is not None:
        for device in service.plan.devices:
            weights[device.node_id] = weights.get(device.node_id, 0) + int(device.weight or device.memory_mb)
    found = []
    for node in service.registry.cluster_nodes():
        resources = node.resources
        free = 0
        backend = node.preferred_backend or "cpu"
        if resources is not None:
            free = resources.ram_available_mb
            for gpu in resources.gpus:
                if (gpu.vram_free_mb or 0) > free:
                    free = gpu.vram_free_mb or 0
                    backend = gpu.vendor or backend
        found.append(AuxCandidate(node.node_id, int(free or 0), backend, weights.get(node.node_id, 0)))
    return found
