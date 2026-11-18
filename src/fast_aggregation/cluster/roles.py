"""Вспомогательные роли: эмбеддинги и реранкер на отдельном узле, не в RPC-разрезе.

Имена файлов проверены в репозиториях Hugging Face gpustack/bge-m3-GGUF
и gpustack/bge-reranker-v2-m3-GGUF.
"""

from __future__ import annotations

from dataclasses import dataclass

AUX_EMBED_PORT = 18081
AUX_RERANK_PORT = 18082
FAILOVER_TRIES = 3

# Оценка VRAM для Q8_0, чтобы планировщик вычел её из бюджета основного разреза.
SUGGESTED_ROLES: dict[str, dict] = {
    "embedding": {
        "role": "embedding",
        "embedding_model_id": "bge-m3-Q8_0",
        "gguf": "bge-m3-Q8_0.gguf",
        "quant": "Q8_0",
        "dim": 1024,
        "pooling": "mean",
        "normalized": True,
        "repo": "gpustack/bge-m3-GGUF",
        "url": "https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf",
        "need_mb": 1400,
        "port": AUX_EMBED_PORT,
        "argv": ["--embedding", "--pooling", "mean"],
        "max_batch": 32,
        "max_input_tokens": 512,
    },
    "rerank": {
        "role": "rerank",
        "embedding_model_id": "bge-reranker-v2-m3-Q8_0",
        "gguf": "bge-reranker-v2-m3-Q8_0.gguf",
        "quant": "Q8_0",
        "dim": None,
        "pooling": None,
        "normalized": False,
        "repo": "gpustack/bge-reranker-v2-m3-GGUF",
        "url": "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf",
        "need_mb": 1200,
        "port": AUX_RERANK_PORT,
        "argv": ["--reranking"],
        "max_batch": 32,
        "max_input_tokens": 512,
    },
}


@dataclass(frozen=True)
class AuxCandidate:
    node_id: str
    free_mb: int
    backend: str
    split_weight: int = 0


def place_auxiliary(candidates: list[AuxCandidate], need_mb: int) -> AuxCandidate | None:
    """Узел с запасом памяти, по возможности не тот, на ком лежит основной разрез.

    Если все подходящие узлы участвуют в разрезе, берётся узел с наименьшей долей.
    """

    fitting = [item for item in candidates if item.free_mb >= need_mb]
    if not fitting:
        return None
    outside = [item for item in fitting if item.split_weight <= 0]
    pool = outside or fitting
    return max(pool, key=lambda item: (item.split_weight <= 0, item.free_mb, item.backend != "cpu"))


def reserve_mb(assignments: dict[str, dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for spec in assignments.values():
        node_id = str(spec.get("node_id") or "")
        if not node_id or spec.get("status") == "down":
            continue
        totals[node_id] = totals.get(node_id, 0) + int(spec.get("need_mb") or 0)
    return totals
