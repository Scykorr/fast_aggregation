from pathlib import Path

import pytest

from fast_aggregation.errors import FastAggError
from fast_aggregation.ragstack import (
    compose_shell,
    disable_cluster,
    enable_cluster,
    merge_env,
    windows_to_wsl,
)


def test_wsl_path_and_single_profile():
    assert windows_to_wsl(r"D:\RAG 5060") == "/mnt/d/RAG 5060"
    core = compose_shell("/mnt/d/RAG 5060", "core")
    local = compose_shell("/mnt/d/RAG 5060", "ollama")
    assert "--profile core" in core and "--profile ollama" not in core
    assert "--profile ollama" in local and "--profile core" not in local
    with pytest.raises(FastAggError):
        compose_shell("/mnt/d/RAG 5060", "core ollama")


def test_env_merge_backup_and_restore(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgres\nOLLAMA_HOST=http://ollama:11434\nOLLAMA_MODEL=qwen3:8b\n", encoding="utf-8")
    backup = enable_cluster(
        env,
        {
            "OLLAMA_HOST": "http://host.docker.internal:11435",
            "OLLAMA_MODEL": "qwen3:8b",
            "EMBEDDING_URL": "http://host.docker.internal:11435/v1/embeddings",
            "RERANK_URL": "http://host.docker.internal:11435/v1/rerank",
            "FASTAGG_TOKEN": "secret",
        },
    )
    assert backup.is_file()
    text = env.read_text(encoding="utf-8")
    assert "DATABASE_URL=postgres" in text
    assert "OLLAMA_HOST=http://host.docker.internal:11435" in text
    assert "FASTAGG_TOKEN=secret" in text
    disable_cluster(env)
    restored = env.read_text(encoding="utf-8")
    assert "OLLAMA_HOST=http://ollama:11434" in restored
    assert "FASTAGG_TOKEN" not in restored
    assert "DATABASE_URL=postgres" in restored
    assert merge_env("A=1\n", {"B": "2"}) == "A=1\nB=2\n"
