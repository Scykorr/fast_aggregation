"""Правки .env и запуск compose RAG-5060. Репозиторий RAG не трогаем git-командами."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fast_aggregation.errors import FastAggError

ENV_KEYS = ("OLLAMA_HOST", "OLLAMA_MODEL", "EMBEDDING_URL", "RERANK_URL", "FASTAGG_TOKEN")
PROFILES = ("core", "ollama")


def windows_to_wsl(path: str) -> str:
    text = path.strip().replace("/", "\\")
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        rest = text[2:].lstrip("\\").replace("\\", "/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    return path.replace("\\", "/")


def parse_env(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        found[key.strip()] = value
    return found


def merge_env(text: str, updates: dict[str, str]) -> str:
    """Меняет только перечисленные ключи. Остальные строки остаются."""

    seen: set[str] = set()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    body = "\n".join(lines)
    if text.endswith("\n") or not text:
        body += "\n"
    return body


def _sidecar(env_path: Path) -> Path:
    return env_path.with_name(".fastagg-env-keys.json")


def enable_cluster(env_path: Path, updates: dict[str, str]) -> Path:
    if not env_path.is_file():
        raise FastAggError(f"В папке RAG нет файла .env: {env_path}")
    unknown = set(updates) - set(ENV_KEYS)
    if unknown:
        raise FastAggError("Можно менять только ключи подключения к fastagg.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = env_path.with_name(f".env.fastagg-{stamp}.bak")
    shutil.copy2(env_path, backup)
    previous = {key: parse_env(env_path.read_text(encoding="utf-8")).get(key) for key in ENV_KEYS}
    _sidecar(env_path).write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    merged = merge_env(env_path.read_text(encoding="utf-8"), updates)
    env_path.write_text(merged, encoding="utf-8")
    return backup


def disable_cluster(env_path: Path) -> None:
    side = _sidecar(env_path)
    if not side.is_file():
        raise FastAggError("Нет сохранённых прежних значений .env. Сначала включите кластер.")
    previous = json.loads(side.read_text(encoding="utf-8"))
    text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    parsed = parse_env(text)
    for key in ENV_KEYS:
        parsed.pop(key, None)
    restored = {key: value for key, value in previous.items() if value is not None}
    # Собрать файл заново нельзя: пропадут комментарии. merge только возвращённые ключи,
    # а отсутствовавшие раньше ключи удалить.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in ENV_KEYS and key not in restored:
                continue
        lines.append(line)
    merged = merge_env("\n".join(lines) + ("\n" if text.endswith("\n") or not lines else ""), restored)
    env_path.write_text(merged, encoding="utf-8")


def ensure_compose(project: Path, kit: Path) -> str:
    if not kit.is_file():
        raise FastAggError(f"В комплекте fastagg нет {kit.name}.")
    target = project / kit.name
    payload = kit.read_text(encoding="utf-8")
    if target.is_file() and target.read_text(encoding="utf-8") == payload:
        return "файл compose уже совпадает с комплектом"
    target.write_text(payload, encoding="utf-8")
    return "docker-compose.fastagg.yml скопирован в папку RAG"


def compose_shell(wsl_path: str, profile: str, *, down: bool = False) -> str:
    if profile not in PROFILES:
        raise FastAggError("Профиль compose должен быть core или ollama, и только один.")
    action = "down" if down else "up -d"
    quoted = wsl_path.replace('"', '\\"')
    return (
        f'cd "{quoted}" && docker compose -f docker-compose.yml '
        f"-f docker-compose.fastagg.yml --profile {profile} {action}"
    )


def cluster_env(api_bind: str, api_port: int, token: str) -> dict[str, str]:
    host = api_bind if api_bind not in {"", "0.0.0.0"} else "host.docker.internal"
    if host in {"127.0.0.1", "localhost"}:
        host = "host.docker.internal"
    base = f"http://{host}:{api_port}"
    return {
        "OLLAMA_HOST": base,
        "OLLAMA_MODEL": "qwen3:8b",
        "EMBEDDING_URL": base + "/v1/embeddings",
        "RERANK_URL": base + "/v1/rerank",
        "FASTAGG_TOKEN": token,
    }
