"""Ollama- и OpenAI-совместимый порт для RAG на той же машине.

Поколение одно за раз. Эмбеддинги и реранкер уходят на узлы ролей,
а не в RPC-разрез основной модели.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from fast_aggregation.api.queue import GenerationQueue
from fast_aggregation.cluster.roles import SUGGESTED_ROLES
from fast_aggregation.version import full_version, product_version

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
RETRY_AFTER_SEC = "2"


class CompatError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def clamp_ctx(requested: int | None, cluster_ctx: int) -> tuple[int, bool]:
    """Эффективный контекст не может быть больше того, с которым поднят llama-server."""

    limit = max(1, int(cluster_ctx or 1))
    if not requested or int(requested) <= 0:
        return limit, False
    asked = int(requested)
    if asked > limit:
        return limit, True
    return asked, False


def strip_think(text: str, think: bool | None) -> str:
    """think=false вырезает блок qwen3. Пропущенный флаг текст не трогает."""

    if think is not False:
        return text
    cleaned = _THINK_BLOCK.sub("", text or "")
    return cleaned.replace("<think>", "").replace("</think>", "").strip()


def ollama_ndjson(pieces: list[str], model: str, *, effective_ctx: int, clamped: bool) -> bytes:
    lines = []
    for piece in pieces:
        lines.append(json.dumps({"model": model, "response": piece, "done": False}, ensure_ascii=False))
    final = {
        "model": model,
        "response": "",
        "done": True,
        "done_reason": "stop",
        "effective_ctx": effective_ctx,
        "clamped": clamped,
    }
    lines.append(json.dumps(final, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def create_compat_app(
    bridge: Any,
    *,
    token: str = "",
    queue: GenerationQueue | None = None,
) -> FastAPI:
    app = FastAPI(title="fastagg ollama compat")
    gate = queue or GenerationQueue()
    app.state.queue = gate

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if token and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "Нужен Bearer-токен fastagg"}, status_code=401)
        return await call_next(request)

    @app.post("/fastagg/roles")
    def assign_role(body: dict):
        from fastapi import HTTPException

        from fast_aggregation.errors import FastAggError

        try:
            return bridge.assign_role(
                str(body.get("role") or ""),
                str(body.get("node_id") or ""),
                str(body.get("model_path") or ""),
            )
        except FastAggError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/fastagg/health")
    @app.get("/api/health")
    def health() -> dict:
        body = bridge.health()
        body["queue"] = gate.snapshot()
        body["version"] = product_version()
        body["fastagg_version"] = full_version()
        return body

    @app.get("/api/version")
    def version() -> dict:
        return {"version": product_version(), "fastagg_version": full_version()}

    @app.get("/api/tags")
    def tags() -> dict:
        return {"models": bridge.tags()}

    @app.get("/api/ps")
    def ps() -> dict:
        return bridge.ps()

    @app.post("/api/show")
    def show(request_body: dict | None = None) -> dict:
        return bridge.show((request_body or {}).get("name"))

    @app.post("/api/pull")
    def pull(request_body: dict) -> JSONResponse:
        name = str((request_body or {}).get("name") or "")
        if name.startswith("http://") or name.startswith("https://") or name.count("/") >= 1 and " " not in name:
            try:
                detail = bridge.pull_hf(name)
            except CompatError as exc:
                return JSONResponse({"error": exc.detail}, status_code=exc.status)
            return JSONResponse({"status": "success", "detail": detail})
        return JSONResponse(
            {
                "error": "Это не реестр Ollama. Укажите ссылку Hugging Face на GGUF "
                "или скачайте файл на вкладке «Модели»."
            },
            status_code=400,
        )

    @app.post("/api/create")
    def create() -> JSONResponse:
        return JSONResponse(
            {"error": "Сборка Modelfile здесь не поддерживается. Положите GGUF в папку моделей и нажмите «Развернуть»."},
            status_code=400,
        )

    @app.post("/api/generate")
    def generate(body: dict):
        return _generate(body, gate, bridge, chat=False)

    @app.post("/api/chat")
    def chat(body: dict):
        messages = body.get("messages") or []
        prompt = _messages_to_prompt(messages)
        merged = dict(body)
        merged["prompt"] = prompt
        if body.get("system") and not any(item.get("role") == "system" for item in messages if isinstance(item, dict)):
            merged["system"] = body.get("system")
        return _generate(merged, gate, bridge, chat=True)

    @app.post("/api/embed")
    @app.post("/api/embeddings")
    def embed_ollama(body: dict):
        return _embed(body, bridge, ollama=True)

    @app.get("/v1/models")
    def openai_models() -> dict:
        data = []
        for item in bridge.tags():
            data.append({"id": item["name"], "object": "model", "owned_by": "fastagg"})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def openai_chat(body: dict):
        messages = body.get("messages") or []
        merged = {
            "model": body.get("model"),
            "prompt": _messages_to_prompt(messages),
            "stream": bool(body.get("stream")),
            "options": {
                "temperature": body.get("temperature"),
                "num_predict": body.get("max_tokens"),
                "stop": body.get("stop"),
                "num_ctx": (body.get("options") or {}).get("num_ctx"),
            },
        }
        return _generate(merged, gate, bridge, chat=True)

    @app.post("/v1/completions")
    def openai_completion(body: dict):
        merged = {
            "model": body.get("model"),
            "prompt": body.get("prompt") or "",
            "stream": bool(body.get("stream")),
            "options": {
                "temperature": body.get("temperature"),
                "num_predict": body.get("max_tokens"),
                "stop": body.get("stop"),
            },
        }
        return _generate(merged, gate, bridge, chat=False)

    @app.post("/v1/embeddings")
    def openai_embeddings(body: dict):
        return _embed(body, bridge, ollama=False)

    @app.post("/v1/rerank")
    def rerank(body: dict):
        try:
            return bridge.rerank(str(body.get("query") or ""), list(body.get("documents") or []), body.get("top_n"))
        except CompatError as exc:
            return JSONResponse({"error": exc.detail}, status_code=exc.status)

    return app


def _messages_to_prompt(messages: list) -> str:
    lines = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or "user"
        content = item.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _options(body: dict) -> dict:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    return options


def _generate(body: dict, gate: GenerationQueue, bridge: Any, *, chat: bool):
    if not bridge.llm_ready():
        return JSONResponse({"error": "Основная модель кластера не запущена"}, status_code=503)
    if not gate.acquire():
        return JSONResponse(
            {"error": "Очередь генерации заполнена"},
            status_code=429,
            headers={"Retry-After": RETRY_AFTER_SEC},
        )
    try:
        options = _options(body)
        requested = options.get("num_ctx")
        effective, clamped = clamp_ctx(requested, bridge.cluster_ctx())
        think = body.get("think")
        temperature = options.get("temperature", body.get("temperature", 0.8))
        num_predict = options.get("num_predict", body.get("num_predict", 256))
        stop = options.get("stop", body.get("stop"))
        model = str(body.get("model") or bridge.primary_name())
        prompt = str(body.get("prompt") or "")
        system = str(body.get("system") or "")
        pieces = list(
            bridge.complete(
                prompt,
                system=system,
                temperature=temperature,
                num_predict=int(num_predict or 256),
                stop=stop,
            )
        )
        text = strip_think("".join(pieces), think if isinstance(think, bool) else None)
        if body.get("stream"):
            payload = ollama_ndjson([text] if text else pieces, model, effective_ctx=effective, clamped=clamped)
            headers = {"X-Fastagg-Effective-Ctx": str(effective)}
            return StreamingResponse(iter([payload]), media_type="application/x-ndjson", headers=headers)
        content_key = "message" if chat else "response"
        body_out: dict[str, Any] = {
            "model": model,
            "done": True,
            "done_reason": "stop",
            "effective_ctx": effective,
            "clamped": clamped,
            "context_length": bridge.cluster_ctx(),
        }
        if chat:
            body_out["message"] = {"role": "assistant", "content": text}
        else:
            body_out[content_key] = text
        return JSONResponse(body_out, headers={"X-Fastagg-Effective-Ctx": str(effective)})
    except CompatError as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status)
    finally:
        gate.release()


def _embed(body: dict, bridge: Any, *, ollama: bool):
    raw = body.get("input", body.get("prompt"))
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list):
        texts = [str(item) for item in raw]
    else:
        texts = []
    spec = SUGGESTED_ROLES["embedding"]
    if len(texts) > int(spec["max_batch"]):
        return JSONResponse({"error": f"Пакет больше {spec['max_batch']}"}, status_code=400)
    try:
        vectors = bridge.embed(texts)
    except CompatError as exc:
        return JSONResponse({"error": exc.detail}, status_code=exc.status)
    identity = bridge.embedding_identity()
    if ollama:
        return {"model": identity["gguf"], "embedding_model_id": identity["embedding_model_id"], "embeddings": vectors, "dim": identity["dim"]}
    return {
        "object": "list",
        "model": identity["gguf"],
        "embedding_model_id": identity["embedding_model_id"],
        "quant": identity["quant"],
        "dim": identity["dim"],
        "normalized": identity["normalized"],
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
    }
