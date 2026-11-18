import hashlib
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fast_aggregation.cli import main
from fast_aggregation.coordinator.app import create_coordinator_app
from fast_aggregation.errors import FastAggError
from fast_aggregation.models_pull import (
    ModelPuller,
    RemoteFile,
    choices_from_files,
    ensure_disk_space,
    files_from_tree,
    parse_hf_url,
    save_token,
    sha_from_headers,
)
from tests.test_console import _service


BODY = b"gguf-model-bytes-0123456789abcdef"
DIGEST = hashlib.sha256(BODY).hexdigest()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True, timeout=5)


def test_parse_blob_resolve_and_repo_links():
    blob = parse_hf_url("https://huggingface.co/org/repo/blob/main/dir/model.Q4_K_M.gguf")
    assert (blob.repo, blob.revision, blob.filename) == ("org/repo", "main", "dir/model.Q4_K_M.gguf")
    resolve = parse_hf_url("https://huggingface.co/org/repo/resolve/v2/model.gguf?download=true")
    assert resolve.filename == "model.gguf"
    assert resolve.revision == "v2"
    repo = parse_hf_url("org/repo")
    assert repo.filename is None and repo.repo == "org/repo"
    with pytest.raises(FastAggError):
        parse_hf_url("https://example.com/org/repo/model.gguf")


def test_tree_lists_quantizations_and_groups_split_parts():
    part_a = hashlib.sha256(b"a").hexdigest()
    part_b = hashlib.sha256(b"b").hexdigest()
    files = files_from_tree(
        "org/repo",
        "main",
        [
            {"path": "README.md", "size": 10},
            {"path": "model-Q4_K_M.gguf", "size": 100, "lfs": {"oid": "sha256:" + DIGEST, "size": 100}},
            {"path": "model-Q8_0-00002-of-00002.gguf", "size": 40, "lfs": {"oid": "sha256:" + part_b, "size": 40}},
            {"path": "model-Q8_0-00001-of-00002.gguf", "size": 60, "lfs": {"oid": "sha256:" + part_a, "size": 60}},
        ],
    )
    choices = choices_from_files(files)
    labels = [choice.label for choice in choices]
    assert "model-Q4_K_M.gguf" in labels
    split = next(choice for choice in choices if "частей" in choice.label)
    assert split.size == 100
    assert [item.path for item in split.files][0].endswith("00001-of-00002.gguf")
    assert split.files[0].sha256 == part_a


def test_disk_space_refuses_when_the_model_does_not_fit(tmp_path: Path):
    with pytest.raises(FastAggError, match="диске"):
        ensure_disk_space(tmp_path, 5_000_000_000, free_bytes=1024)


def test_download_resumes_with_range_and_checks_sha256(tmp_path: Path):
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={"x-linked-size": str(len(BODY)), "x-linked-etag": DIGEST, "content-length": str(len(BODY))},
            )
        seen["range"] = request.headers.get("range")
        start = 0
        if seen["range"]:
            start = int(seen["range"].split("=", 1)[1].split("-", 1)[0])
        chunk = BODY[start:]
        headers = {"x-linked-size": str(len(BODY)), "x-linked-etag": DIGEST}
        if start:
            headers["content-range"] = f"bytes {start}-{len(BODY) - 1}/{len(BODY)}"
            return httpx.Response(206, content=chunk, headers=headers)
        return httpx.Response(200, content=chunk, headers=headers)

    partial = tmp_path / "model.gguf.partial"
    partial.write_bytes(BODY[:8])
    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    view = puller.inspect("https://huggingface.co/org/repo/resolve/main/model.gguf")
    assert view["choices"][0]["files"][0]["sha256"] == DIGEST
    puller.start([__import__("fast_aggregation.models_pull", fromlist=["RemoteFile"]).RemoteFile(**view["choices"][0]["files"][0])])
    puller._thread.join(timeout=5)
    final = puller.view()
    assert final["status"] == "done", final
    assert (tmp_path / "model.gguf").read_bytes() == BODY
    assert seen["range"] == "bytes=8-"
    puller.close()


def test_xet_etag_is_not_used_as_sha256():
    xet = "3fe8c46a" + "ab" * 28
    headers = httpx.Headers({"etag": xet, "x-linked-etag": DIGEST})
    assert sha_from_headers(headers) == DIGEST
    assert sha_from_headers(httpx.Headers({"etag": xet})) is None


def test_checksum_mismatch_moves_partial_aside(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=BODY, headers={"etag": "3fe8c46a" + "ab" * 28, "x-linked-size": str(len(BODY))})

    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    puller.start([RemoteFile("model.gguf", "https://huggingface.co/org/repo/resolve/main/model.gguf", len(BODY), "a" * 64)])
    puller._thread.join(timeout=5)
    view = puller.view()
    assert view["status"] == "error"
    assert (tmp_path / "model.gguf.bad").is_file()
    assert not (tmp_path / "model.gguf.partial").exists()
    puller.close()


def test_incomplete_body_is_retried(tmp_path: Path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.method == "GET" and calls["n"] == 1:
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body", request=request)
        return httpx.Response(200, content=BODY, headers={"x-linked-etag": DIGEST, "x-linked-size": str(len(BODY))})

    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    puller.start([RemoteFile("model.gguf", "https://huggingface.co/org/repo/resolve/main/model.gguf", len(BODY), DIGEST)])
    puller._thread.join(timeout=8)
    assert puller.view()["status"] == "done", puller.view()
    assert calls["n"] >= 2
    puller.close()


def test_bad_sha256_is_reported(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=BODY, headers={"x-linked-etag": "a" * 64, "x-linked-size": str(len(BODY))})

    from fast_aggregation.models_pull import RemoteFile

    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    puller.start([RemoteFile("model.gguf", "https://huggingface.co/org/repo/resolve/main/model.gguf", len(BODY), "a" * 64)])
    puller._thread.join(timeout=5)
    view = puller.view()
    assert view["status"] == "error"
    assert "sha256" in (view["error"] or "")
    puller.close()


def test_offline_inspect_says_there_is_no_internet(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    with pytest.raises(FastAggError, match="интернет"):
        puller.inspect("https://huggingface.co/org/repo")
    puller.close()


def test_pause_and_cancel(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            started.set()
            release.wait(2)
        return httpx.Response(200, content=BODY, headers={"x-linked-size": str(len(BODY)), "x-linked-etag": DIGEST})

    from fast_aggregation.models_pull import RemoteFile

    puller = ModelPuller(tmp_path, tmp_path, client=_client(handler))
    puller.start([RemoteFile("model.gguf", "https://huggingface.co/org/repo/resolve/main/model.gguf", len(BODY), DIGEST)])
    assert started.wait(2)
    assert puller.pause()["status"] == "paused"
    assert puller.cancel()["status"] == "cancelled"
    release.set()
    puller.close()


def test_dashboard_pull_then_lists_the_model(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "/tree/" in str(request.url):
            return httpx.Response(
                200,
                json=[{"path": "tiny.gguf", "size": len(BODY), "lfs": {"oid": "sha256:" + DIGEST, "size": len(BODY)}}],
            )
        return httpx.Response(200, content=BODY, headers={"x-linked-etag": DIGEST, "x-linked-size": str(len(BODY))})

    service = _service(tmp_path, __import__("tests.helpers", fromlist=["machine"]).machine())
    service.models_dir = tmp_path / "models"
    service._model_puller = ModelPuller(service.models_dir, tmp_path, client=_client(handler))
    app = create_coordinator_app(service)
    with TestClient(app) as client:
        saved = client.post("/api/models/hf/token", json={"token": "hf_test"}).json()
        assert saved["saved"] is True
        assert (tmp_path / "hf-token").read_text(encoding="utf-8") == "hf_test"
        inspected = client.post("/api/models/hf/inspect", json={"url": "org/repo"})
        assert inspected.status_code == 200
        files = inspected.json()["choices"][0]["files"]
        started = client.post("/api/models/pull", json={"files": files})
        assert started.status_code == 200
        service._model_puller._thread.join(timeout=5)
        assert service._model_puller.view()["status"] == "done"
        listed = client.get("/api/models").json()
        assert listed["models"][0]["name"] == "tiny.gguf"
        assert "Развернуть" in client.get("/").text


def test_pull_help_exists():
    with pytest.raises(SystemExit) as caught:
        main(["pull", "--help"])
    assert caught.value.code == 0


def test_token_roundtrip_does_not_keep_an_empty_value(tmp_path: Path):
    save_token(tmp_path, "hf_secret")
    assert (tmp_path / "hf-token").read_text(encoding="utf-8") == "hf_secret"
    save_token(tmp_path, "  ")
    assert not (tmp_path / "hf-token").exists()
