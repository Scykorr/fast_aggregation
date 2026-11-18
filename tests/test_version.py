from pathlib import Path

import pytest

from fast_aggregation import __version__
from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.models import Announce
from fast_aggregation.version import (
    android_version_code,
    file_version,
    full_version,
    product_version,
    release_version,
    versions_differ,
    windows_quad,
)


ROOT = Path(__file__).resolve().parents[1]


def test_version_sources_agree():
    init = (ROOT / "src" / "fast_aggregation" / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{__version__}"' in init
    assert __version__ == "0.4.0"
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = {attr = "fast_aggregation.__version__"}' in project
    assert 'version = "0.1.0"' not in project
    iss = (ROOT / "packaging" / "fastagg.iss").read_text(encoding="utf-8")
    assert f'#define MyAppVersion "{__version__}"' in iss
    assert f'#define MyAppVersionFull "{__version__}"' in iss
    assert f'#define SetupBaseName "fastagg-setup-{__version__}"' in iss
    cuda = (ROOT / "packaging" / "fastagg-cuda.iss").read_text(encoding="utf-8")
    assert f'#define SetupBaseName "fastagg-setup-cuda-{__version__}"' in cuda
    gradle = (ROOT / "android" / "app" / "build.gradle.kts").read_text(encoding="utf-8")
    assert "src/fast_aggregation/__init__.py" in gradle
    assert "buildConfig = true" in gradle
    assert "100_000_000" in gradle and "1_000_000" in gradle and "10_000" in gradle
    assert 'versionName = "0.1.0"' not in gradle
    kotlin = (ROOT / "android/app/src/main/java/io/fastagg/node/NodeService.kt").read_text(encoding="utf-8")
    assert "BuildConfig.VERSION_NAME" in kotlin
    assert '"version":"0.1.0"' not in kotlin
    assert product_version() == __version__
    assert android_version_code("0.2.0", 0) == 2_000_000
    assert android_version_code("0.2.0", 14) == 2_000_014
    assert android_version_code("1.2.3", 9) == 102_030_009


def test_build_metadata_and_filename(monkeypatch):
    monkeypatch.delenv("FASTAGG_BUILD", raising=False)
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    assert full_version() == "0.4.0"
    assert file_version() == "0.4.0"
    assert windows_quad() == "0.4.0.0"
    monkeypatch.setenv("GITHUB_RUN_NUMBER", "14")
    assert full_version() == "0.4.0+14"
    assert file_version() == "0.4.0-b14"
    assert windows_quad() == "0.4.0.14"
    assert release_version("0.2.0+14") == "0.2.0"
    assert release_version("0.2.0-b14") == "0.2.0"
    assert versions_differ("0.2.0+99", "0.2.0+1") is False
    assert versions_differ("0.1.0", "0.2.0") is True
    assert versions_differ("", "0.2.0") is True


def test_cli_version_flag(capsys):
    from fast_aggregation.cli import main

    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == f"fastagg {full_version()}"


def test_registry_warns_when_the_node_release_differs():
    registry = NodeRegistry(ttl=30)
    registry.upsert_announce(
        Announce(
            node_id="old",
            name="old",
            advertise_host="10.0.0.2",
            http_port=47000,
            rpc_port=50052,
            os_name="Windows",
            arch="x64",
            version="0.1.0",
        )
    )
    registry.upsert_details("same", name="same", host="10.0.0.3", http_port=47000, version="0.4.0+3")
    rows = {row["node_id"]: row for row in registry.public_nodes()}
    assert rows["old"]["version_mismatch"] is True
    assert "0.1.0" in rows["old"]["version_detail"]
    assert rows["same"]["version_mismatch"] is False


def test_api_version_keeps_the_ollama_field(monkeypatch):
    from fastapi.testclient import TestClient

    from fast_aggregation.api.compat import create_compat_app

    monkeypatch.setenv("FASTAGG_BUILD", "7")

    class Bridge:
        def health(self):
            return {"ok": True, "roles": {}}

    with TestClient(create_compat_app(Bridge())) as client:
        body = client.get("/api/version").json()
        assert body["version"] == "0.4.0"
        assert body["fastagg_version"] == "0.4.0+7"
        health = client.get("/fastagg/health").json()
        assert health["version"] == "0.4.0"
        assert health["fastagg_version"] == "0.4.0+7"
        assert client.get("/api/health").json()["fastagg_version"] == "0.4.0+7"
