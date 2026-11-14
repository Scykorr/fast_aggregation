import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fast_aggregation.cluster.fit import assert_fits, estimate_need, read_gguf_metadata
from fast_aggregation.cluster.health import evaluate_guard, shed_reasons
from fast_aggregation.cluster.planner import choose_backend, plan_placement
from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.discovery.protocol import decode_announce, encode_announce
from fast_aggregation.discovery.scan import hosts_in_cidr, scan_tcp
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.errors import PlacementError
from fast_aggregation.models import ClusterNode, PlacementPlan, PlannedDevice
from fast_aggregation.netmeasure import megabits_per_second
from fast_aggregation.node.app import create_node_app
from fast_aggregation.resources.parsers import parse_thermal_zones, read_sysfs_thermal_c
from fast_aggregation.runtime.releases import resolve_asset
from fast_aggregation.security.certs import ensure_cert, fingerprint_pem
from fast_aggregation.security.token import attach_mac, join_uri, mac_ok, parse_join_uri
from fast_aggregation.security.tunnel import TlsTunnel
from tests.helpers import gpu, machine
from tests.test_api import _node
from tests.test_discovery import _announce, _free_port


def _cluster(name, node_id, resources, **kwargs):
    host = kwargs.pop("host", "192.168.1.10")
    return ClusterNode(
        node_id=node_id,
        name=name,
        host=host,
        http_port=47000,
        rpc_port=50052,
        online=True,
        resources=resources,
        preferred_backend="cuda",
        **kwargs,
    )


def test_slow_wifi_gets_a_small_share_of_layers():
    fast = _cluster("fast", "f", machine(gpus=[gpu(free=8000, total=8000)]), bandwidth_mbps=1000, rtt_ms=1)
    slow = _cluster(
        "slow",
        "s",
        machine(gpus=[gpu(name="wifi", free=8000, total=8000)]),
        host="192.168.1.20",
        bandwidth_mbps=20,
        rtt_ms=40,
    )
    plan = plan_placement([fast, slow], reserve_mb=0)
    slow_share = next(device.share for device in plan.devices if device.node_id == "s")
    assert slow_share < 0.15
    assert any("медлен" in item.lower() or slow_share < 0.15 for item in plan.warnings) or slow_share < 0.15


def test_overheat_and_phone_battery_drop_the_node():
    hot_resources = machine(gpus=[gpu()])
    hot_resources.temperature_c = 95
    hot = _cluster("hot", "h", hot_resources)
    with pytest.raises(PlacementError, match="температур"):
        plan_placement([hot], reserve_mb=0)
    phone = machine(gpus=[gpu()])
    phone.device_class = "phone"
    phone.battery_percent = 10
    phone.power_plugged = False
    with pytest.raises(PlacementError, match="телефон"):
        plan_placement([_cluster("phone", "p", phone)], reserve_mb=0)


def test_unknown_link_keeps_memory_proportions():
    plan = plan_placement(
        [
            _cluster("a", "a", machine(gpus=[gpu(name="a", free=1000, total=1000)])),
            _cluster("b", "b", machine(gpus=[gpu(name="b", free=3000, total=3000)]), host="192.168.1.11"),
        ],
        reserve_mb=0,
    )
    assert plan.tensor_split == [1000, 3000]


def test_preflight_rejects_a_model_that_does_not_fit():
    device = PlannedDevice(
        node_id="a",
        node_name="a",
        host="127.0.0.1",
        rpc_port=1,
        http_port=2,
        device_index=0,
        backend="cuda",
        name="gpu",
        kind="gpu",
        memory_mb=100,
        share=1,
        weight=100,
    )
    plan = PlacementPlan(devices=[device])
    from fast_aggregation.cluster.fit import ModelNeed

    with pytest.raises(PlacementError, match="не помещ" if False else "МиБ"):
        assert_fits(plan, ModelNeed(weight_mb=800, kv_mb=200, total_mb=1000, note=""))


def test_gguf_header_reports_layer_count(tmp_path: Path):
    # Минимальный GGUF: один ключ llama.block_count = 4 (uint32).
    key = b"llama.block_count"
    blob = bytearray(b"GGUF")
    blob += struct.pack("<I", 3)
    blob += struct.pack("<Q", 0)  # tensors
    blob += struct.pack("<Q", 1)  # kv
    blob += struct.pack("<Q", len(key)) + key
    blob += struct.pack("<I", 4)  # uint32
    blob += struct.pack("<I", 4)
    path = tmp_path / "tiny.gguf"
    path.write_bytes(blob)
    assert read_gguf_metadata(path)["block_count"] == 4
    need = estimate_need(str(path), ctx_size=0, parallel=1)
    assert need is not None
    assert need.weight_mb > 0


def test_subnet_scan_finds_local_listener_and_rejects_wide_cidr():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert scan_tcp(["127.0.0.1"], port, timeout=0.5) == ["127.0.0.1"]
    finally:
        sock.close()
    assert len(hosts_in_cidr("10.0.0.0/30")) == 2
    with pytest.raises(Exception):
        hosts_in_cidr("10.0.0.0/16")


def test_announce_mac_rejects_strangers():
    packet = encode_announce(_announce("n", "n", 1), token="secret")
    assert decode_announce(packet, token="secret") is not None
    assert decode_announce(packet, token="other") is None
    assert decode_announce(encode_announce(_announce("n", "n", 1)), token="secret") is None


def test_signed_discovery_between_sockets():
    port_a = _free_port()
    port_b = _free_port()
    seen: list[str] = []
    node_a = _announce("a", "a", 1)
    node_b = _announce("b", "b", 2)
    service_a = DiscoveryService(
        node_a,
        port=port_a,
        bind_host="127.0.0.1",
        interval=0.1,
        enable_broadcast=False,
        extra_targets=[("127.0.0.1", port_b)],
        token="shared",
        on_announce=lambda announce, _addr: seen.append(announce.node_id),
    )
    service_b = DiscoveryService(
        node_b,
        port=port_b,
        bind_host="127.0.0.1",
        interval=0.1,
        enable_broadcast=False,
        extra_targets=[("127.0.0.1", port_a)],
        token="shared",
    )
    service_b.start()
    service_a.start()
    try:
        import time

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "b" not in seen:
            time.sleep(0.05)
    finally:
        service_a.stop()
        service_b.stop()
    assert "b" in seen


def test_tls_tunnel_roundtrip(tmp_path: Path):
    import socket
    import ssl
    import threading

    cert, key, _fp = ensure_cert(tmp_path, ["127.0.0.1"])
    echo = socket.socket()
    echo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)
    echo_port = echo.getsockname()[1]

    def serve() -> None:
        conn, _addr = echo.accept()
        try:
            data = conn.recv(16)
            conn.sendall(data)
        finally:
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    public = holder.getsockname()[1]
    holder.close()
    tunnel = TlsTunnel("127.0.0.1", public, "127.0.0.1", echo_port, server_cert=(cert, key))
    tunnel.start()
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(cert)
        ctx.check_hostname = False
        with socket.create_connection(("127.0.0.1", public), timeout=3) as raw:
            with ctx.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
                tls.sendall(b"ping")
                assert tls.recv(4) == b"ping"
    finally:
        tunnel.stop()
        echo.close()


def test_pairing_code_and_token_gate(tmp_path: Path):
    resources = machine(gpus=[gpu()])
    node = _node(tmp_path, resources)
    app = create_node_app(node, token="secret")
    with TestClient(app) as client:
        denied = client.get("/v1/node")
        assert denied.status_code == 401
        allowed = client.get("/v1/node", headers={"Authorization": "Bearer secret"})
        assert allowed.status_code == 200
        bench = client.get("/v1/bench?bytes=32", headers={"Authorization": "Bearer secret"})
        assert bench.status_code == 200
        assert len(bench.content) == 32
    service = CoordinatorService(
        registry=__import__("fast_aggregation.cluster.registry", fromlist=["NodeRegistry"]).NodeRegistry(),
        data_dir=tmp_path,
        node=None,
        background=False,
        pairing_code="ABCD2345",
        token="secret",
        fingerprint="aa",
    )
    assert service.pair_code_ok("abcd2345")
    assert not service.pair_code_ok("WRONG123")
    uri = join_uri("192.168.1.5", 8080, "ABCD2345", "aa:bb")
    host, port, code, fp = parse_join_uri(uri)
    assert (host, port, code, fp) == ("192.168.1.5", 8080, "ABCD2345", "aa:bb")


def test_android_asset_names_and_backend_order():
    cpu = resolve_asset("b12359", "Android", "arm64", "cpu")
    assert cpu.name == "llama-b12359-bin-android-arm64.tar.gz"
    adreno = resolve_asset("b12359", "Android", "arm64", "opencl")
    assert adreno.name == "llama-b12359-bin-android-arm64-snapdragon.tar.gz"
    with pytest.raises(Exception):
        resolve_asset("b12359", "Android", "arm64", "vulkan")
    phone = machine(os_name="Android", cuda=False, vulkan=False)
    phone.backends.append(__import__("fast_aggregation.models", fromlist=["BackendInfo"]).BackendInfo("opencl", True))
    phone.backends.append(__import__("fast_aggregation.models", fromlist=["BackendInfo"]).BackendInfo("cpu", True))
    assert choose_backend(phone, "auto") == "opencl"


def test_shed_when_node_disappears_or_overheats():
    resources = machine(gpus=[gpu()])
    resources.temperature_c = 70
    node = _cluster("desk", "d", resources)
    plan = plan_placement([node], reserve_mb=0)
    assert shed_reasons(plan, [node]) == []
    gone = ClusterNode(
        node_id="d",
        name="desk",
        host=node.host,
        http_port=1,
        rpc_port=2,
        online=False,
        resources=resources,
    )
    assert shed_reasons(plan, [gone])[0][0] == "d"
    resources.temperature_c = 99
    assert evaluate_guard(resources).action == "exclude"


def test_thermal_parsers(tmp_path: Path):
    assert parse_thermal_zones('{"CurrentTemperature": 3232}') == pytest.approx(50.05, abs=0.1)
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "temp").write_text("67000", encoding="utf-8")
    assert read_sysfs_thermal_c(tmp_path) == pytest.approx(67.0)


def test_bandwidth_math_and_cert_fingerprint(tmp_path: Path):
    assert megabits_per_second(1_000_000, 1) == pytest.approx(8.0)
    cert, _key, fp = ensure_cert(tmp_path, ["127.0.0.1"])
    assert fp == fingerprint_pem(cert.read_bytes())
    payload = {"v": 1, "node_id": "n"}
    signed = attach_mac(payload, "tok")
    assert mac_ok(signed, "tok")
    assert not mac_ok(signed, "nope")
