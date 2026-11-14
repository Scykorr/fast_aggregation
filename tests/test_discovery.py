import json
import socket
import time

from fast_aggregation.constants import DISCOVERY_MAGIC, PROTOCOL_VERSION
from fast_aggregation.discovery.protocol import decode_announce, encode_announce
from fast_aggregation.discovery.udp import DiscoveryService
from fast_aggregation.models import Announce


def _announce(node_id: str, name: str, http_port: int) -> Announce:
    return Announce(
        node_id=node_id,
        name=name,
        advertise_host="127.0.0.1",
        http_port=http_port,
        rpc_port=50052,
        os_name="Linux",
        arch="x64",
        preferred_backend="auto",
    )


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_announce_roundtrip_and_rejection():
    original = _announce("node-a", "alpha", 47000)
    restored = decode_announce(encode_announce(original))
    assert restored == original

    raw = json.dumps({"v": 99, "magic": DISCOVERY_MAGIC, "kind": "announce"}).encode()
    assert decode_announce(raw) is None
    assert decode_announce(b"not-json") is None
    broken = json.loads(encode_announce(original))
    broken["v"] = PROTOCOL_VERSION
    broken["http_port"] = 0
    assert decode_announce(json.dumps(broken).encode()) is None


def test_two_listeners_discover_each_other_over_unicast():
    port_a = _free_port()
    port_b = _free_port()
    seen_a: list[Announce] = []
    seen_b: list[Announce] = []
    node_a = _announce("node-a", "alpha", 47001)
    node_b = _announce("node-b", "beta", 47002)
    service_a = DiscoveryService(
        node_a,
        port=port_a,
        bind_host="127.0.0.1",
        interval=0.15,
        enable_broadcast=False,
        extra_targets=[("127.0.0.1", port_b)],
        on_announce=lambda announce, _addr: seen_a.append(announce),
    )
    service_b = DiscoveryService(
        node_b,
        port=port_b,
        bind_host="127.0.0.1",
        interval=0.15,
        enable_broadcast=False,
        extra_targets=[("127.0.0.1", port_a)],
        on_announce=lambda announce, _addr: seen_b.append(announce),
    )
    service_a.start()
    service_b.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (not seen_a or not seen_b):
            time.sleep(0.05)
    finally:
        service_a.stop()
        service_b.stop()
    assert [item.node_id for item in seen_a] 
    assert seen_a[0].node_id == "node-b"
    assert seen_b[0].node_id == "node-a"
    assert all(item.node_id != "node-a" for item in seen_a)


def test_malformed_datagram_does_not_kill_listener():
    port = _free_port()
    seen: list[Announce] = []
    service = DiscoveryService(
        _announce("self", "self", 1),
        port=port,
        bind_host="127.0.0.1",
        interval=10,
        enable_broadcast=False,
        on_announce=lambda announce, _addr: seen.append(announce),
    )
    service.start()
    try:
        time.sleep(0.2)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(b"garbage", ("127.0.0.1", port))
        foreign = _announce("foreign", "foreign", 9)
        packet = json.loads(encode_announce(foreign))
        packet["v"] = 2
        sock.sendto(json.dumps(packet).encode(), ("127.0.0.1", port))
        sock.sendto(encode_announce(foreign), ("127.0.0.1", port))
        sock.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not seen:
            time.sleep(0.05)
    finally:
        service.stop()
    assert [item.node_id for item in seen] == ["foreign"]
