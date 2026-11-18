from fast_aggregation.cluster.registry import NodeRegistry
from fast_aggregation.models import Announce
from tests.helpers import gpu, machine


def test_announce_then_details_and_ttl():
    registry = NodeRegistry(ttl=10)
    registry.upsert_announce(
        Announce(
            node_id="n1",
            name="desk",
            advertise_host="192.168.0.5",
            http_port=47000,
            rpc_port=50052,
            os_name="Windows",
            arch="x64",
        ),
        now=100,
    )
    assert registry.cluster_nodes(now=105)[0].online is True
    assert registry.cluster_nodes(now=105)[0].resources is None
    assert registry.cluster_nodes(now=120)[0].online is False

    registry.upsert_details(
        "n1",
        resources=machine(gpus=[gpu()]),
        manual=True,
        now=130,
    )
    node = registry.get("n1")
    assert node is not None and node.manual is True
    registry.upsert_announce(
        Announce(
            node_id="n1",
            name="desk",
            advertise_host="192.168.0.9",
            http_port=47000,
            rpc_port=50052,
            os_name="Windows",
            arch="x64",
        ),
        now=131,
    )
    updated = registry.get("n1")
    assert updated is not None
    assert updated.manual is True
    assert updated.host == "192.168.0.9"
    assert updated.resources is not None
