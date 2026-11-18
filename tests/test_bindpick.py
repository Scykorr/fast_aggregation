from fast_aggregation.bindpick import (
    Adapter,
    choose_listening_port,
    is_vpn_name,
    parse_excluded_ranges,
    pick_lan_adapter,
    plan_ports,
    port_excluded,
    route_warning,
    vpn_port_message,
)
from fast_aggregation.cli import ready_payload
from fast_aggregation.errors import FastAggError
import pytest

SAMPLE = """
Protocol tcp Port Exclusion Ranges

Start Port    End Port
----------    --------
     49675       49874
     50000       50059     *
     50158       50257
     61237       61336
"""


def test_excluded_ranges_hide_the_default_rpc_port():
    ranges = parse_excluded_ranges(SAMPLE)
    assert (50000, 50059) in ranges
    assert port_excluded(50052, ranges)
    assert not port_excluded(11435, ranges)
    assert "VPN" in vpn_port_message(47080) or "туннел" in vpn_port_message(47080)


def test_listening_port_skips_excluded_and_takes_ephemeral():
    excluded = [(47080, 47499), (18090, 18189), (28080, 28179)]

    def probe(port, host):
        return port == 34567

    choice = choose_listening_port(
        47080,
        host="127.0.0.1",
        excluded=excluded,
        probe=probe,
        ephemeral=lambda host: 34567,
        label="дашборда",
    )
    assert choice.port == 34567
    assert "47080" in " ".join(choice.notes)


def test_no_bindable_port_names_the_vpn():
    def probe(port, host):
        return False

    with pytest.raises(FastAggError, match="туннел"):
        choose_listening_port(
            47080,
            ranges=((47080, 47080),),
            excluded=[(1, 65535)],
            probe=probe,
            ephemeral=lambda host: 9,
            label="дашборда",
        )


def test_lan_adapter_ignores_wintun_and_warns_on_vpn_route():
    adapters = [
        Adapter("Wintun", "10.8.0.2", vpn=is_vpn_name("Wintun")),
        Adapter("Ethernet", "10.32.237.25", vpn=False),
    ]
    picked = pick_lan_adapter(adapters)
    assert picked is not None and picked.ip == "10.32.237.25"
    assert "Wintun" in route_warning("10.32.237.50", adapters, "10.8.0.2")
    plan = plan_ports(
        dashboard=None,
        api=None,
        rpc=50052,
        discovery=None,
        bind="0.0.0.0",
        advertise="",
        saved={},
        excluded=[(50000, 50059)],
        adapters=adapters,
        probe=lambda port, host: port not in {50052},
        ephemeral=lambda host: 34567,
    )
    assert plan.rpc_port != 50052
    assert plan.advertise == "10.32.237.25"
    assert plan.api_bind == "127.0.0.1"


def test_wait_ready_accepts_a_deployed_model():
    assert ready_payload({"roles": {"llm": {"available": True}}}) is True
    assert ready_payload({"roles": {"llm": {"available": False}}}) is False
    assert ready_payload({}) is False
