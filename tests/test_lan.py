import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from fast_aggregation.coordinator.service import CoordinatorService
from fast_aggregation.discovery.scan import HostProbe, explain_scan, probe_host
from fast_aggregation.node.app import create_node_app
from fast_aggregation.node.pairing import PairSession
from tests.test_safety import _node


def test_closed_port_is_refused_and_timeouts_explain_the_firewall():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    result = probe_host("127.0.0.1", port, timeout=0.3, http_timeout=0.3)
    assert result.state == "refused"
    text = explain_scan([HostProbe(f"10.32.237.{index}", "timeout") for index in range(1, 4)])
    assert "Общедоступная" in text
    assert "47000" not in text or "порт" in text


def test_hello_over_plain_http_is_found():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/hello"):
                raw = json.dumps(
                    {"software": "fast-aggregation", "paired": False, "name": "pc", "node_id": "n", "summary": "cpu"}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        found = probe_host("127.0.0.1", port, timeout=0.4, http_timeout=1.5)
    finally:
        server.shutdown()
    assert found.state == "hello"
    assert found.hello["name"] == "pc"
    assert found.hello["paired"] is False


def test_unpaired_node_answers_hello_and_accepts_an_offer(tmp_path: Path):
    node = _node(tmp_path, __import__("tests.helpers", fromlist=["machine"]).machine())
    session = PairSession(tmp_path, joiner=lambda _pending: {"token": "cluster-token"})
    app = create_node_app(node, token=None, session=session)
    with TestClient(app) as client:
        hello = client.get("/hello")
        assert hello.status_code == 200
        assert hello.json()["paired"] is False
        assert hello.json()["software"] == "fast-aggregation"
        offered = client.post(
            "/v1/pair/offer",
            json={"coordinator": "https://10.32.237.25:47080", "code": "Q9TD93RU", "fingerprint": "aa:bb"},
        )
        assert offered.status_code == 200
        assert offered.json()["status"] == "pending"
        accepted = client.post("/v1/pair/accept")
        assert accepted.status_code == 200
        assert session.auth["token"] == "cluster-token"


def test_pairing_card_uses_lan_addresses_not_loopback(tmp_path: Path, monkeypatch):
    service = CoordinatorService(
        registry=__import__("fast_aggregation.cluster.registry", fromlist=["NodeRegistry"]).NodeRegistry(),
        data_dir=tmp_path,
        background=False,
        pairing_code="Q9TD93RU",
        fingerprint="aa:bb",
    )
    service.public_host = "10.32.237.25"
    service.public_port = 47080
    monkeypatch.setattr(
        "fast_aggregation.netinfo.local_ipv4_addresses",
        lambda: ["127.0.0.1", "10.32.237.25", "192.168.0.8"],
    )
    card = service.pairing_card()
    assert [item["host"] for item in card["links"]] == ["10.32.237.25", "192.168.0.8"]
    assert card["links"][0]["uri"].startswith("fastagg://10.32.237.25:47080?code=Q9TD93RU&fp=aa:bb")
    assert card["links"][0]["command"].startswith('fastagg join --url "fastagg://')
    rotated = service.rotate_pairing_code()
    assert rotated["code"] != "Q9TD93RU"
    assert (tmp_path / "pairing_code").read_text(encoding="utf-8") == rotated["code"]


def test_pair_prompt_accepts_a_link_or_a_code():
    from fast_aggregation.cli import classify_pair_input, want_pair_prompt

    assert classify_pair_input("  ") == "empty"
    assert classify_pair_input("fastagg://10.32.237.25:47080?code=Q9TD93RU&fp=aa") == "link"
    assert classify_pair_input("q9td93ru") == "code"
    assert classify_pair_input("не ссылка") == "invalid"
    args = argparse.Namespace(open=False, interactive=True)
    assert want_pair_prompt(args, "already") is False
    args.open = True
    assert want_pair_prompt(args, "") is False


def test_installer_adds_app_dir_to_path_and_removes_it():
    iss = Path("packaging/fastagg.iss").read_text(encoding="utf-8")
    assert "ChangesEnvironment=yes" in iss
    assert "AddInstallDirToPath" in iss
    assert "RemoveInstallDirFromPath" in iss
    assert "WM_SETTINGCHANGE" in iss


def test_dashboard_shows_the_join_card_and_firewall_is_not_private_only():
    html = Path("src/fast_aggregation/web/index.html").read_text(encoding="utf-8")
    assert html.count("Подключить узел") >= 2
    assert "Новый код" in html
    assert "https://" not in html.lower()
    iss = Path("packaging/fastagg.iss").read_text(encoding="utf-8")
    assert "profile=any" in iss
    assert "profile=private" not in iss
    assert "47000" in iss and "45221" in iss and "47080-47099" in iss
