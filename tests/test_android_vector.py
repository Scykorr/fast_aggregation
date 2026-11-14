"""Вектор подписи UDP, который повторяет unit-тест Android-приложения."""

from fast_aggregation.security.token import canonical_announce, sign


def test_android_announce_vector():
    payload = {
        "advertise_host": "192.168.1.20",
        "arch": "arm64",
        "http_port": 47000,
        "kind": "announce",
        "magic": "FAAG1",
        "name": "phone",
        "node_id": "android-test",
        "os_name": "Android",
        "preferred_backend": "cpu",
        "rpc_port": 50052,
        "software": "fast-aggregation",
        "v": 1,
    }
    canonical = canonical_announce(payload).decode()
    assert canonical.startswith('{"advertise_host":"192.168.1.20"')
    assert sign("secret", canonical.encode()) == "948127acaf412c5f53a7dd944e3e5ab1c71f35a55ddb5328f2978b5636fc7820"
