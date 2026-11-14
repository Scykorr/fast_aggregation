"""Порты, тег сборки llama.cpp и тайминги протокола.

Бинарники официального релиза v1.0.0 на GitHub не приложены: страница
релиза указывает ночную сборку b12359, собранную из того же коммита.
Все узлы кластера обязаны использовать один и тот же тег — протокол RPC
не согласует версии.
"""

from __future__ import annotations

SOFTWARE = "fast-aggregation"
PROTOCOL_VERSION = 1

# Ночная сборка, на которую ссылается релиз llama.cpp v1.0.0 (2026-11-13).
DEFAULT_LLAMA_RELEASE = "b12359"
LLAMA_RELEASE_REPO = "ggml-org/llama.cpp"

DISCOVERY_PORT = 45221
NODE_HTTP_PORT = 47000
RPC_PORT = 50052
# 8080 на Windows часто занят httpd из комплекта EDB Postgres.
COORDINATOR_PORT = 47080
COORDINATOR_PORT_SPAN = 20
ENGINE_PORT = 18080

ANNOUNCE_INTERVAL_SEC = 3.0
PEER_TTL_SEC = 12.0
RESOURCE_REFRESH_SEC = 15.0

# Сколько свободной VRAM оставлять под KV-кэш и драйвер, прежде чем
# отдавать остаток планировщику слоёв.
DEFAULT_VRAM_RESERVE_MB = 512
# Доля свободной ОЗУ, которую CPU-узел готов отдать под слои.
DEFAULT_CPU_RAM_FRACTION = 0.5
MIN_DEVICE_MEMORY_MB = 256

DISCOVERY_MAGIC = "FAAG1"

# Пороги безопасности. Температура — градусы Цельсия, батарея — проценты.
TEMP_THROTTLE_C = 80.0
TEMP_CRITICAL_C = 92.0
BATTERY_LAPTOP_THROTTLE = 40
BATTERY_LAPTOP_CRITICAL = 15
BATTERY_PHONE_MIN = 35
CPU_LOAD_THROTTLE = 90.0

# Канал, относительно которого вес узла не снижается.
GOOD_LINK_MBPS = 200.0
GOOD_LINK_RTT_MS = 8.0
# Медленнее 20% лучшего канала или медленнее этого порога — потолок доли слоёв.
SLOW_LINK_MBPS = 30.0
SLOW_LINK_SHARE_CAP = 0.10

# На телефоне оставляем больше ОЗУ системе.
PHONE_RAM_RESERVE_MB = 1536
PHONE_RAM_FRACTION = 0.35
