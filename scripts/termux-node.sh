#!/data/data/com.termux/files/usr/bin/bash
# Узел fastagg в Termux на Android.
# Репозиторий должен быть уже склонирован, скрипт запускается из его корня.
#
#   pkg install git
#   git clone <url> fast_aggregation
#   cd fast_aggregation
#   bash scripts/termux-node.sh --url 'fastagg://...'
#
# Официальный архив llama.cpp для Android (b12359):
#   arm64 CPU:        llama-b12359-bin-android-arm64.tar.gz  (есть ggml-rpc-server)
#   Adreno OpenCL:    llama-b12359-bin-android-arm64-snapdragon.tar.gz
# Отдельной Vulkan-сборки Android в этом релизе нет. Mali остаётся на CPU,
# пока не появится своя сборка. Нативное приложение в этом репозитории не
# собирается: Termux — рабочий путь, план приложения описан в README.

set -euo pipefail
cd "$(dirname "$0")/.."

pkg install -y python clang openssl libffi
python -m pip install --upgrade pip
python -m pip install -e .

BACKEND="cpu"
if [ -e /sys/class/kgsl/kgsl-3d0 ]; then
  BACKEND="opencl"
  echo "Найден Adreno (kgsl), бэкенд opencl."
else
  echo "kgsl не найден. Берём CPU-сборку. Vulkan для Mali в закреплённом релизе не публикуется."
fi

JOIN_URL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --url) JOIN_URL="$2"; shift 2 ;;
    *) echo "Неизвестный аргумент: $1"; exit 2 ;;
  esac
done

if [ -n "$JOIN_URL" ]; then
  python -m fast_aggregation join --url "$JOIN_URL"
fi

exec python -m fast_aggregation node --backend "$BACKEND"
