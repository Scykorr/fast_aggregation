#!/bin/bash
# Скачивает официальные Android-сборки llama.cpp b12359 и кладёт .so
# в jniLibs, откуда Android разрешает запуск. В git эти файлы не коммитятся.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/app/src/main/jniLibs/arm64-v8a"
TMP="$(mktemp -d)"
mkdir -p "$DEST"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

fetch() {
  local url="$1"
  local dir="$2"
  mkdir -p "$dir"
  curl -fL --retry 3 "$url" | tar -xz -C "$dir"
}

fetch "https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-android-arm64.tar.gz" "$TMP/cpu"
fetch "https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-android-arm64-snapdragon.tar.gz" "$TMP/snap"

find "$TMP/cpu" -type f -name '*.so' -exec cp -n {} "$DEST/" \;
cpu_bin="$(find "$TMP/cpu" -type f -name ggml-rpc-server | head -n 1)"
cp "$cpu_bin" "$DEST/libfastagg-rpc-cpu.so"
chmod 755 "$DEST/libfastagg-rpc-cpu.so"

find "$TMP/snap" -type f -name '*.so' -exec cp -n {} "$DEST/" \;
snap_bin="$(find "$TMP/snap" -type f -name ggml-rpc-server | head -n 1)"
cp "$snap_bin" "$DEST/libfastagg-rpc-opencl.so"
chmod 755 "$DEST/libfastagg-rpc-opencl.so"

echo "jniLibs:" 
find "$DEST" -type f -printf '%f %s\n' | sort
