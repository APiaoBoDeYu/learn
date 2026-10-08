#!/usr/bin/env bash
# Compile the ErfInv TensorRT plugin into a shared library.
#   bash plugins/build.sh      (from code/)
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../artifacts/liberfinv_plugin.so
# sm_110a = Thor. Change -arch for other GPUs (sm_87 Orin, sm_90 H100, ...).
ARCH=${ARCH:-sm_110a}
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="$ARCH" \
     erfinv_plugin.cu -o "$OUT" -lnvinfer
echo "built $OUT"
