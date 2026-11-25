#!/bin/bash
# 用 buildx 构建 amd64 + arm64 双架构镜像，推到本机临时仓库，查看 manifest list。
set -euo pipefail
cd "$(dirname "$0")"

REG=localhost:5000
IMG=$REG/lab07:v1

# 1. 本机临时仓库
docker rm -f lab07-registry >/dev/null 2>&1 || true
docker run -d --name lab07-registry -p 5000:5000 registry:2 >/dev/null

# 2. 安装 QEMU 模拟器（让本机能执行另一种架构的程序）
docker run --privileged --rm tonistiigi/binfmt --install all >/dev/null

# 3. docker-container 驱动的 builder 才支持多平台；network=host 让它能访问 localhost:5000
docker buildx rm lab07-builder >/dev/null 2>&1 || true
docker buildx create --name lab07-builder --driver docker-container \
    --driver-opt network=host --use >/dev/null

# 4. 一条命令构建两个平台并推送 manifest list
docker buildx build --platform linux/amd64,linux/arm64 -t "$IMG" --push .

# 5. 查看 manifest list
echo "== imagetools inspect"
docker buildx imagetools inspect "$IMG"

# 6. 分别运行两个架构
for p in linux/amd64 linux/arm64; do
    echo "== docker run --platform $p"
    docker run --rm --pull always --platform "$p" "$IMG"
done

echo
echo "清理：docker rm -f lab07-registry && docker buildx rm lab07-builder"
