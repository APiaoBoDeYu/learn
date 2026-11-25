#!/bin/bash
# 对比两种 Dockerfile 在"只改了一行源码"时的重建耗时。
# 用法：./run.sh            （在 lab02_cache 目录下执行）
set -euo pipefail
cd "$(dirname "$0")"

build() {  # $1 = Dockerfile 名, $2 = tag
    local start end
    start=$(date +%s.%N)
    docker build -q -f "$1" -t "$2" app >/dev/null
    end=$(date +%s.%N)
    awk -v f="$1" -v s="$start" -v e="$end" 'BEGIN{printf "%-18s %6.2fs\n", f, e-s}'
}

reset_version() { sed -i 's/^VERSION = .*/VERSION = 1  # 改这个数字来模拟"修改源码"/' app/main.py; }
trap reset_version EXIT

echo "== 第一次构建（冷缓存，两者都要 pip install）"
build Dockerfile.bad  lab02:bad
build Dockerfile.good lab02:good

echo "== 修改一行源码后重建"
sed -i "s/^VERSION = .*/VERSION = $RANDOM/" app/main.py
build Dockerfile.bad  lab02:bad
build Dockerfile.good lab02:good

echo "== 用 --progress=plain 看 good 版本哪些步骤是 CACHED"
sed -i "s/^VERSION = .*/VERSION = $RANDOM/" app/main.py
docker build --progress=plain -f Dockerfile.good -t lab02:good app 2>&1 | grep -E '^#[0-9]+ (\[|CACHED)'
