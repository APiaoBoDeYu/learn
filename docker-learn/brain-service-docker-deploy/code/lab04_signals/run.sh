#!/bin/bash
# 五种组合下测量 docker stop 的耗时，并打印容器日志。
# 用法：./run.sh
set -euo pipefail
cd "$(dirname "$0")"

docker build -q -t lab04 . >/dev/null

run_case() {  # $1 = 说明, 其余 = docker run 额外参数
    local desc="$1"; shift
    local name="lab04-case"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" "$@" lab04 >/dev/null
    sleep 2
    local pid1
    pid1=$(docker exec "$name" cat /proc/1/cmdline | tr '\0' ' ')
    local start end
    start=$(date +%s.%N)
    docker stop "$name" >/dev/null
    end=$(date +%s.%N)
    local code
    code=$(docker inspect -f '{{.State.ExitCode}}' "$name")
    echo "================================================================"
    echo "$desc"
    echo "  PID 1     : $pid1"
    awk -v s="$start" -v e="$end" -v c="$code" 'BEGIN{printf "  stop 耗时 : %.1fs   ExitCode=%s\n", e-s, c}'
    echo "  日志:"
    docker logs "$name" 2>&1 | sed 's/^/    /'
    docker rm -f "$name" >/dev/null
}

run_case "1) 包装脚本没有 exec（= run_brain.sh）" \
    -e HANDLE_SIGTERM=1
run_case "2) exec，但程序没有 SIGTERM handler" \
    --entrypoint /app/entrypoint_exec.sh -e HANDLE_SIGTERM=0
run_case "3) exec + 程序处理 SIGTERM（推荐）" \
    --entrypoint /app/entrypoint_exec.sh -e HANDLE_SIGTERM=1
run_case "4) 没有 exec，但加了 --init" \
    --init -e HANDLE_SIGTERM=1
run_case "5) exec + --init，程序没有 handler" \
    --init --entrypoint /app/entrypoint_exec.sh -e HANDLE_SIGTERM=0
