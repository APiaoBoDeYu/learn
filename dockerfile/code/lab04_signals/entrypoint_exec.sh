#!/bin/bash
# 仿照 infra-bin/entrypoint.sh：准备工作之后 exec，程序替换 bash 成为 PID 1
echo "[wrapper pid=$$] preparing..."
exec python /app/app.py "$@"
