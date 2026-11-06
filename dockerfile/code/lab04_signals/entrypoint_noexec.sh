#!/bin/bash
# 仿照 run_brain.sh：做点准备工作，然后"直接调用"程序（没有 exec）
echo "[wrapper pid=$$] preparing..."
python /app/app.py "$@"
echo "[wrapper] app exited"
