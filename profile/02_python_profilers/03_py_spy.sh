#!/usr/bin/env bash
# 02-03 py-spy：采样式 profiler，不修改代码、开销低，可以 attach 到正在运行的进程。
# 运行：bash 02_python_profilers/03_py_spy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/venv/bin/python
SPY=/opt/venv/bin/py-spy
OUT=02_python_profilers/outputs
mkdir -p $OUT

echo "=== 1. record：启动并采样，输出火焰图 SVG（浏览器打开） ==="
$SPY record -r 200 -o $OUT/flame.svg -- $PY 02_python_profilers/workload.py --steps 300 || true

echo; echo "=== 2. record --format speedscope（拖进 https://www.speedscope.app 查看时间线）==="
$SPY record -r 200 -f speedscope -o $OUT/profile.speedscope.json -- $PY 02_python_profilers/workload.py --steps 300 || true

echo; echo "=== 3. record --format raw：折叠栈文本，下面用脚本汇总 ==="
$SPY record -r 200 -f raw -o $OUT/stacks.txt -- $PY 02_python_profilers/workload.py --steps 300 || true
$PY 02_python_profilers/03_py_spy_report.py $OUT/stacks.txt

echo; echo "=== 4. 定位：哪些调用栈经过 .item() 所在的第 150 行 ==="
$PY 02_python_profilers/03_py_spy_report.py $OUT/stacks.txt --grep "pipeline.py:150"
# 注：--native（同时采样 C/C++ 栈）在 aarch64 上不支持：
#   "Collecting stack traces from native extensions (`--native`) is not supported on your platform."

echo; echo "=== 5. dump：查看运行中进程此刻的调用栈 ==="
$PY 02_python_profilers/workload.py --steps 3000 --allow-ptrace > /dev/null &
PID=$!
sleep 7
$SPY dump --pid $PID || true
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
