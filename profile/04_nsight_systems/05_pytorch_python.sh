#!/usr/bin/env bash
# 04-05 nsys 的 PyTorch / Python 集成，以及 GPU metrics
# 运行：bash 04_nsight_systems/05_pytorch_python.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/opt/venv/bin/python
OUT=outputs
CAP="--capture-range=cudaProfilerApi --capture-range-end=stop"

echo "=== 1. --pytorch=autograd-shapes-nvtx：nsys 自动为每个 PyTorch 算子打 NVTX（不用改代码）==="
nsys profile -t cuda,nvtx --pytorch=autograd-shapes-nvtx $CAP -o $OUT/pt_nvtx --force-overwrite true \
    $PY workload_nvtx.py --steps 10 --fast 2>/dev/null | grep "steps,"
nsys stats -q --force-export=true -r nvtx_sum $OUT/pt_nvtx.nsys-rep 2>/dev/null | sed -n '4,14p' | cut -c1-190

echo; echo "=== 2. nvtx_kern_sum：每个 NVTX 区间（算子）发起了哪些 kernel ==="
nsys stats -q -r nvtx_kern_sum:base $OUT/pt_nvtx.nsys-rep 2>/dev/null | grep -E "NVTX Range|-----|aten::(linear|conv2d|layer_norm|_native_multi_head)" | head -8 | cut -c1-230

echo; echo "=== 3. CPU 采样（默认 --sample=process-tree）+ --python-sampling：需要 perf_event_open（nsys status -e）==="
nsys status -e 2>&1 | grep -E "perf_event_open|process-tree"
nsys profile -t cuda,nvtx --python-sampling=true --python-sampling-frequency=1000 $CAP -o $OUT/pysample --force-overwrite true \
    $PY workload_nvtx.py --steps 10 2>&1 | grep -E "steps,|WARN|rror" | head -5
nsys export -t sqlite -o $OUT/pysample.sqlite --force-overwrite true $OUT/pysample.nsys-rep > /dev/null 2>&1
$PY - <<'PYEOF'
import sqlite3
from collections import Counter
c = sqlite3.connect("outputs/pysample.sqlite")
q = lambda sql: c.execute(sql).fetchall()  # noqa: E731
# Python 样本：以 NvtxMark（eventType 34）的形式存放，调用栈在 NVTX 扩展 payload 中（sqlite 导出里 binaryData 为空，只能在 GUI 中看）
print("Python 样本（NvtxMark）:", q("SELECT COUNT(*) FROM NVTX_EVENTS WHERE eventType = 34")[0][0],
      "  payload schema:", [r[0] for r in q("SELECT name FROM NVTX_PAYLOAD_SCHEMAS WHERE name IS NOT NULL")])
# CPU（native）采样：COMPOSITE_EVENTS 每行一个样本，SAMPLING_CALLCHAINS 是它的调用栈（stackDepth 0 = 栈顶）
n = q("SELECT COUNT(*) FROM COMPOSITE_EVENTS")[0][0]
print(f"CPU IP 样本: {n}")
top = q("""SELECT s1.value, s2.value FROM SAMPLING_CALLCHAINS cc JOIN StringIds s1 ON s1.id = cc.symbol
           JOIN StringIds s2 ON s2.id = cc.module WHERE cc.stackDepth = 0""")
mods = Counter(m.split("/")[-1] for _, m in top)
print("  栈顶所在模块（self time）:", ", ".join(f"{m} {100 * k / len(top):.0f}%" for m, k in mods.most_common(6)))
syms = Counter(f"{s[:60]} [{m.split('/')[-1]}]" for s, m in top)
for s, k in syms.most_common(8):
    print(f"  {100 * k / len(top):5.1f}%  {s}")
PYEOF

echo; echo "=== 4. --cudabacktrace：给耗时超过阈值的 CUDA API 调用记录调用栈（这里：同步类 API 超过 5 us）==="
nsys profile -t cuda,nvtx --cudabacktrace=sync:5000 $CAP -o $OUT/bt --force-overwrite true \
    $PY workload_nvtx.py --steps 5 2>&1 | grep -E "steps,|WARN|rror" | head -3
nsys export -t sqlite -o $OUT/bt.sqlite --force-overwrite true $OUT/bt.nsys-rep > /dev/null 2>&1
$PY - <<'PYEOF'
import sqlite3
c = sqlite3.connect("outputs/bt.sqlite")
tabs = [t for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if "CALLCHAIN" in t]
print("调用栈相关的表:", {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tabs})
if "CUDA_CALLCHAINS" in tabs:
    rows = c.execute("""SELECT r.end - r.start, s.value, cc.stackDepth, s2.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                        JOIN StringIds s ON s.id = r.nameId JOIN CUDA_CALLCHAINS cc ON cc.id = r.callchainId
                        JOIN StringIds s2 ON s2.id = cc.symbol ORDER BY r.end - r.start DESC, cc.stackDepth LIMIT 12""").fetchall()
    print("最长的一次同步调用及其调用栈：")
    for dur, api, depth, sym in rows:
        if depth == 0:
            print(f"  {api}  {dur / 1e3:.0f} us")
        print(f"    #{depth} {sym[:90]}")
PYEOF

echo; echo "=== 5. GPU metrics：完整分析见 07_gpu_metrics.py ==="
nsys profile --gpu-metrics-devices=help 2>&1 | head -2
nsys profile --gpu-metrics-set=help 2>&1 | grep -E "gb10b"
