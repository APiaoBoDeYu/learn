#!/usr/bin/env bash
# 5.2 ncu 命令行基础：采集、过滤、锁频、缓存控制、开销、导出。
#   bash 01_ncu_basics.sh 2>&1 | tee outputs/01_ncu_basics.log
set -u
cd "$(dirname "$0")"
PY=/opt/venv/bin/python
W=workload_basic.py
O=outputs
mkdir -p $O
hr() { echo; echo "################ $* ################"; }
# 只保留 ncu 输出中我们关心的行
brief() { grep -E "^==PROF==|^==WARNING==|^==ERROR==|^  [a-zA-Z_].*\(|Duration|SM Frequency|L2 Hit Rate|Memory Throughput|passes" ; }

hr "0. 不用 ncu 时的运行时间"
/usr/bin/time -f "  wall %e s" $PY $W

hr "1. 最简单的用法：默认 basic set（SpeedOfLight/LaunchStats/Occupancy/WorkloadDistribution），采集所有 kernel"
/usr/bin/time -f "  wall %e s" ncu $PY $W 2>&1 | grep -E "^==PROF== Profiling|wall" | sed 's/: 0%.*100%//'

hr "2. 过滤 (a) -k 按 kernel 名（regex:），--launch-skip 跳过前 N 次匹配，-c/--launch-count 只采 N 个"
ncu -k regex:elementwise --launch-skip 3 -c 2 --section SpeedOfLight $PY $W 2>&1 | brief

hr "2. 过滤 (b) --nvtx --nvtx-include：只采 measure 范围内的 kernel（push/pop 范围的写法是 \"名字/\"）"
ncu --nvtx --nvtx-include "measure/" --section SpeedOfLight $PY $W 2>&1 | grep -E "^==PROF== Profiling" | sed 's/: 0%.*100%//'

hr "2. 过滤 (c) 名字太长时：--kernel-name-base demangled 用完整签名匹配（默认 function，只匹配函数名）"
for base in function demangled; do
  echo "--- --kernel-name-base $base -k regex:sum_functor（sum_functor 只出现在模板参数里）"
  ncu --kernel-name-base $base -k regex:sum_functor --section SpeedOfLight $PY $W 2>&1 | grep -E "^==PROF== Profiling|No kernels were profiled" | sed 's/: 0%.*100%//'
done

hr "3. 锁频 --clock-control base（默认）vs none：Thor 上两者都提示 without fixed GPU frequencies"
for cc in base none; do
  echo "--- --clock-control $cc"
  ncu --clock-control $cc --nvtx --nvtx-include "measure/" -k regex:elementwise -c 1 --section SpeedOfLight $PY $W 2>&1 | brief
done

hr "4. 缓存控制 --cache-control all（默认：每个 pass 前清空缓存）vs none：8 MB 的 x*2 第 3 次调用"
for cc in all none; do
  echo "--- --cache-control $cc"
  ncu --cache-control $cc --nvtx --nvtx-include "measure/" -k regex:elementwise --launch-skip 2 -c 1 \
      --section SpeedOfLight --metrics lts__t_sector_hit_rate.pct,lts__d_sectors_fill_sysmem.sum $PY $W 2>&1 \
      | grep -E "Duration|lts__|Memory Throughput"
done

hr "5. 开销：section 越多 pass 越多（每个 pass 都要重放一次 kernel）"
for s in "--section SpeedOfLight" "--set basic" "--set detailed" "--set full"; do
  /usr/bin/time -f "  $s: wall %e s" ncu $s -k regex:elementwise -c 1 $PY $W 2>&1 | grep -E "passes|wall" | sed 's/.*100% - /  /'
done

hr "6. 保存报告 + 命令行查看"
ncu --set full --import-source yes --nvtx --nvtx-include "measure/" -f -o $O/01_basic $PY $W 2>&1 | grep -E "Report"
echo "--- --print-summary per-kernel（同名 kernel 合并统计）"
ncu -i $O/01_basic.ncu-rep --print-summary per-kernel --section SpeedOfLight 2>&1 | grep -E "^  [a-z_]|Duration|Memory Throughput" | head -20
echo "--- --page raw --csv 指定指标（适合脚本处理；这里用 Python 只挑几列）"
ncu -i $O/01_basic.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread 2>/dev/null \
  | $PY -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin))
for r in rows[1:]:
    print(f\"  {r['Kernel Name'][:45]:<45s} {r['gpu__time_duration.sum']:>10s} us  SM% {float(r['sm__throughput.avg.pct_of_peak_sustained_elapsed']):5.1f}  regs {r['launch__registers_per_thread']}\")
"
echo "--- --page source --print-source sass：Source 页（SASS 指令级），按 warp stall 采样数排序，取 reduce_kernel 最热的 6 条指令"
ncu -i $O/01_basic.ncu-rep -k regex:reduce --page source --print-source sass --csv 2>/dev/null | tail -n +2 \
  | $PY -c "
import csv, sys
rows = list(csv.DictReader(sys.stdin))
k = 'Warp Stall Sampling (All Samples)'
tot = sum(int(r[k] or 0) for r in rows)
print(f'  共 {len(rows)} 条 SASS 指令，stall 采样 {tot} 个')
for r in sorted(rows, key=lambda r: -int(r[k] or 0))[:6]:
    print(f\"  {int(r[k]):5d} ({100*int(r[k])/tot:4.1f}%)  {r['Address']}  {r['Source'].strip()[:60]}\")
"
echo "--- 用 Python（ncu_report 模块）读取：见 ncu_tools.py:load"
$PY -c "
import sys; sys.path.insert(0, '.')
import ncu_tools as T
for r in T.load(T.OUT / '01_basic.ncu-rep', metrics=['gpu__time_duration.sum']):
    print(f\"  {r['variant']:<8s} {r['kernel'][:40]:<40s} {r['gpu__time_duration.sum']/1e3:8.1f} us\")
"
