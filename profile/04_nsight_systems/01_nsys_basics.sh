#!/usr/bin/env bash
# 04-01 nsys 基础：采集 → stats 报告 → analyze 专家规则
# 运行：bash 04_nsight_systems/01_nsys_basics.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/opt/venv/bin/python
OUT=outputs
mkdir -p $OUT

for v in slow fast; do
  flag=""; [ $v = fast ] && flag="--fast"
  echo "=== 采集 $v ==="
  # -t: 追踪的 API；--capture-range=cudaProfilerApi: 只采集 cudaProfilerStart/Stop 之间的部分
  # --capture-range-end=stop: 遇到 cudaProfilerStop 后停止采集但让程序继续运行（默认会直接结束程序）
  # --cuda-memory-usage: 记录显存使用曲线
  nsys profile -t cuda,nvtx,osrt,cublas,cudnn --cuda-memory-usage=true \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o $OUT/pipeline_$v --force-overwrite true \
      $PY workload_nvtx.py --steps 30 $flag 2>/dev/null | grep -E "steps,"
done

S="nsys stats -q --force-export=true"
for v in slow fast; do
  R=$OUT/pipeline_$v.nsys-rep
  echo; echo "################ $v ################"
  echo "=== cuda_api_sum：CPU 侧 CUDA API 调用（次数、耗时）==="
  $S -r cuda_api_sum $R 2>/dev/null | sed -n '4,12p'
  echo "=== cuda_gpu_kern_sum：GPU kernel 耗时 Top ==="
  $S -r cuda_gpu_kern_sum:base $R 2>/dev/null | sed -n '4,13p' | cut -c1-200
  echo "=== nvtx_gpu_proj_sum：每个 NVTX 区间投影到 GPU 上的时间 ==="
  $S -r nvtx_gpu_proj_sum $R 2>/dev/null | grep -E "Range  |-----|:(preprocess|infer|postprocess) " | cut -c1-200
  echo "=== analyze：gpu_time_util（GPU 时间利用率）==="
  nsys analyze -q -r gpu_time_util:rows=8 $R 2>/dev/null | grep -v "^$" | tail -12
  echo "=== analyze：gpu_gaps（GPU 空闲 >= 0.2ms 的区间）==="
  nsys analyze -q -r gpu_gaps:gap=0.2:rows=5 $R 2>/dev/null | grep -v "^$" | tail -8
  echo "=== analyze：cuda_memcpy_async（使用 pageable 内存的异步拷贝）==="
  nsys analyze -q -r cuda_memcpy_async:rows=3 $R 2>/dev/null | grep -v "^$" | tail -6
done
