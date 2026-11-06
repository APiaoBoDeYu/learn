"""第 04 章的负载：运行慢/快流水线，只在稳态部分打开 nsys 采集。

- 用 torch.cuda.profiler.start()/stop()（即 cudaProfilerStart/Stop）界定采集区间，
  配合 nsys profile --capture-range=cudaProfilerApi 使用，避开 import、初始化、warmup 阶段
- 流水线的 step() 已经用 common.region() 打了 NVTX 标注：step_i / capture / preprocess / infer / postprocess

用法：
    nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi -o outputs/slow \\
        /opt/venv/bin/python 04_nsight_systems/workload_nvtx.py --steps 20
    选项：--fast（优化版）  --emit-nvtx（为每个 PyTorch 算子生成 NVTX range）
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import PipelineConfig, SlowPipeline, region  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--emit-nvtx", action="store_true", help="torch.autograd.profiler.emit_nvtx()：每个算子一个 NVTX range")
    args = ap.parse_args()

    pipe = SlowPipeline(PipelineConfig.fast() if args.fast else PipelineConfig.slow())
    for i in range(5):                       # warmup，不在采集区间内
        pipe.step(i)
    torch.cuda.synchronize()

    ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True) if args.emit_nvtx else None
    torch.cuda.profiler.start()              # cudaProfilerStart → nsys 从这里开始采集（首次调用有一次性开销）
    if ctx:
        ctx.__enter__()
    for i in range(3):                       # 采集开始后的前几步受 nsys 初始化影响，不计时
        with region(f"settle_{i}"):
            pipe.step(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):
        with region(f"step_{i}"):
            pipe.step(i)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.steps * 1000  # 注意：必须在 profiler.stop() 之前计时，stop() 时 nsys 会落盘数据（约 1 s）
    if ctx:
        ctx.__exit__(None, None, None)
    torch.cuda.profiler.stop()               # cudaProfilerStop → 结束采集
    print(f"[{'fast' if args.fast else 'slow'}{' +emit_nvtx' if args.emit_nvtx else ''}] "
          f"{args.steps} steps, {ms:.3f} ms/step")


if __name__ == "__main__":
    main()
