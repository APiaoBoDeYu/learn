"""03-02 schedule + on_trace_ready：在循环中只采集若干步；用 trace 分析 CPU 在哪个阶段等 GPU。

schedule(wait=1, warmup=2, active=3, repeat=1)：
    step 0          : wait   （不采集）
    step 1-2        : warmup （采集但丢弃，避开 profiler 自身的启动开销）
    step 3-5        : active （真正记录）→ 结束时调用 on_trace_ready
慢流水线的 step() 里已经用 common.region() 标注了 capture/preprocess/infer/postprocess，
在 trace 中显示为 user_annotation。

运行：/opt/venv/bin/python 03_torch_profiler/02_schedule_trace.py
产物：outputs/pipeline_slow.json、outputs/pipeline_fast.json（Perfetto 中打开）
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cupti_fix  # noqa: E402

cupti_fix.ensure()

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, schedule  # noqa: E402

from common import PipelineConfig, SlowPipeline  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
STAGES = ["capture", "preprocess", "infer", "postprocess"]
SYNC_APIS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")  # CPU 阻塞等 GPU


def profile_pipeline(cfg: PipelineConfig, tag: str, active: int = 5) -> Path:
    pipe = SlowPipeline(cfg)
    for i in range(3):
        pipe.step(i)
    path = OUT / f"pipeline_{tag}.json"

    def on_ready(p):
        p.export_chrome_trace(str(path))
        print(f"[{tag}] trace -> {path.name}")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=2, active=active, repeat=1),
                 on_trace_ready=on_ready) as prof:
        for i in range(1 + 2 + active):
            pipe.step(i)
            prof.step()          # 通知 profiler 进入下一步
    return path


def analyze(path: Path, steps: int) -> dict:
    """统计每个阶段的墙钟时间、同步 API（CPU 等 GPU）的次数和耗时、memcpy 次数、kernel launch 次数。

    注意：cudaMemcpyAsync 不一定是等待——设备内拷贝（如 col[t] = x）是异步的；
    .item()/.cpu() 的 D2H 拷贝则会紧跟一次 cudaStreamSynchronize。所以只把 *Synchronize 计为等待。
    """
    ev = [e for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X"]
    regions = [e for e in ev if e.get("cat") == "user_annotation" and e["name"] in STAGES]
    rt = [e for e in ev if e.get("cat") == "cuda_runtime"]
    res = defaultdict(lambda: {"wall": 0.0, "wait": 0.0, "launch": 0, "sync": 0, "memcpy": 0})
    for r in regions:
        r0, r1 = r["ts"], r["ts"] + r["dur"]
        d = res[r["name"]]
        d["wall"] += r["dur"]
        for e in rt:
            if r0 <= e["ts"] < r1 and e["tid"] == r["tid"]:
                if e["name"].startswith(SYNC_APIS):
                    d["wait"] += e["dur"]
                    d["sync"] += 1
                elif e["name"].startswith("cudaMemcpy"):
                    d["memcpy"] += 1
                elif e["name"] == "cudaLaunchKernel":
                    d["launch"] += 1
    return {k: {kk: (v / steps / 1000 if kk in ("wall", "wait") else v / steps) for kk, v in res[k].items()}
            for k in STAGES}


def main():
    steps = 5
    import time
    for tag, cfg in [("slow", PipelineConfig.slow()), ("fast", PipelineConfig.fast())]:
        pipe = SlowPipeline(cfg)
        for i in range(3):
            pipe.step(i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(20):
            pipe.step(i)
        torch.cuda.synchronize()
        print(f"[{tag}] 不开 profiler: {(time.perf_counter() - t0) / 20 * 1000:.2f} ms/step")
    paths = {"slow": profile_pipeline(PipelineConfig.slow(), "slow", steps),
             "fast": profile_pipeline(PipelineConfig.fast(), "fast", steps)}
    for tag, path in paths.items():
        r = analyze(path, steps)
        print(f"\n=== {tag}：每步各阶段（时间 ms/step，次数 /step）===")
        print(f"{'stage':<12s}{'wall':>8s}{'sync耗时':>9s}{'sync次数':>9s}{'memcpy':>8s}{'launch':>8s}")
        for k in STAGES:
            d = r[k]
            print(f"{k:<12s}{d['wall']:8.2f}{d['wait']:9.2f}{d['sync']:9.0f}{d['memcpy']:8.0f}{d['launch']:8.0f}")
        tot = {kk: sum(r[k][kk] for k in STAGES) for kk in ("wall", "wait", "launch", "sync", "memcpy")}
        print(f"{'total':<12s}{tot['wall']:8.2f}{tot['wait']:9.2f}{tot['sync']:9.0f}{tot['memcpy']:8.0f}"
              f"{tot['launch']:8.0f}")


if __name__ == "__main__":
    main()
