"""03-04 显存分析：memory_stats / memory snapshot / 碎片化与 expandable_segments。

A. 慢流水线 realloc 开关的影响：分配次数、峰值、reserved 与 allocated 的差距
B. 导出 memory snapshot（在 https://pytorch.org/memory_viz 中拖入 .pickle 文件查看）
C. 逐步增长的分配（动态 shape），对比默认分配器与 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（子进程中运行）
D. profile_memory=True：按算子统计显存分配

运行：/opt/venv/bin/python 03_torch_profiler/04_memory.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from common import PipelineConfig, SlowPipeline  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
MB = 2 ** 20


def mem_report(tag: str) -> dict:
    s = torch.cuda.memory_stats()
    snap = torch.cuda.memory._snapshot()
    segs = snap["segments"]
    reserved = sum(sg["total_size"] for sg in segs)
    active = sum(b["size"] for sg in segs for b in sg["blocks"] if b["state"] == "active_allocated")
    r = {
        "tag": tag,
        "alloc_calls": s["allocation.all.allocated"],           # 累计的张量分配次数（allocator 层面）
        "cudaMalloc_calls": s["segment.all.allocated"],         # 真正向驱动申请显存的次数
        "peak_alloc_MB": s["allocated_bytes.all.peak"] / MB,
        "peak_reserved_MB": s["reserved_bytes.all.peak"] / MB,
        "cur_reserved_MB": reserved / MB,
        "cur_active_MB": active / MB,
        "segments": len(segs),
        "alloc_retries": s["num_alloc_retries"],                # 显存不够时 free 缓存再重试的次数（OOM 前兆）
        "inactive_split_MB": s["inactive_split_bytes.all.current"] / MB,  # 段内被切开但空闲的碎片
    }
    return r


def part_a_b():
    print("=== A/B. 慢流水线：realloc 开关对显存的影响 ===")
    rows = []
    for tag, cfg in [("realloc=True", PipelineConfig.slow()), ("realloc=False", PipelineConfig.slow(realloc=False))]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        pipe = SlowPipeline(cfg)
        torch.cuda.memory._record_memory_history(max_entries=100000)    # 开始记录每一次分配/释放及其调用栈
        for i in range(30):
            pipe.step(i)
        torch.cuda.synchronize()
        name = "mem_snapshot_" + tag.replace("=", "_") + ".pickle"
        torch.cuda.memory._dump_snapshot(str(OUT / name))
        torch.cuda.memory._record_memory_history(enabled=None)
        rows.append(mem_report(tag))
        print(f"  snapshot -> outputs/{name}")
        del pipe
    keys = [k for k in rows[0] if k != "tag"]
    print(f"\n{'':<20s}" + "".join(f"{r['tag']:>16s}" for r in rows))
    for k in keys:
        print(f"{k:<20s}" + "".join(f"{r[k]:16.1f}" for r in rows))


FRAG_CODE = r"""
import torch, json, os
MB = 2 ** 20
# 模拟动态 shape / 逐步增长的缓冲区（如不断变长的 KV cache）：每一步都申请一个比上一步更大的张量，并释放上一个。
# 默认分配器：缓存里的旧块都比新请求小，复用不了 → 每一步都要 cudaMalloc 一个新段，旧段留在缓存里
# expandable_segments：在同一段虚拟地址上按需扩展（映射更多物理页）→ 旧空间可以被复用
buf = None
for n in range(1, 61):
    buf = torch.empty(n * 4 * MB, dtype=torch.uint8, device="cuda")
s = torch.cuda.memory_stats()
print(json.dumps({
    "conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"),
    "allocated_MB": s["allocated_bytes.all.current"] / MB,
    "reserved_MB": s["reserved_bytes.all.current"] / MB,
    "cudaMalloc_calls": s["segment.all.allocated"],
    "segments": s["segment.all.current"],
}))
"""


def part_c():
    print("\n=== C. 逐步增长的分配（4MB, 8MB, ..., 240MB）：默认分配器 vs expandable_segments ===")
    print(f"{'PYTORCH_CUDA_ALLOC_CONF':<32s}{'allocated':>11s}{'reserved':>11s}{'reserved/alloc':>15s}{'cudaMalloc次数':>14s}")
    for conf in [None, "expandable_segments:True"]:
        env = dict(os.environ)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = conf
        out = subprocess.run([sys.executable, "-c", FRAG_CODE], env=env, capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        ratio = r["reserved_MB"] / r["allocated_MB"]
        print(f"{r['conf']:<32s}{r['allocated_MB']:9.0f}MB{r['reserved_MB']:9.0f}MB{ratio:14.1f}x"
              f"{r['cudaMalloc_calls']:14d}")


def part_d():
    print("\n=== D. profile_memory=True：哪些算子分配显存最多（ToyVLA batch=8 前向）===")
    from common import ToyVLA
    m = ToyVLA().cuda().eval()
    img = torch.randn(8, 3, 224, 224, device="cuda")
    tok = torch.randint(0, 32000, (8, 32), device="cuda")
    with torch.no_grad():
        m(img, tok)
        # 需要带上 CUDA activity 才会显示 CUDA Mem 列（显存统计来自 caching allocator，不依赖 CUPTI 的 kernel 时间）
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            m(img, tok)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=8, max_name_column_width=34))


if __name__ == "__main__":
    part_a_b()
    part_c()
    part_d()
