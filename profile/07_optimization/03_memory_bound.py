"""07-03 Memory-bound / 访存相关的手段：低精度、数据排布（channels_last）、FlashAttention（SDPA）

    /opt/venv/bin/python 07_optimization/03_memory_bound.py

（融合见 02 的 C1 和第 05 章 5.4；这里不重复）
A. 低精度对 Memory-bound 算子：GELU，64M 个元素，fp32 / fp16 / bf16（字节数减半 → 时间减半）
B. channels_last：TinyCNN batch=8，NCHW vs NHWC，fp32 / fp16（cuDNN 的 Tensor Core 卷积偏好 NHWC，见第 04 章练习 1）
C. Attention：朴素实现（显式算出 L×L 的注意力矩阵）vs F.scaled_dot_product_attention，fp16，B=1，H=8，D=64
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TinyCNN, benchmark, compare  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
lines = []


def p(s=""):
    print(s); lines.append(s)


def flush():
    torch.empty(64 * 1024 * 1024, device="cuda").zero_()


def part_a():
    p("== A. GELU，64M 个元素（读一次、写一次）==")
    n = 64 * 1024 * 1024
    r = []
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(n, device="cuda", dtype=dt)
        y = torch.empty_like(x)
        st = benchmark(lambda: F.gelu(x, out=y) if False else torch.nn.functional.gelu(x), name=str(dt).replace("torch.", ""),
                       mode="event", iters=30)
        nbytes = 2 * n * x.element_size()
        st.name = f"{st.name:<9s} {nbytes / st.p50 / 1e6:6.1f} GB/s"
        r.append(st)
        del x, y
    lines.append(compare(r))


def part_b():
    p("\n== B. channels_last：TinyCNN batch=8（eager，no_grad）==")
    r = []
    with torch.no_grad():
        for dt in (torch.float32, torch.float16):
            for fmt in (torch.contiguous_format, torch.channels_last):
                torch.manual_seed(0)
                m = TinyCNN().cuda().eval().to(dt).to(memory_format=fmt)
                x = torch.randn(8, 3, 224, 224, device="cuda", dtype=dt).to(memory_format=fmt)
                name = f"{str(dt).replace('torch.', '')} {'NHWC' if fmt == torch.channels_last else 'NCHW'}"
                r.append(benchmark(lambda: m(x), name=name, mode="event", iters=50))
    lines.append(compare(r))


def naive_attn(q, k, v):
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)     # (B,H,L,L)：显式物化注意力矩阵
    return torch.softmax(s, dim=-1) @ v


def part_c():
    p("\n== C. Attention：朴素 vs SDPA（fp16，B=1，H=8，D=64）==")
    p(f"  {'L':>5s} {'朴素 ms':>9s} {'SDPA ms':>9s} {'加速':>6s} {'朴素峰值显存':>12s} {'SDPA 峰值显存':>13s} {'max|diff|':>10s}")
    for L in (256, 1024, 4096):
        q, k, v = (torch.randn(1, 8, L, 64, device="cuda", dtype=torch.half) for _ in range(3))
        res = {}
        for name, f in (("naive", naive_attn), ("sdpa", F.scaled_dot_product_attention)):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = f(q, k, v)
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
            t = benchmark(lambda: f(q, k, v), mode="event", iters=50).p50
            res[name] = (t, peak, out)
        d = (res["naive"][2].float() - res["sdpa"][2].float()).abs().max().item()
        p(f"  {L:5d} {res['naive'][0]:9.3f} {res['sdpa'][0]:9.3f} {res['naive'][0] / res['sdpa'][0]:5.2f}x "
          f"{res['naive'][1]:10.1f} MB {res['sdpa'][1]:11.1f} MB {d:10.2e}")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        try:
            F.scaled_dot_product_attention(q, k, v); p("  FLASH_ATTENTION 后端可用")
        except RuntimeError as e:
            p(f"  FLASH_ATTENTION 后端不可用：{str(e)[:120]}")


if __name__ == "__main__":
    part_a(); part_b(); part_c()
    (OUT / "03_memory_bound.log").write_text("\n".join(lines) + "\n")
