"""00-03 MFU：GPU "忙" ≠ GPU "算力用满"。

对 ToyVLA / TinyTransformer 在不同 batch、不同精度下：
  - 用 FlopCounterMode 统计一次前向的理论 FLOPs
  - 测延迟 → 计算 MFU = 实际有效 FLOPs/s ÷ 峰值 FLOPs/s
  - 用 NVML 采样 GPU 时间利用率（= nvidia-smi dmon 的 sm 列：采样周期内"有 kernel 在跑"的时间占比）

注：本来想用 torch.profiler 统计 kernel 耗时，但在 Thor 上 torch.profiler 拿不到 GPU 时间戳
（CUPTI 返回的 kernel 记录时间戳为 0，详见第 03 章 §7），所以改用 NVML。

运行：/opt/venv/bin/python 00_theory/03_mfu.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import threading  # noqa: E402

import pynvml  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from common import TinyTransformer, ToyVLA, benchmark, print_env  # noqa: E402

# 来自 02_roofline.py 的实测突发峰值（TFLOPS）
PEAK = {torch.float16: 156.8, torch.float32: 6.39}


def count_flops(model, inputs) -> int:
    """统计一次前向的 FLOPs。

    坑：推理模式下 nn.TransformerEncoderLayer 会走 fastpath 融合算子（_transformer_encoder_layer_fwd），
    FlopCounterMode 不认识它，会悄悄记为 0。统计时临时关掉 fastpath（数学上等价）。
    """
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(*inputs)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    return fc.get_total_flops()


def gpu_busy_ratio(fn, seconds=2.0) -> float:
    """持续运行 fn() 若干秒，期间用 NVML 每 50ms 采样一次 GPU 利用率，取平均。"""
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            samples.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            time.sleep(0.05)

    th = threading.Thread(target=sampler)
    t0 = time.perf_counter()
    th.start()
    while time.perf_counter() - t0 < seconds:
        fn()
    torch.cuda.synchronize()
    stop.set()
    th.join()
    tail = samples[len(samples) // 4:]  # 去掉刚开始的采样（NVML 的利用率是滑动窗口平均）
    return sum(tail) / len(tail) / 100


def run_case(name, model, make_inputs, dtype, batches):
    print(f"\n=== {name} ({str(dtype).split('.')[-1]}) ===")
    print(f"{'batch':>5s} {'GFLOPs/iter':>12s} {'latency(ms)':>12s} {'TFLOPS':>8s} {'MFU':>7s} {'GPU busy':>9s}")
    model = model.to("cuda", dtype).eval()
    for b in batches:
        inputs = make_inputs(b, dtype)
        flops = count_flops(model, inputs)

        @torch.no_grad()
        def fn():
            return model(*inputs)

        st = benchmark(fn, warmup=5, iters=30)
        tflops = flops / (st.p50 / 1e3) / 1e12
        busy = gpu_busy_ratio(fn)
        print(f"{b:5d} {flops / 1e9:12.2f} {st.p50:12.3f} {tflops:8.2f} {tflops / PEAK[dtype]:7.1%} {busy:9.0%}")


def main():
    print_env()

    vla = ToyVLA()
    n_params = sum(p.numel() for p in vla.parameters())
    tok_len = 32

    def vla_inputs(b, dtype):
        return (torch.randn(b, 3, 224, 224, device="cuda", dtype=dtype),
                torch.randint(0, 32000, (b, tok_len), device="cuda"))

    # 经验公式验证：Transformer 前向 FLOPs ≈ 2 × 参数量 × token 数（只计算 GEMM 部分的参数）
    tr = TinyTransformer(512, 8, 4, 2048).cuda()  # 必须在 GPU 上统计：CPU 上的 SDPA 算子 FlopCounter 也不认识
    tr_params = sum(p.numel() for n, p in tr.named_parameters() if p.dim() == 2)
    T = 256
    counted = count_flops(tr, (torch.randn(1, T, 512, device="cuda"),))
    attn = 4 * T * T * 512 * 4  # 4 层，每层 QK^T 和 AV 各 2*T*T*d
    print(f"\n[公式验证] TinyTransformer T={T}: 2×N×T = {2 * tr_params * T / 1e9:.2f} GFLOPs，"
          f"+ attention 4×(2·T²·d)×2 = {attn / 1e9:.2f}，合计 {(2 * tr_params * T + attn) / 1e9:.2f}；"
          f"FlopCounter = {counted / 1e9:.2f} GFLOPs")
    torch.backends.mha.set_fastpath_enabled(True)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        tr.eval()(torch.randn(1, T, 512, device="cuda"))
    print(f"[坑] 不关 fastpath 时 FlopCounter = {fc.get_total_flops() / 1e9:.2f} GFLOPs（漏算了整个 encoder）")
    print(f"[模型] ToyVLA 参数量 {n_params / 1e6:.1f}M")

    run_case("ToyVLA", vla, vla_inputs, torch.float32, [1, 8, 32])
    run_case("ToyVLA", ToyVLA(), vla_inputs, torch.float16, [1, 8, 32, 128])
    run_case("TinyTransformer T=256", TinyTransformer(), lambda b, d: (torch.randn(b, 256, 512, device="cuda", dtype=d),),
             torch.float16, [1, 8, 64])


if __name__ == "__main__":
    main()
