"""06-02 trtexec：构建 FP32/FP16/INT8 engine，并比较几种推理选项的影响。

    /opt/venv/bin/python 06_tensorrt/02_trtexec.py

构建（每个模型 3 种精度）：
    trtexec --onnx=x.onnx --saveEngine=x_<prec>.engine --profilingVerbosity=detailed [--fp16 | --int8 --fp16]
    - fp32：TensorRT 默认允许 TF32（Tensor Core 计算 fp32 矩阵乘），要严格 fp32 需加 --noTF32
    - int8：没有提供校准数据，trtexec 会给所有张量设置一个占位的动态范围 → **只能看性能，精度没有意义**
            （真正部署要用校准器或 Q/DQ 量化的 ONNX，本章不展开）
    - --profilingVerbosity=detailed：保存层的详细信息，03/04 需要它把 layer 和 kernel 对应起来
推理选项：
    默认                   每次推理都做 H2D + 计算 + D2H，报告 Latency 和 GPU Compute Time
    --useCudaGraph         用 CUDA Graph 捕获 enqueue，减少 launch 开销
    --noDataTransfers      不做 H2D/D2H（只看计算）
    --useSpinWait          CPU 自旋等待 GPU 完成（降低同步唤醒延迟，代价是占满一个 CPU 核）
"""
import re
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = ["tinycnn_b8", "toyvla_b1"]
PRECS = {"fp32": [], "fp16": ["--fp16"], "int8": ["--int8", "--fp16"]}
MODES = {"default": [], "useCudaGraph": ["--useCudaGraph"], "noDataTransfers": ["--noDataTransfers"],
         "useSpinWait": ["--useSpinWait"]}


def stat(text, key):
    """解析 trtexec 的 'GPU Compute Time: min = .. ms, max = .. ms, mean = .. ms, median = .. ms, percentile(90%) = ..'"""
    m = re.search(rf"\] {re.escape(key)}: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                  r"percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms", text)
    return dict(zip(["min", "max", "mean", "median", "p90", "p95", "p99"], map(float, m.groups()))) if m else None


def run(args, log):
    r = subprocess.run(["trtexec", *args], capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"trtexec 失败，见 {log}")
    return r.stdout


def main():
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("== 1. 构建 ==")
    p(f"{'model':<12s} {'prec':<5s} {'build_s':>8s} {'engine_MB':>10s}")
    for m in MODELS:
        for prec, flags in PRECS.items():
            eng = OUT / f"{m}_{prec}.engine"
            t0 = time.time()
            run([f"--onnx={OUT / f'{m}.onnx'}", f"--saveEngine={eng}", "--profilingVerbosity=detailed", "--skipInference",
                 *flags], OUT / f"02_build_{m}_{prec}.log")
            p(f"{m:<12s} {prec:<5s} {time.time() - t0:8.1f} {eng.stat().st_size / 1e6:10.1f}")

    p("\n== 2. 推理（每种 3 s；单位 ms；GPU = GPU Compute Time，Latency = H2D + GPU + D2H）==")
    p(f"{'model':<12s} {'prec':<5s} {'mode':<16s} {'GPU_med':>8s} {'GPU_p99':>8s} {'Lat_med':>8s} {'H2D_med':>8s} "
      f"{'Enq_med':>8s} {'qps':>8s}")
    for m in MODELS:
        for prec in PRECS:
            for mode, flags in MODES.items():
                out = run([f"--loadEngine={OUT / f'{m}_{prec}.engine'}", "--duration=3", "--warmUp=500", *flags],
                          OUT / f"02_run_{m}_{prec}_{mode}.log")
                g, lat, h2d, enq = (stat(out, k) for k in ("GPU Compute Time", "Latency", "H2D Latency", "Enqueue Time"))
                qps = float(re.search(r"Throughput: ([\d.]+) qps", out).group(1))
                f = lambda d, k: f"{d[k]:8.3f}" if d else f"{'-':>8s}"  # noqa: E731
                p(f"{m:<12s} {prec:<5s} {mode:<16s} {f(g, 'median')} {f(g, 'p99')} {f(lat, 'median')} {f(h2d, 'median')} "
                  f"{f(enq, 'median')} {qps:8.1f}")
    (OUT / "02_trtexec.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
