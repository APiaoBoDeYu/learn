"""06-05 同一个模型：PyTorch eager / torch.compile / TensorRT（FP32/FP16/INT8，±CUDA Graph）的延迟与精度

    /opt/venv/bin/python 06_tensorrt/05_compare.py      （需要先运行 01、02）

- 输入和 PyTorch fp32 参考输出来自 01 保存的 *_ref.npz；权重用同样的随机种子重建（01_export_onnx.build）
- 计时：CUDA Event，输入已在 GPU 上，只测模型本身（不含 H2D/D2H），取 200 次的中位数
- 精度：与 PyTorch fp32 输出的最大绝对误差。INT8 没有校准，误差没有参考意义（02 的说明）
- TensorRT 用 Python API 执行：context.set_tensor_address + execute_async_v3；
  "+graph" 表示用 torch.cuda.CUDAGraph 捕获 execute_async_v3（与 trtexec --useCudaGraph 等价）
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("exp", HERE / "01_export_onnx.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

LOGGER = trt.Logger(trt.Logger.WARNING)


def gpu_ms(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


class TRTModule:
    def __init__(self, path, inputs):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.inputs = inputs
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        out_name = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(out_name)), device="cuda")
        for n, x in inputs.items():
            self.ctx.set_tensor_address(n, x.data_ptr())
        self.ctx.set_tensor_address(out_name, self.out.data_ptr())
        self.stream = torch.cuda.Stream()
        self.graph = None

    def __call__(self):
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        cur.wait_stream(self.stream)
        return self.out

    def capture(self):
        self()                                   # 先跑一次（TensorRT 会在第一次执行时做一些初始化）
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        return lambda: (self.graph.replay(), self.out)[1]


def main():
    torch.backends.mha.set_fastpath_enabled(False)
    lines = []

    def p(s=""):
        print(s); lines.append(s)

    for name in ("tinycnn_b8", "toyvla_b1"):
        m, _, names = exp.build(name)
        ref = np.load(OUT / f"{name}_ref.npz")
        inputs = {n: torch.from_numpy(ref[n]).cuda() for n in names}
        out_ref = torch.from_numpy(ref["out"]).cuda()
        args = [inputs[n] for n in names]
        m16 = exp.build(name)[0].half()
        args16 = [a.half() if a.is_floating_point() else a for a in args]
        torch._dynamo.reset()
        c_def = torch.compile(m16)
        c_ro = torch.compile(m16, mode="reduce-overhead")
        cands = {
            "torch eager fp32": lambda: m(*args),
            "torch eager fp16": lambda: m16(*args16),
            "torch.compile fp16": lambda: c_def(*args16),
            "torch.compile fp16 reduce-overhead": lambda: c_ro(*args16),
        }
        for prec in ("fp32", "fp16", "int8"):
            mod = TRTModule(OUT / f"{name}_{prec}.engine", inputs)
            cands[f"TensorRT {prec}"] = mod
            cands[f"TensorRT {prec} +graph"] = mod.capture()
        p(f"\n== {name}（输入 {', '.join(f'{n}{tuple(inputs[n].shape)}' for n in names)}）==")
        p(f"  {'方案':<36s} {'GPU ms':>8s} {'vs eager fp32':>14s} {'max|err|':>10s}")
        base = None
        with torch.no_grad():
            for k, fn in cands.items():
                t = gpu_ms(fn)
                base = base or t
                err = (fn().float() - out_ref).abs().max().item()
                note = "（未校准，仅看性能）" if "int8" in k else ""
                p(f"  {k:<36s} {t:8.3f} {base / t:13.2f}x {err:10.2e}{note}")
    (OUT / "05_compare.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
