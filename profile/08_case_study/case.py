"""第 08 章共用组件：逐步优化的流水线 + 测量函数。

阶段（累积式，每一步在上一步的基础上再改一处；顺序来自 01_diagnose.py 的诊断结论）：
    S0 基线                      PipelineConfig.slow()
    S1 后处理向量化              eager_small_ops=False     （诊断：postprocess 是最大的一块，每步 397 个短 kernel）
    S2 预处理放到 GPU            py_preprocess=False       （诊断：_preprocess_python 是第二大的 CPU 热点）
    S3 去掉循环中的 .item()      sync_in_loop=False
    S4 fp16                      fp32=False
    S5 pinned + 预分配           no_pin=False, realloc=False      （= PipelineConfig.fast()）
    S6 channels_last             视觉编码器和输入用 NHWC
    S7 torch.compile             mode="reduce-overhead"（含 CUDA Graph）
    S8 TensorRT                  第 06 章的 toyvla_b1_fp16.engine + CUDA Graph，替换 S6 的模型（不叠加 S7）
    S9 resize 也放到 GPU         在 S8 的基础上，把 cv2.resize 换成 GPU 上的 F.interpolate
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from common import PipelineConfig, SlowPipeline  # noqa: E402

FLOP_PER_STEP = 5.59e9          # ToyVLA batch=1 一次前向的 FLOPs（第 00 章 03_mfu.py，FlopCounterMode）
PEAK_FP16_TFLOPS = 155.6        # 第 00 章实测（新容器）
ENGINE = ROOT / "06_tensorrt" / "outputs" / "toyvla_b1_fp16.engine"


class ChannelsLastPipeline(SlowPipeline):
    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        self.model.vision = self.model.vision.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def infer(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last), self.tokens).float()


class CompiledPipeline(ChannelsLastPipeline):
    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        self._eager = self.model
        self.cmodel = torch.compile(self.model, mode="reduce-overhead")

    @torch.no_grad()
    def infer(self, x):
        return self.cmodel(x.contiguous(memory_format=torch.channels_last), self.tokens).float()


class TRTPipeline(SlowPipeline):
    """推理换成 TensorRT fp16 engine（输入 fp32 图像 + int64 tokens），用 CUDA Graph 捕获 execute_async_v3。"""

    def __init__(self, cfg, **kw):
        import tensorrt as trt
        super().__init__(cfg, **kw)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(ENGINE.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.in_img = torch.empty(1, 3, 224, 224, device="cuda")
        self.out = torch.empty(1, 16, 7, device="cuda")
        self.ctx.set_tensor_address("image", self.in_img.data_ptr())
        self.ctx.set_tensor_address("tokens", self.tokens.data_ptr())
        self.ctx.set_tensor_address("out", self.out.data_ptr())
        self.stream = torch.cuda.Stream()
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)

    @torch.no_grad()
    def infer(self, x):
        self.in_img.copy_(x)                    # fp16 → fp32（engine 的输入是 fp32）
        self.graph.replay()
        return self.out


class TRTGpuResizePipeline(TRTPipeline):
    """在 TRTPipeline 的基础上，把 cv2.resize 也放到 GPU 上（第 07 章 7.1 A4）。"""

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        h, w = self.cfg.cam_hw
        self.pinned_full = torch.empty((h, w, 3), dtype=torch.uint8).pin_memory()

    def preprocess(self, frame):
        self.pinned_full.copy_(torch.from_numpy(frame))
        g = self.pinned_full.to(self.device, non_blocking=True).permute(2, 0, 1)[None].to(self.dtype)
        x = F.interpolate(g, size=(224, 224), mode="bilinear", align_corners=False).div_(255.0)
        return (x - self.mean_t.to(self.dtype)) / self.std_t.to(self.dtype)


S5 = dict(py_preprocess=False, eager_small_ops=False, sync_in_loop=False, fp32=False, no_pin=False, realloc=False)
STAGES = [
    ("S0 基线", lambda: SlowPipeline(PipelineConfig.slow())),
    ("S1 后处理向量化", lambda: SlowPipeline(PipelineConfig.slow(eager_small_ops=False))),
    ("S2 预处理放到 GPU", lambda: SlowPipeline(PipelineConfig.slow(eager_small_ops=False, py_preprocess=False))),
    ("S3 去掉循环中的 .item()", lambda: SlowPipeline(PipelineConfig.slow(py_preprocess=False, eager_small_ops=False,
                                                                       sync_in_loop=False))),
    ("S4 fp16", lambda: SlowPipeline(PipelineConfig.slow(py_preprocess=False, eager_small_ops=False, sync_in_loop=False,
                                                         fp32=False))),
    ("S5 pinned + 预分配", lambda: SlowPipeline(PipelineConfig.slow(**S5))),
    ("S6 channels_last", lambda: ChannelsLastPipeline(PipelineConfig.slow(**S5))),
    ("S7 torch.compile reduce-overhead", lambda: CompiledPipeline(PipelineConfig.slow(**S5))),
    ("S8 TensorRT fp16 + CUDA Graph", lambda: TRTPipeline(PipelineConfig.slow(**S5))),
    ("S9 resize 也放到 GPU", lambda: TRTGpuResizePipeline(PipelineConfig.slow(**S5))),
]


def time_steps(pipe, n=200, warmup=20):
    """端到端 wall 时间（每步结束时 postprocess 里有一次必要的 D2H 同步）。"""
    for i in range(warmup):
        pipe.step(i)
    torch.cuda.synchronize()
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        pipe.step(i)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def phase_times(pipe, n=60):
    """各阶段的时间：在阶段边界同步，把 GPU 工作归到发起它的阶段（会破坏阶段间的重叠，只用于拆分，不用于总延迟）。"""
    acc = {"capture": [], "preprocess": [], "infer": [], "postprocess": []}
    for i in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        f = pipe.capture(i); t1 = time.perf_counter()
        x = pipe.preprocess(f); torch.cuda.synchronize(); t2 = time.perf_counter()
        r = pipe.infer(x); torch.cuda.synchronize(); t3 = time.perf_counter()
        pipe.postprocess(r); torch.cuda.synchronize(); t4 = time.perf_counter()
        for k, a, b in (("capture", t0, t1), ("preprocess", t1, t2), ("infer", t2, t3), ("postprocess", t3, t4)):
            acc[k].append((b - a) * 1000)
    return {k: float(np.median(v)) for k, v in acc.items()}


def power_during(pipe, seconds=3.0):
    """连续运行 seconds 秒，jtop 采样平均功耗（W）和 GPU/CPU 负载。jtop 有约 1 s 延迟，取后半段的样本。"""
    from common.monitor import JtopSampler
    s = JtopSampler(interval=0.2).start()
    t0 = time.perf_counter(); i = 0; n = 0
    while time.perf_counter() - t0 < seconds:
        pipe.step(i); i += 1; n += 1
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    s.stop()
    rec = [r for r in s.records if "p_tot_w" in r and r["t"] > 1.2]
    avg = lambda k: float(np.mean([r[k] for r in rec])) if rec else float("nan")  # noqa: E731
    return {"p_tot_w": avg("p_tot_w"), "p_gpu_w": avg("p_gpu_w"), "gpu_load": avg("gpu_load"), "cpu_max": avg("cpu_max"),
            "ms_per_step": el / n * 1000}


def load_stage_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
