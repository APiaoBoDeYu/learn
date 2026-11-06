"""贯穿全教程的"慢流水线"：相机图像 -> 预处理 -> ToyVLA 推理 -> 后处理 -> 动作(numpy)。

PipelineConfig 里每个开关对应一种典型低效，各章用不同工具去"发现"它们，第 08 章逐个修复：

| 开关              | 低效行为                                   | 瓶颈类型     |
|-------------------|--------------------------------------------|--------------|
| py_preprocess     | Python 逐行循环做归一化 + HWC->CHW         | CPU-bound    |
| sync_in_loop      | 后处理里逐维 .item() 检查越界              | 同步         |
| no_pin            | pageable 内存同步拷贝，每次都新建 GPU 张量 | 拷贝         |
| fp32              | 不开半精度                                 | Compute/Mem  |
| eager_small_ops   | 后处理按维度、按时间步拆成大量小算子       | Launch-bound |
| realloc           | 每次迭代新建大小不一的临时显存             | 分配/碎片    |

用法：
    pipe = SlowPipeline(PipelineConfig.slow())   # 全部开关打开
    pipe = SlowPipeline(PipelineConfig.fast())   # 全部关闭
    actions = pipe.step(i)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .bench_utils import region
from .models import ToyVLA

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PipelineConfig:
    py_preprocess: bool = True
    sync_in_loop: bool = True
    no_pin: bool = True
    fp32: bool = True
    eager_small_ops: bool = True
    realloc: bool = True
    batch: int = 1
    cam_hw: tuple[int, int] = (480, 640)
    n_tokens: int = 32

    @classmethod
    def slow(cls, **kw) -> "PipelineConfig":
        return cls(**kw)

    @classmethod
    def fast(cls, **kw) -> "PipelineConfig":
        base = dict(py_preprocess=False, sync_in_loop=False, no_pin=False, fp32=False,
                    eager_small_ops=False, realloc=False)
        base.update(kw)
        return cls(**base)

    def replace(self, **kw) -> "PipelineConfig":
        return dataclasses.replace(self, **kw)

    def flags(self) -> str:
        names = ["py_preprocess", "sync_in_loop", "no_pin", "fp32", "eager_small_ops", "realloc"]
        on = [n for n in names if getattr(self, n)]
        return ",".join(on) if on else "(all optimized)"


class SlowPipeline:
    def __init__(self, cfg: PipelineConfig | None = None, model: ToyVLA | None = None,
                 device: str = "cuda", seed: int = 0):
        self.cfg = cfg or PipelineConfig.slow()
        self.device = device
        torch.manual_seed(seed)
        self.model = (model or ToyVLA()).to(device).eval()
        self.dtype = torch.float32 if self.cfg.fp32 else torch.float16
        if not self.cfg.fp32:
            self.model = self.model.half()
        rng = np.random.default_rng(seed)
        h, w = self.cfg.cam_hw
        # 预生成几帧"相机图像"，避免把随机数生成算进耗时
        self.frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(8)]
        self.tokens = torch.randint(0, 32000, (self.cfg.batch, self.cfg.n_tokens), device=device)
        self.action_scale = torch.tensor([0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1.0], device=device)
        self.prev_action = torch.zeros(self.model.chunk, 7, device=device)
        # 优化路径使用的预分配缓冲
        b = self.cfg.batch
        self.pinned = torch.empty((b, 224, 224, 3), dtype=torch.uint8).pin_memory()
        self.gpu_u8 = torch.empty((b, 224, 224, 3), dtype=torch.uint8, device=device)
        self.mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.n_warnings = 0

    # ---------------- 1. 采集 ----------------
    def capture(self, i: int) -> np.ndarray:
        return self.frames[i % len(self.frames)]

    # ---------------- 2. 预处理 ----------------
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)  # 两条路径都用 cv2 缩放
        if self.cfg.py_preprocess:
            return self._preprocess_python(small)
        return self._preprocess_gpu(small)

    def _preprocess_python(self, small: np.ndarray) -> torch.Tensor:
        """反面教材：在 CPU 上用 Python 逐行循环，做 /255、减均值、除方差、HWC->CHW。"""
        out = np.empty((3, 224, 224), dtype=np.float32)
        for y in range(224):
            row = small[y].astype(np.float32) / 255.0
            for c in range(3):
                out[c, y] = (row[:, c] - MEAN[c]) / STD[c]
        x = np.repeat(out[None], self.cfg.batch, axis=0)
        if self.cfg.no_pin:
            t = torch.from_numpy(x).to(self.device)           # pageable -> 同步拷贝
        else:
            t = torch.from_numpy(x).pin_memory().to(self.device, non_blocking=True)
        return t.to(self.dtype)

    def _preprocess_gpu(self, small: np.ndarray) -> torch.Tensor:
        """优化版：只拷贝 uint8（数据量是 float32 的 1/4），归一化在 GPU 上一次完成。"""
        if self.cfg.no_pin:
            u8 = torch.from_numpy(np.repeat(small[None], self.cfg.batch, axis=0))
            g = u8.to(self.device)
        else:
            self.pinned.copy_(torch.from_numpy(small))         # 广播到 batch
            g = self.gpu_u8
            g.copy_(self.pinned, non_blocking=True)
        x = g.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        return (x - self.mean_t.to(self.dtype)) / self.std_t.to(self.dtype)

    # ---------------- 3. 推理 ----------------
    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.realloc:
            # 反面教材：每次迭代申请大小不一的临时显存（模拟动态 shape 的中间结果、调试用的缓存等）
            n = (1 + (self._iter % 5)) * 8 * 1024 * 1024
            self._scratch = torch.empty(n, device=self.device)
            self._scratch.zero_()
        return self.model(x, self.tokens).float()

    # ---------------- 4. 后处理 ----------------
    @torch.no_grad()
    def postprocess(self, raw: torch.Tensor) -> np.ndarray:
        a = raw[0]                                            # (chunk, 7)
        if self.cfg.eager_small_ops:
            out = torch.empty_like(a)
            for d in range(a.shape[1]):                       # 按维度拆开
                col = a[:, d] * self.action_scale[d]
                col = torch.clamp(col, -self.action_scale[d], self.action_scale[d])
                for t in range(a.shape[0]):                   # 按时间步做指数平滑 -> chunk*7 次小算子
                    col[t] = 0.8 * col[t] + 0.2 * self.prev_action[t, d]
                if self.cfg.sync_in_loop and col.abs().max().item() > 0.99 * self.action_scale[d].item():
                    self.n_warnings += 1                      # .item() 会强制 CPU 等 GPU
                out[:, d] = col
        else:
            scaled = (a * self.action_scale).clamp(-self.action_scale, self.action_scale)
            out = 0.8 * scaled + 0.2 * self.prev_action
            if self.cfg.sync_in_loop:
                for d in range(out.shape[1]):
                    if out[:, d].abs().max().item() > 0.99 * self.action_scale[d].item():
                        self.n_warnings += 1
        self.prev_action = out
        return out.cpu().numpy()                              # 唯一一次必要的 D2H 同步

    # ---------------- 整体 ----------------
    _iter = 0

    def step(self, i: int = 0) -> np.ndarray:
        self._iter = i
        with region("capture"):
            frame = self.capture(i)
        with region("preprocess"):
            x = self.preprocess(frame)
        with region("infer"):
            raw = self.infer(x)
        with region("postprocess"):
            return self.postprocess(raw)

    def run(self, n: int = 20) -> None:
        for i in range(n):
            with region(f"step_{i}"):
                self.step(i)

