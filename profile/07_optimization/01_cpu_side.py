"""07-01 CPU 侧的瓶颈：Python 预处理、数据加载

    /opt/venv/bin/python 07_optimization/01_cpu_side.py

A. 预处理一帧 480x640 的 uint8 图像 → (1,3,224,224) 归一化张量（与 common/pipeline.py 的两条路径一致）
    A1 Python 逐行循环（慢流水线的 py_preprocess）
    A2 numpy 向量化（一次完成 /255、减均值、除方差、HWC→CHW）
    A3 GPU 归一化：CPU 上 cv2.resize，只拷贝 uint8，在 GPU 上归一化（快流水线的做法）
    A4 全部放到 GPU：拷贝原始 480x640 uint8，在 GPU 上 F.interpolate + 归一化
B. DataLoader：每个样本需要 JPEG 解码 + resize（模拟真实数据集），batch=8
    num_workers = 0 / 2 / 4 / 8，pin_memory = False / True
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import benchmark, compare  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
rng = np.random.default_rng(0)
frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
frame = cv2.GaussianBlur(frame, (9, 9), 3)          # 让图像有一些空间相关性，更像真实图像（JPEG 压缩率也更真实）
mean_t = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
std_t = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
pinned_small = torch.empty((224, 224, 3), dtype=torch.uint8).pin_memory()
pinned_full = torch.empty((480, 640, 3), dtype=torch.uint8).pin_memory()


def a1_python_loop():
    small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    out = np.empty((3, 224, 224), dtype=np.float32)
    for y in range(224):
        row = small[y].astype(np.float32) / 255.0
        for c in range(3):
            out[c, y] = (row[:, c] - MEAN[c]) / STD[c]
    return torch.from_numpy(out[None]).pin_memory().cuda(non_blocking=True)


def a2_numpy():
    small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = ((small.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    return torch.from_numpy(np.ascontiguousarray(x)).pin_memory().cuda(non_blocking=True)


def a3_gpu_normalize():
    small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    pinned_small.copy_(torch.from_numpy(small))
    g = pinned_small.cuda(non_blocking=True)
    x = g.permute(2, 0, 1)[None].float().div_(255.0)
    return (x - mean_t) / std_t


def a4_gpu_all():
    pinned_full.copy_(torch.from_numpy(frame))
    g = pinned_full.cuda(non_blocking=True).permute(2, 0, 1)[None].float()
    x = F.interpolate(g, size=(224, 224), mode="bilinear", align_corners=False, antialias=False).div_(255.0)
    return (x - mean_t) / std_t


# ---------------------------------------------------------------- B
class JpegDataset(torch.utils.data.Dataset):
    def __init__(self, n=512):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.jpg = buf.tobytes()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = cv2.imdecode(np.frombuffer(self.jpg, np.uint8), cv2.IMREAD_COLOR)     # JPEG 解码（CPU）
        small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(small)                                             # uint8 HWC，归一化放到 GPU 上


def loader_throughput(workers, pin, batches=40):
    dl = torch.utils.data.DataLoader(JpegDataset(8 * (batches + 8)), batch_size=8, num_workers=workers, pin_memory=pin,
                                     persistent_workers=False, prefetch_factor=4 if workers else None)
    it = iter(dl)
    for _ in range(4):                      # 跳过 worker 启动
        next(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        x = next(it).cuda(non_blocking=pin)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        n += x.shape[0]
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main():
    lines = []
    ref = a1_python_loop()
    for f in (a2_numpy, a3_gpu_normalize, a4_gpu_all):
        err = (f() - ref).abs().max().item()
        lines.append(f"  {f.__name__:<18s} 与 A1 的最大差异 {err:.2e}")
    print("== A. 预处理一帧（wall 计时，包含 H2D 和 GPU 计算）==")
    r = [benchmark(a1_python_loop, name="A1 Python 逐行循环", iters=100),
         benchmark(a2_numpy, name="A2 numpy 向量化", iters=200),
         benchmark(a3_gpu_normalize, name="A3 CPU resize + GPU 归一化", iters=200),
         benchmark(a4_gpu_all, name="A4 全部在 GPU 上", iters=200)]
    lines.append(compare(r))
    print("\n".join(lines[:3]))
    t_resize = benchmark(lambda: cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR), name="cv2.resize", iters=200)
    s = f"  其中 cv2.resize 480x640→224x224 本身 {t_resize.p50:.3f} ms；A4 与 A3 的差异来自 F.interpolate 的插值实现（未开 antialias）"
    print(s); lines.append(s)

    print("\n== B. DataLoader（每个样本 JPEG 解码 + resize，batch=8；单位：样本/秒）==")
    lines.append("\n== B. DataLoader（每个样本 JPEG 解码 + resize，batch=8；单位：样本/秒）==")
    ds = JpegDataset()
    t1 = benchmark(lambda: ds[0], name="单个样本", iters=100)
    s = f"  单个样本在主进程中处理 {t1.p50:.3f} ms（解码 + resize）→ 单进程理论上限约 {1000 / t1.p50:.0f} 样本/秒"
    print(s); lines.append(s)
    for w in (0, 2, 4, 8):
        for pin in (False, True):
            thr = loader_throughput(w, pin)
            s = f"  num_workers={w}  pin_memory={str(pin):<5s}  {thr:8.0f} 样本/秒"
            print(s); lines.append(s)
    (OUT / "01_cpu_side.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
