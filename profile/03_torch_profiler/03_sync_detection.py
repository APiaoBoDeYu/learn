"""03-03 同步点检测：torch.cuda.set_sync_debug_mode。

PyTorch 会在发生 CPU-GPU 同步的地方（.item()、.cpu()、print(tensor)、nonzero、布尔索引等）
发出警告（"warn"）或直接抛异常（"error"）。不依赖 CUPTI，在 Thor 上可以正常使用。

运行：/opt/venv/bin/python 03_torch_profiler/03_sync_detection.py
"""
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import PipelineConfig, SlowPipeline  # noqa: E402

ROOT = str(Path(__file__).resolve().parents[1]) + "/"


def where() -> str:
    """找到触发同步的、我们自己代码中的那一行。"""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if fr.filename.startswith(ROOT) and "03_sync_detection" not in fr.filename:
            return f"{fr.filename.replace(ROOT, '')}:{fr.lineno}  {fr.line.strip()[:70]}"
    return "?"


def count_syncs(cfg: PipelineConfig, steps: int = 5) -> Counter:
    pipe = SlowPipeline(cfg)
    for i in range(3):
        pipe.step(i)
    torch.cuda.synchronize()
    hits = Counter()
    orig = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" in str(message) and "prototype feature" not in str(message):
            hits[where()] += 1
        else:
            orig(message, category, filename, lineno, file, line)

    warnings.showwarning = hook
    warnings.simplefilter("always")           # 同一位置每次都报，便于计数
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for i in range(steps):
            pipe.step(i)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = orig
    return Counter({k: v / steps for k, v in hits.items()})


def main():
    for tag, cfg in [("slow", PipelineConfig.slow()), ("fast", PipelineConfig.fast())]:
        hits = count_syncs(cfg)
        print(f"\n=== {tag}：每步 {sum(hits.values()):.0f} 次同步 ===")
        for loc, n in hits.most_common():
            print(f"  {n:5.1f}/step  {loc}")

    print("\n=== 常见的隐式同步写法（error 模式下会直接抛异常）===")
    x = torch.randn(1000, device="cuda")
    cases = {
        "x.sum().item()": lambda: x.sum().item(),
        "float(x.max())": lambda: float(x.max()),
        "x.cpu()": lambda: x.cpu(),
        "x[x > 0]（布尔索引）": lambda: x[x > 0],
        "torch.nonzero(x)": lambda: torch.nonzero(x),
        "if x.sum() > 0: ...": lambda: bool(x.sum() > 0),
        "x.tolist()": lambda: x.tolist(),
        "torch.where(x > 0, x, 0)（无同步）": lambda: torch.where(x > 0, x, 0),
        "x.clamp(min=0)（无同步）": lambda: x.clamp(min=0),
    }
    torch.cuda.set_sync_debug_mode("error")
    for name, fn in cases.items():
        try:
            fn()
            print(f"  ✅ 不同步   {name}")
        except RuntimeError as e:
            msg = str(e).split("\n")[0]
            print(f"  ❌ 同步     {name:<36s} {msg[:60]}")
    torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    main()
