"""common 自测：python common/selftest.py

1. 各模型前向一次，打印参数量和输出形状
2. 慢流水线逐个关闭开关，测量每个开关的影响（这张表也是第 08 章的"预期答案"）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import (ManySmallOps, PipelineConfig, SlowPipeline, TinyCNN, TinyTransformer,  # noqa: E402
                    ToyVLA, benchmark, compare, print_env)

FLAGS = ["py_preprocess", "sync_in_loop", "no_pin", "fp32", "eager_small_ops", "realloc"]


def check_models():
    for m, inp in [(TinyCNN(), (torch.randn(2, 3, 224, 224),)),
                   (TinyTransformer(), (torch.randn(2, 128, 512),)),
                   (ManySmallOps(), (torch.randn(1024),)),
                   (ToyVLA(), (torch.randn(2, 3, 224, 224), torch.randint(0, 32000, (2, 32))))]:
        m = m.cuda().eval()
        with torch.no_grad():
            out = m(*[x.cuda() for x in inp])
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{type(m).__name__:<16s} params={n:7.2f}M  out={tuple(out.shape)}")


def check_pipeline():
    slow = PipelineConfig.slow()
    configs = [("slow (all on)", slow)]
    configs += [(f"slow - {f}", slow.replace(**{f: False})) for f in FLAGS]
    configs.append(("fast (all off)", PipelineConfig.fast()))
    results = []
    for name, cfg in configs:
        pipe = SlowPipeline(cfg)
        it = iter(range(10**9))
        results.append(benchmark(lambda: pipe.step(next(it)), name=name, warmup=5, iters=30))
    compare(results)


if __name__ == "__main__":
    print_env()
    check_models()
    print()
    check_pipeline()
