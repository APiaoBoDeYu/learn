"""教程共享组件。

各章 demo 用法：
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from common import benchmark, compare, ToyVLA, SlowPipeline, PipelineConfig
"""
import os

# Thor 环境：v1.0.0 镜像的 triton 3.5 自带的包里缺 ptxas，torch.compile / triton 都要用到它；
# v1.3.0 镜像的 triton 3.8 自带 ptxas-blackwell，不设也能运行，保留此设置无副作用（见 plan/env_report.md）
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

from .bench_utils import Stats, benchmark, compare, cuda_timer, region  # noqa: E402
from .env_info import env_snapshot, print_env  # noqa: E402
from .models import ManySmallOps, TinyCNN, TinyTransformer, ToyVLA  # noqa: E402
from .pipeline import PipelineConfig, SlowPipeline  # noqa: E402

__all__ = [
    "Stats", "benchmark", "compare", "cuda_timer", "region",
    "env_snapshot", "print_env",
    "TinyCNN", "TinyTransformer", "ManySmallOps", "ToyVLA",
    "PipelineConfig", "SlowPipeline",
]
