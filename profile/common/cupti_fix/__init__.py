"""torch.profiler 在 Thor 上的部分修复（详见 cupti_fix.c 的注释）。

    from common import cupti_fix
    cupti_fix.ensure()   # 放在脚本最开头、import torch 之前：如果没有 LD_PRELOAD，就带上它重新启动当前进程

修复后：CPU 侧的 CUDA runtime API 事件（cudaLaunchKernel / cudaStreamSynchronize / cudaMemcpyAsync）可以采集到；
仍然没有：GPU kernel / memcpy / memset 的执行时间（CUPTI 返回的时间戳为 0，属于平台限制）。
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "libcupti_fix.so"


def build() -> Path:
    if not LIB.exists() or LIB.stat().st_mtime < (HERE / "cupti_fix.c").stat().st_mtime:
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(LIB), str(HERE / "cupti_fix.c"), "-ldl"],
                       check=True)
    return LIB


def active() -> bool:
    return str(LIB) in os.environ.get("LD_PRELOAD", "")


def ensure() -> None:
    """若当前进程没有预加载修复库，则带上 LD_PRELOAD 重新 exec 自己。"""
    if active():
        return
    build()
    env = dict(os.environ)
    env["LD_PRELOAD"] = (str(LIB) + " " + env.get("LD_PRELOAD", "")).strip()
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
