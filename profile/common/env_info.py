"""记录运行环境，保证 benchmark 结果可复现。

jtop 读取失败（没有服务、权限不够）时自动降级，不影响 demo 运行。
"""
from __future__ import annotations

import platform

import torch


def env_snapshot(with_jtop: bool = True) -> dict:
    info: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, cc=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                    gpu_mem_gb=round(p.total_memory / 1e9, 1))
    if with_jtop:
        try:
            from jtop import jtop
            with jtop() as j:
                info["nvpmodel"] = str(j.nvpmodel)
                info["jetson_clocks"] = bool(j.jetson_clocks)
                g = next(iter(j.gpu.values()))
                freq = g.get("freq", {})
                info["gpu_freq_mhz"] = freq.get("cur", 0) // 1000
                info["gpu_governor"] = freq.get("governor")
                info["gpu_temp_c"] = j.temperature.get("gpu", {}).get("temp")
        except Exception as e:  # noqa: BLE001
            info["jtop"] = f"unavailable: {type(e).__name__}"
    return info


def print_env(with_jtop: bool = True) -> dict:
    info = env_snapshot(with_jtop)
    print("[env] " + "  ".join(f"{k}={v}" for k, v in info.items()))
    return info


if __name__ == "__main__":
    print_env()
