"""01-05 功耗模式（nvpmodel）对延迟的影响。

默认只读：打印当前模式，在当前模式下测一次 ToyVLA fp16 batch=1 的推理延迟。
加 --apply：依次切换到每个功耗模式测量，结束后恢复原模式。
    ⚠️ --apply 会修改宿主机的全局功耗模式（影响机器上的所有程序），切换过程中可能需要几秒才能稳定。
       部分模式切换需要重启才能生效，jtop 会给出提示；本脚本只测量不需要重启的模式。

运行：/opt/venv/bin/python 01_system_monitor/05_power_modes.py [--apply]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import ToyVLA, benchmark  # noqa: E402


def measure() -> str:
    m = ToyVLA().cuda().half().eval()
    img = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.half)
    tok = torch.randint(0, 32000, (1, 32), device="cuda")
    with torch.no_grad():
        st = benchmark(lambda: m(img, tok), warmup=20, iters=100)
    return st.summary()


def gpu_freq(j) -> str:
    f = next(iter(j.gpu.values()))["freq"]
    return f"GPU {f['cur'] // 1000} MHz（min {f['min'] // 1000} / max {f['max'] // 1000}，{f['governor']}）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际切换功耗模式（修改宿主机全局设置）")
    args = ap.parse_args()
    from jtop import jtop
    with jtop() as j:
        orig = str(j.nvpmodel)
        orig_jc = bool(j.jetson_clocks)          # 切换 nvpmodel 可能让 jetson_clocks 锁定的频率失效，结束时要一起恢复
        orig_freq = gpu_freq(j)
        status = dict(zip(j.nvpmodel.models, j.nvpmodel.status))   # jtop：status 为 False 表示该模式需要重启才能生效
        print(f"当前模式: {orig}  jetson_clocks={orig_jc}  {orig_freq}")
        print(f"可选模式（是否无需重启）: {status}")
        print(f"[{orig}] {measure()}")
        if not args.apply:
            print("\n（只读模式。加 --apply 会依次切换各模式测量，结束后恢复。）")
            return
        try:
            for mode, ok in status.items():
                if mode == orig:
                    continue
                if not ok:
                    print(f"\n跳过 {mode}：需要重启才能生效")
                    continue
                print(f"\n切换到 {mode} ...")
                j.nvpmodel = mode
                time.sleep(5)  # 等待调频稳定
                print(f"[{j.nvpmodel}] jetson_clocks={bool(j.jetson_clocks)}  {gpu_freq(j)}")
                print(f"[{mode}] {measure()}")
        finally:
            j.nvpmodel = orig
            time.sleep(5)
            if bool(j.jetson_clocks) != orig_jc:
                j.jetson_clocks = orig_jc
                time.sleep(5)
            print(f"\n已恢复：{j.nvpmodel}  jetson_clocks={bool(j.jetson_clocks)}  {gpu_freq(j)}")
            print(f"（运行前：{orig}  jetson_clocks={orig_jc}  {orig_freq}）")
            print(f"[恢复后 {j.nvpmodel}] {measure()}")


if __name__ == "__main__":
    main()
