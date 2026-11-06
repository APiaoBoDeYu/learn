"""03-01 torch.profiler 基础：算子表、按 shape 分组、FLOPs、调用栈、导出 trace。

⚠️ Thor 上的限制（见 README §3.8）：拿不到 GPU kernel 的执行时间，CUDA 相关的列为空。
   本脚本开头调用 cupti_fix.ensure()，至少恢复 CPU 侧的 CUDA runtime API 事件（cudaLaunchKernel 等）。

运行：/opt/venv/bin/python 03_torch_profiler/01_basics.py
产物：outputs/basics_trace.json（拖进 https://ui.perfetto.dev 查看）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cupti_fix  # noqa: E402

cupti_fix.ensure()  # 必须在 import torch 之前：带上 LD_PRELOAD 重新启动本进程

import torch  # noqa: E402
from torch._C._profiler import _ExperimentalConfig  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from common import ToyVLA  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    print(f"cupti_fix active: {cupti_fix.active()}")
    model = ToyVLA().cuda().eval()
    img = torch.randn(1, 3, 224, 224, device="cuda")
    tok = torch.randint(0, 32000, (1, 32), device="cuda")
    with torch.no_grad():
        for _ in range(5):          # warmup：不要把 cuDNN 选算法、懒加载 kernel 采进去
            model(img, tok)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,      # 记录输入 shape → 可以按 shape 分组
                 with_flops=True,         # 根据 shape 估算 matmul/conv 的 FLOPs
                 with_stack=True,         # 记录 Python 调用栈 → 可以定位到源码行
                 # 坑：torch 2.9 中只设 with_stack=True 时 FunctionEvent.stack 为空，还需要 verbose=True
                 experimental_config=_ExperimentalConfig(verbose=True),
                 profile_memory=True) as prof:
        with torch.no_grad():
            for _ in range(10):
                with record_function("vla_forward"):   # 自定义区间，会出现在表格和 trace 中
                    model(img, tok)
        torch.cuda.synchronize()

    ka = prof.key_averages()
    print("\n=== 1. 按 CPU 总耗时排序（Top 15）===")
    print(ka.table(sort_by="cpu_time_total", row_limit=15, max_name_column_width=40))

    print("\n=== 2. 按 Self CPU 排序：哪些算子自身的 CPU 开销最大 ===")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=10, max_name_column_width=40))

    print("\n=== 3. group_by_input_shape：同一个算子的不同 shape 分开统计 ===")
    t = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=8,
                                                           max_name_column_width=30, max_shapes_column_width=60)
    print(t)

    print("\n=== 4. FLOPs 统计（with_flops，仅 matmul/conv 类算子）===")
    rows = sorted([e for e in ka if e.flops], key=lambda e: -e.flops)[:6]
    total = sum(e.flops for e in ka if e.flops)
    for e in rows:
        print(f"{e.key:<28s} calls={e.count:4d}  GFLOPs={e.flops / 1e9:8.2f}  ({e.flops / total:5.1%})")
    print(f"{'总计':<28s}            GFLOPs={total / 1e9:8.2f}  → 每次前向 {total / 10 / 1e9:.2f} GFLOPs")

    print("\n=== 5. group_by_stack_n：按 Python 调用栈分组，定位到源码行 ===")
    st = prof.key_averages(group_by_stack_n=8)
    rows = sorted([e for e in st if e.key == "aten::linear"], key=lambda e: -e.cpu_time_total)[:4]
    for e in rows:
        mine = [fr for fr in e.stack if "common/" in fr]       # 只显示我们自己代码里的帧
        print(f"{e.key}  calls={e.count:3d}  cpu_total={e.cpu_time_total / 1000:6.2f}ms  <- {mine[0] if mine else e.stack[:2]}")

    print("\n=== 6. CUDA 侧数据（Thor 上不可用）===")
    n_kernel = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    n_launch = sum(e.count for e in ka if e.key == "cudaLaunchKernel")
    print(f"GPU kernel 事件数: {n_kernel}   cudaLaunchKernel 调用次数: {n_launch}")
    print("→ 能统计到 launch 了多少次 kernel（CPU 侧），但没有 kernel 在 GPU 上的执行时间")

    prof.export_chrome_trace(str(OUT / "basics_trace.json"))
    print(f"\ntrace 已导出: {OUT / 'basics_trace.json'}")


if __name__ == "__main__":
    main()
