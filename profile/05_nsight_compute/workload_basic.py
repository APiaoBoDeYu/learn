"""01_ncu_basics.sh 用的小负载：几种不同的 kernel + NVTX 范围，方便演示各种过滤选项。

    warmup:   x*2 跑 3 次
    measure:  x*2（8 MB，能放进 32 MB 的 L2）、x.sum()、64 MB 的 y*2、matmul
"""
import torch

x = torch.randn(2 * 1024 * 1024, device="cuda")       # 8 MB
y = torch.randn(16 * 1024 * 1024, device="cuda")      # 64 MB
a = torch.randn(1024, 1024, device="cuda", dtype=torch.half)

torch.cuda.nvtx.range_push("warmup")
for _ in range(3):
    x * 2
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("measure")
for _ in range(3):
    x * 2            # 小数组：多次重复时数据在 L2 里
x.sum()
y * 2
a @ a
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
