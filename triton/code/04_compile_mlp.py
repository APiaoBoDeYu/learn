"""Chapter 4: first torch.compile example — a small MLP.

Shows the default `torch.compile` path (Dynamo -> AOTAutograd -> Inductor),
compares eager vs compiled latency across the different `mode=` options, and
inspects the Triton code Inductor generates for the GPU backend.
Run: python3 04_compile_mlp.py
Run with generated-code dump: TORCH_LOGS=output_code python3 04_compile_mlp.py
"""
from common import device_info, check_close, bench_ms

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        out = self.fc2(h)
        return self.ln(out + x)  # residual + norm, like a transformer block


if __name__ == "__main__":
    device_info()
    torch.set_float32_matmul_precision("high")  # allow TF32 tensor cores, as Inductor suggests
    # fp32 输入的 matmul 默认("highest")用严格 fp32 全精度计算，最准但慢。
    # "high" 允许 matmul 走 TF32（TensorFloat-32，Ampere+ GPU 专有格式：指数位和 fp32 相同，
    # 尾数只有10bit）tensor core 路径，速度快很多倍，精度损失通常在 1e-3 量级、可接受。
    # "medium" 更激进，允许 bf16 累加，精度损失更大。
    # torch.compile(Inductor) 检测到 GPU 支持 TF32 但没开这个选项时会打印警告建议开启，即注释里 "as Inductor suggests" 的来源。

    torch.manual_seed(0)
    model = MLP().cuda().eval()  # .eval() 关闭 dropout/batchnorm 的训练态行为（这里模型没用到，但是好习惯）；推理场景

    for batch in [256, 8192]:  # 分别测小/大 batch：torch.compile 的编译产物和输入 shape 相关，不同 batch 会各触发一次独立编译
        print(f"=== batch size {batch} ===")
        x = torch.randn(batch, 1024, device="cuda")

        with torch.no_grad():
            out_eager = model(x)  # 先跑一次纯 eager 输出，作为正确性基准，供下面 check_close 对比

            for mode in ["default", "reduce-overhead", "max-autotune"]:
                # default: 标准编译，兼顾编译耗时和运行速度
                # reduce-overhead: 用 CUDA Graphs 减少 Python/launch 开销，适合小算子、重复密集调用场景
                # max-autotune: 类似 @triton.autotune，为每个融合算子尝试多种 kernel 实现/tile 配置，实测选最快，编译最慢但通常跑得最快
                # a fresh compile per mode: torch.compile caches compiled artifacts
                # keyed by (model, mode, input shapes) under the hood
                compiled_model = torch.compile(model, mode=mode)  # 只是包一层，不会立即编译；返回的是"懒编译"的可调用对象
                out_compiled = compiled_model(x)  # triggers first-time compilation，第一次真正调用才触发 Dynamo 抓图 -> AOTAutograd -> Inductor 生成 kernel
                check_close(out_compiled, out_eager, f"batch={batch} mode={mode}")
                # 正确性校验：逐元素比较 compiled 和 eager 输出是否足够接近（torch.allclose，容差 atol/rtol=1e-2），
                # 这个容差是为了容忍上面开启 TF32 带来的预期内精度损失，不要求逐位相同。
                # 传入的 name 标签把当前 (batch, mode) 组合打印出来，方便在两层循环全部跑完后从日志里
                # 分辨是哪个 batch size、哪种编译 mode 的结果 PASS/FAIL。

                t_eager = bench_ms(lambda: model(x))
                t_compiled = bench_ms(lambda: compiled_model(x))
                speedup = t_eager / t_compiled
                print(f"  eager    : {t_eager:.4f} ms")
                print(f"  compiled : {t_compiled:.4f} ms  (mode={mode}, speedup={speedup:.2f}x)")
                print()
                # Dynamo 会按 (model, mode, 输入shape) 等信息缓存编译产物/guard；不清掉的话，
                # 下一次换 mode 重新 torch.compile 同一个 model 时可能命中/混用旧缓存，
                # 导致测出来的不是这次 mode 真正对应的编译结果，speedup 对比失真。
                # 所以每测完一个 mode 就整体清空 Dynamo 缓存状态，保证下一个 mode 是从头一次干净编译。
                torch._dynamo.reset()  # drop compiled cache before next mode
