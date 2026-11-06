"""
Lab 05 · 跨架构验证：在 ARM 上导出 jit 模型 + 固定输入/输出，拿到 x86 上复现

用法（两步，分别在两台机器上跑）：

    # 第 1 步：在当前这台机器（ARM）上跑，生成三份产物
    /opt/venv/bin/python3 05_cross_arch_export.py export

    # 第 2 步：把整个 artifacts/ 目录拷到 x86 机器（同一份代码、同一个 torch 版本最好），跑
    python3 05_cross_arch_export.py verify

verify 阶段做的事：加载 export 阶段保存的 jit 模型，喂同一份保存好的输入，
把新算出来的输出和 export 阶段保存的输出做数值对比（abs/rel/cosine），
回答两个问题：
  1. ARM 上 torch.jit.save 出来的 .pt 文件，能不能在 x86 上被 torch.jit.load 正常加载/执行？
  2. 同样的输入，两边算出来的数值是否一致（精度是否漂移）？

产物目录（export 阶段生成）：
    artifacts/model.pt       - jit 模型本体（在 ARM 上 script 出来的）
    artifacts/input.pt       - 固定的随机输入张量（torch.save）
    artifacts/reference.pt   - ARM 上跑出来的输出 + 环境信息（架构/torch版本/dtype等）

这个 toy 模型故意选了几种"容易在不同硬件上产生微小数值差异"的算子组合
（conv + batchnorm + 非线性 + 归约），比只用一个 add 更有代表性——
真实模型的 backbone/depth 里也是这几类算子的重复堆叠。
"""
import json
import os
import platform
import sys

import torch
import torch.nn as nn

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_05")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")
INPUT_PATH = os.path.join(ARTIFACTS_DIR, "input.pt")
REFERENCE_PATH = os.path.join(ARTIFACTS_DIR, "reference.pt")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

SEED = 20260710  # 固定随机种子，保证 export 阶段生成的权重/输入是可复现的


class ToyConvNet(nn.Module):
    """故意混合几类常见算子：Conv + BatchNorm + LeakyReLU + 全局归约。

    选它们的理由：
      - Conv/BatchNorm 在不同硬件的浮点累加顺序、SIMD 实现上最容易出现
        "同样输入、极小数值差异"（一般在 1e-6 ~ 1e-5 量级，属于正常的浮点误差，
        不是 bug）。
      - 全局 mean/sum 这种归约操作对累加顺序更敏感，是最能暴露"看似一致、
        实际有微小硬件差异"的算子之一。
    比只测一个 x + 1 更接近真实模型（backbone/depth 里堆的就是这几类算子）。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))
        # 全局池化 + flatten，输出一个定长向量，方便算 abs/rel/cosine
        return y.mean(dim=(2, 3))


def cmd_export() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    model = ToyConvNet()
    model.eval()  # 必须 eval()，否则 BatchNorm 用的是 batch 统计量，不是可复现的固定参数

    scripted = torch.jit.script(model)

    torch.manual_seed(SEED + 1)
    example_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        reference_output = scripted(example_input)

    scripted.save(MODEL_PATH)
    torch.save(example_input, INPUT_PATH)
    torch.save(reference_output, REFERENCE_PATH)

    meta = {
        "exported_on_machine": platform.machine(),      # 例如 aarch64
        "exported_on_platform": platform.platform(),
        "torch_version": torch.__version__,
        "input_shape": list(example_input.shape),
        "input_dtype": str(example_input.dtype),
        "output_shape": list(reference_output.shape),
        "output_dtype": str(reference_output.dtype),
        "seed": SEED,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"[export] 在 {meta['exported_on_machine']} 上导出完成")
    print("=" * 70)
    print(f"模型:          {MODEL_PATH}")
    print(f"固定输入:      {INPUT_PATH}  shape={meta['input_shape']} dtype={meta['input_dtype']}")
    print(f"参考输出:      {REFERENCE_PATH}  shape={meta['output_shape']} dtype={meta['output_dtype']}")
    print(f"环境信息:      {META_PATH}")
    print(f"\n参考输出前几个数值（用于快速目视核对）:\n{reference_output.flatten()[:8]}")
    print(
        f"\n下一步：把整个目录\n  {ARTIFACTS_DIR}\n"
        f"拷到 x86 机器（保持 05_cross_arch_export.py 和 artifacts_05/ 同一相对路径），"
        f"然后在 x86 上跑：\n  python3 05_cross_arch_export.py verify\n"
    )


def cmd_verify() -> None:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(INPUT_PATH) and os.path.exists(REFERENCE_PATH)):
        print(
            f"缺少产物文件，请先在导出机器上跑 `python3 {os.path.basename(__file__)} export`，\n"
            f"再把整个 {os.path.basename(ARTIFACTS_DIR)}/ 目录拷到这台机器、放在脚本同级目录下。"
        )
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    print("=" * 70)
    print(f"[verify] 当前机器: {platform.machine()} ({platform.platform()})")
    print(f"         导出机器: {meta['exported_on_machine']} ({meta['exported_on_platform']})")
    print(f"         导出时 torch 版本: {meta['torch_version']}  当前 torch 版本: {torch.__version__}")
    print("=" * 70)

    # 步骤 1：加载阶段本身能不能成功——这一步验证的是"跨架构的 .pt 文件能不能被
    # torch.jit.load 正常解析"，如果模型引用了自定义 CUDA 算子而对应 .so 在当前机器
    # 没有编译/加载，这一步会直接报错（这是 01_what_is_torchscript.md 1.2b 节提到的
    # "自定义算子 .so 是架构相关"的真实体现，本 toy 模型没有自定义算子，所以不会触发）。
    try:
        loaded_model = torch.jit.load(MODEL_PATH)
        print("[1/3] torch.jit.load 加载成功 —— 跨架构加载本身没有问题")
    except Exception as e:  # noqa: BLE001 - 教学演示，故意宽泛捕获来展示失败模式
        print(f"[1/3] torch.jit.load 加载失败: {e}")
        print("      如果这里失败，通常是自定义算子(.so)在当前架构上缺失/未编译，")
        print("      不是 TorchScript IR 本身跨架构的问题——回去检查 load_library 相关依赖。")
        sys.exit(1)

    example_input = torch.load(INPUT_PATH)
    reference_output = torch.load(REFERENCE_PATH)

    with torch.no_grad():
        new_output = loaded_model(example_input)

    print(f"[2/3] 用同一份输入 (shape={list(example_input.shape)}) 重新推理完成")

    # 步骤 2：数值对比 —— abs / rel / cosine 三个指标，和本仓库
    # torch2engine 的 PTH vs TRT 验证脚本用的是同一套指标口径
    diff = (new_output - reference_output).abs()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    denom = reference_output.abs().clamp_min(1e-8)
    rel_max = (diff / denom).max().item()
    rel_mean = (diff / denom).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        new_output.flatten().double().unsqueeze(0),
        reference_output.flatten().double().unsqueeze(0),
    ).item()

    print("[3/3] 数值对比 (当前机器算出的输出 vs 导出机器保存的参考输出):")
    print(f"      abs_max  = {abs_max:.3e}")
    print(f"      abs_mean = {abs_mean:.3e}")
    print(f"      rel_max  = {rel_max:.3e}")
    print(f"      rel_mean = {rel_mean:.3e}")
    print(f"      cosine   = {cos:.8f}")

    same_arch = platform.machine() == meta["exported_on_machine"]
    print(
        f"\n参考输出前几个数值: {reference_output.flatten()[:8]}\n"
        f"当前输出前几个数值: {new_output.flatten()[:8]}"
    )

    print("\n" + "-" * 70)
    if same_arch:
        print(
            f"注意：当前机器架构 ({platform.machine()}) 和导出机器架构\n"
            f"({meta['exported_on_machine']}) 相同——这次跑的不是真正的跨架构验证，\n"
            f"只是同架构自检（值应该完全相等，cosine=1.0，abs_max≈0）。\n"
            f"要做真正的跨架构验证，需要把 artifacts_05/ 拷到架构不同的机器上再跑 verify。"
        )
    else:
        print(
            f"这是真正的跨架构验证：模型在 {meta['exported_on_machine']} 上导出，\n"
            f"在 {platform.machine()} 上加载执行。\n"
        )
        if abs_max < 1e-3 and cos > 0.9999:
            print(
                "结论：数值高度一致，属于正常的浮点误差范围\n"
                "（不同 CPU/编译器的 SIMD 实现、累加顺序不同，会有 1e-6~1e-4 量级的\n"
                " 正常浮点误差，这不代表模型错了）。TorchScript 跨架构可用性验证通过。"
            )
        else:
            print(
                "结论：数值差异超出正常浮点误差范围，需要进一步排查——\n"
                "常见原因：(a) 两边 torch 版本不一致导致某些算子实现变化；\n"
                "(b) 模型引用了自定义算子，两边编译的 .so 实现不一致；\n"
                "(c) 两边默认 dtype/线程数设置不同。"
            )
    print("-" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("export", "verify"):
        print(f"用法: python3 {os.path.basename(__file__)} [export|verify]")
        sys.exit(1)

    if sys.argv[1] == "export":
        cmd_export()
    else:
        cmd_verify()
