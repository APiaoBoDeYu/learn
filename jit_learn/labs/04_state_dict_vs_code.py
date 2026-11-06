"""
Lab 04 · 权重和分支信息去哪了：state_dict vs .code，运行时 if vs 构造时 if

跑法：
    /opt/venv/bin/python3 04_state_dict_vs_code.py

对应文档：
    jit_learn/02_anatomy_of_pt_file.md, jit_code/skill/jit2model.md (Step 6)
    jit_learn/04_scripted_vs_python_semantics.md（§4.1 分支信息去哪了）

这个 lab 分两部分，主题都是"信息去哪了"：
  1-5. 只需要标准 torch.load，不需要 covariant 环境，直接读真实的
       unified 模型训练 checkpoint，验证 jit2model_guide.md 里
       "checkpoint 顶层是 module，不是直接 state_dict"
       "SE 模块键名有 0/2 的偏移"这些说法。
  6.   只需要 torch，验证 04_scripted_vs_python_semantics.md 的说法：
       跟 Tensor 运行时值有关的 if 会被完整保留进 TorchScript IR；
       只由构造时 Python 对象决定、构造完就不再变化的 if，
       根本不会以"分支"的形式进入编译器。
"""

import torch
import torch.nn as nn

CKPT_PATH = "/root/brain/brain-service/torch2engine/models/unified/model_itr_11800/model_itr_11800.pth"


class RuntimeBranch(nn.Module):
    """if 条件依赖 Tensor 运行时值 (x.shape[0])：分支会被保留进 IR。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 1:
            return x.mean(dim=0, keepdim=True)
        else:
            return x


class ConstructTimeBranch(nn.Module):
    """if 条件依赖构造时的 Python bool：分支在 __init__ 阶段就跑完并消失。"""

    def __init__(self, use_relu: bool):
        super().__init__()
        if use_relu:
            self._act = nn.ReLU()
        else:
            self._act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._act(x)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    section("1. checkpoint 顶层结构：state_dict 藏在哪个 key 下")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"ckpt 的类型: {type(ckpt)}")
    print(f"顶层 key: {list(ckpt.keys())}")
    print(
        "\n验证 jit2model_guide.md 的说法：'state_dict 在 ckpt['module'] 里，不是直接 ckpt'。\n"
        "顶层还有 optimizer/lr_scheduler/global_step/ema_module/grad_scaler 等训练状态——\n"
        "这些是训练框架自己存的，跟模型结构无关，重建模型时完全不需要它们\n"
        "（除了 ema_module，如果训练用了 EMA，推理应该用 ema_module 而不是 module，\n"
        " 这正是本仓库 commit 'd072ae66 解决backbone depth torch模型与框架模型有偏差，\n"
        " 使用了ema参数' 记录的真实教训）。\n"
    )

    sd = ckpt["module"]
    print(f"ckpt['module'] 的类型: {type(sd)}，共 {len(sd)} 个键")

    section("2. RegNet backbone 的键名结构")
    regnet_keys = [k for k in sd if "_backbone_regnet_handler" in k]
    print(f"和 backbone regnet 相关的键: {len(regnet_keys)} 个，前 10 个：")
    for k in regnet_keys[:10]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:75s} {shape}")

    section("3. SE (squeeze-excite) 模块的键名偏移——jit2model_guide.md 里的经典坑")
    se_keys = [k for k in sd if "._core.2._f." in k and "s1.b1" in k]
    print("s1.b1 这个 residual block 里 SE 分支 (._core.2._f.*) 的键：")
    for k in sorted(se_keys):
        v = sd[k]
        print(f"  {k:75s} {tuple(v.shape)}")
    print(
        "\n观察：checkpoint 里 SE 分支只有 _f.0 和 _f.1 两个子模块（各自是一个完整的\n"
        "ConvLayer，带 weight+bias）。如果重建模型时用 nn.Sequential(Conv, Act, Conv, Sigmoid)\n"
        "这种写法，Act/Sigmoid 没有参数，PyTorch 的 state_dict 键名会按*子模块在\n"
        "Sequential 里的下标*编号，于是变成 _f.0（对应 checkpoint 的 _f.0）、\n"
        "_f.2（Act 占了下标 1，所以第二个 Conv 变成下标 2，对应 checkpoint 的 _f.1）。\n"
        "这就是 jit2model_guide.md 表格里 '_f.n._conv -> _f.{n*2}' 这条映射规则的来源——\n"
        "不是随便定的规则，是两种模块组织方式（原框架的显式两段 vs Sequential 隐式编号）\n"
        "在下标计数上的必然结果，画出两边的模块列表对比一下就能推出这条规则，不需要死记。\n"
    )

    section("4. 用 state_dict 的键名重建'这是哪个子模块'的直觉")
    print(
        "一个实用技巧：不需要打开任何源码，只看 state_dict 的键名前缀分布，\n"
        "就能对模型的模块层级有一个大致印象。跟 02_inspect_pt_zip.py 里\n"
        "'数 code/ 文件数看模型规模' 是同一种思路，只是这次数的是 state_dict 键名。\n"
    )
    from collections import Counter

    def top_level_prefix(key: str) -> str:
        return key.split(".")[0]

    prefix_counts = Counter(top_level_prefix(k) for k in sd)
    print("按第一段前缀分组的键数量（近似对应 UnifiedModel 的顶层子模块）：")
    for prefix, cnt in prefix_counts.most_common(15):
        print(f"  {cnt:4d}  {prefix}")

    section("5. 直接验证：这份 checkpoint 里到底有多少参数量")
    total_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    total_bytes = sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "numel"))
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"总字节数 (fp32 存储): {total_bytes / 1e6:.1f} MB")
    print(
        "\n对照第 02 章对 unified.pt 里 archive/data/ 的统计（1075 个 tensor storage，\n"
        "合计约 120 MB）——这里的 240.6 MB (fp32 checkpoint) 正好约等于 120 MB 的两倍，\n"
        "和 'unified.pt 是 fp16 推理模型（每个数占 2 字节），checkpoint 是 fp32 训练权重\n"
        "（每个数占 4 字节）' 完全吻合。这种'交叉核对总字节数、看是否符合精度减半的\n"
        "算术关系'，是判断'我拿到的两份文件是不是同一个模型的不同阶段产物'的一个\n"
        "低成本 sanity check——不需要真的跑一次完整推理对比，先看数量级对不对。\n"
    )

    section("6. 运行时 if vs 构造时 if：分支信息去哪了")
    print("6a. RuntimeBranch —— if 依赖 x.shape[0]，跟 Tensor 运行时值有关：")
    runtime_module = torch.jit.script(RuntimeBranch())
    print(runtime_module.code)
    print(
        "观察：if/else 两条路径原样出现在 IR 里——编译期不知道调用时 x 的\n"
        "shape[0] 究竟是多少，必须把两条路径都保留，运行时再决定走哪条。\n"
    )

    print("6b. ConstructTimeBranch(use_relu=True) —— if 依赖构造时的 Python bool：")
    construct_module = torch.jit.script(ConstructTimeBranch(use_relu=True))
    print(construct_module.code)
    print(
        "观察：__init__ 里的 if use_relu 完全消失，forward 只剩一行确定调用——\n"
        "因为 use_relu 在 torch.jit.script() 被调用之前就已经在纯 Python 里跑完，\n"
        "编译器看到 self._act 时它已经是一个确定的 nn.ReLU 实例，没有分支可言。\n"
        "这和 02_anatomy_of_pt_file.md/04_scripted_vs_python_semantics.md 里\n"
        "Nonlinearity 的 isinstance(nonlin_args, ...) 分支消失是同一种机制。\n"
    )


if __name__ == "__main__":
    main()
