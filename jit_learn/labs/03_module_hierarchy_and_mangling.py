"""
Lab 03 · 模块层级、mangling，以及"折叠噪音行"的小工具

跑法：
    /opt/venv/bin/python3 03_module_hierarchy_and_mangling.py

对应文档：jit_learn/03_reading_generated_code.md

这个 lab 做两件事：
  1. 用一个 toy 模型复现"同一个类因为构造参数不同被 mangle 成多个版本"的现象
     （比 unified.pt 小几百倍，但机制完全一样，方便你一步步跟着看）。
  2. 提供一个 fold_noise() 函数，把第 03 章讲的"module = self.module" /
     "getattr(x, 'N')" 这类噪音行自动折叠，输出一份"骨架视图"——
     你可以把这个函数用在任何一份 000_predict_*.py 上。
"""
import re
import textwrap

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. 复现 mangle 现象
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """和 covariant 的 ConvLayer 类似：同一个类，不同 kernel_size/channel 数
    会让 TorchScript 编译出不同版本（体现在 shape 相关的中间计算上）。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ToyNet(nn.Module):
    """三个 ConvBlock，故意用不同的 kernel_size，观察 mangle 现象。"""

    def __init__(self):
        super().__init__()
        self.block_a = ConvBlock(3, 8, kernel_size=3)
        self.block_b = ConvBlock(8, 8, kernel_size=1)   # 不同 kernel_size
        self.block_c = ConvBlock(3, 8, kernel_size=3)   # 和 block_a 参数完全相同

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.block_a(x)
        b = self.block_b(a)
        c = self.block_c(x)
        return a + b + c


def demo_mangling():
    print("=" * 70)
    print("1. 同一个 ConvBlock 类，被用了 3 次，看编译器怎么处理")
    print("=" * 70)
    scripted = torch.jit.script(ToyNet())

    # torch.jit.save 到内存 buffer，再用 zipfile 直接看 code/ 目录，
    # 复现 jit_learn/02_inspect_pt_zip.py 里做的事，但用的是我们自己的小模型
    import io
    import zipfile

    buf = io.BytesIO()
    torch.jit.save(scripted, buf)
    buf.seek(0)
    z = zipfile.ZipFile(buf)
    # 顶层类 ToyNet 会被存成 archive/code/__torch__.py（不带任何 mangle 后缀，
    # 因为整个文件里只有一个 ToyNet）。它引用的 ConvBlock 有两种编译结果：
    # 第一次遇到的版本（block_a，之后 block_c 参数相同、复用同一份）存成
    # archive/code/__torch__/ConvBlock.py 风格（不带 mangle，是"基准版"）；
    # 遇到第二种不同参数组合（block_b）时，才会分配一个新的 mangle 后缀。
    code_files = [
        n for n in z.namelist()
        if n.startswith("archive/code/") and n.endswith(".py") and "/torch/nn/" not in n
    ]
    print("archive/code/ 下属于本文件 ToyNet/ConvBlock 的条目：")
    for n in sorted(code_files):
        print(" ", n)

    toy_net_code = z.read("archive/code/__torch__.py").decode()
    print("\n--- archive/code/__torch__.py（ToyNet + 未被 mangle 的 ConvBlock 基准版）---")
    print(toy_net_code)

    mangled_files = [n for n in code_files if "___torch_mangle_" in n]
    if mangled_files:
        mangled_code = z.read(mangled_files[0]).decode()
        print(f"--- {mangled_files[0]}（block_b 用的 mangle 版本）---")
        print(mangled_code)

    print(
        "关键观察 1：ToyNet.forward 里 'block_a : __torch__.ConvBlock' 和\n"
        "'block_c : __torch__.ConvBlock' 指向同一个类型注解——block_a 和 block_c\n"
        "（构造参数完全相同）确实共享了同一份编译产物；只有 block_b（kernel_size 不同）\n"
        "被单独分配了一个 ___torch_mangle_N 版本。'mangle 版本数 = 不同构造参数组合数'，\n"
        "这就是 unified.pt 里 372 个 nonlinearity / 345 个 conv_layer 版本背后的机制——\n"
        "并不是模型有 372 个不同的激活函数逻辑，而是有 372 种不同的构造参数组合。\n"
    )

    conv_code = z.read("archive/code/__torch__/torch/nn/modules/conv.py").decode()
    print("--- archive/code/__torch__/torch/nn/modules/conv.py（nn.Conv2d 的编译结果，节选）---")
    print("\n".join(conv_code.splitlines()[:18]))
    print(
        "\n关键观察 2：注意 'kernel_size : Final[Tuple[int, int]] = (3, 3)' 这一行——\n"
        "构造时传入的 kernel_size 数值被直接烧录成了 Final 常量，写进了这个类自己的\n"
        "编译产物里。这正是第 04 章要讲的'构造参数变成不可更改的字面常量'的第一手证据：\n"
        "不需要猜测，直接在 .code 里就能看到 Final[...] 声明和被烧录的具体数值。\n"
    )

    print(
        "\n关键观察 3：路径本身怎么读——去掉前缀 'archive/code/__torch__/'、把 '/' 换成 '.'，\n"
        "剩下的就是这个类在源码里的原始导入路径，可以直接拿去源码树里 find/grep：\n"
        "  archive/code/__torch__/torch/nn/modules/conv.py\n"
        "    -> torch.nn.modules.conv                      (PyTorch 内置类)\n"
        "  archive/code/__torch__/covariant/model/perception/unified/backbone/handlers/backbone_fpn_handler.py\n"
        "    -> covariant.model.perception.unified.backbone.handlers.backbone_fpn_handler\n"
        "    -> 对应源码文件 covariant/model/perception/unified/backbone/handlers/backbone_fpn_handler.py\n"
        "带 ___torch_mangle_N 后缀的文件同理，先去掉这一段文件名后缀再做上述映射——\n"
        "mangle 编号本身不含语义（不是'第几个实例'、不是创建顺序），只是编译器内部的\n"
        "全局去重计数器，'哪个 mangle 对应哪个具体子模块实例'这件事无法从路径反推，\n"
        "要回 data.pkl 里的属性路径（比如 self._backbone_fpn_handler._lateral_fpn_cores.0）\n"
        "或者加载后用 type(submodule).__name__ / submodule.original_name 对号。\n"
    )


# ---------------------------------------------------------------------------
# 2. 折叠噪音行的小工具：把 000_predict_backbone.py 这种 IR 反解析代码
#    "module = self.module" / "getattr(x, 'N')" 类的重复访问折叠掉
# ---------------------------------------------------------------------------

def fold_noise(code: str) -> str:
    """把 IR 反解析代码里的两类结构性噪音折叠成更接近人读的属性访问链：

    1. `moduleN = self.module` (N 可为空) -> 全部替换成对 `self.module` 的直接引用
    2. `varname = getattr(x, "K")` -> 替换成 `varname = x[K]` 风格的下标访问

    这不是一个通用的 TorchScript IR 反解析美化工具，只是教学用的最小示例，
    专门针对 covariant IR 反解析代码里最常见的两种噪音模式。
    """
    lines = code.splitlines()
    # 第一步：找出所有形如 "moduleN = self.module" 的行，统一替换成 self.module
    module_alias_pattern = re.compile(r"^(\s*)(module\d*)\s*=\s*self\.module\s*$")
    aliases = {}
    kept_lines = []
    for line in lines:
        m = module_alias_pattern.match(line)
        if m:
            aliases[m.group(2)] = "self.module"
            continue  # 丢掉这一行，它只是重复声明
        kept_lines.append(line)

    # 第二步：把用到这些 alias 的地方替换回 self.module
    getattr_pattern = re.compile(r'getattr\((\w+),\s*"(\d+)"\)')
    out_lines = []
    for line in kept_lines:
        for alias, target in aliases.items():
            line = re.sub(rf"\b{alias}\b", target, line)
        # getattr(x, "0") -> x[0]  （ModuleList 按位置访问的还原）
        line = getattr_pattern.sub(lambda m: f"{m.group(1)}[{m.group(2)}]", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def demo_fold_noise():
    print("=" * 70)
    print("2. 用 fold_noise() 折叠 000_predict_backbone.py 的真实内容")
    print("=" * 70)

    raw = textwrap.dedent(
        """\
        def predict_backbone(self,
            rgbs: Tensor) -> Dict[str, Tensor]:
          _0 = __torch__.covariant.common.learning_utils.torch_jit_utils.jit_compatible_tensor
          module = self.module
          _backbone_fpn_handler = module._backbone_fpn_handler
          _lateral_fpn_cores = _backbone_fpn_handler._lateral_fpn_cores
          _00 = getattr(_lateral_fpn_cores, "0")
          _conv = _00._conv
          weight = _conv.weight
          module0 = self.module
          _backbone_fpn_handler0 = module0._backbone_fpn_handler
          module1 = self.module
          _backbone_regnet_handler = module1._backbone_regnet_handler
          inputs = torch.unsqueeze(rgbs, 0)
          _1 = (_backbone_regnet_handler).forward(inputs, )
          _2, _3, _4, _5, _6, = _1
          _7 = (_backbone_fpn_handler0).forward(_2, _3, _4, _5, _6, )
          _8, _9, _10, _11, _12, = _7
          x = torch.select(_8, 0, 0)
          x0 = torch.select(_9, 0, 0)
          x1 = torch.select(_10, 0, 0)
          x2 = torch.select(_11, 0, 0)
          x3 = torch.select(_12, 0, 0)
          _13 = {"__len__": _0(5., weight, 4, ), "0": x, "1": x0, "2": x1, "3": x2, "4": x3}
          return _13
        """
    )

    print("--- 原始 IR 反解析代码（来自 jit2engine/jit_code/backbone/000_predict_backbone.py） ---")
    print(raw)
    print("--- fold_noise() 折叠后 ---")
    print(fold_noise(raw))
    print(
        "观察：module/module0/module1 三行重复声明消失了，取而代之的是直接用\n"
        "self.module 引用；getattr(x, \"0\") 变成了 x[0] 这种更接近日常读代码的写法。\n"
        "折叠之后，'RegNet 5 个 blob 喂给 FPN，FPN 输出 5 个特征图去掉 batch 维'\n"
        "这条主线一眼就能看出来，不再被结构性噪音打断视线。\n"
    )


if __name__ == "__main__":
    demo_mangling()
    demo_fold_noise()
