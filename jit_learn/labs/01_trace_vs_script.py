"""
Lab 01 · trace vs script：亲眼看到"分支消失"和"trace 固化"

跑法：
    /opt/venv/bin/python3 01_trace_vs_script.py

对应文档：jit_learn/01_what_is_torchscript.md, jit_learn/04_scripted_vs_python_semantics.md
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 实验 A：Python 构造阶段选好子模块类型 —— "if" 根本不会进入 forward
#
# 这是 covariant 里 Nonlinearity 的真实写法（见
# covariant/common/learning_utils/layers/nonlinearity.py:74-84）：
# __init__ 用 isinstance 分支选出 self._nonlin 到底是 nn.Identity 还是
# nn.LeakyReLU，forward 里只有一行 self._nonlin(x)。这比"forward 里判断 flag"
# 更彻底：TorchScript 编译 forward 时压根没有 if 要处理，因为分支早就在
# 普通 Python 阶段跑完并"物化"成了一个具体的子模块。
# ---------------------------------------------------------------------------

class NonlinLikeCovariant(nn.Module):
    """模拟 covariant 的 Nonlinearity：用哪种激活由构造参数决定，
    决定的动作发生在 __init__（普通 Python），forward 只管调用。
    """

    def __init__(self, use_identity: bool):
        super().__init__()
        self._nonlin: nn.Module
        if use_identity:                       # <- 这个 if 在 __init__ 里，跑在纯 Python 阶段
            self._nonlin = nn.Identity()
        else:
            self._nonlin = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nonlin(x)                 # <- forward 里没有任何 if


def demo_constant_propagation():
    print("=" * 70)
    print("实验 A：'用哪种激活'的 if 判断，会不会出现在编译后的 forward 代码里？")
    print("=" * 70)

    model_identity = torch.jit.script(NonlinLikeCovariant(use_identity=True))
    model_leaky = torch.jit.script(NonlinLikeCovariant(use_identity=False))

    print("\n--- use_identity=True 编译后的 forward 代码 ---")
    print(model_identity.forward.code)

    print("--- use_identity=False 编译后的 forward 代码 ---")
    print(model_leaky.forward.code)

    print(
        "观察：两份 forward 代码长得一模一样，都只是 'return (_nonlin).forward(x)'。\n"
        "forward 里从来没有 if，因为分支从没进入 TorchScript 编译器的视野——\n"
        "它在 __init__ 跑的那一刻（普通 CPython 执行）就已经决定完了，\n"
        "结果是 self._nonlin 这个属性被绑定成两种不同类型的子模块之一。\n"
        "TorchScript 只看到'这里有个子模块，调用它的 forward'，看不到、也不需要知道\n"
        "当初为什么选了 Identity 而不是 LeakyReLU——这正是 backbone FPN 里\n"
        "conv_spec.as_no_nonlinearity() 在 unified.pt 的 IR 反解析代码里彻底消失的真实机制。\n"
        "\n"
        "如果想确认某个位置具体是哪种类型，要去读*那个子模块自己*的 .code\n"
        "（model_identity._nonlin.original_name 是 'Identity'，\n"
        " model_leaky._nonlin.original_name 是 'LeakyReLU'）——\n"
        f"identity 分支: {model_identity._nonlin.original_name}\n"
        f"leaky 分支:    {model_leaky._nonlin.original_name}\n"
    )


# ---------------------------------------------------------------------------
# 实验 B：trace 会把"看起来像变量"的东西固化成字面常量
# ---------------------------------------------------------------------------

class ToyReshape(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]              # 这是"跟输入相关"的量……
        return x.reshape(b, -1)     # ……但 trace 只录制这一次的具体取值


def demo_trace_freezes_shape():
    print("=" * 70)
    print("实验 B：trace 出来的模型，输入 batch=5 时和 batch=3 时的代码有什么不同？")
    print("=" * 70)

    model = ToyReshape()

    example_5 = torch.randn(5, 2, 3)
    traced_5 = torch.jit.trace(model, example_5)
    print("\n--- 用 batch=5 的样例 trace 出来的代码 ---")
    print(traced_5.code)

    # 用 trace=5 的模型去跑一个不同 batch 的输入，看会不会报错/算错
    example_3 = torch.randn(3, 2, 3)
    try:
        out = traced_5(example_3)
        print(f"用 batch=3 的输入喂给 (batch=5 trace 出来的) 模型: 输出 shape = {tuple(out.shape)}")
        print(
            "如果输出 shape 仍然是 (3, 6) 说明这次 reshape 的 batch 维恰好被正确处理了；\n"
            "但工程上不能依赖这种'看起来没问题'——一定要用 script 或显式动态 shape 支持来保证，\n"
            "而不是相信 trace 出来的模型对所有形状都通用。"
        )
    except Exception as e:  # noqa: BLE001 - 教学演示，故意宽泛捕获来展示失败模式
        print(f"用 batch=3 的输入喂给 (batch=5 trace 出来的) 模型: 报错 -> {e}")

    print(
        "\n延伸：project_tools/PITFALLS.md 第17条'当前batch/cams支持是静态shape engine'\n"
        "背后的一层可能原因就是这里——如果某个环节的确是靠 trace 获得的，\n"
        "该环节对'输入形状变化'的鲁棒性要专门验证，不能想当然。\n"
    )


# ---------------------------------------------------------------------------
# 实验 B2：分界线到底在哪——"取 shape 传给 tensor 算子" vs
#          "取 shape 传给 python if"，trace 出来的结果天差地别
#
# 实验 B 里 b = x.shape[0] 接下来传给了 x.reshape(b, -1)：reshape 是个
# tensor 算子，所以 trace 会把"取 shape 这个动作"本身记成图节点
# (aten::size)，运行时重新求值，batch 换了也能算对。
#
# 但如果同样是 b = x.shape[0]，接下来传给的是 python 的 if b > 1：
# b > 1 求值成一个 python bool，不是 tensor 算子，trace 没有对应节点可记，
# 只能记录"这次判断完走的是哪条路"，另一条路连同判断动作本身一起消失。
# 这才是 01_what_is_torchscript.md 里"不作用于张量的控制流"这句话
# 真正想区分的界线：不是"跟 shape 有没有关系"，是"这个值最终有没有
# 变成 tensor 算子的参数，还是被转成了 python 原生的分支跳转条件"。
# ---------------------------------------------------------------------------

class ToyIfBranch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if b > 1:            # b > 1 是 python bool，不是 tensor 比较算子
            return x * 2
        else:
            return x * 100


def demo_trace_vs_tensor_op_boundary():
    print("=" * 70)
    print("实验 B2：x.shape[0] 传给 reshape（tensor 算子）vs 传给 if（python 分支）")
    print("=" * 70)

    # --- 对照组：实验 B 的 reshape 用法，看 graph 里到底是不是死常量 ---
    reshape_model = ToyReshape()
    traced_reshape = torch.jit.trace(reshape_model, torch.randn(5, 2, 3))
    print("\n--- reshape 版本的 graph（注意 aten::size 是运行时会重新执行的算子，不是常量）---")
    print(traced_reshape.graph)
    out3 = traced_reshape(torch.randn(3, 2, 3))
    print(f"喂 batch=3 的输入 -> 输出 shape = {tuple(out3.shape)}  (说明 batch 维是动态取的，不是拍死的)")

    # --- 实验组：if 用法，用 batch=5（5 > 1 为真）去 trace ---
    if_model = ToyIfBranch()
    traced_if = torch.jit.trace(if_model, torch.randn(5, 2, 3))
    print("\n--- if 版本编译后的 forward 代码（用 batch=5 trace，5>1 为真，走的是 *2 分支）---")
    print(traced_if.code)

    # 用 batch=1（本该走 else，应该是 *100）去验证：分支是否真的被拍死了
    out_b1 = traced_if(torch.ones(1, 2, 3))
    got = out_b1[0, 0, 0].item()
    print(
        f"用 batch=1 的输入喂给它（若 if 逻辑真的生效，1>1 为假，应该走 else 得到 *100）：\n"
        f"实际输出 = {got}  (若是 *2 → 2.0；若是 *100 → 100.0)"
    )

    print(
        "\n观察：traced_if.code 里根本没有 if/else，只剩一行 torch.mul(x, 2)——\n"
        "else 分支（*100）从 trace 那一刻起就已经从图里永久消失，不管之后喂什么 batch，\n"
        "这个模型都只会做 *2。而 reshape 版本因为 aten::size 是真实算子，\n"
        "batch=3 依然能算出正确的 (3, 6)。\n"
        "\n"
        "结论：'trace 会不会把这个值拍死'，取决于这个值接下来的去向——\n"
        "  - 去向是 tensor 算子的参数（reshape/expand/view 的 shape 实参等）：\n"
        "    trace 记录的是取值算子本身，运行时可以重新求值，不是死常量。\n"
        "  - 去向是 python 原生的 if/for 条件：trace 只能记录这次判断的结果，\n"
        "    判断动作和没走到的分支，永久不存在于导出后的模型里。\n"
        "PyTorch 对后一种情况会主动抛出 TracerWarning\n"
        "（'Converting a tensor to a Python boolean might cause the trace to be incorrect'），\n"
        "这行警告就是'致命限制'那句话的官方措辞版。\n"
    )


# ---------------------------------------------------------------------------
# 实验 C：script 模型可以有多个具名方法，trace 模型不能——这是判断
#         unified.pt 顶层是哪种模式的直接证据
# ---------------------------------------------------------------------------

class MultiEntryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)
    #用 torch.jit.script(model) 做 script 化（而非 trace）时，编译器默认只会追踪并编译 forward 方法。如果你还想让 predict_a、predict_b 这类非 forward 的方法也被编译进最终的ScriptModule，就必须显式标注 @torch.jit.export
    @torch.jit.export
    def predict_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x) * 2

    @torch.jit.export
    def predict_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x) + 1


def demo_multi_entry_methods():
    print("=" * 70)
    print("实验 C：为什么能有 predict_backbone / predict_depth / predict_segm 三个入口？")
    print("=" * 70)

    scripted = torch.jit.script(MultiEntryModel())
    print("\nscripted 模型除了 forward，还能单独调用具名方法（需要 @torch.jit.export）：")
    print("predict_a.code:")
    print(scripted.predict_a.code)
    print("predict_b.code:")
    print(scripted.predict_b.code)
    print(
        "这正是 extrace_info.py 里能分别读到\n"
        "  jitmodel.predict_backbone.code / jitmodel.predict_depth.code / jitmodel.predict_segm.code\n"
        "三份独立代码的原因之一：script 配合 @torch.jit.export 是做到'多具名方法'的一种方式。\n"
        "但注意：这不是唯一方式——实验 D 会用 torch.jit.trace_module 做出同样效果的多方法\n"
        "模型，而且完全不需要 @torch.jit.export。实测下来 unified.pt 里这三个方法正是用\n"
        "trace_module 分别 trace 出来的（见 covariant/model/utils/export/unified.py 的\n"
        "export_unified_model），所以源码 basic.py 里根本找不到 @torch.jit.export 的痕迹。\n"
    )


# ---------------------------------------------------------------------------
# 实验 D：trace_module —— trace 也能产出多具名方法，且不需要 @torch.jit.export
#
# 这是 unified.pt 里 predict_backbone/predict_depth/predict_segm 三个方法
# 真正的产生方式（见 covariant/model/utils/export/core.py 的 trace_module()
# 封装、covariant/model/utils/export/unified.py 的 export_unified_model()）。
# ---------------------------------------------------------------------------

class MultiMethodModel(nn.Module):
    """跟 MultiEntryModel 结构一样，但完全不用 @torch.jit.export——
    因为我们不打算 script 化它，而是分别对每个方法做 trace。
    """

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(x)

    def predict_a(self, x: torch.Tensor) -> torch.Tensor:  # 没有 @torch.jit.export
        return self.shared(x) * 2

    def predict_b(self, x: torch.Tensor) -> torch.Tensor:  # 没有 @torch.jit.export
        return self.shared(x) + 1


def demo_trace_module_multi_entry():
    print("=" * 70)
    print("实验 D：torch.jit.trace_module —— 不用 @torch.jit.export 也能有多个具名方法")
    print("=" * 70)

    model = MultiMethodModel()
    example = torch.randn(2, 4)

    # inputs 的 key 就是方法名；trace_module 对每个 key 分别喂一次示例输入去录制。
    # "forward" 不是必须的，但很多下游工具（比如 torch.jit.freeze）假设模型有 forward，
    # 这也是 covariant 的 trace_module() 封装里额外塞一个空 forward 输入的原因。
    traced = torch.jit.trace_module(
        model,
        inputs={
            "forward": (example,),
            "predict_a": (example,),
            "predict_b": (example,),
        },
    )

    print("\npredict_a.code:")
    print(traced.predict_a.code)
    print("predict_b.code:")
    print(traced.predict_b.code)

    print(
        "观察：源码里 predict_a/predict_b 完全没有 @torch.jit.export，\n"
        "但 traced 对象上依然能分别调用、分别读到 .code——因为 trace_module 的\n"
        "'要不要编译这个方法'是由 inputs 字典的 key 决定的，跟编译器扫描 forward\n"
        "调用了哪些方法（script 的规则）没有任何关系。\n"
        "\n"
        "跟实验 C 对比着看：\n"
        "  - script + @torch.jit.export：编译器读 Python 源码，能保留 if/for 分支，\n"
        "    但孤立方法需要显式标注才会被纳入编译。\n"
        "  - trace_module：对每个方法名分别'录制一次实际执行'，天然支持多方法，\n"
        "    不需要任何装饰器，但每个方法各自仍然受 trace 的'分支固化'限制（见实验 B）。\n"
    )


# ---------------------------------------------------------------------------
# 实验 E：script 是"静态类型的子集语言"——编译期就会拒绝的三种写法
#
# 对应文档 01_what_is_torchscript.md (b) 小节。"静态类型"不是说你必须手写
# 类型标注（很多时候编译器会自动推断），而是编译器推断出来的类型必须
# 自洽、贯穿始终——同一个函数在不同分支/不同调用里不能变成不兼容的类型。
# "子集语言"则是说 Python 里一些完全合法的动态写法（**kwargs 等）根本不在
# TorchScript 编译器认识的语法范围内，跟类型对不对齐无关，直接语法不支持。
# 三个例子都是：eager 模式下 python 正常跑没有任何问题，torch.jit.script
# 编译期直接报错——这正是 covariant 里 is_in_onnx_export() 这类"导出守卫"
# 存在的原因：训练时的灵活写法过不了 script 编译器，必须切换到类型自洽的
# 代码路径。
# ---------------------------------------------------------------------------

class TrulyBadReturn(nn.Module):
    """if/else 两个分支返回不同、且没有共同类型上界的东西——
    eager 模式下 python 完全不关心这件事，script 编译器会当场拒绝。
    """

    def forward(self, x: torch.Tensor, flag: bool):
        if flag:
            return x * 2               # 这个分支类型是 Tensor
        else:
            return x, x * 2            # 这个分支类型是 Tuple[Tensor, Tensor]


class MixedList(nn.Module):
    """一个 list 字面量里混了 Tensor 和 str——普通 python list 天然支持，
    但 TorchScript 的 List[T] 要求元素类型统一。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        items = [x, "not_a_tensor"]    # 混合类型，被推断成 List[Union[Tensor, str]]
        return items[0]


class UsesKwargs(nn.Module):
    """**kwargs——不是类型不对齐的问题，是这种语法本身就不在
    TorchScript 编译器支持的子集范围内。
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


def _try_script(model, example_call, label):
    print(f"\n--- {label} ---")
    print(f"eager 模式（普通 Python 直接调用）完全没问题: {example_call(model)!r}")
    try:
        torch.jit.script(model)
        print("torch.jit.script 编译成功（意外——不应该走到这里）")
    except Exception as e:  # noqa: BLE001 - 教学演示，故意宽泛捕获来展示编译期报错
        print(f"torch.jit.script 编译失败 -> {type(e).__name__}")
        print(str(e).strip().splitlines()[0])


def demo_script_is_statically_typed_subset():
    print("=" * 70)
    print("实验 E：'静态类型的子集语言'——三种 eager 能跑、script 编译期直接拒绝的写法")
    print("=" * 70)

    _try_script(
        TrulyBadReturn(),
        lambda m: m(torch.ones(2), False),
        "例1：if/else 两个分支返回类型不统一（Tensor vs Tuple[Tensor, Tensor]）",
    )
    _try_script(
        MixedList(),
        lambda m: m(torch.ones(2)),
        "例2：list 字面量元素类型不统一（Tensor 和 str 混在一起）",
    )
    _try_script(
        UsesKwargs(),
        lambda m: m(torch.ones(2), scale=2),
        "例3：forward 用了 **kwargs（可变关键字参数）",
    )

    print(
        "\n观察：三个例子在 eager 模式下 python 都跑得好好的，因为普通 Python 变量\n"
        "没有固定类型、容器可以混装、参数可以随意透传。但 torch.jit.script 是一个\n"
        "真正的编译器：它要求每个函数在所有 return 路径上推导出的类型必须能统一成\n"
        "同一个类型（例1：Tensor 和 Tuple 之间没有共同上界，报错；但如果换成\n"
        "'一次返回 Tensor，一次返回 None'，编译器能自动推断成 Optional[Tensor]，\n"
        "反而是可以编译过的——'静态类型'不等于'必须手写类型标注'，是'推断出的\n"
        "类型必须自洽'）；例2同理，list 被推断成 Union 类型后跟标注的返回类型\n"
        "Tensor 不匹配；例3则是另一个维度的限制——**kwargs 这种语法本身就不在\n"
        "编译器认识的子集范围内，跟类型标不标注无关，直接 NotSupportedError。\n"
        "\n"
        "这正是 covariant 源码里大量 is_in_onnx_export() 导出守卫存在的原因：\n"
        "训练时为了灵活性写的动态返回值/动态参数，过不了 script 编译器，\n"
        "必须在导出路径上切换成类型自洽、参数显式列出的写法。\n"
    )


if __name__ == "__main__":
    demo_constant_propagation()
    demo_trace_freezes_shape()
    demo_trace_vs_tensor_op_boundary()
    demo_multi_entry_methods()
    demo_trace_module_multi_entry()
    demo_script_is_statically_typed_subset()
