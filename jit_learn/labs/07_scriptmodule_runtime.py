"""
Lab 07 · ScriptModule 的本质，以及从 Python 到 C++ 的调用边界

跑法：
    /opt/venv/bin/python3 07_scriptmodule_runtime.py

对应文档：jit_learn/02_anatomy_of_pt_file.md 2.5b 节

这个 lab 用 5 个实验，把"ScriptModule 是 Python 外壳包着 C++ 内核"这句抽象的话
变成可以亲眼看到、亲手验证的现象：

  A. self._c 到底是不是一个独立的 C++ 对象——不是比喻，是真的能 print 出来看类型
  B. Python 的 forward_hook 机制在 ScriptModule 上直接被拒绝注册——因为执行路径
     根本不经过会检查 hook 的 Python __call__
  C. 用 sys.setprofile 数 Python 函数调用次数：eager 模式每个子模块调用都要经过
     Python 解释器，script 模式只在入口/出口露一次脸
  D. GraphExecutor 的 ExecutionPlan 缓存：第一次调用要跑图优化/特化，比后续调用
     慢一到两个数量级——这是"缓存执行计划"不是空话的直接证据
  E. 序列化后的 ScriptModule 在原始 Python 类定义被删除之后依然能加载执行——
     证明 code/*.py 是自给自足的独立描述，不依赖原始 nn.Module 类存在
"""
import io
import sys
import time

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 实验 A：self._c 是不是一个独立的 C++ 对象
# ---------------------------------------------------------------------------

class Sub(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = Sub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub(x) + 1


def demo_c_is_real_cpp_object():
    print("=" * 70)
    print("实验 A：ScriptModule 的 self._c 是不是真的是一个 C++ 对象？")
    print("=" * 70)

    scripted = torch.jit.script(Wrapper())
    c = scripted._c
    plain = Wrapper()  # 对照组：一个普普通通、没有被 script 化的 Python 对象

    print(f"\ntype(scripted) = {type(scripted)}")
    print(f"type(c)        = {type(c)}  (c = scripted._c)")
    print(f"type(plain)    = {type(plain)}  (对照组，纯 Python nn.Module 实例)")

    print("\n--- 检验 1：普通 Python 对象都有 __dict__，c 有吗？ ---")
    print("对照组 plain.__dict__ 的 key 列表:", list(plain.__dict__.keys()))
    try:
        print("c.__dict__ =", c.__dict__)
    except AttributeError as e:
        print(f"报错 -> AttributeError: {e}")

    print("\n--- 检验 2：普通 Python 对象可以随手加一个新属性，c 能吗？ ---")
    plain.some_new_attr = 123  # 对照组：这一行在任何普通 Python 对象上都天经地义地成功
    print("对照组 plain.some_new_attr =", plain.some_new_attr, "（成功）")
    try:
        c.some_new_attr = 123
        print("加上了：c.some_new_attr =", c.some_new_attr)
    except RuntimeError as e:
        print(f"报错 -> RuntimeError: {e}")

    print("\n--- 检验 3：inspect 能看到一个 Python 类的源码，能看到 type(c) 的源码吗？ ---")
    import inspect
    print("对照组 inspect.getsource(Wrapper) 前 2 行:")
    print("\n".join(inspect.getsource(Wrapper).splitlines()[:2]))
    try:
        print(inspect.getsource(type(c)))
    except OSError as e:
        print(f"type(c) 报错 -> OSError: {e}")

    print("\n--- 检验 4：type(c) 的继承链（MRO）里有什么？ ---")
    for klass in type(c).__mro__:
        print(" ", klass)

    print("\nscripted.forward.graph（编译后的图，节点是 C++ 侧的 aten/prim 算子）：")
    print(scripted.forward.graph)

    print(
        "\n观察：\n"
        "  - 检验 1、2 的报错文本本身就是证据——'does not have a field with name\n"
        "    \"__dict__\"'、'does not have an attribute with name ...'，这种报错措辞\n"
        "    是 TorchScript 的 C++ 编译单元在查找字段失败时抛出的，跟 Python 对象\n"
        "    属性缺失的 AttributeError 措辞完全不是一回事。普通 Python 对象——\n"
        "    包括未 script 化的 Wrapper() 实例本身——__dict__ 一定存在，加属性\n"
        "    一定成功，这是 Python 对象的基本权利。c 在这两件事上都直接被拒绝，\n"
        "    说明它根本不是按 Python 对象模型实现的。\n"
        "  - 检验 3 报错 'could not find class definition'：一个用 class 关键字写的\n"
        "    Python 类，inspect 永远能找到它的源文件和行号；找不到，只可能是这个\n"
        "    类型压根不是 Python 源码编译出来的。\n"
        "  - 检验 4 的 MRO 里直接写着 pybind11_builtins.pybind11_object——这是\n"
        "    pybind11（C++/Python 绑定库）生成的基类，字面意义上就是'这个类型的\n"
        "    真身是一段 C++ 代码，pybind11 只是给它套了一层能被 CPython 认出来的\n"
        "    外壳'。self._c 不是某种元数据视图或语义上的比喻，是 torch::jit::Module\n"
        "    这个 C++ 对象本身的 Python 句柄，货真价实。\n"
        "  - graph 里的 'prim::GetAttr[name=\"sub\"]' + 'prim::CallMethod[name=\"forward\"]'\n"
        "    就是 02_anatomy_of_pt_file.md 2.5b 节说的'子模块调用在图里长什么样'——\n"
        "    这条调用从来没有经过 Python 的属性查找/方法解析，是图节点，\n"
        "    GraphExecutor 直接在 C++ 里跳转执行。\n"
    )


# ---------------------------------------------------------------------------
# 实验 B：forward_hook 在 ScriptModule 上直接被拒绝注册
# ---------------------------------------------------------------------------

def _dummy_hook(module, inp, out):
    pass  # 只是用来证明"能不能注册"，不关心内容


def demo_hooks_rejected_on_scriptmodule():
    print("=" * 70)
    print("实验 B：eager 模式的 forward_hook，在 ScriptModule 上还能用吗？")
    print("=" * 70)

    m_eager = Wrapper()
    m_eager.sub.register_forward_hook(_dummy_hook)
    print("\neager 模式：m_eager.sub.register_forward_hook(...) 注册成功（这是 Python 层机制）")

    m_script = torch.jit.script(Wrapper())
    print("\nscript 模式：尝试对 m_script.sub 做同样的注册...")
    try:
        m_script.sub.register_forward_hook(_dummy_hook)
        print("  注册成功（意外——不应该走到这里）")
    except RuntimeError as e:
        print(f"  注册失败 -> RuntimeError: {e}")

    print(
        "\n观察：forward_hook 是 nn.Module.__call__（Python 层 _call_impl）里检查的机制，\n"
        "只有 Python 逐层调用 forward 时才会被触发。ScriptModule 的执行完全在 C++ 侧的\n"
        "GraphExecutor/interpreter 里跑，压根不经过这条 Python 调用路径——所以 PyTorch\n"
        "直接在 API 层面禁止对 ScriptModule 注册 hook，而不是'注册了但不生效'。\n"
        "这是'子模块调用不回 Python'这句话最直接的证据：不是理论上不方便观测，\n"
        "是这条路径在设计上就不存在。\n"
    )


# ---------------------------------------------------------------------------
# 实验 C：数 Python 函数调用次数——eager 每层都要回 Python，script 只在头尾露脸
# ---------------------------------------------------------------------------

class ThreeSubs(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = Sub()
        self.sub2 = Sub()
        self.sub3 = Sub()
        self.sub4 = Sub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub1(x) + self.sub2(x) + self.sub3(x) + self.sub4(x)


def _count_python_calls(fn, x):
    calls = []

    def profiler(frame, event, arg):
        if event == "call":
            code = frame.f_code
            calls.append(f"{code.co_name} @ {code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}")

    sys.setprofile(profiler)
    try:
        fn(x)
    finally:
        sys.setprofile(None)
    return calls


def demo_python_call_count():
    print("=" * 70)
    print("实验 C：同样有 3 个子模块调用，eager 和 script 各触发多少次 Python 函数调用？")
    print("=" * 70)

    m_eager = ThreeSubs().eval()
    m_script = torch.jit.script(ThreeSubs().eval())
    x = torch.randn(4, 4)

    calls_eager = _count_python_calls(m_eager, x)
    calls_script = _count_python_calls(m_script, x)

    print(f"\neager 模式一次 forward 触发的 Python 函数调用数: {len(calls_eager)}")
    for c in calls_eager:
        print("   ", c)

    print(f"\nscript 模式一次 forward 触发的 Python 函数调用数: {len(calls_script)}")
    for c in calls_script:
        print("   ", c)

    print(
        "\n观察：eager 模式下，顶层 forward 调用 sub1/sub2/sub3 三次，每次都要经过\n"
        "nn.Module.__call__ -> _call_impl -> hook 检查 -> 用户 forward 方法体，\n"
        "每一层都是独立的 Python 函数调用，被 profiler 逐个记录下来（数量会随子模块\n"
        "数量线性增长——4 个子模块就会看到更多次）。\n"
        "\n"
        "script 模式下的 3 次不是'3 个子模块各露了一次脸'——把上面列表逐行对着看：\n"
        "三次调用全部发生在同一个入口 model(x) 触发的一次 __call__ 里，且都停在\n"
        "'进入 C++ 之前'这一层（_wrapped_call_impl -> _call_impl -> is_exporting\n"
        "这类 Python 侧的守卫/兼容代码），跟 sub1/sub2/sub3 完全无关。sub1/sub2/sub3\n"
        "三次真正的子模块调用全部在 C++ 的 GraphExecutor/interpreter 里通过\n"
        "prim::CallMethod 完成，Python profiler 根本'看不到'它们，因为它们从来没有\n"
        "以 Python 函数调用的形式发生过——不管子模块堆到 3 个还是 300 个，这份\n"
        "Python 侧调用列表长度都不会变，这才是关键对比：eager 随子模块数量线性增长，\n"
        "script 是一个跟子模块数量无关的固定常数。\n"
        "\n"
        "（这个常数具体是多少、由哪几个函数构成，属于 PyTorch 版本相关的实现细节——\n"
        "比如这次装的版本里是 _wrapped_call_impl/_call_impl/is_exporting 这三个，\n"
        "换个 PyTorch 版本可能是 2 个或 4 个，不用记具体数字，只需要记住'跟子模块数量\n"
        "无关'这个结论。）\n"
        "\n"
        "这正是 02_anatomy_of_pt_file.md 2.8 节'更少的动态分派和 Python 边界开销'\n"
        "的可测量证据，不是空泛的说法。\n"
    )


# ---------------------------------------------------------------------------
# 实验 D：ExecutionPlan 缓存——第一次调用要编译优化，后续调用直接复用
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DeepNet(nn.Module):
    """20 层堆叠，放大'第一次调用要做图优化/特化'这件事的耗时占比。"""

    def __init__(self, n: int = 20):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock() for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def demo_execution_plan_caching():
    print("=" * 70)
    print("实验 D：GraphExecutor 的 ExecutionPlan 缓存——第一次调用慢在哪")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}" + ("" if device == "cuda" else "（没有 CUDA，效果会比 GPU 上弱很多，但方向应该一致）"))

    model = torch.jit.script(DeepNet().to(device).eval())
    x = torch.randn(8, 16, 64, 64, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        sync()
        t0 = time.perf_counter()
        model(x)
        sync()
        t1 = time.perf_counter()

        times = []
        for _ in range(30):
            sync()
            s = time.perf_counter()
            model(x)
            sync()
            times.append(time.perf_counter() - s)

    first_ms = (t1 - t0) * 1e3
    avg_ms = sum(times) / len(times) * 1e3
    print(f"\n第 1 次调用耗时:        {first_ms:8.3f} ms")
    print(f"后续 30 次调用平均耗时: {avg_ms:8.3f} ms")
    print(f"倍数（第1次 / 后续平均）: {first_ms / avg_ms:.1f}x")

    print(
        "\n观察：第一次调用触发的不只是 kernel 计算本身，还包括 GraphExecutor 对这组\n"
        "shape/dtype/device 做的图优化 pass（常量折叠、死代码消除、fusion 等），\n"
        "以及生成并缓存一份 ExecutionPlan。后续输入只要 shape/dtype/device 组合相同，\n"
        "就直接复用这份缓存的 ExecutionPlan，跳过重新优化——这就是'预热/warm-up'\n"
        "这个工程实践背后的真实原理，不是玄学，是有名字、有缓存结构的具体机制。\n"
        "GPU 上这个差距通常是几十到上百倍（CPU 上因为没有 kernel launch/cuDNN\n"
        "算法选择这类额外开销，差距会小很多，甚至不明显——如果你在 CPU 上跑这个\n"
        "实验发现看不出明显差异，换成 CUDA 设备重跑，或者加大模型/输入规模。\n"
    )


# ---------------------------------------------------------------------------
# 实验 E：序列化后的 ScriptModule 脱离原始 Python 类定义依然能跑
# ---------------------------------------------------------------------------

class OnlyDefinedHere(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w


def demo_survives_without_original_class():
    print("=" * 70)
    print("实验 E：把原始 Python 类删掉之后，torch.jit.load 出来的模型还能跑吗？")
    print("=" * 70)

    scripted = torch.jit.script(OnlyDefinedHere())
    buf = io.BytesIO()
    torch.jit.save(scripted, buf)
    buf.seek(0)

    print(f"\n删除前调用一次: {scripted(torch.tensor([2.0]))}")

    # 模拟"换到另一个完全没见过这个类的环境"：把类从当前模块的命名空间里删掉
    globals().pop("OnlyDefinedHere", None)
    still_there = "OnlyDefinedHere" in dir(sys.modules[__name__])
    print(f"当前模块里 OnlyDefinedHere 是否还存在: {still_there}")

    loaded = torch.jit.load(buf)
    out = loaded(torch.tensor([2.0]))
    print(f"\n原始类已被删除，torch.jit.load(buf) 依然成功，调用输出: {out}")
    print("loaded.code:")
    print(loaded.code)

    print(
        "\n观察：这不是巧合，是 02_anatomy_of_pt_file.md 强调的'code/*.py 是自给自足的\n"
        "序列化描述'的直接证明——torch.jit.load 靠的是 archive 里的 code/*.py（IR 文本）\n"
        "+ data.pkl（对象状态）+ data/（tensor 字节）重建出可执行的 ScriptModule，\n"
        "全程没有一步'去 import 原始的 OnlyDefinedHere 类'。这也是为什么 unified.pt\n"
        "能在没有装 covariant 包的 aarch64 目标机上单独跑起来——它从来不依赖\n"
        "covariant.* 的 Python 源码，只依赖注册好的底层算子（内置的 + 自定义 .so）。\n"
    )


if __name__ == "__main__":
    demo_c_is_real_cpp_object()
    demo_hooks_rejected_on_scriptmodule()
    demo_python_call_count()
    demo_execution_plan_caching()
    demo_survives_without_original_class()
