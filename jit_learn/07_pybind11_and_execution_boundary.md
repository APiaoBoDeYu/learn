# 07 · pybind11 封装原理 + "在 Python 里执行" vs "在 C++ 里执行" 的真正含义

## 和已有内容的关系

`02_anatomy_of_pt_file.md` 第 2.5b 节已经讲过 `ScriptModule` 是"Python 句柄 + C++ 实体"，
`self._c` 是那道桥。本章往下再深一层，回答两个更底层的问题：

1. pybind11 具体**怎么**把一个 C++ 类型变成 Python 能创建、能调用的对象——内存布局、
   方法分派、参数类型转换这几步分别在做什么。
2. "在 Python 里执行" 和 "在 C++ 里执行" 这句话，脱离比喻之后精确指什么——不是语言、
   不是速度，而是**由哪个解释器/运行时驱动控制流**。

配套实验脚本仍然是 `jit_learn/labs/07_scriptmodule_runtime.py`；这里补的是那份实验背后的
原理性解释。

## 7.1 pybind11 包的到底是什么

C++ 编译出来的是机器码，本身完全不知道"Python 对象"这个概念存在。pybind11 要做的事情，
准确说是：**在 C++ 编译期，用模板/宏生成一份符合 CPython C-API 规范的 `PyTypeObject`**，
让 CPython 解释器能把某个 C++ 类型当成一个原生 Python 类型来创建实例、查找方法、管理生命周期。

CPython 里，所有堆上的 Python 对象在内存里都必须以一个固定头部开始：

```c
typedef struct {
    Py_ssize_t ob_refcnt;   // 引用计数
    PyTypeObject *ob_type;  // 指向类型对象，决定这个对象"是什么类型"、有哪些方法
} PyObject;
```

`ScriptModule` 对应的 Python 对象，内存布局大致是：

```text
[ PyObject 头：ob_refcnt, ob_type ]
[ 一个指向 C++ torch::jit::Module 对象的句柄/指针 ]
```

也就是说，Python 里拿到的 `m`，是一小块内存：前半段是让 CPython 认得它、能做垃圾回收、
能查 `type(m)` 的"合法证件"；后半段不是重新实现的 Python 数据结构，而是**指向真正的 C++
对象**。C++ 那一侧才存着 Graph、参数 Tensor、方法表；Python 侧的对象自己不存储、不计算
任何东西。

## 7.2 `py::class_<T>` 在编译期做了什么

以 pybind11 里最常见的绑定写法为例：

```cpp
py::class_<Module>(m, "ScriptModule")
    .def("forward", [](Module& self, py::args args) {
        return self.forward_impl(args);
    })
    .def_property_readonly("graph", &Module::get_method_graph);
```

`py::class_<Module>(...)` 这一行在**C++ 编译期**（不是运行时）用模板生成了一整套东西：

- 一个 `PyTypeObject` 实例，包含类型名、大小、方法表指针等 CPython 要求的字段。
- 每个 `.def(...)` 注册的函数/lambda，被包装成一个**trampoline（跳板）函数**——签名固定为
  `PyObject* f(PyObject* self, PyObject* args)`，这是 CPython C-API 规定的所有"C 函数"必须
  遵守的调用约定，不管你原来的 C++ 函数签名是什么。

模块被 import 时（运行时的一次性初始化），pybind11 把这个 `PyTypeObject` 注册进 Python 的
类型系统，之后 Python 里 `ScriptModule` 这个名字就指向它，跟内置类型 `list`/`dict` 在机制上
没有本质区别——都是一个 `PyTypeObject`，只是这个的方法表背后接的是 C++ 逻辑。

## 7.3 一次方法调用，边界上具体发生了什么

`m.forward(x)` 触发的完整链路：

```text
1. CPython 在 m 的类型对象方法表里查到 "forward" 对应的 trampoline 函数指针
2. 调用这个 C 函数，参数是 PyObject* args（Python 元组，里面装着 x 这个 Python Tensor 对象）
3. trampoline 内部：用 pybind11 的 type_caster 把 PyObject* x 转换成 C++ 的 at::Tensor
   （这一步要做真正的类型检查/转换，是有成本的，但只做一次，不是每个算子都做一次）
4. 调用真正的 C++ 逻辑：self.forward_impl(tensor) —— 这一步完全在 C++ 里跑，
   不经过任何 CPython 字节码
5. C++ 返回结果（at::Tensor 或 IValue），再被同一套 type_caster 转换回一个新的
   Python 对象（构造一个新的 PyObject，填好 ob_refcnt/ob_type，内部指向那份 C++ 数据）
6. trampoline 把这个新 PyObject* 返回给 CPython，调用方拿到的就是一个"看起来是原生
   Python 对象"的返回值
```

关键点：**只在第 2 步和第 5 步跨越一次 Python/C++ 边界，做一次参数/返回值的类型转换
（marshalling）**。第 4 步内部无论调用多少层 C++ 函数、算多少个算子，都不会再往回跳
Python——这跟 `02_anatomy_of_pt_file.md` 2.5b 节里 `model(rgbs)` 的调用链路图是同一件事，
这里补的是"边界上那一次转换具体在做什么"。

`type_caster` 是 pybind11 的核心机制之一：每个要暴露给 Python 的 C++ 类型都要注册一个
`type_caster`（PyTorch 给 `at::Tensor`、`c10::IValue` 等类型都写好了），负责双向转换
`PyObject* <-> C++ 类型`。没有注册 caster 的类型是不能跨这条边界的，这也是为什么自定义
C++ 算子想在 Python 侧被使用，必须先给它的输入输出类型找到（或写）对应的 caster/转换路径。

## 7.4 "在 Python 里执行" vs "在 C++ 里执行"：精确含义

容易产生的误解：以为区别是"这段代码写在 .py 文件还是 .cpp 文件里"，或者"哪个更快"。
准确的区别是：**这一步的控制流，由哪个解释器/运行时在驱动**。

### 普通 eager 模式

```python
def forward(self, x):
    y = x + 1
    return y.relu()
```

调用时：

```text
1. CPython 解释器逐条执行这个函数的字节码：
   LOAD_FAST x → LOAD_CONST 1 → BINARY_ADD → STORE_FAST y → LOAD_FAST y → CALL relu → RETURN
2. 跑到 BINARY_ADD：CPython 发现 x 是 Tensor，调用它的 __add__
   → 跨进 C++（ATen 的 add kernel）→ 算完 → 跨回 Python，继续下一条字节码
3. 跑到 y.relu()：同样，跨进 C++ → 算完 → 跨回 Python
```

CPython 的字节码解释器（`ceval.c` 里的 frame evaluation 循环）**全程在场**，每两个算子
之间都要跨一次边界。这就是"Python overhead"的真实来源：不是加法本身慢，是两次 C++
调用之间，CPython 要做字节码取指、分派、构造参数元组、维护引用计数这些工作。

### TorchScript / `ScriptModule` 模式

同一段逻辑被 `torch.jit.script` 编译后：

```text
1. 编译期：Python AST 被静态分析，编译成 Graph（IR）：
   %y = aten::add(%x, %1)
   %2 = aten::relu(%y)
   这份 Graph 整个存在 C++ 对象里，跟原来的 Python 字节码已经没有关系
2. 调用 module.forward(x)：Python 壳只做一次转发（7.3 节的第 2 步），
   把 x 转成 IValue 扔给 C++ 的 GraphExecutor::run()
3. add → relu 两步的调度，由 C++ 的 InterpreterState 驱动
   （这是 TorchScript 自己写的一个小型解释器，逐条执行 Graph 编译出的 Instruction，
   不是 CPython 字节码，操作的是 IValue 不是 PyObject）
4. 整个 forward 跑完，才把最终结果转回一个 Python 对象返回（7.3 节的第 5 步）
```

`add` 和 `relu` 之间**完全不跨回 Python**。

### 对照表

| | 在 Python 里执行 | 在 C++ 里执行 |
|---|---|---|
| 驱动控制流的"虚拟机" | CPython 字节码解释器（frame evaluation） | TorchScript `InterpreterState`（或直接机器码，比如 TensorRT engine） |
| 算子之间要不要跨回 Python | 要，每次都有字节码分派 + 引用计数开销 | 不要，调度全程在 C++ 内部 |
| Python 解释器是否必须存在 | 必须，是执行的驱动者 | 不必须，可以是纯 `libtorch` C++ 进程，Python 从未启动 |
| 谁决定"下一步执行什么" | CPython 的 frame / 字节码 | C++ 侧的 Graph / InterpreterState |
| 数据载体 | `PyObject*`（有对象头、引用计数、属性字典） | `IValue`（tagged union，没有 Python 对象头） |

一句话：**"在哪里执行"问的是"谁在驱动控制流"，不是代码字面写在哪个文件里**。
`ScriptModule` 的 Python 方法（`.def("forward", ...)` 那行 lambda/trampoline）本身严格
说是 C++ 代码，但它也只是"转发一次"；真正被反复驱动、反复分派的执行循环，eager 模式下
是 CPython frame evaluation，TorchScript/TensorRT 模式下是 C++ Graph 执行器或 GPU kernel
调度，Python 解释器从进入 `forward` 主体那一刻起就彻底不在场了。

这正是为什么导出 TorchScript / TensorRT engine 对部署有意义：一旦控制流驱动权从 CPython
转移到 C++，GIL、字节码分派、Python 对象创建这些开销全部消失，剩下的是纯粹的算子调用链
——也是 Thor 平台（无 PyTorch 运行时）能直接跑 `.pt` / TensorRT engine 的根本原因，
呼应 `01_what_is_torchscript.md` 1.1 节最初提出的那个问题。

上面 7.4 节把"C++ 侧的 Graph / InterpreterState"当成一个整体黑盒。7.6 把这个黑盒
打开——`GraphExecutor` 具体拿 Graph 做了什么、`InterpreterState` 到底怎么执行、
以及 TorchScript IR 在 `.pt` 文件里实际是以什么形式存在的。

## 7.6 TorchScript IR 在 `.pt` 里到底存在哪、以什么形式存在

`02_anatomy_of_pt_file.md` 已经讲过 `.pt` 归档里 `code/` 目录下有一份 `.py` 文本。
这里补一个容易搞错的细节：**这份 `.py` 文本不是"反编译出来给人看的镜像"，它本身就是
TorchScript IR 的序列化形式**——`torch.jit.save` 在导出时会把内存里的 `Graph`（IR，
节点是 `aten::`/`prim::` 算子调用的 SSA 形式）**打印**成这种类 Python 语法的文本，
直接写进 `code/*.py`；旁边配套一个 `.py.debug_pkl`，存的是打印用的字符串表和行号映射，
给调试/报错时定位用，不是另一份独立的 Graph 存储。

拿真实归档验证过（`jit2engine/jit_model/unified.pt`，注意这个路径下的 `unified.pt`
是目录形式，内部真正的 zip 是 `e779c45986c16bb78c851fb56a38c1a9-15` 这个文件）：

```text
archive/
├── code/__torch__/.../xxx.py           ← 某个类型某个方法的 TorchScript IR（打印成文本）
├── code/__torch__/.../xxx.py.debug_pkl ← 配套调试信息（字符串表 + 行号）
├── data.pkl                             ← 模块树实例数据：属性值、子模块引用、类型名
├── constants/                           ← 常量 Tensor（非 parameter 的固定值）
└── data/                                ← 权重 Tensor 的实际存储（按编号平铺）
```

`data.pkl` 只存**实例数据**（这个模块有哪些子模块、哪些属性、类型叫什么名字），完全不
包含方法体。方法体（Graph）保存在 `code/*.py` 里，按类型的 `qualified_name` 组织，
一个类型的方法只存一份，被这个类型的所有实例共享。

`torch.jit.load` 反序列化时：

1. 按 `data.pkl` 里的 `qualified_name` 去 `code/` 目录找到对应类型的 `.py` 文本
2. 用 TorchScript 自己的编译器前端（`torch/csrc/jit/frontend/` 的 Lexer/Parser）把这份
   文本**重新解析**成内存里的 `Graph` 对象
3. 反序列化 `data.pkl`，重建出模块树的具体实例（子模块指针、权重 Tensor），实例的类型
   指针指向第 2 步解析出的 `Graph`/方法表

注意第 2 步是"解析"不是"编译"——这份文本在导出时已经是类型标注齐全、控制流展开完毕的
确定性描述，重新解析只是忠实地还原出等价的 `Graph`，不会重新做类型推断或优化决策，
真正的类型推断只发生在最初 `torch.jit.script`/`trace` 那一次。

一句话：**编译（推断类型、决定分派）发生在导出时一次性完成，`.pt` 里存的是编译产物
（Graph，打印成文本形式）+ 数据（权重），加载时只是把这两者重新解析/反序列化并关联
起来，没有重新编译这一步。**

## 7.7 GraphExecutor：Graph 到可执行状态之间还有什么

7.4 节说"TorchScript `InterpreterState` 驱动控制流"，这句话省略了中间一大段——`Graph`
不会被直接拿去逐节点执行，中间要经过 `GraphExecutor` 的优化和特化。完整链路：

```text
Method.forward(x)                         [Python 侧最后一次转发]
  → torch::jit::Function::run(Stack&)      = GraphFunction::run
  → get_executor().run(stack)              GraphExecutor（每个 Method 首次调用时惰性创建）
  → GraphExecutorImplBase::run(Stack&)
       → getPlanFor(stack)                 拿/建 ExecutionPlan（可能触发编译优化）
       → InterpreterState(plan.code).run(stack)   真正执行
```

`GraphExecutor` 现在默认走的是 `ProfilingGraphExecutorImpl`（新执行器 + profiling 模式，
非移动端默认开启；老的无 profiling 版本 `GraphExecutorImpl` 代码还在但基本不会被走到）。
它对 Graph 做两阶段处理：

**阶段一：profiling warm-up**——第一次调用时，先跑一份插了 `prim::profile` 节点的
**未优化、未融合**图，用真实输入跑一遍纯粹为了记录下实际观察到的 shape/dtype/device/
requires_grad。**这意味着一个刚 load 出来的 `ScriptModule`，第一次调用的性能和数值路径
可能和之后稳定下来的调用不完全一样**——如果要拿 PTH 侧结果去跟 TensorRT engine 做精度
对齐，应该先 warm-up 几次再采样参考值，避免比较到这份未融合的第一次运行结果。

**阶段二：用 profiling 收集到的信息做真正优化**，按顺序大致包含：inline 子函数调用、
死代码消除、常量传播、shape/dtype 特化、公共子表达式消除、peephole 优化、循环展开、
**算子融合**（把连续 pointwise 算子合并成一个 `prim::FusionGroup`，减少 dispatcher 调用
和显存读写，现在默认走 NNC/TensorExpr 融合器），最后把 profile 节点转成运行时
`prim::Guard`——检查真实输入是否还符合当初特化时的假设，不符合就触发 `prim::BailOut`
退化到一份不依赖该特化的兜底子图重跑。

优化产出的结果缓存在一个 `ExecutionPlan` 里（图 + 编译好的 `Code`）；在
`ProfilingGraphExecutorImpl` 下大致是"一份主优化 plan + 有限次数的降级 bailout 路径"，
不是无限增长的按 shape 建表（那是老执行器的行为）。

## 7.8 InterpreterState 执行的 Code/Instruction 到底是什么

`ExecutionPlan.code` 是 `Graph` 线性化后的产物，包含：

- `vector<Instruction>`——真正的指令序列，opcode 是 TorchScript 自己定义的一套
  （`OP` 调用算子、`LOAD`/`MOVE`/`STORE` 寄存器读写、`JMP`/`JF`/`LOOP` 控制流、
  `GUARD`/`FAIL_GUARD` 特化检查、`CALL` 子函数调用……）
- 常量表、类型表
- **operator_table_**——每个图节点对应的算子，在 `Code` 构建阶段就已经解析好放进表里，
  执行时按下标直接拿，不再做字符串/schema 查找

`InterpreterState` 是每次执行时的可变状态：一个**寄存器堆**存长生命周期的中间 `IValue`，
外加每次算子调用时临时用一下的 `Stack`。设计上明确不做寄存器分配（官方文档原话是为了
让调试更容易），值用完就靠 `MOVE` 指令尽快释放引用计数。

执行循环就是标准的字节码解释器写法：`switch (inst.op) { case OP: ...; case LOAD: ...; }`，
其中 `OP` 分支直接调用 `operator_table_[inst.X](stack)`。

**算子调用是否走正常 ATen dispatcher**：走的，逐条走，包括 dtype/device/backend/autograd
的 dispatch key 解析，跟 eager 模式完全一样。**唯一例外是融合区域内部**——一旦几个
pointwise 算子被 `FusionGroup` 合并，融合后整块变成一条指令、对应一个生成的融合 kernel，
被合并掉的那些单独算子节点不再存在，也就不会分别经过 dispatcher 了。

`torch.jit.trace` 和 `torch.jit.script` 产出的 Graph，最终都走同一套
`GraphExecutor → ExecutionPlan → Code → InterpreterState` 流程，执行引擎层面没有区别，
差异只在于两种前端产出的 Graph 长什么样（trace 通常没有控制流分支，除非内部调用了
已经 script 化的子函数）。`torch.jit.load` 也是同理：反序列化只影响 Graph 怎么被
重建出来（7.6 节的重新解析），一旦内存里有了 `Module` 对象，后面 profiling、优化、
`Code` 构建、解释执行全都是同一套机制，不存在"加载出来的模型用另一套引擎跑"这种事。

## 7.9 TorchScript Interpreter 和 CPython 解释器：什么关系

容易产生的疑问：C++ 是编译型语言，为什么"进入 C++ 内部"之后还会冒出一个"解释器"？

答案是分清楚**谁在解释谁**。`interpreter.cpp` 这份代码本身是 C++ 源码，会被 gcc/clang
编译成机器码——这一层是纯编译型的，没有任何解释发生。但**这份编译好的原生代码，运行时
扮演的角色是一个字节码解释器**：它读取 7.8 节说的 `vector<Instruction>`，逐条
`switch` 分发执行。它解释的对象不是 C++ 源码，而是 **TorchScript Graph 编译出的自定义
指令序列**——这套指令集是 PyTorch 团队自己设计的，跟 x86/ARM 机器码无关，也跟 Python
bytecode 无关。

类比 JVM：JVM 本身用 C++ 写、编译成原生代码，但运行时解释/JIT 执行的是 `.class` 字节码。
TorchScript Interpreter 和 JVM 的角色是同一种东西——**用编译型语言实现的一个虚拟机，
去执行另一种自定义的、更贴近张量计算图语义的指令集**。

和 CPython 解释器（`ceval.c`）的关系：架构模式相同（都是"虚拟机 + 自定义字节码"），
但完全是两套独立系统，互不调用：

| | CPython 解释器 | TorchScript Interpreter |
|---|---|---|
| 实现语言 | C | C++ |
| 解释的字节码 | Python bytecode（`LOAD_FAST`/`CALL_FUNCTION`/...） | TorchScript `Instruction`（`OP`/`LOAD`/`GUARD`/...） |
| 字节码来源 | `.py` 源码经 CPython 编译器编译 | `Graph`（IR）经 `Code` 构建线性化 |
| 数据载体 | `PyObject*` | `IValue` |

两者唯一的关系是**调用时序上的接力**：`scripted_module(x)` 这行代码先由 CPython 解释器
驱动（7.3 节的 trampoline 转发那一段），一路调用到 `callee.run(stack)`——从这一刻起，
执行权从 CPython 解释器交给 TorchScript Interpreter，完全脱离 Python 字节码，是纯 C++
的 `switch(inst.op)` 循环在跑，直到 `forward` 整个执行完，才把结果转回 Python 对象、
控制权还给 CPython。这正是 7.4 节对照表里"谁在驱动控制流"这条的具体化：eager 模式下
驱动者全程是 CPython；script 模式下，进入 `forward` 之后驱动者切换成了 TorchScript
Interpreter，两个解释器前后接力，没有代码复用或调用关系。

## 7.10 TorchScript Interpreter 为什么比 CPython 快

容易先入为主地归因成"C++ 比 Python 快"，但算子本身（矩阵乘、卷积……）在 eager 模式和
script 模式下调的是**同一个 ATen kernel**，计算本体的速度不会因为用不用 TorchScript
而变化。真正的差别在**算子之间的调度开销**，原因分四层：

**① CPython 每次都要做"不知道下一步是什么"的通用分派。**
`y = x + 1` 对应字节码 `BINARY_ADD`，但 CPython 在运行时才知道 `x` 的具体类型——可能是
`int`、`Tensor`，或任何自定义了 `__add__` 的对象。所以每次执行到这一行都要重新查
`x` 的类型对象、找 `__add__`（走 MRO）、不行再退回试 `__radd__`、构造参数元组、调用、
维护引用计数——哪怕 `x` 每次都是同一种类型，这套通用分派流程也要重复跑一遍。这是动态
语言灵活性的代价。

**② TorchScript `Instruction` 执行的是"编译期已经查好表"的结果。**
回顾 7.8 节：`Code` 构建阶段就已经把每个算子节点解析成 `operator_table_` 里一个具体的
可调用对象。运行到 `OP` 指令时：

```cpp
case INST(OP): {
    frame.function->operator_table_[inst.X](stack);
}
```

直接按下标取函数指针调用，不再做"这次是什么类型该调哪个实现"的判断——因为 7.7 节讲的
shape/dtype 特化 + guard 机制，已经在编译期把这个决策做完并固化进了这条指令。相当于把
CPython 每次运行时都要做的查表工作，提前挪到编译期只做一次，运行时只是照本回放。

**③ 没有 `PyObject` 装箱开销。**
CPython 里每个值都是 `PyObject*`——带引用计数、类型指针、可能还有 `__dict__`，哪怕整数
加法也要经过装箱/拆箱。TorchScript Interpreter 操作的是 `IValue`（tagged union，参见
7.4 节对照表），寄存器堆直接存 `IValue`，没有动态属性字典这类开销。

**④ 算子融合进一步减少调度次数本身，不只是让单次调度更便宜。**
7.7 节讲的 `FusionGroup` 把连续的 pointwise 算子合并成一条指令、一次 kernel 调用。
eager 模式下 `x.add(1).relu()` 是两次完整的 CPython 字节码分派 + 两次跨 C++ 边界
（7.4 节的边界转换）；TorchScript 不仅省去 CPython 分派，融合后甚至把两次算子调用也
合并成了一次，边界跨越次数直接减半。

**结论**：调度开销省得多，不代表计算本身变快。模型里小算子/多分支越多，调度开销占比越
高，TorchScript 的收益越明显；如果模型主体就是几个大 GEMM，调度开销本来占比就低，收益
有限。这也解释了 TensorRT 为什么通常比 TorchScript 还快一截——TensorRT 的融合更激进
（跨更多算子、专门为目标硬件生成 kernel），把调度开销压得更接近于零，这也是
`thor-engine-convert` 流程里 TorchScript 之后还要再编译成 TensorRT engine 的意义所在。

## 7.11 和 `torch.Tensor` 的类比

同一套"Python 外壳 + C++ 内核"模式在 PyTorch 里到处都是，不是 `ScriptModule` 特有的：

- `torch.Tensor`（Python 对象）包着 `c10::TensorImpl`（C++ 对象，持有 storage 指针、
  shape、stride、dtype）。
- `ScriptModule`（Python 对象）包着 `torch::jit::Module`（C++ 对象，持有 Graph、
  子模块树、参数）。

两者的调用模式完全一致：Python 侧的属性访问/方法调用，本质是"查表找到对应的 trampoline
→ 转换参数类型 → 调 C++ 方法 → 转换返回值类型"，跟 7.3 节描述的链路是同一个模式的两个
实例。理解了其中一个，另一个不需要重新学。

## 常见追问

**Q：那 Python 里写的 `for i in range(10): model(x)` 这个循环，是不是也全程在 C++ 里跑？**

不是。`for` 循环本身、每次调用 `model(x)` 这个"调用发生"的动作，仍然是 CPython 在驱动
（取 `model` 这个名字、构造调用、进入 `__call__`）。变化的只是**进入 `forward` 内部之后**
的那一段：eager 模式下内部还是 CPython 逐行跑，script 模式下内部转移给了 C++
`InterpreterState`。循环本身的 10 次跨语言调用开销（对应 7.3 节的边界转换）依然存在，
且发生 10 次——JIT 编译省的是"每次调用**内部**多少次跨边界"，不是"要不要发起调用"。

**Q：`self._c.forward(...)` 里的 `self._c` 每次访问都要跨边界转换吗？**

`self._c` 本身只是访问一个已经存在的 Python 属性（一个 `PyObject*`，内部包着指向 C++
`Module` 的指针），访问它不需要类型转换，跟访问任何 Python 属性一样便宜。真正发生
7.3 节所说的类型转换的，是**调用** `.forward(x)` 时对参数 `x` 和返回值做的转换，
不是访问 `_c` 这个属性本身。

**Q：能不能自己写代码验证"eager 模式跨边界次数多、script 模式跨边界次数少"？**

可以，`jit_learn/labs/07_scriptmodule_runtime.py` 里用 `sys.setprofile` 实测过：同一段
逻辑，eager 模式下 CPython 记录到的 Python 函数调用次数明显多于 script 模式（15 次 vs
3 次），这就是"每次算子跨回 Python"和"进入 forward 后不再跨回"两种模式的直接证据，
不需要凭空相信上面的链路图。
