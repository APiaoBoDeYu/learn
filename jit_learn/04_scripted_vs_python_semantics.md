# 04 · Scripted 语义 vs 普通 Python 语义：信息去哪了

前两章讲的是"怎么读语法"，这一章讲更根本的问题：**为什么有些信息从源码到
`.pt` 文件的路上会彻底消失**，这是语义层面的、不可逆的信息损失，
不是"你没读懂"的问题。

## 4.1 真实机制：分支往往根本没进入 TorchScript 编译器的视野

容易先入为主地以为"TorchScript 编译器做了常量传播，把 `if` 优化掉了"。
实际验证下来（见 `jit_learn/labs/01_trace_vs_script.py` 实验 A）更准确的说法是：
**很多"分支"根本不是发生在 `forward` 里的 `if`，而是发生在 `__init__` 里的
普通 Python 逻辑**，`forward` 从头到尾只有一行确定的调用。

covariant 里 `Nonlinearity` 的真实写法
（`covariant/common/learning_utils/layers/nonlinearity.py:74-84`）就是这个模式：

```python
class Nonlinearity(nn.Module):
    def __init__(self, nonlin_args: NonlinArgs):
        super().__init__()
        self._nonlin: nn.Module
        if isinstance(nonlin_args, IdentityNonlinArgs):
            self._nonlin = nn.Identity()
        elif isinstance(nonlin_args, LeakyReluArgs):
            self._nonlin = nn.LeakyReLU(negative_slope=nonlin_args.negative_slope, inplace=nonlin_args.inplace)
        # ...
        else:
            raise NotImplementedError

    def forward(self, x):
        return self._nonlin(x)
```

**这个 `if` 跑在 `__init__` 里，是纯 Python 代码，跑在 `torch.jit.script(model)`
被调用之前，跟 TorchScript 编译器完全无关**。它执行的结果是：
`self._nonlin` 这个属性被绑定成 `nn.Identity()` 或 `nn.LeakyReLU(0.2)`
**这两种不同类型的对象之一**。等到之后 `torch.jit.script` 去编译 `forward` 时，
它看到的 `forward` 里从始至终只有一行 `return self._nonlin(x)`——
**编译器根本没有 `if` 要处理，因为分支从未存在于它的输入里**。

**结果**：无论 `nonlin_args` 是哪种类型，编译出的 `.pt` 文件里
`Nonlinearity.forward` 的 IR 反解析代码都长一个样：

```python
def forward(self, x: Tensor) -> Tensor:
    _nonlin = self._nonlin
    return (_nonlin).forward(x, )
```

**"为什么选了 Identity 而不是 LeakyReLU"这个判断逻辑（`isinstance(nonlin_args, ...)`）、
以及 `nonlin_args` 这个变量本身，永远不会出现在 `.pt` 里**——不是因为编译器把它优化掉了，
而是因为它从一开始就只存在于构造阶段的 Python 世界，构造完成后就被丢弃、
只留下"结果"（`self._nonlin` 指向哪个具体子模块实例）。



（如果你确实需要在 `forward` 里写一个跟随 `Tensor` 值变化的运行时 `if`——
比如 `if x.shape[0] > 1`——那种 `if` 会被完整保留在 TorchScript IR 里，
IR 反解析代码里能看到 `if`。两者的区别是：**跟 Tensor 运行时值有关的分支会被保留；
只由构造时 Python 对象决定、构造完就不再变化的分支，根本不会以"分支"的形式
进入编译器**。区分这两者，是读 JIT 代码时判断"这里有没有可能藏着一个我不知道的
分支"的关键直觉。）

## 4.1b 第二种更直接的机制：`Final` 属性——构造参数本身被烧录成字面常量

`Nonlinearity` 例子里消失的是**分支逻辑**；还有一种更直接的消失，
不涉及任何分支，纯粹是**构造参数的具体数值被烧录进编译产物**。

`jit_learn/labs/03_module_hierarchy_and_mangling.py` 里可以亲眼验证：
`nn.Conv2d(in_ch, out_ch, kernel_size=3, ...)` 构造完之后，`torch.jit.script`
编译出的 `Conv2d.code` 长这样（这是真实跑出来的输出，不是示意）：

```python
class Conv2d(Module):
  ...
  kernel_size : Final[Tuple[int, int]] = (3, 3)
  in_channels : Final[int] = 3
  out_channels : Final[int] = 8
  stride : Final[Tuple[int, int]] = (1, 1)
  padding : Final[Tuple[int, int]] = (1, 1)
  ...
```

`Final[T] = <具体值>` 是 TorchScript 类型系统里"编译期常量属性"的显式声明：
一旦某个属性被判定为 `Final`（对不可变的 Python 内建类型，如
`int`/`str`/`Tuple[int, ...]`，且构造后不再被赋值，TorchScript 会自动推断出它是
`Final`），**这个属性的值就永久嵌入这份编译产物本身**，不再是"运行时从某个
`nn.Module` 实例上读出来的属性"，而是**这份 `.code` 文件自带的常量**。

这解释了 mangle 机制的本质：**mangle 后缀的数量 = 这个类被实例化时用到的、
不同 `Final` 属性取值组合的数量**。`unified.pt` 里 341 个 `torch.nn.modules.conv`
版本，直接对应 341 种不同的 `(in_channels, out_channels, kernel_size, stride, ...)`
组合——每一种组合的具体数值都完整保留在对应那份 `.code` 里，**这一点反而是
"没有消失"的信息**：只要你找到某个具体 Conv2d 实例对应的那份 mangled `.code`，
就能直接读出它的 `kernel_size`/`stride` 等构造参数，不需要回源码。

**和 4.1 节 `Nonlinearity` 例子的区别**：
- `Nonlinearity` 消失的是"为什么选 Identity 还是 LeakyReLU"这个**判断逻辑**
  （`isinstance` 分支本身），但选中的**那个子模块类型**依然可读。
- `Conv2d` 的 `Final` 属性消失的是"这个 `kernel_size=3` 当初是从哪个配置字段算出来的"
  这个**来源**，但**数值本身**完整保留、直接可读，不需要回源码。

两种情况共同的教训是一样的：**TorchScript 保真地记录了"构造完成后的最终状态"，
但从不记录"为什么会是这个状态"**。数值超参数（`Final` 属性）通常能直接读到；
决策逻辑和依据（配置字段来源、`isinstance` 判断）永远读不到。

## 4.2 三种"消失方式"，对应三种找回策略

| 消失方式 | 例子 | 找回策略 |
|---------|------|---------|
| **分支发生在 `__init__`（纯 Python），从未进入 `forward`/编译器** | `isinstance(nonlin_args, ...)` 的判断逻辑本身消失，`forward` 里只剩一行调用 | 回源码找决定走哪条分支的配置/逻辑（`config.yaml` + handler 构造代码） |
| **构造参数被烧录成 `Final` 常量** | `kernel_size=3` 变成 `Final[Tuple[int,int]] = (3, 3)`；`nn.GroupNorm(eps=1e-5)` 的 `eps` 同理 | 数值本身可以直接从该子模块的 `.code` 读到，**但"为什么是这个值"要回源码/`config.yaml`** |
| **子模块被实例化成某个具体类型，但类型本身不携带"为什么"** | `self._end` 是 `Identity` 还是 `LeakyReLU` 各自都合法，能在 IR 反解析代码里认出类型，但认不出决策依据 | 读该子模块自己的 `.code`（能看出类型），若需要知道决策依据回源码 |

**关键实践含义**：`jit2model_guide.md` 里"Step 2 读源码 handler"和
"Step 3 读配置文件"不是可选步骤、不是"更仔细一点会更好"——
它们是**唯一**能找回"分支/决策逻辑本身"这类消失信息的路径，因为这类信息在
`.pt` 文件里物理上不存在了，无论你多仔细读 IR 反解析代码都不可能读出来。
而"构造参数的具体数值"和"子模块的具体类型"理论上都可以从 IR 反解析代码里抠出来
（去读该子模块自己的 `.py`），但**效率远低于直接读源码**——
covariant 源码有类名、有注释、有配置字段名，IR 反解析代码只有
`___torch_mangle_239.Conv2d` 这种无意义的名字，两条路径殊途同归时，
永远选源码；只有当你需要"验证某个具体实例到底吃了什么参数"时，
才值得去点开那份 mangled `.code` 直接确认数值。

## 4.3 一个可以亲手验证的最小复现


## 4.4 trace 模式下的另一种信息消失：形状/常量固化

如果 `unified.pt` 里某个子模块是 `trace` 出来的（不是 `script`），
还会有第二种消失：**任何跟输入张量的形状/取值相关但没被识别成"跟随输入变化"
的中间量，会被固化成字面常量**。比如：

```python
def forward(self, x):
    b = x.shape[0]              # trace 时 x.shape[0] == 5（假设 trace 用了 5 个相机）
    return x.view(b, -1)        # trace 出来的代码里，5 可能被写成字面量 5，不是 x.shape[0]
```

这是 `project_tools/PITFALLS.md` 第 17 条"当前 batch/cams 支持是静态 shape engine"
背后更早的一层原因——如果某个环节是靠 trace 拿到的，输入相机数、分辨率这些形状信息
在源头上就有被写死的风险，后面到 ONNX/TensorRT 阶段的"静态 shape"问题，
很多时候根源要往前追到这一步的 trace 行为。**看到 IR 反解析代码里出现具体数字
（而不是 `.shape[i]` 这种表达式）时，要警惑：这个数字究竟是"设计上就是常量"
（比如卷积核大小 3），还是"trace 那一次输入形状恰好如此、被误固化的变量"**。
区分方法：换一个不同 shape 的输入重新 trace/推理，看这个数字会不会变——
不变则是真常量，变了说明原来的模型对形状不是真正动态的。

## 4.5 小结

- Script 编译期常量传播 → **决策逻辑本身**（`if` 条件、为什么选这个配置）永久消失，
  只能回源码/配置文件找。
- 子模块类型（Identity vs LeakyReLU）→ 类型本身理论上能从 IR 反解析代码认出来，
  但读源码效率更高，应优先读源码。
- Trace 模式下的形状/取值 → 可能被静默固化成字面常量，是"部署时发现动态维度
  支持不了"这类问题的常见根源，需要主动怀疑、主动验证。

下一章开始动手：不看别人分析，自己跑一遍最小实验，把这三章的结论
在自己手上验证一次。
