# 第 5 章：torch.compile 进阶

## 5.1 Graph Break：为什么 torch.compile 比 TorchScript 更"抗造"

完整代码见 [`code/05_graph_break_demo.py`](code/05_graph_break_demo.py)。

先看一个包含数据依赖控制流的函数：

```python
def has_graph_break(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    if y.sum().item() > 0:   # .item() 强制 GPU->CPU 同步，拿到具体 Python int
        z = y + 1
    else:
        z = y - 1
    return z * 2
```

如果这是 `torch.jit.trace`：给定一次输入，trace 只会记录**实际走过的那个分支**，之后不管
输入怎么变，永远执行录制时走的那条路——这是一个典型的"看起来能跑，结果却是错的"陷阱。

`torch.compile` 的处理方式不同：用 `torch._dynamo.explain` 可以看到它具体做了什么。

### 实际运行

```bash
python3 code/05_graph_break_demo.py
```

关键输出（已省略冗长的 guard 列表）：

```
=== torch._dynamo.explain on a function WITH a graph break ===
Graph break from `Tensor.item()`, consider setting:
    torch._dynamo.config.capture_scalar_outputs = True
Graph break: from user code at:
  File ".../05_graph_break_demo.py", line 18, in has_graph_break
    if y.sum().item() > 0:

Graph Count: 2
Graph Break Count: 1
Op Count: 3
Break Reasons:
  Break Reason 1:
    Reason: Unsupported Tensor.item() call with capture_scalar_outputs=False
    Explanation: Dynamo does not support tracing `Tensor.item()` with
                 config.capture_scalar_outputs=False.
```

**`Graph Count: 2`，`Graph Break Count: 1`**：Dynamo 把这个函数拆成了两段图，中间那个
`.item()` 触发同步的地方用 eager 执行。整个函数依然能跑出正确结果（对，也对错，两个分支
都会按实际输入选择），只是没有被融合成一个连续的编译图——**这就是"graph break"**：
性能可能打折扣，但正确性不受影响。这是 torch.compile 相比 `trace` 的核心优势。

### 用 `fullgraph=True` 强制定位 break 点

如果你想确保关键路径完全没有 graph break（比如追求 `reduce-overhead`/CUDA Graph 模式的
最大收益），可以用 `fullgraph=True` 让 break 直接变成异常，方便定位：

```python
compiled_strict = torch.compile(has_graph_break, fullgraph=True)
compiled_strict(x)  # 抛异常，而不是静默 break
```

实测报错：

```
Raised as expected: Unsupported: Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
```

把 `.item()` 换成纯 tensor 操作（`torch.where`）后，同一个函数变成零 break：

```python
def no_graph_break(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    z = torch.where(y.sum() > 0, y + 1, y - 1)  # 保持在 tensor 世界里，不触发同步
    return z * 2
```

```
=== torch._dynamo.explain on the rewritten, break-free function ===
Graph Count: 1
Graph Break Count: 0
Op Count: 6
```

`fullgraph=True` 在这个版本上直接跑通，不再抛异常。

## 5.2 动态 Shape 与重新编译

完整代码见 [`code/05_dynamic_shapes.py`](code/05_dynamic_shapes.py)。

torch.compile 编译出的代码会针对**具体输入 shape**特化（这样才能做 tile 大小等编译期决策）。
输入 shape 变化时，之前编译的代码不再适用，触发**重新编译**，由一套"guard"机制检测：

```python
model = nn.Linear(128, 128).cuda().eval()
compiled = torch.compile(model)
for batch in [8, 16, 32]:
    x = torch.randn(batch, 128, device="cuda")
    compiled(x)
```

用 `TORCH_LOGS=recompiles` 能看到触发重新编译的具体原因：

```bash
TORCH_LOGS=recompiles python3 code/05_dynamic_shapes.py
```

实测输出：

```
=== Static shapes: a new batch size triggers a recompile each time ===
Recompiling function inner in .../torch/_dynamo/external_utils.py:66
    triggered by the following guard failure(s):
    - 0/0: tensor 'args[0]' size mismatch at index 0. expected 8, actual 16
```

有意思的是：只出现了**一条**重新编译日志（8→16 那次），16→32 那次**没有**触发——这不是
脚本的 bug，是 torch.compile 的默认行为：`automatic_dynamic_shapes` 机制在检测到同一个
维度第二次变化后，会自动把那个维度标记为动态（用符号变量代替具体数值编译），之后这个维度
的进一步变化就不再需要重新编译了。这是一种"渐进式动态化"的启发式策略——静态编译换更好的
特化性能，动态编译换更少的重编译开销，torch.compile 默认尝试在两者间自动取得平衡。

### 主动标记动态维度：`mark_dynamic`

如果你在部署前就知道某个维度会变化（比如 serving 场景的 batch size），可以提前显式标记，
类似 TensorRT 的 optimization profile（预先指定 min/opt/max shape 范围）：

```python
x0 = torch.randn(8, 128, device="cuda")
torch._dynamo.mark_dynamic(x0, 0)  # dim 0（batch）标记为动态
compiled2(x0)
for batch in [16, 32, 64]:
    x = torch.randn(batch, 128, device="cuda")
    compiled2(x)  # 不应触发新的 recompile
```

### 实际运行

```
=== torch._dynamo.mark_dynamic: mark batch dim dynamic up front ===
(with mark_dynamic, no further recompiles should appear above)
```

实测确认：这一段没有再打印任何 `[__recompiles]` 日志，`mark_dynamic` 生效。

**与 TensorRT 的对照**：TensorRT 的 optimization profile 是在 build engine 时**离线**指定
形状范围，之后运行时超出范围会直接报错；torch.compile 的动态 shape 是**运行时按需**决定
（第一次遇到变化才决定要不要动态化），超出范围不会报错，只是可能触发一次新的编译。
这是 AOT 与 JIT 两种编译时机在处理动态性上的直接体现。

## 5.3 把手写的 Triton kernel接入 torch.compile

这是本章、也是本教程 Triton 与 torch.compile **关系最紧密**的例子。完整代码见
[`code/05_custom_triton_in_compile.py`](code/05_custom_triton_in_compile.py)。

场景：你写了一个 Triton kernel（比如第 2 章的 add，这里做成 `add + relu` 融合版），
想让它作为 `nn.Module.forward` 里的一步，同时让 torch.compile 继续融合它前后的其他算子。
用 `torch.library.triton_op` 把 kernel 注册成一个自定义算子：

```python
@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, tl.maximum(z, 0.0), mask=mask)


@torch.library.triton_op("tutorial::add_relu", mutates_args={})
def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    torch.library.wrap_triton(add_relu_kernel)[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelWithCustomKernel(torch.nn.Module):
    def forward(self, x, y):
        z = add_relu(x, y)  # 我们手写的 Triton kernel
        return z * 2 - 1    # 普通 op，Inductor 会尝试融合它们
```

`torch.library.triton_op` + `torch.library.wrap_triton` 的作用：告诉 Dynamo"把这一整个
调用当成一个不透明节点来追踪，不要试图深入解析 kernel 内部"——这与你给 TensorRT 写自定义
plugin 时"告诉 builder 这一层是黑盒，别试图拆解融合"的思路是一致的。区别是这里整个过程
仍在 Python/torch.compile 的图里，图的其余部分依然能被 Inductor 正常融合。

### 实际运行

```bash
python3 code/05_custom_triton_in_compile.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
[PASS] custom_triton_op_eager: allclose=True, max_abs_err=0.000e+00
[PASS] custom_triton_op_compiled_fullgraph: allclose=True, max_abs_err=0.000e+00
eager (custom kernel + 2 ops)    : 0.0630 ms
compiled (kernel fused with rest): 0.0713 ms

fullgraph=True succeeded: Dynamo traced straight through our
hand-written Triton kernel without a graph break.
```

关键结论：`fullgraph=True` **没有报错**——证明 Dynamo 完整地追踪穿过了我们手写的 Triton
kernel，没有产生 graph break。这个例子里编译版本反而略慢（0.071 ms vs 0.063 ms），
因为算子本身极小、graph 融合的空间有限，编译开销（额外的 dispatcher 调用/guard 检查）
盖过了收益——诚实呈现：**接入自定义 kernel 不保证一定更快，价值在于让你的手写 kernel
能参与到更大图的融合决策里**，在算子更大、融合空间更多的真实模型中才能体现优势。

> **踩坑记录**：这个例子最初在 `triton==3.6.0` 下运行失败，报
> `ImportError: cannot import name 'specialize_impl'`——这是 torch 2.9.1 的
> `torch._higher_order_ops.triton_kernel_wrap` 内部依赖的 triton 符号在 3.6.0 里发生了变化。
> 降级到 `triton==3.5.0` 后问题消失（见第 1 章）。这类"自定义 kernel 接入 torch.compile"的
> 高阶特性对 torch/triton 版本匹配的要求比单纯用 Triton 写 kernel 更严格，遇到类似
> `ImportError` 时先检查版本搭配。

## 5.4 小结

- Graph break 是 torch.compile 用来保证正确性优先的机制：追踪不了就退回 eager，不会像
  `trace` 那样静默产生错误结果；`fullgraph=True` 可以把 break 变成显式报错，便于定位
- 默认的动态 shape 策略是"渐进式"的：同一维度第二次变化后自动转为动态，减少重复编译；
  `mark_dynamic` 可以提前显式声明，效果类似 TensorRT 的 optimization profile 但作用于运行时
- `torch.library.triton_op`/`wrap_triton` 是 Triton 手写 kernel 接入 torch.compile 图的
  标准方式，让自定义 kernel 也能参与更大范围的图级融合——这是本教程里 Triton 和
  torch.compile 结合得最紧密的一环

下一章：[第 6 章：综合对比](06_benchmark_comparison.md) —— 把 TorchScript、torch.compile、
手写 Triton、ONNX+TensorRT 放在同一个模型上做真实 benchmark
