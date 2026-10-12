# 第 11 章：torch.compile 与 TensorRT —— 执行原理到底差在哪

> 前面的章节一直在比较各条路线**跑得多快**，这一章回答一个更底层的问题：
> 两者**分别在做什么**？GPU 上最终执行的到底是什么？
>
> 一句话概括：torch.compile 是 PyTorch 内置的 **JIT 编译器**，目标是在不离开
> Python/PyTorch 的前提下加速训练和推理。TensorRT 是 NVIDIA 的
> **AOT 推理优化器 + 运行时**，目标是在 NVIDIA GPU 上把推理性能压到极致。

## 11.1 torch.compile 的执行流程

```
Python 代码
   │  TorchDynamo：在 Python 字节码层拦截 (PEP 523 frame evaluation hook)
   ▼
FX Graph（遇到不支持的代码就 graph break，那部分回退到 eager 执行）
   │  AOTAutograd：同时生成前向和反向图，算子分解为 ATen/Prims
   ▼
Backend（默认 TorchInductor）
   │  算子融合、内存规划
   ▼
生成 Triton kernel（GPU）/ C++/OpenMP（CPU）
```

几个关键点：

- **Guards 机制**：编译结果会附带一组检查条件（shape、dtype、Python 对象状态等）。
  每次调用先检查 guard，不满足就重新编译。
- **Graph break**：遇到 `print`、依赖数据的控制流、第三方库调用等，会把图切开，
  中间那段用 Python 正常执行。所以它**不要求整个模型都能编译**，兼容性很强
  （第 5 章的 `ctrlflow` 实验就是这个行为）。
- **动态 shape**：可以通过 `dynamic=True` 或自动检测，生成带符号 shape 的 kernel。
- **可插拔后端**：`backend="inductor"`（默认）、`"cudagraphs"`，
  甚至 `"tensorrt"`（Torch-TensorRT 提供）。

## 11.2 TensorRT 的执行流程

```
模型 (ONNX / Torch-TensorRT / TRT 网络 API)
   │  解析成 TRT Network（静态计算图）
   ▼
Builder 优化阶段（离线，可能耗时数分钟）
   ├─ 图优化：层融合 (Conv+BN+ReLU)、消除冗余层、张量拼接消除
   ├─ 精度校准：FP16 / INT8 (PTQ 校准或 QAT) / FP8 / INT4
   ├─ Kernel auto-tuning：对每层在目标 GPU 上实测多种实现，选最快的 tactic
   └─ 内存复用规划
   ▼
序列化 Engine (.engine / .plan)：与 GPU 型号、TRT 版本强绑定
   ▼
TRT Runtime 加载执行（可以不依赖 Python 和 PyTorch）
```

几个关键点：

- **整图静态编译**：要求整个图都能被 TRT 支持。遇到不支持的算子，需要写 Plugin
  （第 10 章），或者在 Torch-TensorRT 里回退到 PyTorch 执行（第 4、5 章）。
- **动态 shape**：通过 Optimization Profile 指定 min/opt/max 范围，
  范围之外的 shape 不支持（第 9 章）。
- **核心性能来源**：NVIDIA 手写和调优的 kernel 库（cuDNN/cuBLAS/CUTLASS/Myelin），
  加上低精度量化（第 8 章）。

## 11.3 对比表

| 维度 | torch.compile | TensorRT |
|---|---|---|
| 编译时机 | JIT，首次运行时编译 | AOT，离线 build 出 engine |
| 适用场景 | 训练 + 推理 | 只做推理 |
| 图捕获方式 | 字节码拦截，允许 graph break | 需要完整静态图（ONNX 等） |
| Kernel 来源 | 自动生成 Triton/C++ 代码 | 预置的高度调优 kernel 库 + auto-tuning |
| 量化 | 有限（依赖 torchao 等） | 原生 INT8/FP8/INT4，带校准工具 |
| 动态 shape | 灵活，符号化 shape | 需要预定义 profile 范围 |
| 硬件 | NVIDIA/AMD GPU、CPU 等 | 只支持 NVIDIA GPU |
| 部署依赖 | 需要 Python + PyTorch | 可以纯 C++ runtime，体积小 |
| 可移植性 | 代码可移植，缓存不一定能复用 | engine 绑定 GPU 架构和 TRT 版本 |
| 易用性 | 一行 `torch.compile(model)` | 需要导出、调试算子支持、处理精度问题 |
| 典型加速 | 1.3~2x（相对 eager） | 2~5x+（尤其开启 FP16/INT8 时） |

> 注意："典型加速"一行是业界常见的经验范围，**不是本机实测**。
> 本机的实测数据见第 6 章，INT8 在小 batch 下反而更慢这类反例见第 8 章。

## 11.4 GPU 上最终跑的到底是什么

一个常见的理解是："TensorRT engine 执行的是 cubin 机器码，torch.compile 执行的是
Backend 生成的 cubin 加上一些 eager 代码。"这个说法**基本对，但需要修正几处**。

GPU 上跑的最终都是 cubin（SASS 机器码），eager 模式也一样。
两者真正的区别在于：**kernel 是谁生成的**，以及 **CPU 端由谁来调度这些 kernel**。

### TensorRT engine 里有什么

engine 不全是 cubin，它更像一份"执行计划"：

- **权重**：可能已经转换成 FP16/INT8 等格式
- **每层选定的 tactic**：auto-tuning 后选出的 kernel 实现，记录成 ID
- **内存复用方案**、层的执行顺序
- **部分嵌入的 cubin**：比如 Myelin 在 build 时针对融合子图生成的 kernel

大量手写调优的 kernel 并不在 engine 里，而是在 `libnvinfer.so` 中，
engine 只按 tactic ID 去引用。有些 tactic 还会直接调用 cuBLAS/cuDNN。
如果用了 Plugin，执行的是 plugin `.so` 里的代码。

运行时由 **TRT Runtime（C++）** 按计划依次 launch 这些 kernel，全程不经过 Python。

### torch.compile 执行的内容

可以分三部分：

1. **Inductor 生成的 kernel**：流程是 Triton 源码 → PTX → ptxas → cubin，
   结果缓存在 `/tmp/torchinductor_*`。主要是逐元素操作、reduction 这类融合后的算子。
2. **外部库 kernel**：matmul 和 conv 默认**不**由 Triton 生成，而是调用
   cuBLAS/cuDNN（走 aten 的 extern kernel）。只有开 `mode="max-autotune"` 时，
   才会去比较 Triton 模板和 cuBLAS 的性能，选更快的那个。
3. **graph break 处的 eager 代码**：由 Python 解释器执行，调用的是 ATen 预编译好的
   kernel，本质上也是 `libtorch_cuda.so` 里的 cubin。

另外还有两块容易忽略的 **CPU 端开销**：

- **Guard 检查**：每次调用都要在 Python 里检查一遍。
- **Wrapper 代码**：Inductor 默认生成一个 **Python wrapper** 来依次 launch kernel。
  可以用 `cpp_wrapper=True` 或 AOTInductor（R6）改成 C++ wrapper，
  也可以用 `mode="reduce-overhead"` 借助 CUDA Graphs 把这部分 launch 开销基本消掉。

### 补充：Guard 到底是什么

**问题来源**：Dynamo 编译的是**某一次具体调用**时看到的那段 Python 代码。为了能生成
一张静态图，它会把很多东西当成常量“烧”进图里：输入的 shape、dtype、device，
Python 标量参数的值，全局变量 `torch.relu` 指向哪个函数，等等。
但 Python 是动态语言，下一次调用时这些前提都可能变了。

**Guard 就是这些前提条件的清单**。每份编译产物都附带一组 guard，每次调用时：

```
调用 compiled_fn(x, scale)
   │  逐个检查该函数缓存里各编译版本的 guard
   ├─ 全部通过 → 直接运行这份编译好的代码
   └─ 都不通过 → 重新编译（recompile），新版本加入缓存
                 超过 recompile_limit（本机默认 8）→ 放弃编译，回退 eager
```

所以 guard 的作用是**保证正确性**：用缓存的编译结果之前，确认“编译时的假设现在仍然成立”。
否则 shape 变了还在用旧 kernel，或者 `scale` 变了还在用烧进去的旧常量，结果就是错的。

**实测**：用下面这个最小例子，打开 `TORCH_LOGS="guards,recompiles"`：

```python
def f(x, scale):
    return torch.relu(x) * scale
cf = torch.compile(f)
cf(x, 2.0)                       # x.shape = [4, 8]，首次编译
cf(x, 3.0)                       # scale 变了
cf(torch.randn(16, 8, ...), 3.0) # shape 变了
```

第一次编译生成的 guard（日志摘录，有删减）：

```
TENSOR_MATCH: check_tensor(L['x'], ..., torch.float32, device=0,
              requires_grad=False, size=[4, 8], stride=[8, 1])
EQUALS_MATCH: L['scale'] == 2.0             ← Python 标量被当成常量烧进了图
ID_MATCH:     ___check_obj_id(G['torch'].relu, ...)   ← torch.relu 没被替换
GLOBAL_STATE: ___check_global_state()       ← grad 模式、autocast 等全局状态
Guard eval latency = 17.00 us
```

后两次调用分别触发了重新编译，日志直接给出了是哪个 guard 失败：

```
Recompiling function f  triggered by: - 0/0: scale == 2.0
Recompiling function f  triggered by: - 0/1: tensor 'x' size mismatch at index 0. expected 4, actual 16
```

还能看到 Dynamo 的**自动动态化**（`automatic_dynamic_shapes=True`）：
`scale` 第二次变了之后，新版本里的 guard 从 `scale == 2.0` 放宽成只检查类型；
shape 第二次变了之后，新版本的 guard 变成 `size=[None, 8]` 加上 `2 <= L['x'].size()[0]`，
也就是第 0 维变成符号 shape，以后 batch 再变就不用重新编译了。

**Guard 带来的三个实际影响**：

1. **每次调用的固定 CPU 开销**。guard 树已经用 C++ 实现（`TREE_GUARD_MANAGER`），
   上面单次评估是 17–45 us 的量级（这是开日志时打印的数字，不开日志会更快）。
   对比一下：本机上对一个 4 元素张量做 `x + 1`，eager 是 5.7 us/次，
   compile 后反而是 32.7 us/次。这里包含了 guard、Python wrapper 和 Triton launch，
   不全是 guard 的开销，但说明**小模型、小 batch 下 CPU 端开销会盖过 kernel 本身**。
2. **意外的重新编译**。把会变的 Python 标量（学习率、step 计数、温度系数）当参数传进去，
   每个新值都会触发一次 recompile，可能卡顿几秒。排查用 `TORCH_LOGS="recompiles"`；
   解决办法是改成传 tensor，或者对 shape 用 `torch._dynamo.mark_dynamic`。
3. **超过上限后静默回退**。重新编译次数超过 `recompile_limit` 后，该函数会直接走 eager，
   不报错，只有一条 warning，性能就悄悄掉回去了。

**和 TensorRT 的对比**：TensorRT engine 没有 guard 这个概念。它的“前提”在 build 时
就固定了：输入 shape 必须落在 Optimization Profile 范围内，否则直接报错，不会自动重新 build。
这也是它运行时 CPU 开销低的原因之一：调用时只做极少量的 shape 检查。
torch.compile 用 guard 换来了“什么 Python 代码都能试着编译”的灵活性，代价是每次调用都要查一遍。
`torch.export` 系列路线（R4、R6）同样不在运行时做 Python guard，
而是在导出时把 shape 约束固定下来，运行时只检查输入是否满足这些约束。

### 修正后的说法

| | GPU 端执行的 kernel | CPU 端调度 |
|---|---|---|
| TensorRT | TRT 库 kernel + build 时生成的 kernel + cuBLAS/cuDNN + plugin | TRT C++ Runtime |
| torch.compile | Triton 生成的 cubin + cuBLAS/cuDNN + graph break 部分的 ATen kernel | Python wrapper + guard + Python 解释器（eager 部分） |

所以 TensorRT 的优势主要来自两点：

1. **kernel 质量更高**，并且有低精度量化；
2. **CPU 端调度更轻**，没有 Python 开销。

在 Jetson 这类 CPU 较弱的设备上，第二点往往更明显。

## 11.5 两者可以结合：Torch-TensorRT

**Torch-TensorRT**（R4）把 TensorRT 作为 torch.compile 的一个后端：

```python
import torch_tensorrt
model = torch.compile(model, backend="tensorrt")
# 或者走 AOT 路径：torch.export → torch_tensorrt.dynamo.compile → 保存
```

TRT 能支持的子图交给 TRT 执行，不支持的部分回退到 PyTorch。
这样既有 TRT 的性能，又保留了 PyTorch 的灵活性。细节见第 4 章。

## 11.6 怎么选

- **训练加速、研究迭代、模型经常改动** → 用 torch.compile
- **生产推理部署、追求最低延迟或最高吞吐、需要 INT8/FP8、边缘设备** → 用 TensorRT
- **想要两者兼顾** → 用 Torch-TensorRT

本教程的环境是 **Jetson（Thor）**。在 Jetson 上做推理部署，TensorRT 基本是首选：
JetPack 自带 TRT，还能利用 INT8 等低精度，相比 torch.compile 在边缘端的优势更明显。
另外，Jetson 上 build engine 比较慢，建议 build 一次后把 engine 序列化缓存下来。

不过要记得第 6 章的数据：不需要 TRT 的时候，AOTInductor（R6）同样能去掉 Python 依赖，
并且在含不支持算子的模型上是全表最快的路线。"边缘端 = TensorRT"不是绝对的。
