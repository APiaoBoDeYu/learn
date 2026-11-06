# 01 · TorchScript 是什么，为什么需要它

## 1.1 起点问题：普通 PyTorch 模型有什么"缺陷"

你平时写的 `nn.Module` 是这样的：

```python
class Net(nn.Module):
    def forward(self, x):
        if x.shape[0] > 1:
            x = self.a(x)
        else:
            x = self.b(x)
        return x
```

这段代码**必须靠 CPython 解释器逐行执行**才能跑：`if` 判断、属性查找 `self.a`、
函数调用，全部是 Python 运行时的事。这意味着：

- 你不能把这个模型丢给一个**不装 Python、不装 PyTorch** 的 C++ 程序去跑
  （而 Thor 上部署的目标就是"不依赖 Torch 的运行时"——见 `torch2engine/README.md`）。
- 你不能让编译器/优化器提前分析整个计算图做算子融合，因为 `if` 分支要运行时才能确定走哪条。
- 你不能简单地把模型发给另一台机器执行——Python 源码 + 环境依赖太重、太脆弱。

TorchScript 就是为了解决"把 PyTorch 模型变成一份**不依赖 Python 解释器执行 forward 主体逻辑**
的可执行描述"而存在的一套子系统：通过**编译**（`script`）或**追踪**（`trace`）两种方式之一，
把 Python 模型转换成 TorchScript IR（`Graph`），再由一套独立于 CPython 的 **C++ 运行时**
（`GraphExecutor` 优化 + JIT interpreter 执行）来跑这份 IR。

这里有一个常见误解需要先澄清："受限的 Python 子集"这个说法**只精确适用于 `script`**：
`script` 是一个真正的编译器，要**静态解析**你的 Python 源码/AST，所以它必须能"看懂"你写的
每一行语法，落在支持子集之外的写法会在编译期直接报错（细节见 1.2(b)）。`trace` 完全是另一套
机制——它不解析、不检查你的 Python 代码，就是老老实实按 eager 模式跑一遍，把**这一次实际执行
过的张量算子序列**记录下来；你可以在被 trace 的函数里写任意 Python 语法（`print`、第三方库、
复杂 `if/for`），trace 都不会因为"语法不在子集内"而报错。trace 的问题不是"受限"，而是
"失真"：不作用于张量的 Python 控制流（比如条件依赖 shape 而不是 tensor 本身）会被按当次
取值直接拍死成一条路径，这是 1.2(a) 要讲的"致命限制"，跟 script 的"子集语言编译期报错"
是两种完全不同性质的限制，不要混为一谈。

`unified.pt` 就是这套子系统的产物：它是一个 **TorchScript Module**，被序列化到磁盘，
可以用 `torch::jit::load`（C++）或 `torch.jit.load`（Python）加载并执行，
全程不需要 covariant 项目的任何 Python 源码——这也是为什么它能被单独发布、
单独用来做 depth/backbone/segmentation 的 profiling（见你仓库里
`covariant/service/services/inference/tests/test_performance.py` 这类脚本的动机）。

## 1.2 两种生成方式：trace 与 script

TorchScript 有两条完全不同的"编译"路径，`unified.pt` 很可能是两者的混合（老实说，
混合是常态，纯 script 或纯 trace 的大模型很少见）。

### (a) `torch.jit.trace`

**做法**：给模型喂一次真实输入，*录制*这次前向传播中实际执行过的张量操作序列。

```python
traced = torch.jit.trace(model, example_input)
```

**心智模型**：像是"录屏"。它不理解你的 Python 代码逻辑，只记录"张量 A 经过了 conv，
变成张量 B；B 经过了 relu，变成张量 C……"。

**致命限制**：所有**不作用于张量**的 Python 控制流（`if x.shape[0] > 1`、
`for i in range(self.num_layers)` 这种 `self.num_layers` 是 Python int 而不是 Tensor 的情况）
在 trace 时会被**按这一次的实际取值直接展开/固化**，其他分支被永久丢弃。
这就是为什么你在读 `000_predict_backbone.py` 时看不到任何 `if`——
如果这段是 trace 出来的，所有条件分支早就在导出模型时被"拍死"成了一条路径。

**分界线到底在哪——不是"跟 shape 有没有关系"，是"这个值最后去哪了"。**
`x.shape[0]` 这类取值本身，trace 是否固化取决于它接下来被喂给了谁：

- **喂给 tensor 算子**（比如 `x.reshape(b, -1)` 的 `b`）：trace 记录的是
  `aten::size` 这个取值算子**本身**，是运行时会重新执行的图节点，不是死常量。
  实测跑一遍就能看到（`jit_learn/labs/01_trace_vs_script.py` 实验 B / B2）：

  ```python
  class ToyReshape(nn.Module):
      def forward(self, x):
          b = x.shape[0]
          return x.reshape(b, -1)

  traced = torch.jit.trace(ToyReshape(), torch.randn(5, 2, 3))
  ```

  `traced.graph` 里看到的是：

  ```
  %5 : int = aten::size(%x, %4)      # 运行时动态取 x 的第 0 维
  %b : Long = prim::NumToTensor(%5)
  ...
  %16 = aten::reshape(%x, %15)       # reshape 用的是上面动态算出来的 b
  ```

  用 `batch=5` trace 出来的模型喂 `batch=3` 的输入，输出 shape 正确变成 `(3, 6)`——
  batch 维不是拍死的常量。

- **喂给 python 原生的 `if`/`for` 条件**：`b > 1` 求值成一个**python bool**，
  不是 tensor 比较算子，trace 没有对应节点可记，只能记录"这次判断完走的是哪条路"，
  判断动作本身和没走到的分支一起永久消失。同样实测一下（同一个 lab 脚本的实验 B2）：

  ```python
  class ToyIfBranch(nn.Module):
      def forward(self, x):
          b = x.shape[0]
          if b > 1:            # b > 1 是 python bool，不是 tensor 比较算子
              return x * 2
          else:
              return x * 100

  traced = torch.jit.trace(ToyIfBranch(), torch.randn(5, 2, 3))   # 5 > 1 为真，走 *2
  ```

  编译出来的 `traced.code` 只剩一行：

  ```python
  def forward(self, x: Tensor) -> Tensor:
      return torch.mul(x, CONSTANTS.c0)   # 就是 x * 2，if/else 全部消失
  ```

  用 `batch=1`（本该走 `else`，该输出 `*100`）去喂这个模型，实际输出还是 `*2`
  的结果——不管之后喂什么 batch，这个模型永远只做 `*2`，`else` 分支从被
  trace 那一刻起就不存在了。PyTorch 会在 trace 时主动抛出警告，
  这行警告就是这条"致命限制"的官方措辞版：

  ```
  TracerWarning: Converting a tensor to a Python boolean might cause the trace
  to be incorrect. We can't record the data flow of Python values, so this
  value will be treated as a constant in the future.
  ```


### (b) `torch.jit.script`

**做法**：用一个**真正的编译器**（有词法分析、类型推断）读你的 Python 源码
（`forward` 方法体本身），编译成 TorchScript IR。它**理解**你的 `if`/`for`/递归，
只要你写的代码落在 TorchScript 支持的 Python 子集内。

```python
scripted = torch.jit.script(model)
```

**代价**：TorchScript 是**静态类型**的子集语言。普通 Python 里"什么都能塞"的灵活写法
（比如一个函数有时返回 `Tensor`，有时返回 `None`；比如动态往 `self` 上加属性；
比如用 `**kwargs`）大量不支持，写 script 兼容的代码需要显式类型标注、
避免动态特性——这也是为什么 covariant 项目里能看到大量
`is_in_onnx_export()` 风格的"导出守卫"代码：模型作者要**同时**兼顾训练时的灵活 Python
和导出时的 TorchScript 子集，写法上必须做妥协。

**"子集语言"和"静态类型"是两条独立的限制，值得分开理解：**

- **子集语言**：`torch.jit.script` 内部是一个真正的编译器（AST 解析 + 类型推断），
  只认 Python 语法/语义的一个被显式定义好的子集。落在这个子集之外的写法，
  哪怕跟类型完全无关，编译期直接拒绝——不是运行时才报错。
- **静态类型**：编译器要给每个变量、函数签名、返回值都推导出一个**确定且贯穿
  始终**的类型，运行时不允许变。这不等于"你必须手写类型标注"——很多时候编译器
  会自动推断（比如一次返回 `Tensor`、一次返回 `None`，会被自动推断成
  `Optional[Tensor]`，可以编译过）；真正的要求是**推断出来的类型必须自洽**，
  同一个函数在不同分支/不同调用路径上不能变成两个不兼容的类型。

  三个 eager 模式下完全合法、但 `torch.jit.script` 编译期直接报错的例子
  （见 `jit_learn/labs/01_trace_vs_script.py` 实验 E，可实际跑一遍看报错）：

  ```python
  # 例1：if/else 两个分支返回类型没有共同上界 —— Tensor 和 Tuple 之间无法统一
  class TrulyBadReturn(nn.Module):
      def forward(self, x, flag):
          if flag:
              return x * 2               # 类型 Tensor
          else:
              return x, x * 2            # 类型 Tuple[Tensor, Tensor]
  # RuntimeError: Previous return statement returned a value of type Tensor
  # but this return statement returns a value of type Tuple[Tensor, Tensor]
  ```

  ```python
  # 例2：list 字面量里混了 Tensor 和 str —— 普通 python list 天然支持混装，
  # 但 TorchScript 的 List[T] 要求元素类型统一，取出来后类型变成 Union[Tensor, str]
  class MixedList(nn.Module):
      def forward(self, x: torch.Tensor) -> torch.Tensor:
          items = [x, "not_a_tensor"]
          return items[0]
  # RuntimeError: Return value was annotated as having type Tensor
  # but is actually of type Union[Tensor, str]
  ```

  ```python
  # 例3：**kwargs —— 跟类型对不对齐无关，是"子集语言"这条限制本身，
  # 这种语法压根不在编译器认识的范围内
  class UsesKwargs(nn.Module):
      def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
          return x
  # NotSupportedError: Compiled functions can't take variable number of
  # arguments or use keyword-only arguments with defaults
  ```

  三个例子在 eager 模式下用 python 直接调用都完全没问题（普通变量没有固定类型、
  容器可以混装、参数可以随意透传）；一旦 `torch.jit.script`，例1、例2 是"静态类型"
  维度的报错（分支/容器类型不自洽），例3 是"子集语言"维度的报错（语法本身不支持，
  跟类型标注无关）。这正是 `is_in_onnx_export()` 这类导出守卫存在的原因：训练时
  为了灵活性写的动态返回值/动态参数，过不了 script 编译器，必须在导出路径上切换
  成类型自洽、参数显式列出的写法。

### (c) `torch.jit.trace_module`：trace 也能产出多具名方法

上一节说"trace 出来的模型只有一个 `forward`，没有额外具名方法"——这是针对
`torch.jit.trace(model, example_input)` 这个最常见调用形式说的，**不是 trace 机制本身的
限制**。PyTorch 还提供了 `torch.jit.trace_module`，专门用来对**同一个模型的多个方法**
分别录制：

```python
traced_module = torch.jit.trace_module(
    module,
    inputs={                       # key 是方法名，value 是该方法的示例输入
        "predict_a": (example_a,),
        "predict_b": (example_b,),
    },
)
traced_module.predict_a(...)       # 编译产物上确实有这两个具名方法
traced_module.predict_b(...)
```

`inputs` 字典的每个 key 对应模型上一个已存在的方法名，`trace_module` 会**对每个方法名
分别喂一次示例输入去录制**，最终产出的 `ScriptModule` 上就会有这些方法名对应的编译后
方法——跟 `@torch.jit.export` 完全无关，因为这条路径根本不经过"编译器分析 `forward`
调用了哪些方法"这一步，`inputs` 的 key 集合本身就是"要不要编译这个方法"的唯一依据。

这正是本仓库 `covariant/model/utils/export/unified.py` 里 `export_unified_model()`
真正采用的方式（封装成了 `covariant/model/utils/export/core.py` 里的 `trace_module()`
辅助函数）：

```python
inputs = {
    "predict_backbone": [rgbs],
    "predict_segm": [rgb, im_size, resnet_feats, obj_thresh, bin_thresh],
    "predict_depth": [rgbs, im_size, intrinsics, inv_intrinsics, ref_to_cam, near, far, resnet_feats],
}
traced_model = trace_module(model, inputs, strict=False)  # 内部调用 torch.jit.trace_module
```

`unified.pt` 里的 `predict_backbone`/`predict_depth`/`predict_segm` 三个方法，就是这样
分别单独 trace 出来的——**不是 script，源码里自然也不会有 `@torch.jit.export`**。
（`core.py` 里还多做了一步：先用一个 `WrapperModule` 把这三个方法名动态挂到一个新对象
上再 trace，是为了绕开 `trace_module` 要求模型有个可用 `forward` 的限制，跟"要不要具名
方法"这个核心问题无关。）

**这里 `resnet_feats` 传的是什么，是不是要先跑一遍 `predict_backbone` 拿真实输出？**
不需要——`unified.py` 的 `_prepare_inputs()` 里这段：

```python
resnet_feats=tuple(
    torch.randn(ncam, c, height // s, width // s, device=device, dtype=dtype)
    for c, s in zip(model.output_backbone_spec.channels, model.output_backbone_spec.integer_strides)
),
```

是**纯随机数**（`torch.randn`），shape 按 `model.output_backbone_spec` 里的 channel 数/stride
算出来，跟 backbone 真实产出的数值完全无关。



注意 trace 的"致命限制"（1.2(a) 提到的分支固化）对 `trace_module` 里的**每一个方法**
依然独立成立——`predict_backbone`、`predict_segm`、`predict_depth` 各自内部如果有
`if`/`for` 这类依赖 Python 值的控制流，各自都会被按当次示例输入固化，不会因为用了
`trace_module` 而豁免。

### (d) 混合：`script` 里嵌 `trace`，或反过来，或像 `unified.pt` 这样多方法都用 `trace_module`

一个大模型的顶层如果需要 `if is_in_onnx_export()` 之类**依赖 Python 值的分支**被完整
保留，就只能用 `script`；但如果只是需要"多个具名入口方法"，`trace_module`（见上）同样
能做到，不必上升到 `script`。`unified.pt` 暴露出的 `predict_backbone` / `predict_depth`
/ `predict_segm` 三个方法，实际验证下来正是 `trace_module` 分别 trace 出来的（见上一节），
不是 script——**"多具名方法"本身不能用来判断一个 `.pt` 是 script 还是 trace 出来的**，
真正能区分两者的是"方法内部有没有被完整保留的、依赖 Python 值的控制流"。
除此之外，一个模型内部某些确定不含分支、纯粹是张量运算的子模块也可能是单独
`trace`/`trace_module` 出来再拼进整体 `script` module 里的——`script` 和 `trace`
在同一个 `.pt` 内混用是常态。

官方unified模型导出确实用的是torch.jit.trace_module（是trace，不是纯script），但get_z_order/postprocess_detections这两个函数在003_segmentation_pipeline_standalone.py源码里本身已经用@torch.jit.script装饰过（我之前读代码时也确认了这一点，postprocess_detections定义前有@torch.jit.script）。

这是trace和script混合的经典模式：
- trace_module对predict_segm整体外壳（模块调用编排、张量搬运）做trace——这部分本身没有数据依赖分支，trace没问题
- 但当trace过程执行到调用get_z_order/postprocess_detections这两个已经被@torch.jit.script预编译的函数时，trace机制不会把它们内联展开成具体走过的那条分支路径的裸算子序列，而是把这个已经存在的、保留完整if/for控制流的ScriptFunction原样嵌入最终的trace图里，作为一个不透明的子图调用节点

这解释了为什么部署态JIT模型能对不同N、不同z_order分布的真实数据都正确工作——不是trace本身很聪明，是这两个关键的、真正含数据依赖控制流的函数根本没有被trace"看到"内部细节，它们是作为已经script好的黑盒子图被trace机制原样保留下来的。这与我之前看到的predict_segm图里那117个CallMethod/CallFunction节点是一致的——那些正是"被trace保留为不透明调用"的script/子模块边界，不是被内联展开的普通算子。

## 1.2b 跨架构：能不能在 x86 训练服务器上导出，给 ARM 板子用

这是实际部署时绕不开的问题：训练/导出通常在 x86 服务器上做，但目标硬件
（比如 Thor）是 ARM（`aarch64`）。答案是**可以**，而且不需要任何"交叉编译"
`.pt` 文件本身——原因要回到第 2 章会详细解剖的文件结构：

`.pt` 文件是一个 zip，里面只有两类东西：
- `archive/code/*.py`：**文本形式的 TorchScript IR**（`.code` 属性看到的那些 `def forward(...)`）
- `archive/data/*`：**纯数值的 tensor 字节**（float32/float16 数组）

**这两类东西都不含任何 CPU 架构相关的机器码**。

```python
import io, torch, torch.nn as nn

class Toy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

m = torch.jit.script(Toy())
buf = io.BytesIO()
torch.jit.save(m, buf)
print(buf.getvalue()[:120])
# b'PK\x03\x04...archive/data.pkl...\x80\x02c__torch__\nToy\nq\x00)...'
```

开头 `PK\x03\x04` 就是标准 ZIP 文件头，后面是 pickle 字节流和 IR 文本——
**没有 x86/ARM 的区分**。`torch.jit.script`/`trace` 做的是"把 Python 计算图
编译成一份平台无关的中间表示"，不是"编译成某个 CPU 的机器指令"，
所以这份产物天然就是跨架构的。

**这里容易产生一个误解：`archive/code/*.py` 里那些 `def forward(...)` 文本，
是不是从某种更底层的东西"反编译"出来的？不是——顺序完全反过来。**
`torch.jit.script(model)` 编译完成的那一刻，内存里首先得到的就是 TorchScript
IR（一张图：节点是 `aten::conv2d`/`aten::relu` 这类算子调用，边是 tensor 数据流），
这个 IR 本身才是"本体"。`torch.jit.save()` 序列化时，PyTorch 把这份 IR **按固定
规则打印**成一段类 Python 语法的文本，直接写进 `archive/code/*.py`——这是一个
**确定性的正向操作**（同一份 IR 永远打印出同一份文本），跟"从不透明二进制里
逆向猜测还原代码"的反编译（decompile）完全是两个方向。可以实测验证这个顺序：

```python
m = torch.jit.script(Toy())
print(m.code)              # 编译完成、还没 save 之前，IR 打印结果已经存在
torch.jit.save(m, "toy.pt")
# zip 里 archive/code/__torch__.py 的文本，跟上面 m.code 打印出来的逐字一致
loaded = torch.jit.load("toy.pt")
print(loaded.code)          # 重新 load 后打印结果仍然一致——只是原样读回，不是重新计算
```

`.code`/`.graph` 这两个属性名字里都没有"decompile"，官方叫法是
un-parse（反解析）/ pretty-print（打印）——这也是为什么本仓库文档统一改叫
"IR 反解析代码"，不再用"反编译代码"这个容易引起误解的说法（真正需要靠
by-layer 比对去"猜"内部结构的是编译后的 TensorRT `.engine` 二进制，那才是
这个仓库里名副其实的逆向工程场景，见下表和 `project_tools/PITFALLS.md`）。

**真正区分架构的是"加载并执行这份文件"的运行时**，不是文件本身：

| 阶段 | 产物 | 是否架构相关 | 需要什么 |
|------|------|------------|---------|
| 训练 | `.pth` checkpoint | 否 | Python + PyTorch（任意架构都行） |
| 导出 TorchScript | `.pt`（本文档讨论的对象） | **否**——纯 IR + 权重字节 | Python + PyTorch（导出机器的架构，训练用的那台就行） |
| **加载/执行 `.pt`** | 推理结果 | **是**——要看目标机器装的运行时 | ARM 板子上要有 **ARM 架构编译的 PyTorch/LibTorch** |

流程因此是：
1. 在 x86 服务器上正常 `torch.jit.script(model).save("model.pt")`——不需要
   任何 ARM 相关的工具链，就是普通的模型导出。
2. 把 `model.pt` 直接 `scp` 到 ARM 板子上——就是文件拷贝，跟拷一张图片没区别。
3. ARM 板子上需要有能"看懂"这份 IR 并在**当前这块 CPU/GPU 上**翻译执行的运行时：
   - 装了 ARM 版 PyTorch（Python 环境）：`torch.jit.load("model.pt")` 直接能跑；
   - 或者只装 LibTorch（C++ 运行时，不需要 Python）：
     `torch::jit::load("model.pt")`，编译链接 `libtorch.so`（ARM 版）即可。

这跟你在本仓库更熟悉的 **TensorRT `.engine` 文件正好是反例**，值得对照记住：

| | `.pt`（TorchScript） | `.engine`（TensorRT） |
|---|---|---|
| 内容 | 平台无关 IR + 权重 | 针对具体 GPU 型号**提前编译好**的 kernel 调度计划 |
| x86 生成、ARM 直接用 | ✅ 可以，直接拷贝 | ❌ 不行，必须在目标 GPU（或同型号）上用 `trtexec`/builder 重新构建 |
| 硬件指令翻译发生在哪一步 | **加载时**，由 PyTorch/LibTorch 现场完成 | **构建时**，已经针对具体 GPU 架构（SM 版本）、显存做了 tactic 选择 |

这也是为什么 `torch2engine_method.md` 反复强调"目标环境：aarch64 / NVIDIA Thor
/ TRT 10.14"——`.engine` 换了硬件就可能要重新构建；而 `.pt`/`.onnx` 这类
"IR 层"产物换硬件通常不需要重新生成，只需要目标机器上有对应的运行时。

**一个例外，需要单独注意**：如果模型引用了自定义 CUDA 算子（比如你们的
`soft_bin_ops`、`roi_upsample_cuda`），对应的 `.so` **是真正的机器码，是架构相关的**，
必须单独在 ARM 上编译出 ARM 版 `.so`，再用 `torch.ops.load_library` 加载——
这个依赖和 `.pt` 文件本身无关，是"自定义算子"这个额外依赖带来的；
PyTorch 内置算子（conv/linear/relu 等）已经被编译进官方发行的 ARM 版
PyTorch/LibTorch 包里，不需要你自己操心。

**一个实际的坑，提前说一下:** torch.jit.load 有一个隐性依赖：如果模型引用了自定义算子（像你们的 soft_bin_ops、roi_upsample_cuda），这些 .so 是真正的编译产物、是架构相关的机器码，必须在 ARM 上单独编译一份 ARM 版的 .so，再用 torch.ops.load_library 加载。这一点和 .pt 本身无关，是"自定义 CUDA 算子"这个附加依赖带来的，纯 PyTorch 内置算子（conv/linear/relu 等）不存在这个问题，因为它们已经被编译进 PyTorch/LibTorch 官方发行包里了。

### 待验证任务：unified 模型在 x86 导出、ARM 加载后分割结果差异很大（未结案）

这是一次真实遇到的故障，先记录排查进展，方便下次接着查，不要重新从头猜一遍。

**现象**：`unified.pt` 在 x86 上导出，拷到 ARM（本机）上用
`torch.jit.load` + Python 加载执行（跟 `extrace_info.py` 同样的方式），
`predict_segm` 的分割结果和 x86 上跑出来的相差很大（未量化到具体
abs/cosine 数值，肉眼可见明显不对）。自定义 CUDA 算子（`soft_bin_ops`、
`roi_upsample_cuda` 等）是在 ARM 上单独重新编译的 `.so`，用
`torch.ops.load_library` 加载。

**已排除的可能性**：
- `.pt` 文件本身跨架构可用性——按本节原理，IR + 权重字节不含机器码，
  `torch.jit.load` 在 ARM 上确认能正常加载不报错。
- `roi_upsample`（`covariant/common/torch_ops/roi_upsample/roi_upsample_cuda_kernel.cu`）
  的 CUDA kernel 源码本身：逐行核对过，CPU/CUDA 两版实现的坐标换算、
  双线性插值逻辑完全一致，没有架构敏感写法（无 warp intrinsic、
  无硬编码 SM 版本假设）；Bazel 注册宏（`my_common.h` 里的
  `DISPATCH_CPU_OR_CUDA`）也是普通设备判断，没有可疑之处。
- `torchvision.ops.roi_align`（`predict_segm` 里通过
  `ops.torchvision.roi_align(..., 7, 7, 2, False)` 调用，见
  `jit2engine/jit_code/segmentation/000_predict_segm.py` 第 281-285、
  452-456 行）——已在两台机器上用同样的固定输入实测过（x86 torchvision
  0.17.2+cu121，ARM torchvision 0.24.1，版本号相差 7 个 minor 版本，
  一度是最大嫌疑），**结果一致**，可以排除。

**尚未验证、下次应该查的方向**（按怀疑程度排序）：
1. `soft_bin_ops`（depth 用得多，但 segmentation 是否间接依赖需要确认——
   `predict_segm` 的完整算子引用列表还没有像 `roi_align` 那样逐个跑过）、
   `patch_outerprod`、`demosaicing`、`sym3eig` 这几个自定义算子，
   逐个用固定输入在 x86/ARM 上分别跑一遍数值对比（可以参考
   `jit_learn/labs/05_cross_arch_export.py` 的 export/verify 模式，
   针对单个算子而不是整个模型）。
2. x86 和 ARM 上完整的 `torch`/`torchvision`/CUDA/cuDNN 版本矩阵是否
   完全一致（目前只确认了 torchvision 版本不同但恰好无影响，torch
   本身版本、cuDNN 版本尚未系统比对）——版本不同不代表一定有问题
   （这次 torchvision 差 7 个版本反而没事），但仍是需要先排除的基础项。
2b. 特别检查**这几个自定义算子有没有依赖 cuDNN/cuBLAS 里某些在
   ARM 上表现不同的融合路径**，而不是只看自己的 `.cu` 源码——源码逐行一致
   不代表调用的底层 BLAS/cuDNN kernel 在 ARM Tensor Core 架构上数值行为一致。
3. 用 `jit2model_guide.md` 的"逐层二分定位法"，在 ARM 上对 `predict_segm`
   做分阶段中间张量 dump（参考 backbone/depth 案例里的
   `--debug-stage-outputs` 思路），同步在 x86 上跑同样的中间张量，
   找到"从哪一层开始数值就已经不对"，再回头看那一层用了哪个算子——
   这比继续对着算子源码猜更可靠，应该是下次排查的第一步，而不是最后一步。
4. 排查模型加载后是否被显式或隐式转换了 dtype/device 相关的默认行为
   （例如 ARM 上某些 GPU 默认开启的 TF32/降精度路径与 x86 不同），
   这类问题往往不在自定义算子里，而在 PyTorch 自带 op 的精度模式上。

## 1.3 为什么 IR 反解析代码里"看不到超参数"

这是 `jit2model_guide.md` 里反复强调、也是最容易踩坑的点，这里从原理上解释一次：

- TorchScript **子模块之间的调用关系**（`self._backbone_fpn_handler.forward(...)`）
  是**结构性**的：子模块在 `__init__` 时被注册为 `nn.Module` 的属性，
  这个"树状结构"被完整保留在序列化文件里（下一章会看到 `code/` 目录里每个子模块
  一个 `.py` 文件，就是这棵树的体现）。所以 Step 1（读 JIT 代码抓调用骨架）永远可靠。

- 但**子模块内部"是否有激活函数"、"用了什么归一化"这类由 Python 端`if`/配置对象
  决定的结构**，如果这个决定发生在**构造 `nn.Module` 树的阶段**（也就是
  `__init__`/`__attrs_post_init__` 跑的时候），那么这个决定的产物只是"这个位置上挂了
  一个 `nn.LeakyReLU(0.2)` 还是一个 `nn.Identity()`"——**这两者对 TorchScript 编译器来说
  都是合法的、已经实例化好的子模块，编译器只管把它们的 `forward` 编译进去，
  不会去问"你当初为什么选了这个"**。所以 IR 反解析代码只会写
  `_act = self._end; y = (_act).forward(x)`，你看到的是"调用了 `_end`"，
  但 `_end` 到底是哪种激活，要么靠 IR 反解析代码里那个子模块自己的 `.code`
  （下一章会看），要么就干脆看不出来——如果它是 `Identity`，
  TorchScript 甚至可能把这一步直接优化掉，代码里连调用都不出现。

  这正是 `jit2model_guide.md` 里 backbone FPN 那个坑的根本原因：
  `conv_spec.as_no_nonlinearity()` 这行 Python 代码只在**构造阶段**执行了一次，
  它的"决定"永久固化成了"这个位置放的是 `Identity` 模块"，
  这个决定本身（"为什么是 Identity"）不会出现在任何 `.pt` 文件里。

## 1.4 小结：三层信息，三个来源

| 信息层 | 例子 | 保留在 `.pt` 里？ | 去哪里找 |
|--------|------|------------------|---------|
| 模块调用结构 | "FPN 消费 RegNet 的 5 个 blob" | ✅ 完整保留 | JIT IR 反解析代码（`000_predict_*.py`） |
| 已实例化的子模块类型 | "这个位置是 `LeakyReLU` 还是 `Identity`" | ✅ 保留（但要看该子模块自己的 `.code` 或 `state_dict` 里有没有对应参数） | 子模块自己的 IR 反解析代码 + 源码 handler |
| 决定该用哪种子模块/数值超参的**逻辑和来源** | "为什么 slope 是 0.2 不是 0.01" | ❌ 完全丢失 | 只能回 covariant 源码 + `config.yaml` |

下一章开始真正解剖 `unified.pt` 这个 zip 文件，把上面这张表格里"✅"对应的东西
在文件系统上指给你看。
