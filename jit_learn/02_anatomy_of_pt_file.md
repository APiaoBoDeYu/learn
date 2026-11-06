# 02 · 解剖 unified.pt：一个 TorchScript 文件到底装了什么

本章直接对着真实的 `unified.pt` 做拆解。目标不是把每个 pickle 字节都解释完，而是建立一张稳定的地图：

```
TorchScript archive
├── code/          # 类型、函数、方法体：Python-like TorchScript 程序文本
├── data.pkl       # module object state：属性、子模块、parameter/buffer 引用关系
├── constants.pkl  # 方法代码里嵌入的常量
└── data/          # 大块 Tensor storage：权重/缓冲区/tensor attribute 的原始字节
```

一句话先定调：

> `torch.jit.script/trace` 产出的是 `ScriptModule` 或 `ScriptFunction`，不是一份裸 `Graph`；这些对象内部的每个函数/方法才有自己的 TorchScript IR，也就是 C++ 侧的 `torch::jit::Graph`。`torch.jit.save` 保存的是整个 JIT module object graph，而不是把某个 Graph 原样 dump 到磁盘。

## 2.1 `.pt` 文件的真实身份：ZIP archive

`torch.jit.save` 生成的 `.pt` 文件本质是一个 zip。可以不依赖 PyTorch，直接用系统工具或 Python `zipfile` 查看。

```bash
file /root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15
# → Zip archive data, at least v0.0 to extract, compression method=store
```

对这个 `unified.pt`，核心条目大致是：

```text
archive/code/                   # Python-like TorchScript 源码；通常一个编译后的类/函数一个 .py
archive/data.pkl                # module object state：属性、子模块、parameter/buffer 引用关系和值
archive/constants.pkl           # 方法代码里嵌入的常量；不是 module 的 parameter/buffer
archive/data/                   # Tensor storage 原始字节：0, 1, 2, ...
archive/traced_inputs.pkl       # trace 时留下的样例输入记录
archive/traced_inputs/0..13     # trace 样例输入的原始张量
archive/byteorder               # 张量存储字节序
archive/.data/version           # TorchScript 序列化格式版本号
archive/.data/serialization_id  # 序列化 ID
```

更通用的 TorchScript archive 结构可以理解成：

```text
model.pt
├── code/
│   ├── __torch__.py
│   ├── __torch__.py.debug_pkl
│   └── package/submodule.py
├── data.pkl
├── constants.pkl
├── data/
│   ├── 0
│   ├── 1
│   └── ...
├── version
└── byteorder   # 新版本中可能存在
```

`unified.pt` 的差别主要是多了一层 `archive/` 前缀，所以会看到 `archive/code/`、`archive/data.pkl`、`archive/data/0` 这种路径。


## 2.2 先分清内存对象：ScriptModule、ScriptFunction、Graph

很多误解来自把 `ScriptModule`、`ScriptFunction` 和 `Graph` 混成一个东西。更准确的层次是：

```text
ScriptModule / ScriptFunction     # 可调用对象、容器、对外接口
        ↓
Function / Method                 # 某个具体函数或方法
        ↓
torch::jit::Graph                 # 这个函数/方法内部的计算图 IR
```

如果 script/trace 的是 `nn.Module`，产物是 `ScriptModule`。一个 `ScriptModule` / C++ 侧的 `torch::jit::Module` 大致长这样：

```text
ScriptModule / torch::jit::Module
├── type / class schema
├── methods
│   ├── forward
│   ├── exported methods
│   └── 每个 method 内部有 TorchScript Function / Graph
├── submodules
│   ├── self.layer1
│   ├── self.encoder
│   └── 本质上是 module attribute，值是另一个 ScriptModule object
├── parameters
│   ├── self.weight
│   ├── self.bias
│   └── Tensor / Parameter state
├── buffers
│   ├── running_mean
│   └── running_var
└── normal attributes
    ├── int / float / bool / str
    ├── list / tuple / dict
    ├── Optional
    └── Tensor / custom TorchScript class object
```

这里要区分两类信息：

- `methods` 是“怎么计算”，内部有 TorchScript `Function` / `Graph`。
- `submodules`、`parameters`、`buffers`、`attributes` 是“对象持有什么状态”，保存时主要靠 `data.pkl` 和 `data/` 恢复。

如果 script/trace 的是普通函数，产物是 `ScriptFunction`：

```python
import torch

def f(x, y):
    return x + y * 2

sf = torch.jit.script(f)
```

可以近似理解为：

```text
ScriptFunction
├── name: f
├── schema: f(Tensor x, Tensor y) -> Tensor
├── code: Python-like TorchScript code
└── graph: torch::jit::Graph
```

所以 `ScriptFunction` 是可调用对象，`Graph` 是它内部函数体的计算表示。

`script` 和 `trace` 最终都会形成 Graph，但来源不同：

```text
script function -> ScriptFunction -> Graph
script module   -> ScriptModule   -> method Graphs

trace function  -> ScriptFunction -> Graph
trace module    -> ScriptModule   -> method Graphs
```

`torch.jit.script` 从 Python 源码/AST 编译出 TorchScript IR，能保留 TorchScript 支持的控制流：

```python
def f(x, flag: bool):
    if flag:
        return x + 1
    else:
        return x - 1
```

script 后的 Graph 里会出现 `prim::If`。也就是说，`if` 作为控制流节点被保留下来。

`torch.jit.trace` 用样例输入跑一遍函数/模型，记录这次实际执行过的 tensor ops：

```python
traced = torch.jit.trace(model, example_input)
```

trace 不真正理解 Python 控制流。如果分支依赖输入数据，它通常只捕获 example input 当时走过的那条路径。

## 2.3 `code/`：TorchScript 源码，不是原始 Python 源码

查看 `code/` 目录：

```bash
python3 - <<'EOF'
import zipfile

p = "/root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15"
z = zipfile.ZipFile(p)
code_files = [n for n in z.namelist() if n.startswith("archive/code/") and n.endswith(".py")]

print("code/ 下 .py 文件数:", len(code_files))
for n in code_files[:20]:
    print(" ", n)
EOF
```

你会看到类似：

```text
archive/code/__torch__/covariant/model/perception/unified/backbone/handlers/backbone_fpn_handler.py
archive/code/__torch__/torch/nn/modules/conv/___torch_mangle_239.py
archive/code/__torch__/covariant/common/learning_utils/layers/conv_layer/___torch_mangle_238.py
```

`code/*.py` 的准确定位是：

> 它是序列化阶段保存的 Python-like TorchScript 程序文本，用来在 `torch.jit.load` 时重建 TorchScript 类型、函数和 Graph。它不是原始训练源码，也不是 `torch::jit::Graph` 的裸 dump，更不是从机器码反编译出来的 Python 源码。

这点很重要。反编译通常指从不透明的二进制/字节码猜测还原高级语言代码；TorchScript archive 里的 `code/*.py` 不是这样来的。它是 TorchScript serialization 的 code-like 表示，可以被 TorchScript 编译器重新解析。

一个 `code/*.py` 文件里可能有两种东西：

- **编译后的类**：可能包含多个方法，例如 `WrapperModule` 里同时有 `predict_backbone`、`predict_segm`、`predict_depth`、`forward`。
- **模块级函数**：例如 `archive/code/__torch__/covariant/common/torch_ops/roi_upsample.py` 里可能只有顶层 `def _roi_upsample_script(...)`，没有 class。

哪些方法会出现在 `code/` 里，取决于它有没有被实际编译到：

- `script` 通常来自 `forward` 调用链和 `@torch.jit.export` 方法。
- `trace_module` 来自传给它的 `inputs` 字典 key。
- 没被编译到的方法/分支不会凭空出现在 `code/*.py` 里。

### 为什么有大量 `___torch_mangle_NNN`

这个模型里有大量 `nn.Conv2d` 实例。它们在 Python 里属于同一个类 `torch.nn.modules.conv.Conv2d`，但不同实例的 `kernel_size`、`stride`、`in_channels` 等构造参数不同，编译后的 TorchScript 类型/方法体也可能不同。

TorchScript 的编译单元更接近“编译后的类型”，不是“Python 源码里的一个类名”。为了区分这些编译产物，序列化时会出现：

```text
Conv2d
___torch_mangle_239.Conv2d
___torch_mangle_245.Conv2d
...
```

它们本质上都来自“看起来一样的 Conv2d”，但被当成不同的编译后类型保存。

这直接影响模型还原：不能靠 `code/` 里的 mangled 类名去猜它对应源码里的哪个子模块。稳定线索是 `data.pkl` 里的属性路径，例如：

```text
self._backbone_regnet_handler.s1.b1...
```

`jit_learn/labs/03_module_hierarchy_and_mangling.py` 用 toy 网络复现了这个规则：两个构造参数完全相同的 `ConvBlock` 可能共享同一份 `__torch__.ConvBlock`，第三个参数不同的实例会被分配到 `__torch__.___torch_mangle_1.ConvBlock`。脚本还会打印出类似：

```python
kernel_size : Final[Tuple[int, int]] = (3, 3)
```

这说明构造时传给 `nn.Conv2d(..., kernel_size=3)` 的参数已经被编译器固化成 TorchScript 类型里的 `Final` 常量声明。

### 怎么把 `code/` 路径读回源码路径

`archive/code/__torch__/...` 这条路径不是随意生成的，它就是这个类原始 Python 导入路径的直接映射。解析规则：

1. 去掉前缀 `archive/code/__torch__/`，把剩下路径里的 `/` 换成 `.`；
2. 如果文件名带 `___torch_mangle_N` 后缀，先去掉这一段再做上一步；
3. 得到的字符串就是这个类在源码里的 import path，可以直接拿去项目源码树里 `find`/`grep`。

```text
archive/code/__torch__/torch/nn/modules/conv.py
  -> torch.nn.modules.conv                                  （PyTorch 内置类）

archive/code/__torch__/covariant/model/perception/unified/backbone/handlers/backbone_fpn_handler.py
  -> covariant.model.perception.unified.backbone.handlers.backbone_fpn_handler
  -> 对应源码文件 covariant/model/perception/unified/backbone/handlers/backbone_fpn_handler.py

archive/code/__torch__/covariant/common/learning_utils/layers/normalization/___torch_mangle_259.py
  -> （先去掉 ___torch_mangle_259 这段）
  -> covariant.common.learning_utils.layers.normalization
  -> 对应源码文件 covariant/common/learning_utils/layers/normalization.py
```

**mangle 编号本身不含语义**——不是"第几个实例"，不是创建顺序，只是编译器内部全局递增的去重计数器。所以"这个 `___torch_mangle_259.py` 具体对应哪个实例（哪一层的 GroupNorm）"这件事无法从路径本身反推，只能回上一节说的 `data.pkl` 属性路径（例如 `self._backbone_fpn_handler._lateral_fpn_cores.0`），或者在 `torch.jit.load` 之后用 `type(submodule).__name__` / `submodule.original_name` 逐层对号。

一句话总结这条映射关系的边界：**路径前半段（去掉 mangle 后缀）能直接告诉你"这是源码里的哪个类"，但不能告诉你"这是这个类的哪个实例"**——前者是静态类型信息，后者要靠对象树（`data.pkl`）才能确定。

### 这个模型的 `code/` 规模

跑 `jit_learn/labs/02_inspect_pt_zip.py` 第 3 部分可以看到：

```text
code/ 目录一共有 2599 个 .py 文件
其中 2550 个带 mangle 后缀
只有 49 个不带 mangle 后缀
```

按“去掉 mangle 后缀、只看类家族”归类后，实例最多的几类是：

```text
nonlinearity                    372
conv_layer                      345
torch.nn.modules.conv           341
normalization                   315
torch.nn.modules.container      289
```

不用打开任何一行 TorchScript 源码，只看这些数量也能看出模型结构：这是一个由大量卷积、归一化、激活和容器模块堆出来的深 CNN，包括 backbone RegNet、FPN、depth 的 3D UNet、segmentation 的多头结构。

## 2.4 `data.pkl`、`data/`、`constants.pkl`：状态和数值在哪里

### `data.pkl`：module object state

查看 `data.pkl` 的开头：

```bash
python3 - <<'EOF'
import zipfile

p = "/root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15"
z = zipfile.ZipFile(p)
raw = z.read("archive/data.pkl")

print(raw[:300])
EOF
```

它是 pickle 字节流，但里面不是普通 Python 对象，而是 TorchScript 反序列化器能理解的对象状态。它描述了重建 `WrapperModule` / `UnifiedModel` 这棵对象图需要的信息：

- 顶层 module 和 submodule 的对象关系。
- attributes 的名字、类型和值。
- parameter / buffer / tensor attribute 对 storage 的引用。
- 普通 attribute，例如 int、float、list、dict、Optional、TorchScript class object。

开头可以看到类似：

```text
__torch__.covariant.model.utils.export.core.WrapperModule ... module ...
__torch__.covariant.model.perception.unified.orchestration.basic.UnifiedModel ...
```

这和 `000_predict_backbone.py` 里的访问链对应：

```python
module = self.module
_backbone_fpn_handler = module._backbone_fpn_handler
```

`self` 是最外层 `WrapperModule`，`self.module` 是 `UnifiedModel` 实例，`_backbone_fpn_handler` 是它的子模块属性。这条属性访问链和训练代码里的 `model.module._backbone_fpn_handler` 是同一种对象关系，只是现在它被固化在 TorchScript archive 的对象状态里。

不要直接用普通 `pickle.load(open(..., "rb"))` 解这个文件。它引用的类路径是 `__torch__.covariant...`，普通 Python 环境里没有这些类；必须通过 `torch.jit.load` 这个专用反序列化器，它知道如何把 TorchScript 类型系统和对象状态接起来。

这也解释了为什么加载模型前要先注册自定义算子：

```python
torch.ops.load_library(...)
torch.jit.load(...)
```

如果模型 Graph 里引用了 `roi_upsample_cuda`、`patch_outerprod`、`sym3eig` 之类 C++/CUDA op，而对应 `.so` 没有提前注册到 dispatcher，反序列化或执行时就会报“找不到算子”。

### `data.pkl` 不是 `__init__` 的完整快照

`data.pkl` 保存的是 TorchScript 反序列化时需要恢复的 module object state，不是原始
Python `__init__` 执行后的完整 `__dict__`，也不是 `__init__` 里出现过的所有局部变量。

判断一个东西会不会进 `data.pkl`，不要只问“它有没有写在 `__init__` 里”，而要问：

- 它是不是被编译后的 `ScriptModule` 持有的 module state？
- 它是不是 submodule、parameter、buffer 或 TorchScript 支持序列化的 attribute？
- 它的类型能不能进入 TorchScript 类型系统并被 archive 反序列化器恢复？

通常会作为对象状态保存的，是这类挂到 `self` 上的东西：

```python
self.foo = SomeModule()
self.weight = torch.nn.Parameter(...)
self.register_buffer("running_mean", torch.zeros(3))
self.scale = 0.5
self.names = ["a", "b"]
```

其中 submodule、parameter、buffer 是最典型的 module state。普通 attribute 也可能保存，
但前提是它被 TorchScript 接纳为 attribute，并且类型可序列化，例如 int、float、bool、
str、Tensor、list、dict、Optional 或 TorchScript class object。

不应该指望这些东西一定保存：

- 普通 Python 对象，例如 `object()`、文件句柄、logger、复杂 helper。
- 只用于导出过程的临时属性。
- 没有进入编译后 `ScriptModule` 类型系统的 Python 侧状态。
- `__init__` 里的局部变量，因为它们本来就不是 module attribute。

### forward 没用到的属性会不会保存

会不会保存，主要取决于它是不是编译后 module state，而不是取决于当前 `forward`
有没有读它。

例如：

```python
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Conv2d(3, 8, 3)
        self.unused = torch.nn.Linear(10000, 10000)

    def forward(self, x):
        return self.used(x)
```

如果 `self.unused` 作为 submodule 被保留下来，它仍然可能出现在 `data.pkl` / `data/`
里，因为它是 module object state。`forward` 没有用到它，并不自动等于保存时删掉它。
同理，未被当前 `forward` 使用的 parameter、buffer，也可能作为 state 保存。

但这不意味着每次 JIT 推理都会执行它。JIT 执行的是当前 method 的 Graph。如果 Graph
里没有这些节点：

```text
prim::GetAttr[name="unused"]
prim::CallMethod[name="forward"](%unused, ...)
aten::xxx(%unused_weight, ...)
```

那 `unused` 就不会参与当前 `forward` 的运行时计算。

因此，forward 未使用但被保存的 module state 通常影响的是：

- `.pt` 文件体积。
- `torch.jit.load` 加载时间。
- 运行时内存占用。
- `state_dict()` 大小和参数遍历成本。

通常不影响的是：

- 当前 `forward` Graph 的节点数量。
- JIT interpreter 每次执行的 instruction 数量。
- 当前推理的 dispatcher 调用次数。
- 当前推理的 kernel 计算时间。

例外是：如果当前 Graph 间接读取了这个属性，例如读取 `self.flag` 控制分支，或者读取某个
buffer 参与 shape/scale 计算，那它就不是“没用到”，会体现在 Graph 中，并可能影响优化路径
或运行时开销。

### `data.pkl` 不记录函数调用关系

`data.pkl` 记录对象状态，不记录“forward 里调用了哪些函数/算子”。反过来，`forward`
里直接调用的函数/算子不会作为对象树节点出现在 `data.pkl` 里：

```python
torch.ops.custom.patch_outerprod(...)
some_script_function(x)
```

在这个 `unified.pt` 里可以这样验证：

```text
soft_bin_ops     搜 data.pkl: 未出现   搜 code/*.py: 也未出现
roi_upsample     搜 data.pkl: 未出现   搜 code/*.py: 出现
patch_outerprod  搜 data.pkl: 未出现   搜 code/*.py: 出现
sym3eig          搜 data.pkl: 未出现   搜 code/*.py: 出现
```

结论：只搜 `data.pkl` 会漏掉函数式自定义算子引用。要判断某个算子是否被模型使用，必须同时搜 `data.pkl` 和 `code/*.py`。

### `data/`：Tensor storage 原始字节

查看 `archive/data/`：

```bash
python3 - <<'EOF'
import zipfile

p = "/root/brain/brain-service/jit2engine/jit_model/unified.pt/e779c45986c16bb78c851fb56a38c1a9-15"
z = zipfile.ZipFile(p)
data_files = [n for n in z.namelist() if n.startswith("archive/data/")]

print("data/ 下条目数:", len(data_files))
print("最大的几个:")
for sz, n in sorted(((z.getinfo(n).file_size, n) for n in data_files), reverse=True)[:5]:
    print(f"  {sz / 1e6:8.2f} MB  {n}")
EOF
```

`archive/data/<N>` 是纯二进制 tensor storage。shape、dtype、stride 等元信息不直接写在这个二进制文件名里，而是通过 `data.pkl` 里的反序列化记录关联回去。

这就是为什么 `torch.jit.load` 后仍然能拿到类似下面的 `state_dict()`：

```python
{
    "_backbone_regnet_handler.stem._core._conv.weight": tensor(...),
    ...
}
```

无名的 `data/0`、`data/1` 会通过 `data.pkl` 里的属性路径重新挂回具体 parameter / buffer / tensor attribute。

### `constants.pkl` 和 `traced_inputs.pkl`

`constants.pkl` 保存的是方法代码里嵌入的常量。它不是 module 的 parameter/buffer；parameter/buffer 属于 module object state 和 tensor storage。

`archive/traced_inputs.pkl` 和 `archive/traced_inputs/0..13` 是这个模型经过 `torch.jit.trace` 的直接物证。纯 `script` 模块通常不会留下“当时喂过的样例输入”记录；trace 需要样例输入，所以 archive 里可能保留这些 traced input 信息。

## 2.5 保存和加载：archive 如何变回 ScriptModule

把前面的内容串起来：

```text
torch.jit.save(script_module, path)
  = 保存 code-like 信息到 code/*.py
  + 保存 module object state 到 data.pkl
  + 保存方法内嵌常量到 constants.pkl
  + 保存大块 Tensor storage 到 data/
```

加载时：

```text
torch.jit.load(path)
  = 解析 code/*.py 里的 TorchScript 源码
  + 重建 TorchScript 类型、函数、方法和对应 Graph
  + 恢复 data.pkl 里的 module object state
  + 把 data/ 里的原始 storage 包装回 Tensor
  → 得到可调用的 ScriptModule
```

注意加载的结果不是原始 Python `nn.Module`，而是 `ScriptModule`。它不依赖原始 `covariant.*` Python 源码来执行 `forward`，但依赖用到的底层算子已经注册好。

整体执行链路可以画成：

```text
ScriptModule / ScriptFunction
  ↓ method / function
torch::jit::Graph
  ↓ GraphExecutor 优化、特化、生成执行计划
torch::jit::Code / Instruction
  ↓ JIT interpreter 解释执行
PyTorch dispatcher
  ↓
ATen / cuDNN / cuBLAS / custom C++/CUDA kernel
```

## 2.5b `ScriptModule` 到底是不是 C++ 对象、调用时怎么跨语言

上面的链路图把 `ScriptModule` 画成起点，但它本身是什么、Python 调用它时具体发生了什么，值得单独说清楚。

**准确说法：`ScriptModule` 是一个 Python 对象，但它的核心状态和方法体全部是 C++ 对象——Python 层只是一层薄壳。**

```text
Python 侧:  torch.jit.ScriptModule / RecursiveScriptModule (Python 类)
              │
              │ 持有一个属性 self._c
              ▼
C++ 侧:    torch::jit::Module (C++ 对象，通过 pybind11 暴露给 Python)
              │
              ├── methods: 每个方法是 torch::jit::Function，内部有 Graph
              ├── submodules: 递归的 torch::jit::Module 树
              ├── parameters/buffers: 真正的 at::Tensor（C++ Tensor）
              └── attributes
```

- `torch::jit::Module` 是货真价实的 C++ 类。`torch.jit.load` 底层调用的是 C++ 侧的 `torch::jit::load()`，反序列化出来的**首先**是这个 C++ 对象，不是 Python 对象。
- Python 侧的 `ScriptModule` 是用 `pybind11` 包了一层，`self._c` 这个属性直接是那个 C++ `Module` 对象的 Python 句柄——**不是拷贝，是同一份内存**。
- 这跟 `torch.Tensor` 的实现模式完全一致：Python `Tensor` 对象包着 C++ `at::Tensor`/`c10::TensorImpl`，`ScriptModule` 包着 `torch::jit::Module`，都是"Python 外壳 + C++ 内核"。

以 `feats = model(rgbs)` 为例，跨语言调用的完整链路：

```text
1. Python: model(rgbs)                            ← 触发 Python 的 __call__
2. Python: ScriptModule.__call__ 内部调用 self._c.forward(...)   ← pybind11 绑定
3. 跨越 Python/C++ 边界（GIL 内完成一次 C++ 函数调用）
4. C++: torch::jit::Module::forward(...)           ← 找到 "forward" 这个 Method
5. C++: GraphExecutor::run(...)
     - 第一次遇到这组 shape/dtype：跑优化 pass（常量折叠、死代码消除、fusion——见 2.7 节），
       生成一份 ExecutionPlan 并缓存
     - 后续同类型输入：直接复用缓存的 ExecutionPlan，跳过重新优化
6. C++: InterpreterState 逐条执行 Instruction        ← 操作 IValue 栈，不是 Python 对象
7. C++: dispatcher 按 device/dtype 分发到具体 kernel
8. C++: cuDNN / cuBLAS / ATen native kernel 实际执行  ← 结果是 C++ 侧的 at::Tensor
9. 结果 IValue（包着 at::Tensor）跨回 Python/C++ 边界
10. Python: 包装成 Python 侧 torch.Tensor 返回给调用者
```

几个关键点：

- **只在第 2→3 步和第 9→10 步跨越一次 Python/C++ 边界**，中间第 4-9 步全程在 C++ 里跑——不会像 eager 模式那样每个算子都要回一次 Python。这正是 2.6 节"更少的动态分派和 Python 边界开销"的具体机制。
- 第 5 步的 `ExecutionPlan` 缓存是关键优化点：`GraphExecutor` 按输入的 shape/dtype/device 组合做特化缓存，同一组合第二次调用直接跳过优化 pass。这也解释了为什么第一次调用（warm-up）通常比后续调用慢。
- 第 6 步的 `InterpreterState` 操作的是 `IValue`（PyTorch 内部的 tagged union，能装 Tensor/int/List/Tuple/…），不是 Python 的 `PyObject`——没有 Python 对象头、没有 Python 语义的引用计数、没有属性字典查找，这是 JIT 比 eager 快的底层原因。
- 如果 Graph 里有自定义算子（`torch.ops.xxx`），第 7 步的 dispatcher 查表会分发到 `torch.ops.load_library` 注册的 C++/CUDA kernel，同样不回 Python。

一句话总结：`ScriptModule` 是"Python 句柄 + C++ 实体"的组合，`self._c` 就是那道桥；`model(x)` 调用时 Python 只在**入口和出口**露一下脸，中间"跑图"的全过程——优化、缓存执行计划、逐条执行指令、分发到 kernel——完全发生在 C++ 侧，这也是 2.6 节"不回到 Python"这句话的具体所指：不是完全不碰 Python，而是 forward 主体逻辑的执行权从 CPython 解释器转移到了 C++ JIT runtime。

上面每一条结论都可以亲手验证，不需要凭空相信：`jit_learn/labs/07_scriptmodule_runtime.py` 用 5 个实验逐条实测——`self._c` 的真实类型、`register_forward_hook` 在 `ScriptModule` 上为什么直接报错而不是"注册了但不生效"、用 `sys.setprofile` 数出 eager 和 script 模式下 Python 函数调用次数的差异（15 次 vs 3 次）、`GraphExecutor` 首次调用比后续调用慢一到两个数量级（GPU 上实测约 60 倍）、以及序列化后的模型在原始 Python 类被删除后依然能 `torch.jit.load` 并正确执行。

## 2.6 为什么 trace 之后也叫 JIT 模型

`JIT` 是 Just-In-Time compilation。这里的“编译”不要狭义理解成“把整个模型翻译成机器码”；在 TorchScript 语境里，它主要指：

> 在运行时附近，把 Python 模型/函数转换成 TorchScript Graph，并由 JIT runtime 做优化、特化、生成和缓存执行计划。

`torch.jit.trace` 是 JIT/TorchScript 系统的一种前端。它用样例输入跑一遍 Python 模型，把这次实际执行过的 tensor ops 记录成 TorchScript Graph，最后产出 `ScriptModule` 或 `ScriptFunction`。

普通 Python 模型和 JIT 模型的关键区别是：**谁负责执行 `forward` 的主体逻辑**。

```text
普通 Python 模型:
nn.Module + Python forward
  ↓ Python 解释器进入 model.__call__ / forward，一行行执行 Python 代码
ATen / cuDNN / cuBLAS kernel

JIT / trace 模型:
ScriptModule + TorchScript Graph
  ↓ 进入 PyTorch C++ JIT runtime，由 GraphExecutor / interpreter 执行 Graph
ATen / cuDNN / cuBLAS kernel
```

所以 trace 后叫 JIT 模型，是因为模型已经从：

```text
Python 对象 + Python forward
```

变成了：

```text
ScriptModule + TorchScript Graph
```

从 Python 里调用一个 JIT 模型，大致是：

```text
Python 调用 ScriptModule.__call__
  ↓ 进入 PyTorch C++ JIT runtime
C++ GraphExecutor / interpreter 执行 TorchScript Graph
  ↓ 遇到 aten::conv / aten::relu / aten::matmul
C++ dispatcher 调用 ATen / cuDNN / cuBLAS kernel
  ↓ Graph 执行完成
结果 Tensor 返回给 Python
```

这里的“不回到 Python”指的是不会再逐行执行原始 `nn.Module.forward()`。入口和出口仍然可以在 Python。如果 Graph 里残留 `PythonOp` / Python fallback，执行时可能回 Python；但一个正常保存、跨机器加载、C++ 可加载的 TorchScript 模型通常应该避免这种依赖。

自定义算子如果是通过 `torch.ops.xxx` 注册的 C++/CUDA op，JIT 执行时走的是 dispatcher 到 C++/CUDA 实现，不是回 Python 源码。

## 2.7 优化和算子融合发生在哪里

JIT interpreter 最后确实可能是逐条执行 instruction，但它执行的是**优化后的执行计划**，不是原始 trace/script Graph。

关键点：

> 算子融合不是 interpreter 执行到一半临时决定的，而是在 interpreter 执行之前，由 `GraphExecutor` 的优化 / fusion pass 先改写 Graph。

简化流程：

```text
1. trace/script 得到原始 Graph
   aten::add
   aten::relu
   aten::mul

2. GraphExecutor 在执行前优化 Graph
   常量折叠
   死代码消除
   peephole optimization
   shape/type specialization
   fusion pass

3. fusion pass 把可融合的连续小算子替换成 fused node
   aten::add + aten::relu + aten::mul
   ↓
   prim::FusionGroup / TensorExprGroup / CUDAFusionGroup 等

4. 生成 Code / Instruction
   interpreter 后面看到的已经不是三条普通 aten instruction，
   而是一条 fused group 调用。

5. 执行 fused group
   调用 fuser 生成/缓存的融合 kernel，或者调用对应融合后端。
```

所以不是：

```text
interpreter 执行到 aten::add，临时发现后面有 relu/mul，于是现场合并
```

而是：

```text
执行前先扫描 Graph，把 aten::add -> aten::relu -> aten::mul
改写成一个 fused node；interpreter 后面只负责执行这个 fused node。
```

“知道全图长什么样”的价值就在这里：优化 pass 可以提前看到连续子图，判断这些节点是否都是 elementwise/pointwise、shape/device/dtype 是否兼容、中间结果有没有被别处使用、有没有副作用或 alias 风险，然后安全地把一段子图替换成一个融合节点。

这个融合能力有边界：

- 通常容易融合：连续逐元素算子、简单 broadcast、部分 reduction。
- 通常不这样融合：`conv`、`matmul` 这类大算子，因为它们本来就会分发到 cuDNN/cuBLAS/MKL/ATen 的高性能现成 kernel。
- 自定义 op、复杂 indexing、动态 shape、alias/副作用风险高的子图，也可能无法融合。

因此 TorchScript JIT 的主路径仍然是：

```text
解释优化后的执行计划
  ↓
分发到现成 ATen / cuDNN / cuBLAS / custom kernel
```

fuser/NNC/nvFuser/oneDNN Graph 这类局部优化后端只是对特定子图的额外优化层，不是全模型默认生成机器码。

## 2.8 为什么编译后可能更快

TorchScript 比 eager PyTorch 快，并不是因为卷积、矩阵乘这些 kernel 换了一套。很多时候真正计算的底层 kernel 是同一份：

```text
conv2d  -> cuDNN / oneDNN / ATen
matmul  -> cuBLAS / MKL / ATen
relu    -> ATen / fused kernel
```

差异主要来自“到达 kernel 之前”的路径。

普通 eager 执行 `self.conv(x)`，每次都要经过：

```text
nn.Module.__getattr__
  ↓
nn.Module.__call__
  ↓
forward_pre_hooks / forward_hooks 检查
  ↓
Python 函数调用建栈帧
  ↓
Python object 和 C++ Tensor 边界转换
  ↓
dispatcher 选择 kernel
  ↓
实际 kernel
```

TorchScript 编译后，很多事情可以提前解析：

- `self.conv` 这种属性查找可以变成固定槽位/索引。
- 构造参数可以固化成 `Final` 常量，例如 `kernel_size : Final[Tuple[int, int]] = (3, 3)`。
- GraphExecutor 可以做常量折叠、死代码消除、peephole 优化。
- 推理场景通常配合 `no_grad` / `inference_mode` 避免 autograd 记录。
- 连续小算子可能通过 fuser 减少中间 tensor 读写和 kernel launch。

JIT interpreter 也是“逐条解释”，但它解释的是 C++ runtime 的 `Instruction`，不是 CPython 的 bytecode。一条 instruction 操作的是轻量的 `IValue` 栈，没有 Python 对象头、Python 属性字典查找、hook 遍历和 GIL 竞争。

所以更准确的对比是：

```text
eager:
Python 解释器解释 Python forward
  ↓
大量动态语言开销
  ↓
dispatcher
  ↓
kernel

TorchScript/JIT:
C++ JIT runtime 执行优化后的 Code / Instruction
  ↓
更少的动态分派和 Python 边界开销
  ↓
dispatcher
  ↓
kernel
```

这份 `unified.pt` 有 2599 个编译后的 `.py` 文件，说明模型里有大量小模块、小算子和重复结构。对这种调用次数多、单次 Python 调度成本容易累积的模型，TorchScript 带来的固定开销削减会更明显。

## 2.9 eager 模式和本章链路的最终对比

Eager 模式是 PyTorch 默认执行方式：写到哪行，Python 就执行到哪行；执行过程中边走边调用 PyTorch 算子，不预先构建和优化整张 Graph。

```text
Eager 模式:
nn.Module.forward Python 源码
  ↓ Python 解释器逐行执行
  ↓ 每次动态属性查找、hook 检查、Python 函数调用
  ↓ dispatcher
  ↓ kernel
```

TorchScript/JIT 模式：

```text
script/trace
  ↓
ScriptModule / ScriptFunction
  ↓
method/function 内部的 TorchScript Graph
  ↓ GraphExecutor 优化、特化、生成执行计划
  ↓ JIT interpreter 执行 Code / Instruction
  ↓ dispatcher
  ↓ kernel
```

你现在读的 `unified.pt` 是已经编译好的 TorchScript archive。`torch.jit.load` 之后跑的是 Graph/JIT runtime/interpreter 这条路径，不是 eager 模式；文档里反复拿 eager 做对比，是为了说明同一个 `covariant.*` 模型源码在两种跑法下，执行路径和开销来源到底差在哪里。

## 2.10 JIT interpreter / dispatcher 的开销通常有多大

JIT interpreter 和 PyTorch dispatcher 没有固定占比。它们通常是微秒级固定开销，真正占总耗时多少，取决于模型是“大 kernel 主导”还是“小算子 / 调度主导”。

粗略量级可以这样理解：

```text
JIT interpreter 执行一次 op 调用:
  亚微秒到几微秒级

PyTorch dispatcher 选择并调用 kernel:
  几百纳秒到几微秒级

一次小 CUDA kernel launch:
  数微秒到十几微秒级

一次大 conv / matmul kernel:
  几十微秒、几百微秒，甚至毫秒级
```

所以同样是几微秒调度开销，放在不同算子后面的意义完全不同：

```text
大 conv:
  调度 2us + kernel 500us
  overhead ≈ 0.4%

小 relu:
  调度 2us + kernel 5us
  overhead ≈ 28%

极小 view / shape op:
  调度 2us + 实际计算接近 0
  overhead 可能就是主体
```

工程上可以按三类场景估：

```text
大算子主导模型:
  conv / matmul kernel 时间可能占 80% - 95%+
  JIT interpreter + dispatcher 可能只占几个百分点

小算子密集模型:
  add / mul / relu / sigmoid / slice / cat / indexing 很多
  JIT interpreter + dispatcher + kernel launch 可能占 20% - 60%

CPU 小 tensor 场景:
  真正计算太少
  dispatcher / interpreter / Python 边界等固定开销可能占 50% 以上
```

这也解释了 JIT 和 engine 的差异：如果模型主要由少量大 `conv` / `matmul` 组成，JIT interpreter 和 dispatcher 占比通常低，engine 只有在 kernel selection、layout、precision、fusion 上做得更好时才会明显快；如果模型有大量小 op、多分支、后处理、pointwise 链路，engine 靠静态调度、fusion、内存规划减少这些固定开销，就更容易明显加速。

不要靠猜，应该用 profiler 验证：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    for _ in range(20):
        y = model(x)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

重点看：

- `aten::xxx` 的 `self_cpu_time_total`。
- `cuda_time_total` 和 CPU time 的比例。
- kernel launch 数量。
- 小 op 数量，尤其是 view、slice、cat、permute、index、pointwise。
- 是否有大量 CPU 侧调度时间但 CUDA kernel 很短的算子。

一句话总结：

> JIT interpreter + PyTorch dispatcher 通常是微秒级固定开销；在大 conv/matmul 模型里可能只占几个百分点，在小算子密集、小 batch 或 CPU 小 tensor 场景里可能占到几十个百分点，甚至成为主要瓶颈。

下一章开始逐行读 `000_predict_backbone.py`，把 `code/*.py` 里的 Python-like TorchScript 程序文本语法讲透。
