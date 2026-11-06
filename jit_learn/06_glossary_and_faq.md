# 06 · 术语表 + 常见问题

## 术语表

| 术语 | 含义 |
|------|------|
| **TorchScript** | PyTorch 提供的、Python 的一个受限静态类型子集 + 对应的编译器/运行时，可以脱离 Python 解释器独立执行模型。 |
| **`torch.jit.trace`** | 通过实际运行一次模型、录制执行到的张量操作序列来生成 TorchScript 模块。不理解 Python 控制流，所有非张量分支在录制时被按当次取值固化。 |
| **`torch.jit.script`** | 用真正的编译器分析 Python 源码（`forward` 等方法体），编译成 TorchScript IR。理解 `if`/`for`，但要求代码落在 TorchScript 支持的 Python 子集内。 |
| **`ScriptModule`** | `torch.jit.trace`/`torch.jit.script` 的输出对象类型，可以 `.save()` 到磁盘、可以有多个具名方法（`@torch.jit.export` 标注）、可以脱离原 Python 类定义单独 `torch.jit.load()` 使用。 |
| **`.code`** | `ScriptModule`（或其某个方法）的一个属性，返回该方法编译后的 IR **反解析（un-parse）/ 打印（pretty-print）** 成的可读 Python 风格文本。是调试视图，不是驱动执行的东西本身——**不是反编译**：这段文本是 IR 编译完成的那一刻就已经存在的确定性打印结果，`torch.jit.save` 只是把它原样写进磁盘文件，不是加载后另外逆向猜测出来的。 |
| **反解析 / un-parse（不是"反编译"）** | 把 TorchScript 内部的图（IR）表示，按固定规则打印成看起来像 Python 代码的文本，供人读。这是一个**确定性的正向操作**（IR → 文本，跟"编译源码 → IR"方向相反但同样是无损、规则明确的转换），不是从不透明二进制里逆向猜测结构的"反编译/逆向工程"。这个打印过程会引入大量教学中反复提到的"结构性噪音"（重复的属性访问、无意义的临时变量名），但内容本身是对 IR 的忠实复述，不是猜测。 |
| **Mangling / mangle** | 当同一个 Python 类被实例化多次、且不同实例的构造参数导致编译结果不同时，TorchScript 给每种不同的编译结果分配一个独立的类名（形如 `___torch_mangle_N`），以示区分。相同构造参数的实例共享同一份未 mangle 的"基准版"编译产物。 |
| **`Final[T]`** | TorchScript 类型系统里"编译期常量属性"的声明。对不可变类型（`int`/`str`/`Tuple[...]` 等）、且构造后不再被重新赋值的属性，编译器会自动推断为 `Final` 并把具体数值直接烧录进编译产物，不再是运行时可变的实例状态。 |
| **`archive/data.pkl`** | `.pt` 文件（zip）里记录整个模块树结构（每个子模块实例、属性、引用关系）的 pickle 字节流，需要通过 `torch.jit.load` 专用反序列化器解析，不能直接用 `pickle.load`。 |
| **`archive/code/`** | `.pt` 文件里，每个被编译的"类"对应一份 IR 反解析文本文件的目录。 |
| **`archive/data/`** | `.pt` 文件里，每个 Tensor 的原始二进制存储（storage），按数字编号，不含 shape/dtype 信息（这些信息在 `data.pkl` 里）。 |
| **`archive/traced_inputs.pkl`** | 如果模块内部有 `trace` 环节，这里记录当时喂给 trace 的样例输入；是"这个模型混用了 trace"的直接证据。 |
| **custom op / 自定义算子** | 用 C++/CUDA 实现、通过 `torch.ops.load_library(".../xxx.so")` 注册进 PyTorch 算子表的操作（例如本仓库的 `soft_bin_ops`、`roi_upsample_cuda`）。模型里引用了这些算子时，必须先加载对应 `.so`，否则 `torch.jit.load` 会在反序列化引用到该算子的节点时报错。 |
| **state_dict** | `nn.Module.state_dict()` 返回的 `{参数/buffer 名: Tensor}` 字典，按模块层级用 `.` 拼接键名。checkpoint 里的 state_dict 键名和你重建模型后的键名往往不同，需要写映射函数对齐。 |
| **`WrapperModule` / `UnifiedModel`** | 本仓库 `unified.pt` 里实际的顶层类结构：最外层是 `covariant.model.utils.export.core.WrapperModule`，它的 `self.module` 属性才是真正的 `covariant.model.perception.unified.orchestration.basic.UnifiedModel`。IR 反解析文本里反复出现的 `self.module` 就是指向这一层。 |

## 常见问题

**Q1：为什么不直接看 covariant 源码，非要折腾 JIT 反解析代码？**

两种场景各有用处，不是互相替代关系：
- **只有 `.pt` 文件、没有对应版本源码**时（比如模型是很久以前训练/导出的，
  当时的源码已经被后续改动覆盖，或者你拿到的 `.pt` 来自另一个环境），
  JIT 反解析代码是唯一能确认"这个具体文件的实际计算逻辑"的依据——
  源码可能已经和它不一致了。
- **想优化/改造/导出模型**时，需要先用 JIT 代码确认"计算图骨架"，
  再回源码补全"为什么是这个骨架里的这些超参"，两者配合才能重建出
  一份既结构正确、又超参正确的独立实现（这正是 `jit2model_guide.md`
  整套方法论存在的原因）。

**Q2：如果 JIT 代码和源码对不上怎么办？**

先确认版本：`.pt` 文件是哪次训练/哪个 commit 导出的，源码是否是对应版本
（`covariant/model/perception/unified/...` 这类源码文件可能在导出之后又被改动过）。
如果版本确认一致仍对不上，大概率是本文档第 4 章讲的某种"信息消失"导致你
误读了 JIT 代码的字面含义（比如把噪音行当成了主逻辑），回去按第 3 章的方法
重新折叠一遍再对照。

**Q3：为什么有的 `.code` 里能看到具体数值（比如 `Final[int] = 8`），
有的却完全看不出超参数？**

区别在于这个超参数是"被烧录进某个子模块自己的构造属性"（能看到，见第 4 章
4.1b 节），还是"决定了要不要构造某个子模块、或构造成哪种类型"的**决策逻辑本身**
（看不到，见 4.1 节）。数值本身几乎总能找到；"为什么是这个数值"的逻辑链条
经常找不到。

**Q4：`extrace_info.py` 里为什么要先 `torch.ops.load_library` 再 `torch.jit.load`？
顺序能反过来吗？**

不能反过来。`torch.jit.load` 反序列化 `data.pkl` 的过程中，一旦遇到某个节点
引用了自定义算子（比如 `torch.ops.covariant.roi_upsample`），它需要**立刻**
能在当前进程的算子注册表里查到这个算子的实现，才能继续往下解析。
如果这个查找发生在 `load_library` 之前，注册表里还没有这个算子，
会直接报错（类似 `unknown builtin op` 或 `Unknown operator`）。
这也是为什么这个顺序在 `extrace_info.py` 里被放在最前面、写成一个固定模式，
而不是"用到了才加载"——反序列化阶段就需要，不是等实际调用 `forward` 才需要。

**Q5：`torch.jit.trace` 和 `torch.jit.script` 该选哪个？**

这是训练/导出模型的作者要做的选择，不是读模型的人能改的（`unified.pt`
已经是别人导出好的产物）。但如果未来你要自己导出新模型，一般原则是：
模型的主干计算（卷积、矩阵乘法等）用什么方式都行，但只要有**依赖非 Tensor
Python 值的控制流**（`if`/`for` 循环次数不是常量）、或者需要**多个具名导出方法**
（像这里的 `predict_backbone`/`predict_depth`/`predict_segm`），就必须用
`script`（至少顶层包装类要用 `script`）。纯粹是固定计算图、没有分支、
只有一个入口的简单子网络，`trace` 更省事、兼容性通常也更好
（`script` 对 Python 语法子集的限制更严格）。这也是 `torch2engine/`
和 `jit2engine/` 两条路线分道的原因之一：`torch2engine` 走的是
"直接从 covariant 源码重新导出 ONNX"，不需要过 TorchScript 这一步；
`jit2engine` 走的是"从已有的 TorchScript `unified.pt` 反向理解结构"，
两条路线互相独立，互为交叉验证（参考 `torch2engine/README.md`
和本 repo commit 历史里 backbone/depth/segmentation 的精度对比工作）。

**Q6：这份材料和 `jit2model_guide.md` / `jit2model.md` 的关系，再确认一次？**

- `jit_learn/`（本目录）：原理课，回答"为什么"，配可执行 lab，
  不要求你已经懂 TorchScript。
- `jit2engine/jit_code/jit2model_guide.md`：backbone 案例的详细记录（较早版本）。
- `jit2engine/jit_code/skill/jit2model.md`：三个案例（backbone/depth/segmentation）
  汇总后的操作手册（当前维护的版本），回答"怎么做"，假设你已经具备本目录的背景知识。

读的顺序建议：先读完 `jit_learn/00-05`，跑完 labs，再去读
`skill/jit2model.md` 走一遍真实的重建流程。
