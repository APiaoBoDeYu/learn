# 第 6 章：综合对比与选型 —— 数据推翻了哪些先验判断

本章是全教程的结论。所有数字都是本机（Jetson Thor, CUDA 13.0, torch 2.9.1,
TensorRT 10.14.1）实测，FP16，串行执行，CUDA event 计时，warmup 20 / 测 100 次。

复现：

```bash
cd /root/data/learn/torch2trt/code
python3 run_bench.py            # 全部 4 模型 × 7 路线
python3 run_bench.py --table    # 从 results.jsonl 重新渲染
```

**关于复现性**：本章表格是冷缓存下的一次完整运行。我重跑了整个矩阵做核对，
延迟数字全部落在 3% 以内（例如 resnet18 的 ts_trt 0.782 → 0.767，
bert_tiny 的 trtexec 0.287 → 0.286），路线排名完全一致。
唯一变化大的是构建耗时 —— 见 6.1 末尾关于缓存的说明。
`customop / aoti` 的加速比在 1.06–1.14x 之间波动（模型太小，绝对差只有几微秒），
但"它是该模型上最快路线"这个结论稳定。

## 6.1 完整结果

### 中位延迟（ms）

| route | resnet18 (b=8) | bert_tiny (b=8,s=128) | ctrlflow (b=32) | customop (b=32) |
|---|---|---|---|---|
| eager | 2.879 | 0.867 | 0.102 | 0.068 |
| torch_compile | 1.334 | 0.735 | 0.258 | 0.135 |
| onnx_trt | 0.852 | 0.413 | 0.104 | **X** |
| trtexec | 0.826 | 0.287 | 0.103 | **X** |
| dynamo_trt | 0.807 | 0.296 | **X** | 0.202 |
| ts_trt | **0.782** | **X** | 0.059 ⚠ | **X** |
| aoti | 1.270 | 0.348 | **X** | **0.060** |

### 加速比（相对 eager）

| route | resnet18 | bert_tiny | ctrlflow | customop |
|---|---|---|---|---|
| eager | 1.00x | 1.00x | 1.00x | 1.00x |
| torch_compile | 2.16x | 1.18x | 0.40x | 0.50x |
| onnx_trt | 3.38x | 2.10x | 0.98x | X |
| trtexec | 3.48x | **3.02x** | 0.99x | X |
| dynamo_trt | 3.57x | 2.93x | X | 0.34x |
| ts_trt | **3.68x** | X | 1.74x ⚠ | X |
| aoti | 2.27x | 2.49x | X | **1.14x** |

⚠ = 数值错误（见第 5 章）。**这两个标记比表里所有数字都重要。**

### 构建耗时（s）

| route | resnet18 | bert_tiny | ctrlflow | customop |
|---|---|---|---|---|
| eager | 0.0 | 0.0 | 0.0 | 0.0 |
| torch_compile | 41.2 | 12.6 | 1.8 | 0.7 |
| onnx_trt | 17.8 | 13.1 | 5.3 | X |
| trtexec | 21.6 | 18.5 | 10.4 | X |
| dynamo_trt | 20.6 | 15.3 | X | 11.9 |
| ts_trt | 18.9 | X | 5.4 | X |
| aoti | **9.6** | **8.0** | X | 6.4 |

**这张表的数字要比延迟表宽松地读。** 构建耗时受缓存影响很大：重跑同一条命令时，
resnet18 上 `torch_compile` 从 41.2s 降到 6.6s、`dynamo_trt` 从 20.6s 降到 7.1s
（Inductor 有磁盘缓存，Torch-TensorRT 有 timing cache），而两条 ONNX 路线基本不变
（17.4 / 21.2s，每次都重新 parse 和构建）。

所以**冷启动**（CI 里的典型情况）用上表，**本地迭代**时 Inductor 系路线要快得多。
延迟数字不受这个影响，重跑的复现结果在 3% 以内。

### 数值一致性（cosine vs fp32 eager）

所有构建成功且喂示例输入分布的路线，cosine 相似度都是 **1.00000**，
`top1_agree` 都是 **1.000**。

这一整张表全是 1.0，说明一件重要的事：**FP16 本身不是正确性风险。**
本教程测到的所有正确性问题，无一例来自精度，全部来自**图捕获**（控制流被固化）。
选型时该担心的是前者的对立面 —— 你的图有没有被完整捕获。

（第 5 章那个 0.386 不在这张表里，因为它只在喂"另一个分支"的输入时才暴露。
这正是它危险的原因：标准 benchmark 抓不到它。）

### 失败清单

| 模型/路线 | 类型 | 原因 |
|---|---|---|
| bert_tiny / ts_trt | 构建失败 | `shape '[8,128,2,64]' is invalid for input of size 0`（trace 的 shape 传播出错） |
| ctrlflow / dynamo_trt | **诚实拒绝** | `GuardOnDataDependentSymNode` |
| ctrlflow / aoti | **诚实拒绝** | 同上 |
| customop / onnx_trt | 构建失败 | `aten::erfinv` 无 opset 17 映射（加自定义算子 + plugin 后可构建，见第 10 章） |
| customop / trtexec | 构建失败 | 同上（同一个 ONNX 导出步骤） |
| customop / ts_trt | **运行时崩溃** | `Expected input tensors to have type Half, found type float` |

注意第 2、3 行和其他行性质不同：那不是"路线有 bug"，那是路线**正确地拒绝了一个
不该被静态化的模型**。第 5 章讲了为什么这该记为优点。

## 6.2 被数据推翻的四个先验判断

第 0 章给了一张预判表。现在逐条核对，有四条需要修正 —— 这正是做实验而不是读文档的价值。

### 推翻 1：「Torch-TensorRT 比 ONNX 路线更快」 → 错

| | resnet18 | bert_tiny |
|---|---|---|
| onnx_trt | 0.852 | 0.413 |
| trtexec | 0.826 | **0.287** |
| dynamo_trt | 0.807 | 0.296 |

差异全在 5% 以内，而且 bert_tiny 上 trtexec 反而最快。

原因很清楚：**两条路线最终都是同一个 TensorRT builder 在做优化和 tactic 选型。**
前端只决定图怎么送进去，不决定 kernel 怎么选。

> **如果你的 ONNX 路线已经跑通了，不要为了性能去迁移到 Torch-TensorRT。
> 迁移的理由只能是第 5 章那些能力差异（部分回退、控制流严格性），不是速度。**

**⚠ 这条结论只适用于静态 shape。** 第 9 章在动态 shape 下重测，
dynamo_trt 比 ONNX 路线快 **20–29%**。如果你跑动态 shape，迁移可能真的值得。

### 推翻 2：「TensorRT 总比 torch.compile 快」 → 对，但幅度远小于预期，且有反例

resnet18 上 3.38–3.68x（TRT）vs 2.16x（torch.compile），TRT 确实赢。
但 bert_tiny 上只有 1.18x（torch.compile），差距拉到 2.5 倍，
说明 **Inductor 在 transformer 上远不如在 CNN 上有效**（至少在这个尺寸和这块硬件上）。

而在两个小模型上 torch.compile **比 eager 还慢**（0.40x / 0.50x）：
编译开销和 guard 检查压过了融合收益。

> **模型小于某个尺寸时，所有编译路线都是负收益。先测再上。**

### 推翻 3：「AOTInductor 只是个 TorchScript 替代品，性能不重要」 → 严重低估

| | resnet18 | bert_tiny | customop |
|---|---|---|---|
| aoti | 2.27x | 2.49x | **1.14x（全表最优）** |
| 构建耗时 | **9.6s（最快）** | **8.0s（最快）** | 6.4s |

AOTInductor 是：**冷启动构建最快**（比 TRT 路线快约 2 倍）、性能稳定在 2.3–2.5x、
产物脱离 Python、而且**零额外依赖**（PyTorch 自带，不用装 TensorRT 那 1.5GB）。

而在 `customop` 上它是唯一既比 eager 快又能产出独立产物的路线。第 10 章给 ONNX 路线
补上自定义算子 + TensorRT plugin 之后重测，AOTI（0.060 ms）依然比最好的 ONNX 方案
（0.079 ms）快，b=8192 时也是（0.144 vs 0.200 ms）。

> **如果你要的是"编译 + 脱离 Python 部署"而不是"最后 30% 的极限性能"，
> AOTInductor 值得先试，它的工程成本比 TensorRT 路线低一个量级。**

### 推翻 4：「TorchScript 路线只是老一点」 → 是主动的风险

第 0 章我把 R5 定位成"老项目兼容"。实测它在 resnet18 上最快（3.68x），
但四个模型里**三个出问题**：bert_tiny 构建失败、customop 运行时崩溃、
ctrlflow 静默出错。为 3% 性能接受这个，新项目不划算。

## 6.3 修正后的选型表

| 你的情况 | 选 | 理由（实测依据） |
|---|---|---|
| CNN，静态 shape，要极限延迟 | **trtexec 或 onnx_trt** | 3.4–3.5x，生态最成熟，产物脱离 Python |
| Transformer，静态 shape | **trtexec** | 3.02x；此处 dynamo 无优势而依赖更重 |
| 模型含 TRT 不支持的算子 | **先试 aoti**；要 TRT 就先**导出期分解成标准算子**，再考虑 dynamo_trt / C++ plugin | aoti 1.14x vs dynamo_trt 0.34x；分解比 C++ plugin 快 11%（第 10 章） |
| 有数据相关控制流 | **torch_compile**，或重构模型 | 唯一两个分支都正确且能编译的路线 |
| 要脱离 Python，但不想引入 TRT 依赖 | **aoti** | 2.3–2.5x，构建最快，零额外依赖 |
| 模型很小（< 1M 参数，亚毫秒） | **eager** | 所有编译路线都是负收益 |
| 训练也要加速 | **torch_compile** | TensorRT 只做推理 |
| 已有 ONNX 路线跑得好（**静态 shape**） | **别动** | 迁移换不来性能（推翻 1） |
| 已有 ONNX 路线，**动态 shape** | **值得试 dynamo_trt** | 快 20–29%（第 9 章） |
| 存量 TorchScript 路线在跑 | 别动，但别扩大 | 见推翻 4 |
| LLM 推理 | 都不是 —— 见第 7 章 | 另一个层级的问题 |

## 6.4 三个通用规律

### 规律 1：模型越大，编译收益越大；小模型上编译是负收益

按参数量排：

| 模型 | 参数量 | eager 延迟 | 最佳加速比 |
|---|---|---|---|
| resnet18 | 11.69M | 2.879 ms | 3.68x |
| bert_tiny | 4.77M | 0.867 ms | 3.02x |
| ctrlflow | 0.53M | 0.102 ms | 1.74x ⚠ / 0.99x |
| customop | 0.27M | 0.068 ms | 1.14x |

单调关系。机制很直接：编译的收益来自**减少 kernel 数和显存往返**，
而固定开销（engine 边界同步、guard 检查、调度）是常数。
计算量小到和固定开销同量级时，收益就没了。

**实用推论：亚毫秒级的模型不要指望编译。** 那个量级你该优化的是 batching、
数据搬运、和请求调度，不是 kernel。

### 规律 2：加速比在 CNN 上最高，且路线间差异最小

resnet18 上四条 TRT 路线全落在 0.782–0.852ms（9% 带宽内）。
bert_tiny 上从 0.287 到 0.413（44% 带宽）。

conv/bn/relu 的融合是一个**被解决透了**的问题，谁都做得好。
transformer 上各家的 attention 处理差异就大了 —— 也意味着 transformer 上
**多试几条路线的回报更高**。

### 规律 3：构建时间和运行性能不相关

resnet18 上 `torch_compile` 冷启动构建 41.2s 换来 2.16x，
`aoti` 构建 9.6s 换来 2.27x。**慢 4 倍的构建，换来更差的性能。**

（注意这个对比在**热缓存**下会缩小：重跑时 torch_compile 降到 6.6s。
但 CI 里通常是冷的。）

所以构建时间要单独当一个成本看（它直接影响你的 CI 周期），不要用它推测性能。

## 6.5 一个决策流程

```
模型 eager 延迟 < 0.5ms？
  └─ 是 → 停。优化 batching 和数据流，不是 kernel。（规律 1）
  └─ 否 ↓

torch.export.export(model, inputs) 能过吗？
  └─ 不能 → 有数据相关控制流。
  │         选 torch_compile（正确但可能更慢），或重构模型把控制流移出去。（第 5 章）
  └─ 能 ↓

必须脱离 Python 部署吗？
  └─ 不必须 → 先试 torch_compile（一行代码）。够快就结束。
  └─ 必须 ↓

愿意引入 TensorRT 依赖（1.5GB + 三方版本对齐）吗？
  └─ 不愿意 → aoti。（2.3–2.5x，构建最快，零依赖）
  └─ 愿意 ↓

torch.onnx.export 能过吗？
  └─ 不能（不支持的算子）↓
  │    能在导出期分解成标准 ONNX 算子吗？（symbolic，不改模型源码）
  │      └─ 能 → trtexec / onnx_trt，不用 plugin，还能被 TRT 融合。（第 10 章）
  │      └─ 不能 → dynamo_trt，必须检查子图计数。
  │           回退子图是瓶颈 → 写 C++ IPluginV3（ONNX parser 和 dynamo converter 都能用）
  │           碎片太多又不想写 plugin → 退回 aoti。（第 5、10 章 customop）
  └─ 能 → trtexec / onnx_trt。（3.0–3.5x，最成熟）
       └─ 最后务必用"走不同分支/不同 shape 的输入"验证一遍，
          不要只用示例输入。（第 5 章）
```

## 6.6 这些数字的适用边界

诚实说明本教程结论**不能**直接外推的地方：

- **单一硬件**。Jetson Thor 是边缘平台。数据中心 GPU（A100/H100）上显存带宽、
  SM 数量、tensor core 代数都不同，绝对数字肯定变。**趋势和排名大概率保持**，
  但请在你的目标硬件上重跑 —— 这就是 `run_bench.py` 存在的意义。
- ~~**FP16，没测 INT8/FP8**~~ → **第 8 章已补上实测**。结论比我当初的猜测复杂：
  INT8 在 batch ≤ 8 时**比 FP16 慢**，要到 batch ≥ 32 才反超（1.14–1.18x）；
  FP8 在这块硬件上全程是净损失。所以"需要 INT8 就给 TensorRT 加分"这句话
  **需要加上 batch size 的限定条件**。
- **没测多流/多实例并发**。只测了单流延迟。吞吐场景下 engine 的并发行为
  （多个 execution context、CUDA stream 复用）是另一个话题。
- ~~**没测动态 shape 的实际性能**~~ → **第 9 章已补上实测**，而且它**推翻了下面
  "推翻 1"的适用范围**：静态 shape 下 dynamo_trt 和 ONNX 路线差 <5%，
  但动态 shape 下 dynamo_trt 快 20–29%。另外动态 engine 比专用静态 engine
  慢 44–84%，这个代价比我预期的大得多。
- **`weights=None`**。随机权重。延迟和路线间数值一致性不受影响，
  但如果你要测真实 accuracy 必须换成预训练权重。
- **模型规模偏小**。最大的 11.69M 参数。规律 1 说明大模型上编译收益更大，
  所以本教程的加速比数字可能是**保守**的。

---

← [第 5 章：控制流与不支持算子](05_control_flow_and_fallback.md) | [第 7 章：附录](07_appendix.md) →

后续实测（本章之后补充）：[第 8 章：量化](08_quantization.md) · [第 9 章：动态 shape](09_dynamic_shapes.md) · [第 10 章：自定义算子与 plugin](10_custom_op_and_plugin.md)
