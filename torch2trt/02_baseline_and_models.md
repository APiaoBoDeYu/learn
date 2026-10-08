# 第 2 章：基准与模型设计 —— 怎么做一个不骗自己的对比实验

本章不讲任何一条转换路线。本章讲**怎么测**。

理由很直接：这个教程的全部价值取决于第 6 章那张对比表可信不可信。如果计时方法有偏、
正确性判据选错、或者四个模型的选择有盲区，那张表会给出一个自信但错误的结论，
而且你没有办法发现。所以先把方法论钉死。

## 2.1 四个反面教材：benchmark 最常见的四种骗法

我在写本教程的代码时，前三个都真实踩过一遍。

### 骗法 1：用 wall-clock 计时，不同步

```python
t0 = time.time()
out = model(x)
print(time.time() - t0)     # 错
```

CUDA kernel 是**异步下发**的。`model(x)` 返回时 GPU 很可能一个 kernel 都没算完，
你测到的是 Python 下发指令的时间。模型越大、这个数字越离谱地小。

正确做法是用 **CUDA event**，让 GPU 自己给自己打时间戳：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record(); fn(); end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
```

`common.py` 的 `bench_ms()` 就是这个。

**但要注意反过来的陷阱**：有时候你关心的**恰好就是** host 端开销 —— 比如比较
eager（每个算子一次 Python dispatch）和一条"整张图一次调用"的路线时，host 开销
正是主要差异。这时 CUDA event 会把你想测的东西藏起来。所以 `common.py` 同时提供
`bench_wall_ms()`，两个都在，按问题选。

### 骗法 2：不 warmup

第一次调用包含：CUDA context 初始化、cuDNN benchmark 选算法、
torch.compile 触发 JIT、TensorRT 反序列化 engine、显存分配器冷启动。这些都是
**一次性**成本，混进稳态延迟里就是纯噪声。本教程统一 warmup 20 次、测 100 次。

### 骗法 3：只报平均值

平均值会被一次调度抖动带偏。本教程报**中位数**作头条数字（抗抖动），
同时报 **p90** —— 因为对一个线上推理服务，尾延迟比平均延迟更能决定 SLA。

### 骗法 4（最隐蔽）：正确性基准选错

这个我踩得最狠，值得完整复盘，因为**症状看起来完全不像 bug**。

我的第一版 `run_bench.py` 这样验证正确性：跑一次 fp32 模型拿参考输出，再跑被测路线，
比 cosine 相似度。结果 **eager 路线和自己的 fp32 参考的 cosine 相似度是 -0.02**。

```
[DIFF] customop/eager: max_abs_err=1.918e+00, cos_sim=-0.022819, top1_agree=0.031
```

eager 路线根本没做任何转换，它和参考输出的差异**必须**是纯 fp16 舍入误差，
cosine 应该是 0.999+。-0.02 意味着两个输出毫无关系。

原因是 `models.get()` 每次调用都重新随机初始化权重，所以"被测模型"和"参考模型"
是两个**不同的网络**。整套正确性数据全是噪声。

修法是在 `get()` 里固定种子，并且把输入生成改成"先在 fp32 下抽样再 cast"
（否则 fp16 和 fp32 的 RNG 抽取路径不同，输入也会不一样）：

```python
def get(name: str, seed: int = 0, **kw):
    torch.manual_seed(seed)
    return ZOO[name](**kw)
```

修完之后：

```
[PASS] customop/eager: max_abs_err=3.174e-03, cos_sim=0.999999, top1_agree=1.000
```

**方法论教训**：任何对比实验都必须有一个**自我一致性检查** —— 一个你确切知道答案的
用例。这里就是「eager 对 fp32 参考应该几乎完全一致」。这一条不成立，别的数字一个都
不能信。本教程把 eager 行永久留在每张表里，它既是加速比的分母，也是这个自检。

## 2.2 正确性判据：为什么 `allclose` 不够

FP16 下 `torch.allclose` 是个很差的信号。它会因为**单个**离群元素 fail，
而那个元素可能完全不影响模型预测。所以 `common.compare()` 同时报三个指标：

| 指标 | 回答什么问题 | 什么时候看它 |
|---|---|---|
| `max_abs_err` | 最坏的单点偏差有多大 | 排查数值爆炸 |
| `cos_sim` | 输出向量方向还一致吗 | **主要指标**，对整体形变敏感 |
| `top1_agree` | argmax 还一样吗 | **上线判据**，分类任务真正在乎的 |

一个真实例子，第 6 章 resnet18 的数据：

```
[PASS] resnet18/dynamo_trt: max_abs_err=1.027e-02, cos_sim=0.999999, top1_agree=1.000
```

`max_abs_err` 是 1e-2 —— 单看这个数字挺大，1000 类 logit 里有个元素偏了 0.01。
但 `cos_sim=0.999999` 且 `top1_agree=1.000`：**分类结果一个都没变**。
如果只看 `max_abs_err` 并设个 1e-3 的阈值，你会拒绝一个完全可用的 engine。

## 2.3 四个模型：每个负责证明一件事

单模型 benchmark 是本教程要反驳的核心错误。路线的**排名会随模型改变**，
所以只测 resnet 得到的结论不能往 transformer 上套。

### `resnet18` —— 对照组

静态 shape，纯 conv/bn/relu。所有路线都该能跑，而且这是 TensorRT 的最佳战场
（conv+bn+relu 融合 + 成熟的 tactic 库）。它的作用是给出**加速比的上界**，
以及在别的模型上出问题时告诉你"是模型的问题，不是环境的问题"。

用 `weights=None` 随机初始化：本教程只测延迟和路线间的数值一致性，
从不测真实 accuracy，所以随机权重足够。

### `bert_tiny` —— transformer 与动态 shape

matmul / layernorm / softmax 主导，计算特征和 CNN 完全不同。

这里有个设计决定值得解释：**我手写了 encoder block，没用
`nn.TransformerEncoderLayer`**。因为那个模块内部有对 `is_causal` 和 mask 类型的
分支判断，还会走 fused SDPA 路径，导致**不同路线导出的图结构不一样** ——
那就不是在比路线，而是在比"哪条路线碰巧走到了哪个 kernel"。手写的 block 在所有
路线下导出结果一致，对比才干净。

它的输入是 **int64 token ids**，这会触发 TensorRT 路线的 int64→int32 处理
（`truncate_double` / `truncate_long_and_double` 参数），第 3 章会讲。

### `ctrlflow` —— 数据相关控制流

```python
def forward(self, x):
    if x.mean() > 0:
        x = F.relu(self.a(x))
    else:
        x = torch.tanh(self.b(x))
    return self.out(x)
```

`x.mean() > 0` **无法**在导出期确定。这是全教程最重要的模型，因为它是唯一能分辨
「静默出错」和「诚实报错」的方法：

- trace 类路线（`torch.jit.trace`、传统 ONNX 导出）：导出**成功**，但只记录了
  示例输入走的那个分支。对另一半输入，产物是错的，**没有任何警告**。
- `torch.export` / AOTInductor：**拒绝**导出并报错。
- `torch.compile`：在 `if` 处 graph break，两个分支都正确。

第 5 章用 `exp_control_flow.py` 把这个变成一个测量结果 —— 分别喂正均值和负均值的
输入，看哪条路线在**没走过的那个分支**上翻车。

### `customop` —— 无 converter 的算子

含 `torch.erfinv`：正常的 aten 算子（eager 和 Inductor 都没问题），但**没有 ONNX
opset 映射，也没有 Torch-TensorRT converter**。这让它成为"不支持算子"这个问题的
干净探针，而不是一个人为构造的怪例。

## 2.4 统一接口：让对比真正 apples-to-apples

`routes.py` 里每条路线都是同一个签名：

```python
def build(model, inputs, **kw) -> (runnable, meta)
```

- `runnable`：和原模型同签名的可调用对象
- `meta`：`build_s` + 路线特有信息（engine 大小、TRT 子图数、回退子图数……）

这样 `run_bench.py` 能用**完全相同**的计时和校验代码对待所有路线。表格里唯一变化的
量是路线本身，不是测量方法。

处理不支持的情况用一个专门的异常：

```python
class RouteUnsupported(Exception):
    """路线确实处理不了这个模型 —— 这是一个实验结果，不是 bug。"""
```

"ONNX 导出拒绝了这个模型"正是本教程要测量的事实之一，所以 driver 捕获它、
记成表格里的一个 `X`，而不是让整个 benchmark 崩掉。

## 2.5 结果落盘：JSONL 而不是内存里的 dict

每条结果一产生就 append 进 `artifacts/results.jsonl`：

```python
def save_result(tag, record):
    with open(ARTIFACTS / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
```

原因是实际跑下来，**总会有某条路线挂掉或者卡住**。append-only 保证前面辛苦跑出来的
行不会因为后面一条路线 OOM 而全部白费。`run_bench.py --table` 可以随时从文件重新
渲染表格，不用重跑。

## 2.6 跑一遍基准

```bash
cd /root/data/learn/torch2trt/code

# 全部四模型 × 七路线（慢，建议后台跑）
python3 run_bench.py

# 单模型
python3 run_bench.py --models resnet18

# 单路线
python3 run_bench.py --routes eager,dynamo_trt

# 只重新渲染表格
python3 run_bench.py --table
```

注意 driver 是**严格串行**的，每条路线之间会 `empty_cache()` + 同步 + sleep 2 秒。
第 1 章讲过原因：TensorRT 构建期吃满 GPU，并行跑会得到毫无意义的数字。

---

← [第 1 章：环境搭建](01_setup.md) | [第 3 章：ONNX → TensorRT 路线](03_onnx_trt_route.md) →
