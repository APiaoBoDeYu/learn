# 第 03 章 框架级 Profiling：torch.profiler

> 📎 对应笔记：doc/ 中没有 torch.profiler 的专门主题；与之对应的是 [doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「区别联系」：框架级 Profiler（TRT / torch）回答"哪层慢"，懂模型语义但看不到硬件，所以要和 Nsight 配合使用。
>
> 漏斗模型的第 2 层（L2）：**从 PyTorch 算子的角度看"哪个算子慢、哪里有同步、显存用在哪"**。
>
> ⚠️ **Thor 平台限制**：torch.profiler 在本机**拿不到 GPU kernel 的执行时间**（CUPTI 返回的 GPU 时间戳为 0，属于平台问题，排查过程见 [3.8](#38-thor-上的限制与排查过程)）。本章围绕可用的功能展开：CPU 侧的算子统计、CUDA runtime API、同步检测、显存分析、torch.compile 诊断。GPU 侧的时间线交给[第 04 章 nsys](../04_nsight_systems/README.md)。

## 学习目标

1. 掌握 `torch.profiler` 的核心 API：activities、record_shapes、with_stack、with_flops、profile_memory、schedule；
2. 读懂 `key_averages().table()`，会按 shape、调用栈分组定位问题；
3. 用 `set_sync_debug_mode` 和 runtime API 统计找到所有同步点；
4. 用 memory snapshot 和 memory_stats 分析显存峰值与碎片，理解 `expandable_segments` 的作用；
5. 用 dynamo 计数器诊断 graph break，判断 `torch.compile` 是否真的生效；
6. 知道 profiler 自身的开销会改变被测程序的行为。

## 目录

- [3.1 基础 API 与算子表](#31-基础-api-与算子表)
- [3.2 schedule、trace 导出与阶段分析](#32-scheduletrace-导出与阶段分析)
- [3.3 同步点检测](#33-同步点检测)
- [3.4 显存分析](#34-显存分析)
- [3.5 torch.compile 诊断](#35-torchcompile-诊断)
- [3.6 TensorBoard 与 HTA](#36-tensorboard-与-hta)
- [3.7 profiler 的开销与观察者效应](#37-profiler-的开销与观察者效应)
- [3.8 Thor 上的限制与排查过程](#38-thor-上的限制与排查过程)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 |
|---|---|
| [01_basics.py](01_basics.py) | 算子表、按 shape 分组、FLOPs、调用栈、导出 trace |
| [02_schedule_trace.py](02_schedule_trace.py) | schedule、按 `record_function` 区间统计同步和 launch |
| [03_sync_detection.py](03_sync_detection.py) | `set_sync_debug_mode` 找出所有同步点 |
| [04_memory.py](04_memory.py) | memory_stats、snapshot、`expandable_segments` |
| [05_compile_diag.py](05_compile_diag.py) | graph break、mode 对比、重编译 |
| [common/cupti_fix/](../common/cupti_fix/) | Thor 上 CUPTI 的部分修复（恢复 CUDA runtime API 事件） |

---

## 3.1 基础 API 与算子表

```python
from torch.profiler import profile, ProfilerActivity, record_function
from torch._C._profiler import _ExperimentalConfig

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,       # 记录输入 shape
             with_flops=True,          # 根据 shape 估算 matmul/conv 的 FLOPs
             with_stack=True,          # 记录 Python 调用栈
             experimental_config=_ExperimentalConfig(verbose=True),   # ⚠️ torch 2.9 中缺了它，stack 会是空的
             profile_memory=True) as prof:
    with record_function("vla_forward"):      # 自定义区间
        model(img, tok)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")         # 在 https://ui.perfetto.dev 中打开
```

### 读算子表（[01_basics.py](01_basics.py)，ToyVLA fp32 batch=1 前向 10 次）

```
                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg  ...  # of Calls  Total MFLOPs
                             vla_forward        39.11%      68.527ms        99.98%     175.187ms      17.519ms  ...          10            --
    aten::_transformer_encoder_layer_fwd         2.15%       3.776ms        36.09%      63.229ms       1.054ms  ...          60            --
      aten::_native_multi_head_attention         2.43%       4.263ms        19.51%      34.187ms     569.783us  ...          60            --
                            aten::linear         1.29%       2.259ms        10.41%      18.242ms     121.613us  ...         150            --
                             aten::addmm         6.78%      11.888ms         8.72%      15.277ms      76.385us  ...         200     24495.882
                            aten::conv2d         0.37%     654.000us         8.58%      15.039ms     125.325us  ...         120     17774.346
                             aten::empty         8.15%      14.273ms         8.15%      14.273ms      12.744us  ...        1120            --
```

| 列 | 含义 |
|---|---|
| Self CPU | 算子**自身**的 CPU 时间（不包括子算子） |
| CPU total | 包括子算子在内的 CPU 时间 |
| Self CUDA / CUDA total | 算子发起的 kernel 在 GPU 上的执行时间（⚠️ **Thor 上不可用**，这两列不会出现） |
| CPU Mem / CUDA Mem | 分配的内存/显存（负值表示释放） |
| # of Calls | 调用次数 |
| Total MFLOPs | `with_flops` 估算的计算量 |

从表中能读出：
- `aten::_transformer_encoder_layer_fwd`：`nn.TransformerEncoderLayer` 在推理时走的 **fastpath 融合算子**，6 层 × 10 次 = 60 次调用。
- `aten::empty` 调用了 1120 次：每个算子输出张量都要分配内存，这是 eager 模式固有的开销。
- `cudaLaunchKernel` 调用了 1110 次（按 Self CPU 排序的表中可以看到），平均每次 6.9 μs，**即每次前向要 launch 111 个 kernel**。

### 按 shape 分组、按调用栈分组

```python
prof.key_averages(group_by_input_shape=True).table(...)    # 同一个算子不同 shape 分开统计
prof.key_averages(group_by_stack_n=8)                      # 按调用栈分组，可以定位到源码行
```

```
aten::_native_multi_head_at...   60  [[1, 97, 512], [1, 97, 512], [1, 97, 512], ...]   ← 序列长度 97 = 49 视觉 + 32 文本 + 16 动作
aten::linear  calls= 20  cpu_total=  2.58ms  <- common/models.py(61): forward
```

### with_flops 的局限

```
aten::addmm                  calls= 200  GFLOPs=   24.50  (57.6%)
aten::conv2d                 calls= 120  GFLOPs=   17.77  (41.8%)
总计 → 每次前向 4.25 GFLOPs
```

第 00 章用 `FlopCounterMode`（关闭 fastpath）统计的结果是 **5.59** GFLOPs。`with_flops` 只认识 addmm、mm、conv 等少数算子，fastpath 融合算子内部的 attention 计算没有算进去。**FLOPs 估算要用多种方法交叉验证**。

## 3.2 schedule、trace 导出与阶段分析

在训练或推理循环里，不要从头采到尾。用 `schedule` 只采集几步：

```python
from torch.profiler import schedule, tensorboard_trace_handler
with profile(activities=[...],
             schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
             on_trace_ready=tensorboard_trace_handler("outputs/tb")   # 或自定义回调：lambda p: p.export_chrome_trace(...)
            ) as prof:
    for i in range(8):
        pipe.step(i)
        prof.step()           # 每一步结束时调用
```

| 阶段 | 行为 |
|---|---|
| wait | 不采集 |
| warmup | 采集但丢弃（避开 profiler 自身的启动开销） |
| active | 正式记录，结束时调用 `on_trace_ready` |
| repeat | 以上周期重复几次（0 表示一直重复） |

### 按阶段统计同步与 launch（[02_schedule_trace.py](02_schedule_trace.py)）

慢流水线的 `step()` 里用 `common.region()` 标注了四个阶段，它们在 trace 中显示为 `user_annotation`。脚本统计了每个区间内的 CUDA runtime API 调用：

```
[slow] 不开 profiler: 10.80 ms/step
[fast] 不开 profiler: 3.39 ms/step

=== slow：每步各阶段（时间 ms/step，次数 /step）===
stage           wall   sync耗时   sync次数  memcpy  launch
capture         0.02     0.00        0       0       0
preprocess      3.99     0.00        1       1       0
infer          10.68     0.00        0       0     112
postprocess    21.20     0.09       15     127     378
total          35.89     0.09       16     128     490

=== fast：每步各阶段（时间 ms/step，次数 /step）===
stage           wall   sync耗时   sync次数  memcpy  launch
preprocess      1.26     0.00        0       1       6
infer          10.56     0.00        0       0      97
postprocess     0.40     0.00        1       1       6
total          12.24     0.00        1       2     109
```

结论：
- **后处理是 Launch-bound 的元凶**：每步 378 次 kernel launch，加上 127 次 memcpy（`col[t] = ...` 触发的设备内拷贝），优化版只有 6 次 launch。
- **同步点**：慢版每步 16 次 `cudaStreamSynchronize`（后处理 15 次，预处理 1 次），优化版只有 1 次（最后的 `.cpu()`）。
- **sync 耗时几乎为 0？** 这是 profiler 的观察者效应，见 [3.7](#37-profiler-的开销与观察者效应)。

## 3.3 同步点检测

### `torch.cuda.set_sync_debug_mode`（推荐，不依赖 CUPTI）

```python
torch.cuda.set_sync_debug_mode("warn")    # 每次发生同步时发出警告（带调用位置）
torch.cuda.set_sync_debug_mode("error")   # 直接抛异常 → 在 CI 中防止有人引入新的同步
torch.cuda.set_sync_debug_mode("default") # 关闭
```

[03_sync_detection.py](03_sync_detection.py) 拦截警告，统计每个同步点的位置：

```
=== slow：每步 16 次同步 ===
   14.0/step  common/pipeline.py:150  if self.cfg.sync_in_loop and col.abs().max().item() > 0.99 * self.acti
    1.0/step  common/pipeline.py:112  t = torch.from_numpy(x).to(self.device)           # pageable -> 同步拷贝
    1.0/step  common/pipeline.py:161  return out.cpu().numpy()                              # 唯一一次必要的 D2H 同步
=== fast：每步 1 次同步 ===
    1.0/step  common/pipeline.py:161  return out.cpu().numpy()                              # 唯一一次必要的 D2H 同步
```

结果和 3.2 节中 runtime API 的统计完全一致（16 次 vs 1 次）。注意第 112 行：**从 pageable 内存做 H2D 拷贝也会同步**。

### 常见的隐式同步

```
  ❌ 同步     x.sum().item()
  ❌ 同步     float(x.max())
  ❌ 同步     x.cpu()
  ❌ 同步     x[x > 0]（布尔索引）          ← 输出 shape 依赖数据，必须先算出有多少个 True
  ❌ 同步     torch.nonzero(x)               ← 同上
  ❌ 同步     if x.sum() > 0: ...            ← Python 的 if 需要一个 CPU 上的 bool
  ❌ 同步     x.tolist()
  ✅ 不同步   torch.where(x > 0, x, 0)       ← 输出 shape 固定
  ✅ 不同步   x.clamp(min=0)
```

修复思路：**把控制流留在 GPU 上**（`torch.where` 代替 `if`/布尔索引），或者**把需要的标量攒在一起，最后只取一次**（例如把 7 个维度的越界标志 `stack` 起来，一次 `.cpu()`）。

> ⚠️ PyTorch 会提示：`Synchronization debug mode is a prototype feature and does not yet detect all synchronizing operations`。它并不能检测出所有的同步，要结合 runtime API 统计（3.2）或 nsys（第 04 章）交叉验证。

## 3.4 显存分析

### memory_stats 与 snapshot（[04_memory.py](04_memory.py) A/B 部分）

```python
torch.cuda.memory._record_memory_history(max_entries=100000)   # 开始记录每一次分配/释放及其 Python 调用栈
run(...)
torch.cuda.memory._dump_snapshot("snap.pickle")                # 拖进 https://pytorch.org/memory_viz 查看
torch.cuda.memory._record_memory_history(enabled=None)         # 停止记录
s = torch.cuda.memory_stats()                                  # 各种计数器
```

慢流水线 30 步，打开和关闭 `realloc`（每步申请一块大小不一的临时显存）的对比：

```
                        realloc=True   realloc=False
alloc_calls                  15165.0         15133.0
cudaMalloc_calls                21.0            14.0     ← 真正向驱动申请显存的次数
peak_alloc_MB                  457.6           180.0     ← 峰值：实际被张量占用的显存
peak_reserved_MB               682.0           202.0     ← 峰值：caching allocator 持有的显存
cur_active_MB                  329.1           169.1
segments                        21.0            16.0
alloc_retries                    0.0             0.0     ← >0 说明显存紧张，allocator 在释放缓存后重试（OOM 前兆）
inactive_split_MB                8.9             6.9     ← 段内被切开、暂时空闲的碎片
```

- `realloc` 让峰值 reserved 从 202 MB 涨到 **682 MB**（3.4 倍），但第 00 章测得它对延迟只有约 1.1 倍的影响。caching allocator 让"分配"本身很便宜（`alloc_calls` 有 1.5 万次，而 cudaMalloc 只有 21 次），**代价转嫁到了显存占用上**。
- 在 memory_viz 中打开 `mem_snapshot_realloc_True.pickle`，在 "Active Memory Timeline" 视图里能看到周期性出现的大块（32~160 MB 的 `_scratch`），点击还能看到分配时的 Python 调用栈。
- ⚠️ aarch64 上不支持记录 C++ 调用栈（`record_context_cpp is not support on non-linux non-x86_64 platforms`），snapshot 中只有 Python 调用栈。

### 动态 shape 与 `expandable_segments`（C 部分）

模拟一个逐步增长的缓冲区（4 MB、8 MB、…、240 MB，每次释放上一个），这是动态 shape、不断变长的 KV cache 等场景的典型模式：

```
PYTORCH_CUDA_ALLOC_CONF           allocated   reserved reserved/alloc  cudaMalloc次数
(default)                             240MB     7312MB          30.5x            58
expandable_segments:True              240MB      700MB           2.9x             0
```

- **默认分配器**：每次请求都比缓存里的任何一个空闲块大，只能新申请一个段，旧段留在缓存中不归还，最后 reserved 是实际使用量的 30 倍。
- **`expandable_segments:True`**：在一大段虚拟地址空间上按需映射物理页，段可以原地"长大"，旧空间可以复用（它不走"申请新段"的统计口径，所以 cudaMalloc 次数显示为 0）。
- 用法：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，必须在 CUDA 初始化之前设置。在统一内存的 Jetson 上，被 caching allocator 占住的显存，CPU 侧也用不了，所以这一点尤其重要。

### profile_memory：按算子统计（D 部分）

```
                              Name  ...      CUDA Mem  Self CUDA Mem    # of Calls
                       aten::empty  ...      87.89 MB      87.89 MB           114
           aten::cudnn_convolution  ...      54.33 MB      54.33 MB            12
                   aten::clamp_min  ...      48.59 MB      48.59 MB             9   ← ReLU 的输出
                       aten::addmm  ...      46.94 MB      46.94 MB            19
```

> 需要同时开启 `ProfilerActivity.CUDA` 才会显示 CUDA Mem 列。显存统计来自 caching allocator，在 Thor 上正常可用。

## 3.5 torch.compile 诊断

### graph break：编译真的生效了吗？（[05_compile_diag.py](05_compile_diag.py) A 部分）

```
[post_slow] explain: graphs=0 breaks=-1 | counters: graph_break=0 unimplemented=1 编译出的图=0
     1 x  Unsupported Tensor.item() call with capture_scalar_outputs=False
   → 一个图都没编译出来：graph break 发生在 for 循环内部时，dynamo 会放弃整个函数，退回 eager 执行（不报错，非常隐蔽）
[post_fast] explain: graphs=1 breaks=0 | counters: graph_break=0 unimplemented=0 编译出的图=1
```

- `post_slow` 的 `.item()` 在 for 循环里，dynamo 无法把它放进图中，**于是放弃了整个函数**。`torch.compile(post_slow)` 不报错，但实际上**完全没有编译**。
- `torch._dynamo.explain` 在这种情况下给出了 `breaks=-1` 这样的异常值（torch 2.9），更可靠的方法是读取 `torch._dynamo.utils.counters` 中的 `graph_break`、`unimplemented` 和 `stats.unique_graphs`。
- 命令行诊断：`TORCH_LOGS="graph_breaks,recompiles" python xxx.py`。

### 编译效果（B 部分）

```
[ManySmallOps(200), 1K elems]
eager                             1.660 ms    1.00x
compile default                   0.050 ms   33.21x     ← 200 个逐元素算子融合成 1 个 kernel
reduce-overhead                   0.076 ms   21.85x

[post_fast (16x7)]
eager                             0.046 ms    1.00x
compile default                   0.051 ms    0.90x     ← 计算量太小，guard 检查的开销盖过了收益
reduce-overhead                   0.073 ms    0.63x

[ToyVLA fp16 batch=1]
eager                             2.342 ms    1.00x
compile default                   1.552 ms    1.51x     ← 融合 + 去掉 Python 开销
reduce-overhead                   0.790 ms    2.96x     ← 再加上 CUDA Graph：消除 launch 开销
```

> 首次编译耗时：ManySmallOps 约 4 s，ToyVLA 约 6 s；之后命中 Inductor 缓存（`/tmp/torchinductor_root`），不到 1 s。Inductor 还提示 `Not enough SMs to use max_autotune_gemm mode`：Thor 只有 20 个 SM，GEMM 的 autotune 模式不可用。

| mode | 做了什么 | 适用场景 |
|---|---|---|
| `default` | 算子融合（Triton kernel）、消除 Python 开销 | 通用 |
| `reduce-overhead` | 在 default 的基础上加 **CUDA Graph** | **Launch-bound**（batch 小、kernel 多）；输入 shape 需要固定 |
| `max-autotune` | 对 GEMM/卷积做 autotune，并启用 CUDA Graph | 追求极致性能，编译很慢；Thor 上 GEMM autotune 不可用 |

**ToyVLA batch=1 时 `reduce-overhead` 带来了 2 倍于 default 的收益**，这直接证明了第 00 章的判断：batch=1 时是 Launch-bound（GPU busy 只有 54%）。

### 重编译（C 部分）

```
默认（静态 shape）         6 种 batch 大小 → 编译出的图 2 个
dynamic=True         6 种 batch 大小 → 编译出的图 1 个
```

默认模式下，第一次遇到新 shape 时会以 dynamic 方式重编译一次。如果 shape 是有限的几种，可以用 `torch._dynamo.mark_dynamic(x, 0)` 指定动态维度，或者对输入做 padding/分桶，把 shape 固定下来（`reduce-overhead` 要求 shape 固定）。

## 3.6 TensorBoard 与 HTA

这两个工具都**基于 GPU trace**，在 Thor 上无法使用。这里只介绍原理，在独显平台上可以直接用。

- **TensorBoard 插件**（`torch-tb-profiler`）：`on_trace_ready=tensorboard_trace_handler(dir)`，然后运行 `tensorboard --logdir dir`。提供 Overview（GPU 利用率、各类耗时的占比）、Operator、Kernel、Trace、Memory 等视图。注意该插件已经不再积极维护。
- **HTA（Holistic Trace Analysis）**：读取 Kineto trace，做更深入的统计：

  ```python
  from hta.trace_analysis import TraceAnalysis
  ta = TraceAnalysis(trace_dir="traces/")          # 目录下每个 rank 一个 json
  ta.get_temporal_breakdown()                       # GPU 时间分解：计算 / 通信 / 内存拷贝 / 空闲
  ta.get_idle_time_breakdown()                      # GPU 空闲的原因：host 等待、kernel 间依赖等
  ta.get_gpu_kernel_breakdown()                     # kernel 分类统计
  ta.get_cuda_kernel_launch_stats()                 # launch 延迟统计（定位 Launch-bound）
  ta.get_comm_comp_overlap()                        # 多卡训练：通信与计算的重叠率
  ```

  本机实测：在没有 GPU kernel 的 trace 上调用 `get_temporal_breakdown()`，直接报 `IndexError: single positional indexer is out-of-bounds`。

## 3.7 profiler 的开销与观察者效应

| | 不开 profiler | 开 profiler（CPU+CUDA activity） | 放大倍数 |
|---|---|---|---|
| 慢流水线 | 10.80 ms/step | 35.89 ms/step | 3.3x |
| 优化流水线 | 3.39 ms/step | 12.24 ms/step | 3.6x |
| ToyVLA 前向（record_shapes + with_stack + profile_memory） | 约 3.6 ms | 17.5 ms | 4.9x |

profiler 会给**每一个**算子和 runtime API 调用都加上记录开销（约几微秒），算子越碎，放大越严重。这会**改变 CPU 和 GPU 之间的速度平衡**：

- 不开 profiler 时，慢流水线 CPU 侧的耗时和 GPU 相当，`.item()` 要真正等 GPU（第 02 章 line_profiler 测得每次约 0.2 ms）；
- 开了 profiler 后，CPU 慢了 3 倍，GPU 总是先跑完，于是 3.2 节测得的 sync 耗时**几乎为 0**。

**结论**：torch.profiler 适合看"**有什么**"（哪些算子、多少次调用、在哪里同步、什么 shape），对"**花了多长时间**"要谨慎，尤其是同步等待的时间和 CPU/GPU 的重叠关系。需要准确的时间线时，用开销更低的 nsys。另外，要尽量只打开必要的选项（`record_shapes`、`with_stack`、`profile_memory` 都会增加开销）。

## 3.8 Thor 上的限制与排查过程

这是一次完整的工具排查案例，展示了"当工具本身出问题时"该怎么定位。

**现象**：`profile(activities=[CPU, CUDA])` 采集不到任何 CUDA 事件，Kineto 日志（`KINETO_LOG_LEVEL=0`）报 `GPU trace is empty!`。

**排查步骤**：

| # | 假设 | 验证方法 | 结论 |
|---|---|---|---|
| 1 | torch 版本有问题 | 换用 vla 环境里的 torch 2.14 | ❌ 同样为空 |
| 2 | CUPTI 本身坏了 | 写一个最小的 C 程序直接调用 CUPTI activity API | ❌ 能拿到记录（后来发现时间戳有问题） |
| 3 | Kineto 调用 CUPTI 的方式有问题 | 用 `LD_PRELOAD` 拦截库打印 Kineto 调用的每一个 CUPTI 函数，再在 C 程序中按相同顺序复现 | ✅ **找到原因 1**：采集后设置 `CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER` 会让已采集的 buffer 全部丢失 |
| 4 | 修好原因 1 就够了 | 拦截库跳过这个属性 → runtime API 事件恢复了，但 kernel 被判为 `Out-of-range` | ⚠️ 只修好了一部分 |
| 5 | kernel 时间戳有问题 | C 程序中打印 kernel 记录的 start/end | ✅ **原因 2**：GPU 侧记录（kernel、memset）的**时间戳全部为 0**；CPU 侧 runtime 记录正常 |
| 6 | 是不是权限导致的 | 在宿主机上（root、完整权限）运行同一个程序 | ❌ 同样为 0 |
| 7 | 换 CUPTI 版本 | 2025.3.1（CUDA 13.0）、2026.2.0、nsys 自带的 13.1 | ❌ 全部为 0；`cuptiActivityEnableHWTrace` 返回 NOT_SUPPORTED |

**结论**：CUPTI activity API 在这台 Thor（JetPack 7.2）上拿不到 GPU 时间戳，属于平台限制。nsys 有自己的采集机制，能拿到正确的 kernel 时间。

> **新容器复测（2026-11-06，privileged，CapEff=000001ffffffffff）**：结论不变。[cupti_timestamp_test.cu](cupti_timestamp_test.cu)（最小 CUPTI activity 程序，`nvcc -O2 -arch=sm_110 cupti_timestamp_test.cu -lcupti`）的输出见 [outputs/cupti_timestamp_test.log](outputs/cupti_timestamp_test.log)：5 条 kernel 记录、memset、memcpy 的 start/end **全部为 0**，runtime 记录正常；torch.profiler 的 CUDA 设备侧事件数仍为 0。**有了完整权限也一样**，说明不是权限问题，而是平台问题。

**部分修复**：[common/cupti_fix/](../common/cupti_fix/) 用 `LD_PRELOAD` 跳过那个有问题的属性设置，恢复 CPU 侧的 CUDA runtime/driver API 事件（`cudaLaunchKernel`、`cudaStreamSynchronize`、`cudaMemcpyAsync`），3.2 节的统计就依赖它：

```python
from common import cupti_fix
cupti_fix.ensure()     # 放在 import torch 之前；会带上 LD_PRELOAD 重新启动当前进程
import torch
```

**可用性一览**：

| 功能 | Thor | 说明 |
|---|---|---|
| CPU 算子统计、shape、调用栈、FLOPs 估算 | ✅ | |
| `record_function` 区间 | ✅ | |
| CUDA runtime API 事件 | ⚠️ 需要 `cupti_fix` | |
| GPU kernel / memcpy 的执行时间 | ❌ | 用 nsys |
| 显存统计（profile_memory、snapshot、memory_stats） | ✅ | 不依赖 CUPTI |
| `set_sync_debug_mode` | ✅ | 不依赖 CUPTI |
| torch.compile 诊断 | ✅ | |
| TensorBoard 插件 / HTA | ❌ | 依赖 GPU trace |

**和 nsys 配合使用**：`torch.autograd.profiler.emit_nvtx()` 会为每个算子生成一个 NVTX range，于是在 nsys 的时间线上，每个 GPU kernel 都能对应到发起它的 PyTorch 算子，第 04 章会用到。

## 常见坑

1. **`with_stack=True` 但调用栈是空的**：torch 2.9 中还要加上 `experimental_config=_ExperimentalConfig(verbose=True)`。
2. **没做 warmup 就开始采集**：首次调用的 cuDNN 选算法、懒加载 kernel 会污染结果，用 `schedule` 的 warmup 阶段避开。
3. **把 profiler 下的耗时当成真实耗时**：profiler 会让程序慢 3~5 倍，还会改变 CPU/GPU 的速度平衡（3.7）。
4. **`with_flops` 漏算**：融合算子不在它的统计范围内（3.1）。
5. **`torch.compile` 悄悄没生效**：for 循环里的 graph break 会让整个函数退回 eager，要检查 `counters["stats"]["unique_graphs"]`（3.5）。
6. **`reduce-overhead` 用在计算量很小的函数上反而变慢**（3.5 B 部分）。
7. **动态 shape 导致 reserved 显存暴涨**：用 `expandable_segments:True`（3.4）。
8. **在 Jetson 上期待看到 GPU 时间**：Thor 上不可用，用 nsys（3.8）。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- torch.profiler 从算子层面回答"有什么"：算子、shape、调用次数、调用栈、显存、同步点。
- 同步检测首选 `set_sync_debug_mode`；显存分析用 snapshot、memory_stats，动态 shape 场景开启 `expandable_segments`。
- torch.compile：先确认真的编译了（counters），再选 mode；`reduce-overhead` 对 Launch-bound 效果显著（ToyVLA b=1：2.96 倍）。
- profiler 的开销会改变被测程序的行为，涉及时间的结论要用低开销的工具复核。
- Thor 上 GPU kernel 的时间要用 nsys 看：[第 04 章 Nsight Systems](../04_nsight_systems/README.md)。
