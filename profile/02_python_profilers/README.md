# 第 02 章 Python 级 Profiling：cProfile / line_profiler / py-spy / VizTracer

> 📎 对应笔记：[doc/程序性能分析工具.xmind](../doc/程序性能分析工具.xmind) →「viztracer」（导出火焰图、用 Perfetto 查看）
>
> 漏斗模型的第 1 层（L1）：第 01 章发现"GPU 不忙、某个 CPU 核很忙"之后，**用 Python profiler 找出 CPU 时间花在了哪里**。

## 学习目标

1. 掌握 4 种 Python profiler 的原理、用法和开销，知道什么场景用哪个；
2. 用它们在慢流水线中找出 CPU 侧的热点（Python 预处理、后处理里的小算子、`.item()`）；
3. 理解 **CUDA 异步执行会让 Python profiler 把 GPU 时间算错地方**，并知道怎么识别和规避；
4. 避开 VizTracer `include_files` 的剪枝坑、py-spy 在容器中的权限问题。

## 目录

- [2.1 四种工具的原理与对比](#21-四种工具的原理与对比)
- [2.2 cProfile：函数级统计](#22-cprofile函数级统计)
- [2.3 line_profiler：逐行耗时](#23-line_profiler逐行耗时)
- [2.4 py-spy：采样、无侵入、可以 attach](#24-py-spy采样无侵入可以-attach)
- [2.5 VizTracer：时间线](#25-viztracer时间线)
- [2.6 异步陷阱：Python profiler 看到的 GPU 时间](#26-异步陷阱python-profiler-看到的-gpu-时间)
- [2.7 开销对比与选型](#27-开销对比与选型)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 |
|---|---|
| [workload.py](workload.py) | 公共负载：慢流水线 N 步（本章所有工具都分析它） |
| [01_cprofile.py](01_cprofile.py) | cProfile + pstats：tottime / cumtime / callers |
| [02_line_profiler.py](02_line_profiler.py) | 不修改源码，逐行分析两个热点函数 |
| [03_py_spy.sh](03_py_spy.sh) + [03_py_spy_report.py](03_py_spy_report.py) | 火焰图 / speedscope / 文本汇总 / dump |
| [04_viztracer.py](04_viztracer.py) | 时间线、过滤、自定义事件、异步陷阱 |

被分析的对象是慢流水线（[common/pipeline.py](../common/pipeline.py)），不开任何工具时约 **10.6~10.9 ms/step**：

```bash
$ /opt/venv/bin/python 02_python_profilers/workload.py --steps 50
[slow] 50 steps, 10.851 ms/step
$ /opt/venv/bin/python 02_python_profilers/workload.py --steps 50 --fast
[fast] 50 steps, 3.264 ms/step
```

---

## 2.1 四种工具的原理与对比

| 工具 | 原理 | 粒度 | 开销（本机实测） | 输出 | 典型用途 |
|---|---|---|---|---|---|
| **cProfile** | 确定性：`sys.setprofile` 钩住每一次函数调用和返回 | 函数 | +43% | 统计表、`.prof` | 第一步：找到热点函数 |
| **line_profiler** | 确定性：钩住指定函数的每一行 | 行 | +60%（只分析 2 个函数） | 逐行统计表 | 第二步：看函数内部哪一行慢 |
| **py-spy** | **采样**：外部进程定期读取目标进程内存中的 Python 栈 | 栈 | **+5%** | 火焰图、speedscope、raw | 线上进程、不能改代码、长时间运行 |
| **VizTracer** | 确定性：在 C 层钩住函数调用，记录每次调用的起止时间 | 每一次调用 | +48%（全量）/ +1~11%（过滤后） | **时间线** JSON（Perfetto） | 看时序：谁先谁后、哪里有空隙、多线程/多进程 |

- **确定性 vs 采样**：确定性 profiler 记录每一次调用，不会漏掉，但开销和调用次数成正比；采样 profiler 开销固定、很低，但短于采样间隔的调用可能看不到，结果是统计意义上的。
- **统计表 vs 时间线**：cProfile、line_profiler 只给出累计值，看不出时序；VizTracer 能看到每一次调用的先后和间隔，这对分析流水线和异步行为至关重要。
- **共同的局限**：**它们都只能看到 CPU**。因为 CUDA 是异步的，Python 函数返回时 GPU 可能还没算完（见 2.6）。

## 2.2 cProfile：函数级统计

```bash
# 命令行（不改代码）
/opt/venv/bin/python -m cProfile -o outputs/pipeline.prof 02_python_profilers/workload.py --steps 50
/opt/venv/bin/python -m pstats outputs/pipeline.prof     # 交互式查看：sort tottime / stats 20
snakeviz -s -H 0.0.0.0 -p 8080 outputs/pipeline.prof     # 浏览器可视化（容器内需要映射端口）
```

```python
# 代码内（只分析关心的部分）
prof = cProfile.Profile(); prof.enable()
run(pipe, 50)
prof.disable()
pstats.Stats(prof).strip_dirs().sort_stats("tottime").print_stats(12)
```

[01_cprofile.py](01_cprofile.py) 的输出：

```
慢流水线 50 步（开着 cProfile）: 15.57 ms/step

=== 函数自身耗时 tottime（按 tottime 排序，前 12 行）===
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.168    0.003    0.190    0.004 pipeline.py:140(postprocess)
       50    0.117    0.002    0.157    0.003 pipeline.py:103(_preprocess_python)
      300    0.045    0.000    0.045    0.000 {built-in method torch._transformer_encoder_layer_fwd}
    25250    0.038    0.000    0.038    0.000 module.py:1951(__getattr__)
      300    0.033    0.000    0.213    0.001 transformer.py:781(forward)
9000/3300    0.028    0.000    0.036    0.000 module.py:2823(named_modules)
       50    0.025    0.001    0.025    0.001 {resize}
    ...
=== 谁调用了 Tensor.item（print_callers）===
{method 'item' of 'torch._C.TensorBase' objects}  <-     700    0.010    0.010  pipeline.py:140(postprocess)
```

读表方法：

| 列 | 含义 |
|---|---|
| `ncalls` | 调用次数。`9000/3300` 表示递归：总调用次数 / 非递归的原始调用次数 |
| `tottime` | **函数自身**的耗时（不包括它调用的子函数）。找"热点代码"看这一列 |
| `cumtime` | 累计耗时（包括子函数）。找"热点调用路径"看这一列 |

结论：`postprocess` 和 `_preprocess_python` 的**自身耗时**排在前两位（合计 0.29 s / 50 步 ≈ 5.7 ms/step），它们是 CPU 侧的两大热点，下一步用 line_profiler 看内部。

另外两个值得注意的现象：
- `module.py:__getattr__` 被调用了 25250 次。`nn.Module` 访问子模块和参数都要经过 `__getattr__`，这是 PyTorch eager 模式固有的 Python 开销（`torch.compile` 可以消除它，见第 07 章）。
- `.item()` 一共只算了 0.010 s。但它其实是让 CPU 等 GPU 的同步点，等待的时间有一部分算到了 `postprocess` 自身（tottime）里。cProfile 对 C 扩展函数的计时边界不够精细，2.3 节的 line_profiler 能更清楚地看到这一点。

## 2.3 line_profiler：逐行耗时

```bash
# 方式 1：kernprof（需要在函数上加 @profile 装饰器）
kernprof -l -v script.py
```

```python
# 方式 2：代码内注册，不修改源码（02_line_profiler.py 的做法）
from line_profiler import LineProfiler
lp = LineProfiler()
pipe._preprocess_python = lp(type(pipe)._preprocess_python).__get__(pipe)
pipe.postprocess = lp(type(pipe).postprocess.__wrapped__).__get__(pipe)   # 被 @torch.no_grad() 包了一层，要取 __wrapped__
run(pipe, 30)
lp.print_stats(output_unit=1e-3)
```

输出（节选）：

```
Function: SlowPipeline._preprocess_python at line 103
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   106      6750          9.3      0.0      5.5          for y in range(224):
   107      6720         26.7      0.0     15.9              row = small[y].astype(np.float32) / 255.0
   108     26880         38.5      0.0     22.9              for c in range(3):
   109     20160         82.1      0.0     48.7                  out[c, y] = (row[:, c] - MEAN[c]) / STD[c]
   112        30         10.0      0.3      6.0              t = torch.from_numpy(x).to(self.device)

Function: SlowPipeline.postprocess at line 140
   148      3570          5.5      0.0      3.3                  for t in range(a.shape[0]):
   149      3360        100.7      0.0     61.0                      col[t] = 0.8 * col[t] + 0.2 * self.prev_action[t, d]
   150       210         45.2      0.2     27.4                  if self.cfg.sync_in_loop and col.abs().max().item() > ...
```

结论：

- **预处理**：最内层的通道循环（第 109 行）占 49%。问题在于"Python 循环 + 小块 numpy 操作"：每次操作只处理 224 个元素，numpy 的调用开销远大于计算本身。修复方法是向量化（一次处理整张图）或者放到 GPU 上做（第 07 章）。
- **后处理第 149 行**：执行了 3360 次（16 个时间步 × 7 个维度 × 30 步），占 61%。每次执行会启动约 5 个极小的 GPU kernel（索引、乘、加、写回），这就是 **Launch-bound**：CPU 不停地发起 kernel，每个 kernel 本身只要几微秒。
- **后处理第 150 行**：只执行了 210 次，却占 27%，**每次 0.2 ms**，比第 149 行每次的耗时高约 7 倍。原因是 `.item()` 要把 GPU 上的值拷回 CPU，必须等前面所有已提交的 kernel 跑完，这就是**同步点**。

## 2.4 py-spy：采样、无侵入、可以 attach

```bash
py-spy record -r 200 -o flame.svg -- python workload.py           # 启动并采样，输出火焰图
py-spy record -f speedscope -o p.json -- python workload.py       # speedscope 格式（https://www.speedscope.app）
py-spy record -f raw -o stacks.txt -- python workload.py          # 折叠栈文本，便于脚本处理
py-spy dump --pid <PID>                                           # 打印运行中进程此刻的调用栈（卡住时特别有用）
py-spy top --pid <PID>                                            # 类似 top 的实时视图
```

[03_py_spy.sh](03_py_spy.sh) 依次演示以上用法；[03_py_spy_report.py](03_py_spy_report.py) 把 raw 输出汇总成"文本版火焰图"。

### 火焰图怎么读

- 横轴是**样本占比**（不是时间先后）：越宽，说明 CPU 在这个函数上花的时间越多；
- 纵轴是调用栈：下层调用上层（py-spy 的 SVG 默认根在上方，从上往下长）；
- 先找"宽的平顶"：自身耗时最多的函数。

### 文本汇总（只统计稳态的 `step` 调用栈，排除 `import torch` 和模型初始化）

```
总样本数 1074，其中 'step (pipeline.py' 内的稳态样本 689（64%），以下百分比均相对于稳态样本

  self%  function（自身耗时最多）
  30.3%  postprocess (pipeline.py:149)        <- 后处理的小算子循环
  15.8%  _preprocess_python (pipeline.py:109) <- 预处理的通道循环
  10.3%  forward (transformer.py:922)
   7.8%  _conv_forward (conv.py:544)
   6.8%  postprocess (pipeline.py:150)        <- .item() 同步
   5.2%  _preprocess_python (pipeline.py:107)

 total%  function（累计耗时，含子调用，按函数聚合）
 100.0%  step (pipeline.py)
  40.6%  postprocess (pipeline.py)
  29.3%  infer (pipeline.py)
  28.6%  preprocess (pipeline.py)
```

> 注意：采样文件中有 36% 的样本来自启动阶段（`import torch` 就占了约 19%）。对于短时间运行的脚本，**一定要排除启动阶段**，否则热点会被 import 和初始化淹没。

py-spy 的结论和 cProfile、line_profiler 一致，但开销只有约 5%（10.85 → 11.39 ms/step），而且**不需要修改代码**。

### dump：程序"卡住"时看它在干什么

```
Thread 19102 (active+gil): "MainThread"
    postprocess (common/pipeline.py:149)
    decorate_context (torch/utils/_contextlib.py:120)
    step (common/pipeline.py:175)
    run (workload.py:37)
```

### ⚠️ 容器中的权限问题

```
$ py-spy dump --pid 12345
Error: Permission denied (os error 13)
```

py-spy 通过 ptrace 读取目标进程的内存。本容器的 `kernel.yama.ptrace_scope = 1`（只允许追踪自己的子进程），而且容器没有 `CAP_SYS_PTRACE` 权限。三种解决办法：

| 方法 | 做法 | 适用场景 |
|---|---|---|
| 让 py-spy 作为父进程 | `py-spy record -- python xxx.py` | 能控制启动方式（本章大部分用法） |
| 目标进程主动允许被追踪 | 在目标进程里调用 `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`（`workload.py --allow-ptrace`） | 不能改启动方式，但能改代码 |
| 给容器加权限 | `docker run --cap-add SYS_PTRACE ...` | 一劳永逸，需要重建容器 |

> **新容器实测（2026-11-06，`--privileged`）**：`ptrace_scope` 仍然是 1，但进程有 CAP_SYS_PTRACE，`py-spy dump --pid` 可以直接 attach 一个不是自己子进程、也没有调用 `prctl` 的进程，returncode 0，见 [03_py_spy_attach.py](03_py_spy_attach.py) 和 [outputs/py_spy_dump_privileged.log](outputs/py_spy_dump_privileged.log)。在非特权环境中，上面的前两种方法仍然适用。

另外，**`--native`（同时采样 C/C++ 栈）在 aarch64 上不支持**：
```
Collecting stack traces from native extensions (`--native`) is not supported on your platform.
```

## 2.5 VizTracer：时间线

```bash
viztracer -o result.json workload.py --steps 20         # 命令行
vizviewer result.json                                   # 本地查看（会启动一个 web 服务）
vizviewer --flamegraph result.json                      # 火焰图视图
# 或者把 json 拖进 https://ui.perfetto.dev （推荐，大文件也能流畅打开）
```

```python
from viztracer import VizTracer
tracer = VizTracer(output_file="result.json", max_stack_depth=6, ignore_c_function=True)
tracer.start()
run(pipe, 20)
tracer.stop(); tracer.save()
```

### A/B. 开销与过滤（[04_viztracer.py](04_viztracer.py)）

```
不开 tracer:                        10.57 ms/step
A 默认（全部追踪）:                  15.63 ms/step  开销 +48%  事件数 47665
B1 include=[common/]               10.84 ms/step  开销  +2%  事件数      0  记录到的阶段 []
B2 include=[common/, 本目录]          11.70 ms/step  开销 +11%  事件数   4683  记录到的阶段 ['capture', 'preprocess']
B3 depth<=6 + ignore_c             11.05 ms/step  开销  +5%  事件数   1872  记录到的阶段 ['capture', 'preprocess', 'infer', 'postprocess']
B4 depth<=6 + min_dur 20us         11.71 ms/step  开销 +11%  事件数    805  记录到的阶段 ['preprocess', 'infer', 'postprocess']
```

| 参数 | 作用 | 注意 |
|---|---|---|
| `max_stack_depth` | 只记录栈深度不超过 N 的调用 | **最安全**的降开销方式，推荐 |
| `ignore_c_function` | 不记录 C 函数（如 `torch.conv2d`） | 能大幅减少事件数，但就看不到 torch 算子了 |
| `min_duration`（μs） | 丢弃短于阈值的调用 | 短函数会消失（B4 中 `capture` 不到 20 μs，被丢弃了） |
| `include_files` / `exclude_files` | 只追踪或排除某些文件 | ⚠️ **会剪掉整棵子树**（见下文） |
| `log_sparse` + `@log_sparse` | 只记录用装饰器标记的函数 | 开销最低，适合长时间运行 |
| `tracer_entries` | 环形缓冲区的大小 | 运行时间长时，超出部分会覆盖最早的事件 |

**⚠️ include_files 的剪枝坑**（B1、B2 的实测）：
- B1 只写了 `common/`，事件数是 **0**。`run()` 在 `workload.py` 里，不在列表中，VizTracer 遇到不在列表中的函数时，会**连同它的整棵调用子树一起跳过**，于是 `run()` 调用的流水线函数一个都没被记录。
- B2 把本目录也加进去后，`infer` 和 `postprocess` 仍然缺失。它们被 `@torch.no_grad()` 装饰过，实际调用链是 `step → decorate_context（在 torch 的文件里）→ infer`，`decorate_context` 不在列表中，又被剪掉了。

→ **使用 include_files 时，列表中必须包含调用链上所有的文件，包括装饰器所在的文件**。用 `max_stack_depth` 更省心。

### C. 自定义事件

```python
with tracer.log_event(f"frame {i}"):                 # 自定义区间（在时间线上显示为一个块）
    pipe.step(i)
tracer.add_counter("pipeline_state", {"n_warnings": n, "cuda_mem_mb": m})   # 数值曲线（counter 轨道）
tracer.log_instant("checkpoint", args={"step": i})   # 瞬时事件（一条竖线）
```

用 Perfetto 打开 `outputs/viztracer_custom.json`，可以看到每一帧的区间、显存曲线和 checkpoint 标记。这比在日志里打印时间戳直观得多。

### 在 Perfetto 里看什么

- **每一帧的阶段分布**：preprocess / infer / postprocess 各占多宽；
- **重复的小块**：后处理里密密麻麻的 `__mul__`、`__setitem__`，每一个都很短，但数量巨大，这是 Launch-bound 在 CPU 侧的表现；
- **长条的 `item`**：CPU 在这里等 GPU。

## 2.6 异步陷阱：Python profiler 看到的 GPU 时间

```
=== D. 各阶段的 CPU 侧耗时（ms/step），来自 trace 统计 ===
[子进程 CUDA_LAUNCH_BLOCKING=1] 31.31 ms/step
stage               默认(异步)  LAUNCH_BLOCKING
capture               0.00             0.01
preprocess            3.70             3.84
infer                 2.77             9.90
postprocess           4.85            16.91
sum                  11.33            30.65
```

**默认（异步）模式下**，`infer` 只花了 2.77 ms，这只是 CPU 把模型的 kernel 一个个放进队列的时间，函数返回时 GPU 还在算。后处理中**第一个 `.item()` 会等这些 GPU 工作跑完**，所以 `postprocess` 看起来很贵（4.85 ms），其中一部分其实是推理的 GPU 时间。

**设置 `CUDA_LAUNCH_BLOCKING=1` 后**，每次 kernel launch 都要等 GPU 执行完才返回，GPU 时间就"留在"了发起它的函数里：`infer` 变成 9.90 ms。但整体也慢了 3 倍（31.31 ms/step）：每个 kernel 都要额外付出一次同步的往返延迟，而 ToyVLA 一次前向有几百个 kernel，后处理的小算子更多。

结论：

| 你看到的 | 实际情况 |
|---|---|
| 某个函数在 Python profiler 里很快 | 可能只是 launch 很快，GPU 工作还在后面排队 |
| 某个同步点（`.item()`、`.cpu()`、`print(tensor)`）很慢 | 它在为**前面所有**的 GPU 工作买单 |
| `CUDA_LAUNCH_BLOCKING=1` 下的耗时 | 定性地说明"GPU 时间属于谁"，但**绝对值被严重放大**，不能当作真实数据 |

→ 想知道 GPU 时间的真实分布，要用能同时看到 CPU 和 GPU 时间线的工具：**Nsight Systems（第 04 章）**。（torch.profiler 本来也能做到，但在 Thor 上拿不到 GPU 数据，见第 03 章。）

## 2.7 开销对比与选型

| 工具 | 慢流水线 ms/step | 开销 |
|---|---|---|
| 不开任何工具 | 10.6 ~ 10.9 | 基准 |
| py-spy record（200 Hz） | 11.39 | **+5%** |
| VizTracer（depth ≤ 6 + ignore_c） | 11.05 | +5% |
| VizTracer（全量） | 15.63 | +48% |
| cProfile | 15.57 | +43% |
| line_profiler（只分析 2 个函数） | 17.35 | +60% |

> 开销和代码特征强相关：函数调用越多、越短，确定性 profiler 的开销越大。慢流水线的后处理有大量极短的调用，属于对确定性 profiler 不友好的情况。

**选型建议**：

```
程序在跑 / 不能改代码 / 线上环境        → py-spy（record / dump / top）
第一次分析，想知道"哪个函数慢"          → cProfile（或 py-spy）
已经知道是哪个函数，想知道"哪一行慢"    → line_profiler
关心时序、并发、流水线的阶段和空隙      → VizTracer + Perfetto
涉及 GPU                                → 先用以上工具看 CPU 侧，再用 nsys 看完整的时间线
```

## 常见坑

1. **把 Python profiler 的耗时当成 GPU 耗时**（2.6）：异步执行会让 GPU 时间算到后面的同步点上。
2. **采样结果里混进了启动阶段**：`import torch` 就占了约 19% 的样本，要先做 warmup，并且只统计稳态部分。
3. **VizTracer 的 `include_files` 剪掉整棵子树**（2.5），包括装饰器所在的文件。
4. **容器里 py-spy 没有权限**（2.4）：用 `record --`、`prctl` 或 `--cap-add SYS_PTRACE` 解决。
5. **确定性 profiler 的开销让热点"变形"**：开销和调用次数成正比，小函数会被高估。可以用 py-spy 交叉验证。
6. **line_profiler 分析被装饰的函数**：要取 `__wrapped__`，否则分析的是装饰器本身。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- 4 种工具：cProfile（函数级）→ line_profiler（行级）→ py-spy（低开销采样）→ VizTracer（时间线）。
- 慢流水线 CPU 侧的三大热点：后处理里的小算子（Launch-bound）、Python 预处理（CPU-bound）、`.item()`（同步）。
- Python profiler 看不到 GPU。异步执行会把 GPU 时间算到同步点上，需要用 nsys 来验证。
- 下一章：[第 03 章 torch.profiler](../03_torch_profiler/README.md)，从框架层面看算子。
