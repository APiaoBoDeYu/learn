# 第 7 章：调试技巧、选型速查与常见坑

## 7.1 `TORCH_LOGS` 速查表

`torch.compile` 的可观测性主要通过环境变量 `TORCH_LOGS` 控制（也可以用
`torch._logging.set_logs(...)` 在代码里设置）。本教程实测用到的几个：

| 选项 | 作用 | 本教程对应章节 |
|---|---|---|
| `output_code` | 打印 Inductor 生成的完整 Python/Triton 源码 | 第 4 章 |
| `graph_breaks` | 打印每个 graph break 的位置和原因 | 第 5 章（也可用 `torch._dynamo.explain`） |
| `recompiles` | 打印触发重新编译的 guard 失败原因 | 第 5 章 |
| `dynamo` | Dynamo 追踪过程的详细日志（字节码级） | 深入调试用，本教程未展开 |
| `aot` | AOTAutograd 阶段的图（前向+反向） | 训练场景调试用 |

组合使用（逗号分隔）：

```bash
TORCH_LOGS="graph_breaks,recompiles" python3 your_script.py
```

## 7.2 缓存位置与清理

**Triton 编译缓存**：`~/.triton/cache`（本机实测约 9.5MB，随着跑过的 kernel 数量增长）。
每个子目录对应一个 `(kernel 源码 hash, 参数签名)` 的编译产物。清理：

```bash
rm -rf ~/.triton/cache
```

**Inductor 编译缓存**：`/tmp/torchinductor_<username>/`（本机是
`/tmp/torchinductor_root/`），存放 Inductor 生成的 Python wrapper 代码和调用的 Triton
kernel。用 `TORCH_LOGS=output_code` 时日志末尾会打印具体文件路径，比如：

```
Output code written to: /tmp/torchinductor_root/6n/c6nlsnfvmgnoycoibc6crjj32axt2uvt4ngabffiujxgatsoyzmq.py
```

直接打开这个 `.py` 文件比看日志更方便，是一份完整可读的普通 Python 文件。清理：

```bash
rm -rf /tmp/torchinductor_*
```

## 7.3 首次编译耗时 vs 缓存后调用

torch.compile 是 JIT，**第一次调用**包含了 Dynamo 追踪 + Inductor codegen +
Triton 编译（可能还有 autotune 搜索）的全部耗时；后续调用直接用缓存的编译产物。
本机实测（第 4 章的 MLP，`mode=default`，`batch=256`）：

```
first call (compile + run): 907.5 ms
second call (cached):       0.76 ms
```

差距接近 1200 倍。这带来两个实践含义：

1. **不要在生产推理请求路径上让第一次调用发生**——推理服务启动后应该先用一批
   典型 shape 的假数据"预热"（warm up）一遍，再开始接收真实流量
2. **对比 eager vs compiled 的性能一定要排除首次编译时间**，只比较预热后的稳定态延迟
   （本教程所有 benchmark 脚本的 `bench_ms` 都在计时前做了 warmup，正是为此）

## 7.4 常见坑

**坑 1：全局变量/闭包变量导致意外 recompile**
Dynamo 会给读取到的全局变量加 guard；如果某个全局配置在训练循环里被频繁修改，
可能触发意料之外的重新编译。排查方式：`TORCH_LOGS=recompiles` 看 guard 失败的具体原因。

**坑 2：`.item()` / `print()` / 非 tensor 的 Python 控制流触发 graph break**
第 5 章已经用真实代码演示过 `.item()` 的情况。经验规律：任何"需要把 tensor 变成
具体 Python 标量才能继续往下走"的操作（`.item()`、`if tensor_condition:`、
`len()` 作用在动态 shape 上等）都是 graph break 高发点。用 `torch._dynamo.explain`
或 `fullgraph=True`（第 5 章 5.1 节）定位。

**坑 3：新 GPU 架构下 Triton 内置 `ptxas`/头文件版本滞后**
本教程第 1 章完整记录了这个真实踩坑过程（Jetson Thor / sm_110a）。核心症状：

- `ptxas fatal: Value 'sm_XXX' is not defined for option 'gpu-name'` → Triton 内置
  `ptxas` 版本太旧，不认识你的架构，需要通过 `TRITON_PTXAS_PATH`（或 `_BLACKWELL`
  变体，取决于 Triton 版本）指向系统 CUDA 工具链的 `ptxas`
- `fatal error: cuda.h: No such file or directory` → 把 CUDA 头文件目录加进 `CPATH`

**坑 4：torch/triton 版本不匹配导致的 `ImportError`**
第 5 章接入自定义 Triton kernel 时实测遇到的
`ImportError: cannot import name 'specialize_impl'`，本质是 torch 的 Inductor 内部
代码调用了某个 triton 版本才有/才被移除的符号。没有强制的版本锁定机制，遇到类似
`ImportError`/`AttributeError` 且看起来"不像是我代码写错了"，优先怀疑版本搭配，
换一个相邻版本的 triton 试试。

**坑 5：`torch.onnx.export` 默认导出器变了**
本教程第 6 章实测，较新版本 PyTorch 的 `torch.onnx.export` 默认走基于 `torch.export`
的新导出器（`dynamo=True`），依赖额外的 `onnxscript` 包。如果只是想用最省事的方式导出
且不想装新依赖，显式传 `dynamo=False` 回退到老的 TorchScript-based 导出器。

## 7.5 延伸阅读：AOTInductor（不在本教程展开）

如果你的场景更接近"用 torch.compile 的思路，但要做成 TensorRT engine 那种可以脱离
Python、离线产出部署产物的形式"，PyTorch 提供了 **AOTInductor**（`torch.export` +
Inductor 的 AOT 模式）：离线把模型编译成一个 `.so`，可以在 C++ 里直接加载调用，
不需要 Python/Dynamo runtime。这与 TensorRT 的"离线 build engine，运行时用 C++ API
加载"在使用体验上是同一类东西，只是生成的产物内部是 Inductor/Triton 编译链而不是
TensorRT builder。本教程未实测这条路径，如果你需要"torch.compile 的开发体验 + AOT 部署"
可以作为下一步的学习方向。

## 7.6 章节速查

| 章节 | 你能学到什么 | 关键实测数字 |
|---|---|---|
| [00 总览](00_overview.md) | Triton/torch.compile 在你已有知识体系里的坐标 | — |
| [01 环境搭建](01_setup.md) | 安装、新架构踩坑与修复 | — |
| [02 Triton 基础](02_triton_basics.md) | program/block/grid、mask、归约 | vector add 0.0249ms，softmax 0.0567ms |
| [03 Triton 进阶](03_triton_intermediate.md) | autotune、matmul、算子融合、profiling | fused LN 比 naive 快 5.2x |
| [04 torch.compile 基础](04_torch_compile_basics.md) | Dynamo/AOTAutograd/Inductor、mode 对比 | 加速 1.09x~1.17x |
| [05 torch.compile 进阶](05_torch_compile_advanced.md) | graph break、动态 shape、自定义 kernel 接入 | fullgraph 从报错到通过 |
| [06 综合对比](06_benchmark_comparison.md) | 五条路径真实 benchmark 与选型 | TensorRT 4.84x，torch.compile 1.12x |
| 07 附录（本章） | 调试开关、缓存、常见坑 | 首次编译 907ms vs 缓存 0.76ms |
