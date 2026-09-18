# 第 1 章：环境搭建

## 1.1 安装 Triton

Triton 通过 pip 安装：

```bash
pip3 install triton
```

在 x86_64 + 常见 GPU 上，这一步装完基本就能直接用。但本教程的实测机器是
**NVIDIA Jetson Thor (aarch64, sm_110a)** —— 一块相当新的架构，装完之后遇到了两个
值得记录的真实问题，因为它们代表了一类"Triton 在新硬件/新平台上常见的坑"。

### 问题 1：`triton==3.6.0` 与 `torch==2.9.1` 的内部 API 不匹配

torch 2.9.1 的 Inductor 在处理"用户自定义 Triton kernel 接入 torch.compile"这条路径时
（第 5 章会讲），需要调用 `triton.runtime.jit.specialize_impl`，但这个符号在 triton 3.6.0
里已经被重命名/移动，实测报错：

```
ImportError: cannot import name 'specialize_impl' from 'triton.runtime.jit'
```

这是版本没对齐导致的，不是代码写法的问题。查看 PyPI 上的可用版本并回退：

```bash
pip3 install "triton==3.5.0"
```

**经验**：torch 官方 release 会在文档里标注"配套的 triton 版本区间"，装 triton 前最好先确认
`torch.__version__` 对应哪个 triton 版本（`torch` 的 `METADATA` 里只写了"install the matching
version"，没写死具体号码，实测踩坑后手动尝试才定下 3.5.0 能用）。

### 问题 2：新 GPU 架构下，Triton 内置的 `ptxas` 不认识 `sm_110a`

Triton wheel 里自带了一份预编译的 `ptxas`（PTX → cubin 的汇编器），版本落后于系统装的
CUDA 13.0 工具链。跑最简单的 kernel 时报错：

```
ptxas fatal   : Value 'sm_110a' is not defined for option 'gpu-name'
```

而系统里 `/usr/local/cuda-13.0/bin/ptxas`（配合驱动版本走的）是认识这个架构的。Triton 提供了
环境变量钩子，可以让它使用外部 `ptxas` 而不是内置的（这个变量名在不同 Triton 版本里不同，
3.6.0 是 `TRITON_PTXAS_BLACKWELL_PATH`，3.5.0 统一用 `TRITON_PTXAS_PATH`，可在
`triton/knobs.py` 源码里搜 `env_nvidia_tool` 确认当前版本用哪个）：

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### 问题 3：Triton 的 JIT 编译步骤找不到 `cuda.h`

Triton 运行时会用系统 `gcc` 现场编译一个launcher C 扩展来调用 CUDA Driver API，
如果 `cuda.h` 不在 `gcc` 默认搜索路径里会报错：

```
fatal error: cuda.h: No such file or directory
```

解决方式是把 CUDA 头文件/库目录加进 `CPATH`/`LIBRARY_PATH`（这是 `gcc` 认的标准环境变量，
不是 Triton 专属的）：

```bash
export CPATH=/usr/local/cuda-13.0/targets/sbsa-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LIBRARY_PATH
```

### 汇总：最终生效的环境变量

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CPATH=/usr/local/cuda-13.0/targets/sbsa-linux/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:$LIBRARY_PATH
```

本教程把这几行放进了 [`code/common.py`](code/common.py) 的顶部，用 `os.environ.setdefault`
在 `import triton` **之前**设置好，这样每个示例脚本自己 `python3 xxx.py` 就能跑，不依赖你
提前 `export` 或 `source ~/.bashrc`——如果你在别的机器上（尤其是较新/较旧的 GPU 架构，
或者 conda/docker 环境），Triton 内置 `ptxas` 版本和驱动/CUDA 工具链不匹配是最常见的
"装完就是跑不起来"的原因，遇到报错先看是不是这个问题。

> 如果你在常见的 x86_64 + 主流 GPU（如 A100/4090/H100 等已发布一段时间的架构）上，
> 大概率不会遇到问题 1/2/3——这些坑本质是"新架构/新平台跑得比生态适配快"导致的。

## 1.2 验证安装：最小 kernel

[`code/common.py`](code/common.py) 提供了 `device_info()` 辅助函数，打印环境信息；
下面是一个独立的最小验证（等价于第 2 章 vector add 的核心逻辑）：

```python
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

x = torch.rand(1024, device='cuda')
y = torch.rand(1024, device='cuda')
out = torch.empty_like(x)
add_kernel[(triton.cdiv(1024, 256),)](x, y, out, 1024, BLOCK_SIZE=256)
torch.cuda.synchronize()
assert torch.allclose(out, x + y)
```

### 实际运行输出（本机）

```
torch      : 2.9.1
triton     : 3.5.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
allclose: True
SUCCESS: Triton kernel compiled and ran on this GPU architecture
```

## 1.3 本教程的目录结构

```
/root/data/learn/
├── 00_overview.md              第 0 章：全局地图
├── 01_setup.md                 本章
├── 02_triton_basics.md
├── 03_triton_intermediate.md
├── 04_torch_compile_basics.md
├── 05_torch_compile_advanced.md
├── 06_benchmark_comparison.md
├── 07_appendix.md
└── code/
    ├── common.py                共享工具：环境变量修复、计时、正确性校验
    ├── 02_vector_add.py
    ├── 02_softmax.py
    ├── 03_matmul_autotune.py
    ├── 03_fused_layernorm.py
    ├── 03_profiling.py
    ├── 04_compile_mlp.py
    ├── 05_graph_break_demo.py
    ├── 05_dynamic_shapes.py
    ├── 05_custom_triton_in_compile.py
    └── 06_benchmark_all.py
```

每个 `code/*.py` 都可以独立运行：

```bash
cd /root/data/learn/code
python3 02_vector_add.py
```

下一章开始正式学习 Triton 编程模型。→ [第 2 章：Triton 基础](02_triton_basics.md)
