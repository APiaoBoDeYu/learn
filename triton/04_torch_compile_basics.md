# 第 4 章：torch.compile 基础

## 4.1 与 TorchScript 的关键差异

你已经用过 `torch.jit.trace`/`torch.jit.script`，这里直接对比：

| | `torch.jit.trace` | `torch.jit.script` | `torch.compile` |
|---|---|---|---|
| 捕获方式 | 跑一遍真实输入，录制执行的算子序列 | 静态分析 Python AST，编译成 TorchScript IR | **Dynamo**：解释执行 Python 字节码，边执行边捕获成 FX Graph |
| 遇到数据依赖的控制流 | **静默**烙死当次走过的分支，之后输入变了也不会走另一分支——隐蔽的正确性 bug | 支持，但对 Python 语法子集要求严格，很多写法直接报错"不支持" | 支持：能追踪的部分正常捕获，追踪不了的地方**graph break**，用 eager 执行那一小段，结果总是对的 |
| 失败模式 | 悄悄给错误结果 | 编译时报错（早发现） | 运行时正常出结果，但可能因为 break 太多而没什么加速 |
| 何时编译 | 你手动调用时（AOT） | 你手动调用时（AOT） | 第一次实际调用时（JIT），之后复用缓存 |

第 5 章会用真实代码演示 graph break 的行为；这一章先看 torch.compile 最基础的使用方式。

## 4.2 第一个例子：MLP block

完整代码见 [`code/04_compile_mlp.py`](code/04_compile_mlp.py)。模型是一个典型的
transformer 风格 block（Linear → GELU → Linear → 残差 → LayerNorm）：

```python
class MLP(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        out = self.fc2(h)
        return self.ln(out + x)

model = MLP().cuda().eval()
compiled_model = torch.compile(model, mode="default")
out = compiled_model(x)   # 第一次调用触发编译
```

用法上比 `torch.jit.script(model)` 还简单——**不需要**改代码结构、不需要处理"这段 Python
语法 script 不支持"的报错。这是 torch.compile 相对 TorchScript 最大的可用性提升。

## 4.3 三种 `mode`

- `default`：默认策略，编译速度快，通用优化（算子融合、内存规划）
- `reduce-overhead`：额外用 CUDA Graph 降低 Python/launch overhead，适合小 batch、
  launch-overhead 敏感的场景，但会占用额外显存且对输入变化更敏感
- `max-autotune`：为矩阵乘/卷积等重计算算子做 Triton kernel 级 autotune（本质上就是
  第 3 章讲的 `@triton.autotune`，只是候选集合和搜索由 Inductor 自动生成），编译时间显著变长

### 实际运行（batch=256 和 batch=8192 两种规模）

```bash
python3 code/04_compile_mlp.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
=== batch size 256 ===
[PASS] batch=256 mode=default: allclose=True, max_abs_err=9.835e-07
  eager    : 0.2856 ms
  compiled : 0.2535 ms  (mode=default, speedup=1.13x)

[PASS] batch=256 mode=reduce-overhead: allclose=True, max_abs_err=9.835e-07
  eager    : 0.2960 ms
  compiled : 0.2664 ms  (mode=reduce-overhead, speedup=1.11x)

[PASS] batch=256 mode=max-autotune: allclose=True, max_abs_err=9.835e-07
  eager    : 0.2953 ms
  compiled : 0.2534 ms  (mode=max-autotune, speedup=1.17x)

=== batch size 8192 ===
[PASS] batch=8192 mode=default: allclose=True, max_abs_err=2.912e-05
  eager    : 4.2899 ms
  compiled : 3.9389 ms  (mode=default, speedup=1.09x)

[PASS] batch=8192 mode=reduce-overhead: allclose=True, max_abs_err=2.912e-05
  eager    : 4.3021 ms
  compiled : 4.2348 ms  (mode=reduce-overhead, speedup=1.02x)

[PASS] batch=8192 mode=max-autotune: allclose=True, max_abs_err=2.915e-05
  eager    : 4.2609 ms
  compiled : 4.2222 ms  (mode=max-autotune, speedup=1.01x)
```

如实报告：在这个模型规模、这块 GPU 上，加速在 1.0x～1.2x 之间，**不是"编译就能翻倍"**。
这是符合预期的真实结果——这个 MLP block 的主要耗时是两次大矩阵乘（已经调用高度优化的
cuBLAS/tensor core 路径），Inductor 能优化的空间主要是**算子融合**（把 GELU、残差加、
LayerNorm 这些"胶水"算子合并成更少的 kernel launch），对矩阵乘本身通常不会有突破性提升。
日志里还能看到一行有意思的信息：

```
Not enough SMs to use max_autotune_gemm mode
```

这是 Inductor 检测到当前 GPU 的 SM（流多处理器）数量较少（Jetson Thor 是嵌入式/边缘 GPU，
SM 数远少于 A100/H100），认为对 GEMM 做 max-autotune 的收益不足以覆盖搜索开销，自动跳过了
——这本身就是一个很好的例子，说明 torch.compile 的优化决策会**感知实际硬件**，
不是无脑套用同一套策略。

## 4.4 查看 Inductor 生成的 Triton 代码

用 `TORCH_LOGS=output_code` 可以把 Inductor 生成的完整代码打印出来：

```bash
TORCH_LOGS=output_code python3 code/04_compile_mlp.py
```

实测输出里能找到形如下面的融合 kernel（GELU 与残差加融合到了一个 kernel 里）：

```python
@triton.jit
def triton_poi_fused_addmm_gelu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4096)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
```

对照第 2 章你学的语法：`tl.program_id`、`tl.arange`、`tl.load`/`tl.store`——**这就是
Triton 代码本身**，只是变量名是 Inductor 自动生成的（`tmp0`, `tmp1`...），逻辑是把
`bias add + GELU`（`0.5 * x * (1 + erf(x / sqrt(2)))`，GELU 的精确定义）揉进了一个 kernel。
这印证了第 0 章的类比：**学会读 Triton，就能读懂 torch.compile 在 GPU 上生成的一切**。

另外还能看到一个单独的 `triton_per_fused_add_addmm_native_layer_norm_1` kernel，
把"残差加 + LayerNorm 的均值方差归约 + 归一化 + 仿射变换"全部融合在一起——
这正是我们在第 3 章手写的 fused LayerNorm kernel 在做的事情，Inductor 自动生成了等价逻辑，
只是变量命名和排布方式不同。

## 4.5 小结

- `torch.compile` 用起来比 `torch.jit.script` 更简单：不需要改代码结构，遇到不支持的
  语法会 graph break 而不是直接报错
- 三种 `mode` 在小模型上差异不大，`max-autotune` 的收益取决于硬件（本机 SM 数量不足时会
  自动跳过 GEMM autotune）
- 实测加速在 1.0x～1.2x，主要来自算子融合而非替换矩阵乘实现——这是诚实的结果，不是
  "compile 万能加速"的营销话术
- `TORCH_LOGS=output_code` 能让你直接看到 Inductor 生成的 Triton 源码，第 2/3 章学的语法
  在这里能直接对上号

下一章：[第 5 章：torch.compile 进阶](05_torch_compile_advanced.md) —— graph break、
动态 shape、把你自己写的 Triton kernel 接入 torch.compile
