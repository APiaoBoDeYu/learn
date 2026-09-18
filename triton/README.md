# OpenAI Triton 与 torch.compile 系统教程

面向已掌握 **TorchScript / ONNX / CUDA / TensorRT** 的读者，系统学习 Triton 和
torch.compile，并理解它们与已有技术栈的区别与联系。

所有代码在真实环境跑通验证，文档里的延迟数字都是本机实测数据，不是估算。

## 环境

- 硬件：NVIDIA Thor (Jetson, aarch64)，compute capability (11, 0) / sm_110a
- CUDA 13.0
- PyTorch 2.9.1
- Triton 3.5.0（**注意**：不是最新的 3.6.0，原因见 [第 1 章](01_setup.md)）
- TensorRT 10.14.1（第 6 章对比用）

如果你在别的机器上运行本教程代码，大概率不需要第 1 章里那些新架构专属的环境变量
workaround——那些是这块特定硬件（较新的边缘 GPU 架构）暴露出的问题，代表了一类
"生态适配滞后于新硬件发布"的通用现象，值得了解但不是普遍必需的步骤。

## 学习路线图

| 章节 | 内容 | 建议用时 |
|---|---|---|
| [00 全局地图](00_overview.md) | Triton/torch.compile 与 TorchScript/ONNX/CUDA/TensorRT 的对比表、架构图、三个关键类比 | 20 分钟 |
| [01 环境搭建](01_setup.md) | 安装 Triton，解决新架构下的编译报错 | 15 分钟 |
| [02 Triton 基础](02_triton_basics.md) | 编程模型（program/block/grid/mask），vector add，softmax | 45 分钟 |
| [03 Triton 进阶](03_triton_intermediate.md) | autotune，tiled matmul，fused LayerNorm（核心：kernel 融合的真实收益），profiling | 60 分钟 |
| [04 torch.compile 基础](04_torch_compile_basics.md) | Dynamo → AOTAutograd → Inductor 流水线，MLP 实例，查看生成的 Triton 代码 | 45 分钟 |
| [05 torch.compile 进阶](05_torch_compile_advanced.md) | graph break 机制，动态 shape/recompile，**手写 Triton kernel 接入 torch.compile** | 60 分钟 |
| [06 综合对比](06_benchmark_comparison.md) | TorchScript / torch.compile / 手写 Triton / ONNX+TensorRT 五条路径真实 benchmark，选型指南 | 30 分钟 |
| [07 附录](07_appendix.md) | `TORCH_LOGS` 速查、缓存清理、常见坑、AOTInductor 延伸阅读 | 需要时查阅 |

建议按顺序阅读：第 2、3 章打好 Triton 基础后，第 4、5 章会反复引用"这就是第 2/3 章学的
语法"，第 6 章的综合对比依赖前面所有章节的结论。

## 快速开始

```bash
cd /root/data/learn/code
python3 02_vector_add.py      # 验证环境跑通
```

每个 `code/*.py` 都能独立运行，产生的输出应与对应章节文档里贴出的"实际运行输出"一致
（延迟数字会因硬件负载有小幅波动，属于正常范围）。

## 目录结构

```
/root/data/learn/
├── README.md                    本文件
├── 00_overview.md ~ 07_appendix.md   八个章节
└── code/
    ├── common.py                 共享工具：环境变量修复、计时、正确性校验
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

## 核心结论速览（细节见各章）

- Triton ≈ 块级（block-level）的 CUDA：省去手动线程索引和边界判断，代价是你仍需自己写
  归约、数值稳定性等逻辑
- torch.compile ≈ 更宽容的 TorchScript：Dynamo 用字节码级捕获代替 trace/script，
  遇到追踪不了的代码自动 graph break 退回 eager，正确性优先
- torch.compile 在 GPU 上默认生成 Triton kernel（Inductor 后端）——学 Triton 能让你
  读懂 `TORCH_LOGS=output_code` 的输出
- 手写 Triton kernel 的真实价值在**融合**（本教程实测 fused LayerNorm 比多 kernel 版快
  5.2x），且需要通过 `torch.library.triton_op` 接入 torch.compile 的图，才能同时获得
  "融合正确"与"覆盖全图"两个收益
- TensorRT（AOT，离线 build）与 torch.compile（JIT，运行时编译）不是替代关系，是不同
  部署场景的选择：本教程实测同一模型 TensorRT 比 eager 快 4.84x，torch.compile 快
  1.07x~1.13x——AOT 换来的是更充分的离线优化空间，JIT 换来的是零部署链路复杂度
