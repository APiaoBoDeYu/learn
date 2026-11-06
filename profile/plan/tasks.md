# Tasks：任务拆解与进度

> 依据 [plan.md](plan.md)。状态：⬜ 未开始 / 🔄 进行中 / ✅ 完成 / ⚠️ 完成但有偏差（见实测记录）
> 每个任务的 DoD 见 [spec.md §7](spec.md#7-验收标准definition-of-done)。🚩 = 检查点，完成后向用户汇报。

## 阶段 A：基础设施

| ID | 任务 | 产出 | 依赖 | 状态 |
|---|---|---|---|---|
| T00 | 环境检查：Python 包、pip 联网、nsys/ncu/trtexec/tegrastats、锁频权限、磁盘 | `plan/env_report.md`；回答 Q1/Q4/Q6 | - | ⚠️ |
| T00' | 新容器 v1.3.0_thor（privileged）环境检查与重装 | `plan/env_report.md` 新增一节；更新 Q2/Q3/Q6 | - | ✅ |
| T01 | 共享组件：`bench_utils.py`、`models.py`、`pipeline.py`、`env_info.py`、`requirements.txt`、目录骨架、`.gitignore` | `common/` 可导入，自测脚本通过 | T00 | ✅ |

🚩 检查点 1：汇报环境情况和 common 设计

## 阶段 B：工具章节

| ID | 任务 | 关键 demo | 依赖 | 状态 |
|---|---|---|---|---|
| T02 | 第 00 章 基础理论 | 计时对错对比、Roofline 实测、MFU 计算 | T01 | ✅ |
| T03 | 第 01 章 系统监控 | jtop logger、dmon logger、负载模式对比 | T01 | ✅ |
| T04 | 第 02 章 Python profilers | cProfile / line_profiler / py-spy / VizTracer | T01 | ✅ |
| T05 | 第 03 章 torch.profiler | 算子表、trace、schedule、memory、同步检测、compile | T01 | ⚠️ |

🚩 检查点 2：完成 03 章后汇报

| ID | 任务 | 关键 demo | 依赖 | 状态 |
|---|---|---|---|---|
| T06 | 第 04 章 Nsight Systems | NVTX、stats/analyze、stream overlap、CUDA Graph、GPU metrics | T01 | ✅ |
| T07 | 第 05 章 Nsight Compute | 4 类瓶颈 + bank conflict kernel，优化前后对比 | T01 | ✅ |

🚩 检查点 3：完成 05 章后汇报

| ID | 任务 | 关键 demo | 依赖 | 状态 |
|---|---|---|---|---|
| T08 | 第 06 章 TensorRT | ONNX 导出、trtexec、dumpProfile、三步法 | T01、T06、T07 | ✅ |
| T09 | 第 07 章 优化手段 | 对照表中每种手段的最小前后对比 | T02–T08 | ✅ |

## 阶段 C：综合与收尾

| ID | 任务 | 产出 | 依赖 | 状态 |
|---|---|---|---|---|
| T10 | 第 08 章 综合实战 | ToyVLA 慢流水线全流程、`report_template.md`、实测报告 | T02–T09 | ✅ |
| T11 | 附录 | 速查表、工具对比、可视化流程、常见坑（汇总实测）、参考资料 | T02–T10 | ✅ |
| T12 | 收尾 | 顶层 README 加各章链接、通读校对、清理大产物、最终 DoD 核对 | 全部 | ✅ |

🚩 检查点 4：最终交付

---

## 实测记录（每完成一个任务追加）

<!-- 格式：
### Txx 章节名（日期）
- 结果：
- 发现 / 与 spec 的偏差：
- 遗留问题：
-->

### T00 环境检查（2026-11-05）⚠️
- 结果：Jetson AGX Thor，JetPack 7.2，MAXN + jetson_clocks，GPU 锁在 1575 MHz，20 个 SM（sm_110）。详见 [env_report.md](env_report.md)
- 安装了 viztracer、line_profiler、snakeviz、nvtx、HTA（Q1：pip 可联网 ✅）；`packaging` 已回滚到 23.2 以兼容 streamlit/mlflow
- 偏差 1：**GPU 性能计数器无权限**（`RmProfilingAdminOnly=1`，容器没有 CAP_SYS_ADMIN）→ ncu 和 nsys GPU metrics 不可用（Q2/Q3），**阻塞 T07**，需要用户在宿主机侧处理
- 偏差 2：容器内没有 tegrastats/nvpmodel/jetson_clocks 命令（Q4/Q6），用 jtop API 替代
- 偏差 3：triton/torch.compile 需要 `TRITON_PTXAS_PATH`，已在 `common/__init__.py` 中统一设置

### T01 共享组件（2026-11-05）✅
- 产出：`common/{bench_utils,env_info,models,pipeline,selftest}.py`、`requirements.txt`、`.gitignore`、各章空目录
- 自测（`python common/selftest.py`）：slow 11.37ms → fast 3.34ms（3.40x）。逐个关闭开关后的加速比：eager_small_ops 1.51x > py_preprocess 1.33x > fp32 1.21x > sync_in_loop 1.18x > realloc 1.11x > no_pin 1.03x
- 发现：**no_pin 和 realloc 对延迟几乎没有影响**（Thor 是 CPU/GPU 统一内存；caching allocator 会复用显存块）。这两个开关保留下来，作为"凭直觉判断的瓶颈不一定是真瓶颈"的反例；realloc 主要在第 03 章 memory snapshot 中观察显存峰值和碎片

### T02 第 00 章 基础理论（2026-11-05）✅
- 产出：`00_theory/{README.md, exercises.md, 01_timing_pitfalls.py, 02_roofline.py, 03_mfu.py}`，`outputs/roofline.png`
- 实测：DRAM 约 230 GB/s；FP32 6.39 TFLOPS（理论 8.06）；TF32 65；FP16/BF16 约 157/162 TFLOPS（突发）；拐点约 690 FLOP/B
- 发现 1：Thor 的 L2 有 32 MB，GEMV 4096² fp16 不刷 L2 时测出 930 GB/s（是 DRAM 的 4 倍），已加入 `flush_l2()`
- 发现 2：功耗墙。FP16 持续满载时吞吐从 157 降到约 104 TFLOPS，jtop 显示的频率不变，VDD_GPU 约 96 W
- 发现 3：`FlopCounterMode` 不认识 TransformerEncoderLayer 的 fastpath，CPU 上的 SDPA 也会漏算，统计时需关闭 fastpath 并在 GPU 上跑
- **重大偏差：torch.profiler 在 Thor 上拿不到 GPU 数据**。排查过程：
  1. Kineto 报 `GPU trace is empty!`；torch 2.9.1（/opt/venv）和 torch 2.14（vla 环境）表现相同
  2. 写最小 C 程序直接调用 CUPTI：采集后设置 `CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER` 会导致 buffer 丢失（Kineto 停止采集时正好会做这一步）→ 用 LD_PRELOAD 跳过后，记录能取回来，但被判为 Out-of-range
  3. 根因：**CUPTI 返回的 GPU 侧记录（kernel、memset）时间戳全是 0**，CPU 侧的 runtime 记录正常。宿主机（root、完整权限）同样如此；CUPTI 2025.3.1、2026.2.0、nsys 附带的 13.1 都一样；`cuptiActivityEnableHWTrace` 返回 NOT_SUPPORTED
  4. 结论：这是平台限制，torch.profiler 只能用 CPU 侧功能（算子、shape、调用栈、显存），GPU 侧改用 nsys（nsys 能正常拿到 kernel 时间）
  - 影响：00 章的 GPU busy 改用 NVML 采样；**03 章需要调整内容**（CPU 侧功能 + 内存 + 平台限制案例 + 用 emit_nvtx 配合 nsys）；HTA 依赖 GPU trace，只能做原理说明

### T03 第 01 章 系统监控（2026-11-05）✅
- 产出：`01_system_monitor/{README.md, exercises.md, 01~05 demo}`，`common/monitor.py`（JtopSampler/NvmlSampler），`outputs/load_patterns.png`
- 发现：jtop 数据有约 1 s 延迟；Thor 上 EMC load 和 `dmon mem` 始终为 0；NVML 的 memory/clock/throttle 字段 Not Supported；Thor 没有 DLA；tegrastats 的 `GR3D_FREQ` 没有负载百分比
- 发现：GPU load 区分不了 memory-bound 和 compute-bound（都约 98%），GPU 功耗可以（16 W vs 100 W）；队列满时 launch 调用会自旋占满一个 CPU 核
- 偏差：tegrastats 在宿主机上采集了 3 秒样例（只读）；05_power_modes 的 `--apply` 会修改宿主机全局功耗模式，**没有执行**，只验证了只读路径

### T04 第 02 章 Python profilers（2026-11-05）✅
- 产出：`02_python_profilers/{README.md, exercises.md, workload.py, 01_cprofile.py, 02_line_profiler.py, 03_py_spy.sh, 03_py_spy_report.py, 04_viztracer.py}`
- 实测开销（慢流水线，基准约 10.7 ms/step）：py-spy +5%、VizTracer（depth≤6+ignore_c）+5%、VizTracer 全量 +48%、cProfile +43%、line_profiler +60%
- 发现：VizTracer `include_files` 会剪掉不在列表中的函数的整棵子树（包括 @torch.no_grad 的 decorate_context）；1.1 版本没有 VizCounter，改用 add_counter
- 发现：py-spy 在容器中 attach 非子进程会报 Permission denied（ptrace_scope=1 且没有 CAP_SYS_PTRACE），用 prctl(PR_SET_PTRACER_ANY) 解决；`--native` 在 aarch64 上不支持；record 结束时偶发 `No child process`（产物正常）
- 发现：CUDA_LAUNCH_BLOCKING=1 能把 GPU 时间归回到发起它的函数上，但整体慢了 3 倍，只能定性使用

### T05 第 03 章 torch.profiler（2026-11-05）⚠️
- 产出：`03_torch_profiler/{README.md, exercises.md, 01~05 demo}`，`common/cupti_fix/`（LD_PRELOAD 拦截库 + ensure()）
- 偏差：Thor 上拿不到 GPU kernel 时间（见 T02 的根因记录），内容调整为 CPU 侧算子、runtime API、同步检测、显存、compile 诊断；TensorBoard/HTA 只做原理说明（实测 HTA 在 CPU-only trace 上报 IndexError）
- 发现：torch 2.9 中 with_stack 需要配合 `_ExperimentalConfig(verbose=True)`；with_flops 漏算融合算子（4.25 vs 5.59 GFLOPs）；profiler 让程序慢 3.3~4.9 倍，导致 sync 耗时被测成约 0（观察者效应）
- 发现：for 循环里的 `.item()` 会让 dynamo 放弃整个函数（0 个图、不报错），explain 返回 breaks=-1；`capture_scalar_outputs=True` 也救不回来，改成张量运算后编译快了 7.5 倍
- 发现：逐步增长的分配，默认分配器 reserved 7312 MB，expandable_segments 只要 700 MB；aarch64 上 snapshot 不支持 C++ 调用栈
- 实测：ToyVLA fp16 b=1：compile default 1.51x，reduce-overhead 2.96x；ManySmallOps 33x
- 检查点 2：用户要求连续完成第 00~04 章，合并到第 04 章完成后一起汇报

### T06 第 04 章 Nsight Systems（2026-11-05）⚠️
- 产出：`04_nsight_systems/{README.md, exercises.md, workload_nvtx.py, 01_nsys_basics.sh, 02_analyze_sqlite.py, 03_streams_overlap.py, 04_cuda_graph.py, 05_pytorch_python.sh, 06_op_gpu_time.py}`，`outputs/timeline.png`
- 实测：nsys 开销约 +32%（每次 API 调用多约 2 μs）；慢版 GPU busy 35.6%、快版 21.2%（快版仍然是 Launch-bound）；CUDA Graph：ManySmallOps 5.79x、ToyVLA b1 1.98x、memory-bound 1.00x
- 发现：fp32 卷积有大量 nchwToNhwcKernel（每步 17 次，占 6.9%），fp16 路径下消失
- 发现：Thor 上 H2D 拷贝和计算**严格交替、没有重叠**（asyncEngineCount=2，原因不明）；2-block 的长 kernel 用多 stream 可以快 6.68x，短 kernel 用多 stream 反而更慢
- 自己的 bug：计时写在 profiler.stop() 之后，误把 nsys 落盘的约 1 s 算成了开销（已修复，并写入文档作为踩坑案例）
- 新工具：`06_op_gpu_time.py` 用 `--pytorch=autograd-nvtx` + correlationId 得到每个 aten 算子的 GPU 时间，替代 torch.profiler 在 Thor 上缺失的 CUDA time 列
- 偏差：GPU metrics 不可用（计数器权限）；CPU 采样 / Python 采样不可用（容器内 `perf_event_open` 失败，需要 `--cap-add PERFMON` 或在宿主机上运行）

### 交接（2026-11-06）
- 用户在宿主机上新建了 privileged 容器 `brain_thor_yhy`（镜像 v1.3.0_thor），后续工作转到新容器中进行
- 交接文档：[../resume/README.md](../resume/README.md)（含：做不了的事及原因、新容器的环境准备步骤、下一步任务、踩坑汇总、新会话的开场白）
- 旧容器的记忆文件备份：`resume/claude_memory/`

### T00' 新容器环境检查（2026-11-06）✅
- 结果：版本基本相同（torch 2.9.1 / CUDA 13.0 / cuDNN 9.17 / TRT 10.14.1.48），**triton 3.5.0 → 3.8.0**，不设 TRITON_PTXAS_PATH 也能运行。详见 [env_report.md](env_report.md#t00-新容器-v130_thor-环境报告2026-11-06)
- 权限：CapEff=000001ffffffffff；nsys CPU 采样 OK、GPU metrics 列出 Thor、ncu 能采计数器 → **T07 解除阻塞**
- 安装：教程依赖已重装；packaging 回滚到 23.2，pytest 降到 8.4.2；**镜像里没有 jtop 客户端**，服务端是 4.5.2，PyPI 上没有该版本，从 GitHub 提交 11140a0 安装
- 回归：selftest slow 10.39 / fast 3.33 ms（旧 11.37 / 3.34）；带宽 239.6 GB/s（旧 226.4）；FP16 155.6、BF16 155.4 TFLOPS，持续满载 95.4（旧 104.1）。差异在波动范围内，第 00~04 章的数字不改
- 发现：ncu `--clock-control base` 在 Thor 上不生效（与 none 结果相同）；`dram__*` 指标为 n/a → 留给 T07 研究

### T07 第 05 章 Nsight Compute（2026-11-06）✅
- 产出：`05_nsight_compute/{README.md, exercises.md, cuda/bottlenecks.cu, kernels.py, ncu_tools.py, workload_basic.py, 01_ncu_basics.sh, 02~07 demo, 06_plot_roofline.py}`，`outputs/roofline_ncu.png`
- 偏差 1：没有用 `load_inline`，改为 nvcc → `.so` + ctypes（load_inline 实测首次编译 102 s，nvcc 直接编译约 5 s），README 中说明了原因
- 偏差 2：Register-bound 的结论与大纲的预期不同：ILP 高时 Occupancy 33% 已达 80% 带宽；`__maxnreg__(64)` spill 2.8 GB，慢 3 倍；提高 Occupancy（25%→83%）没有加速（0.99x），软件预取（Occupancy 降到 16.7%）快 1.62x。用 Little 定律解释，增加了 B 组"低 ILP"实验
- 发现（Thor）：没有 `dram__*` 指标；SOL Memory% 对满带宽的 kernel 只显示约 17%，并给出"延迟问题"的错误结论；用 `lts__d_sectors_fill_sysmem`（读）+ `min(写请求, 4×写 miss)`（写，估算）换算 DRAM 流量；L2 写请求会重复计数非合并写（转置 naive 8 倍）；`--clock-control base` 不生效
- 实测收益：合并访存 6.5x、融合 2.9x（ncu 单看每个 kernel 都满带宽）、warp divergence 1.96x（IPC 仍是 3.99、Branch Efficiency 99.4%，笔记中的判据失效；要看 Active Threads Per Warp）、排序 1.90x（但排序本身 2.25 ms 比节省的多）、bank conflict 修复 1.07x（合并写贡献 1.48x）、GEMM K=4095 时 cuBLAS 退回 Kernel2，慢 6.5x
- 真实 kernel：慢版 fp32 GEMM `Kernel2` 占 infer 的 47.7%（Occupancy 16.7%、Tail Effect）；快版 fp16 中逐元素/BN/LayerNorm 合计超过 45%，下一步是融合
- 遗留：`outputs/` 约 236 MB（07 的两个报告最大），T12 清理；第 02~04 章的权限回补（resume §4.2）尚未开始

### 回补第 02~04 章（resume §4.2，2026-11-06）✅
- 第 04 章：新增 `07_gpu_metrics.py` 和 4.8 节；`05_pytorch_python.sh` 补上 CPU 采样、Python 采样、`--cudabacktrace` 的实测；4.1、4.5、4.7、常见坑、小结、练习 4 已更新 → T06 改为 ✅
  - GPU metrics（gb10b，20 kHz）：慢版 infer 阶段 GR 88% → SMs 76% → Issue 29% → Tensor 2%；GPU 忙时只有约 2/3 的 SM 在工作；没有 DRAM 带宽指标；开启 GPU metrics 的开销可以忽略
  - **4.5 节"拷贝与计算不重叠"的原因已查明**：≥256 KB 的 cudaMemcpyAsync（H2D/D2H/D2D）由 SM 执行（Copy Engine 活动为 0，SMs Active 约 99.8%），和 GEMM 抢 SM；<256 KB 的拷贝走 Copy Engine，可以和 GEMM 并发，但带宽只有约 14~20 GB/s，总时间反而更长
  - 发现（观察者效应）：nsys 下慢版的 pageable H2D 拷贝 33 次中有 9 次需要 1~4 ms，不开 nsys 时没有；估计约占慢版 nsys 开销的三分之一
  - Python 采样能采集，但调用栈在 NVTX 二进制 payload 中，sqlite 导出不展开 → 命令行分析继续用 py-spy
  - `--cudabacktrace` 直接定位到 `.item()`（`_local_scalar_dense_cuda` → `cudaStreamSynchronize` 393 us）
- 第 02 章：privileged 下 `py-spy dump --pid` 可以直接 attach 非子进程（新增 `03_py_spy_attach.py`）
- 第 03 章：新增 `cupti_timestamp_test.cu`，privileged 下 GPU 时间戳仍全部为 0 → 结论不变，T05 保持 ⚠️
- 第 00 章：0.3 节的指标表加上了实测链接
- ⚠️ 安全：nsys 报告包含目标进程的完整环境变量（本环境中有云服务密钥和 token），已记录到 env_report.md，T12 时处理

### T08 第 06 章 TensorRT（2026-11-06）✅
- 产出：`06_tensorrt/{README.md, exercises.md, 01_export_onnx.py, 02_trtexec.py, 03_layer_profile.py, 04_three_steps.py, 05_compare.py}`
- 偏差：新镜像中没有 onnxscript / polygraphy / onnxruntime / torch2trt / modelopt → ONNX 用 `dynamo=False` 导出；精度直接与 PyTorch 输出比较；INT8 未校准（只看性能）；6.6 节的 Polygraphy、Plugin、torch2trt 只做说明
- 偏差：nsys 2025.3 没有 `-t tensorrt`（大纲与笔记过时），TensorRT 自己发 NVTX（domain `TensorRT` + `myelin-exec`）；笔记中的 `--workspace` 在 TRT 10 中已移除
- 发现：TRT 10.14 在 Thor 上把整张图交给 Myelin（一个 ForeignNode）；没有独立的 Reformat 层，由 `__myl_CastConc`（fp32→fp16 + 3 通道 pad 到 4）代替；逐层 profile 比真实时间多 7~34%，且看不到层间间隙
- 发现：ncu 的 NVTX 过滤与层名中的 `/` 冲突 → `regex:myelin-exec@^<层名，/ 换成 .>$/`
- 三步法：TinyCNN 的 stem 卷积占 27.6%，间隙只有 7% → ncu：tensor pipe 75% 但只发挥峰值的约 17%（3 通道、旧 mma 路径，tc pipe 0）；ToyVLA 没有热点层，nsys 发现 kernel 间隙占 27%、enqueue 411 us ≈ GPU 跨度 → Launch-bound，CUDA Graph 1.39x
- 对比（ToyVLA b=1）：eager fp32 3.555 ms → compile reduce-overhead 0.758 → TRT fp16 + CUDA Graph 0.345 ms（10.3x，误差 < 1e-3）；TinyCNN b=8：TRT fp16 + graph 6.60x
- 遗留：`06_tensorrt/outputs/` 约 356 MB（engine、nsys/ncu 报告），T12 清理；nsys 报告中有环境变量（见 env_report.md）

### T09 第 07 章 优化手段（2026-11-06）✅
- 产出：`07_optimization/{README.md, exercises.md, 01_cpu_side.py, 02_sync_launch.py, 03_memory_bound.py, 04_alloc_dynamic.py, 05_llm_decode.py}`；README 7.0 是"瓶颈 → 手段 → 实测收益 → 无效/负优化"总表，前面各章已测的直接引用
- 实测：预处理 numpy 2.13x / GPU 4.49x / 全 GPU 11.87x（瓶颈转移到 cv2.resize）；DataLoader 8 worker 11x，pin_memory +0~3%；同步推迟 1.10x；ManySmallOps CUDA Graph 6.38x、compile 33.8x；Memory-bound 上 Graph 1.02x、compile 16.5x；GEMM 上都是 1.00x；fp16 GELU 2.07x；channels_last 1.35x/1.56x；SDPA L=4096 12.9x（显存 516→4 MB）；默认 allocator reserved 14.6 GB vs expandable 1.38 GB
- 发现（负优化）：小 MLP 上 compile 稳态比 eager 慢 1.9x；已融合时 reduce-overhead 比 default 慢；分桶让稳态变慢；inductor 磁盘缓存让编译时间差 4~10 倍（冷/热）
- LLM decode（101.8M 参数 fp16）：prefill 24.7 TFLOPS；decode 2.0 ms/token，只达带宽下限（0.848 ms）的 42%（GEMV 效率，与第 00 章 GEMV 48% 吻合）；KV cache prompt 128 时 1.28x、512 时 2.14x；batch 64 吞吐 10.8x；decode CUDA Graph 只有 1.11x
- 未实测：GPU 解码（NVJPG/DALI）、TRT optimization profile、投机解码、权重量化 kernel（留作练习）

### T10 第 08 章 综合实战（2026-11-06）✅
- 产出：`08_case_study/{README.md, exercises.md, case.py, 01_diagnose.py, 02_optimize.py, 03_accuracy.py, report_template.md, REPORT.md}`，`outputs/stages.png`
- 诊断（L0~L3 实测，L4 引用第 05 章）决定修复顺序：后处理 → 预处理 → 同步 → fp16 →（pinned/预分配，反例）→ channels_last → compile → TensorRT → GPU resize
- 结果：p50 10.321 → 0.747 ms（13.8x），p99 10.846 → 1.210 ms，能耗 353 → 32 mJ/step，MFU 0.35% → 4.81%，最大误差 9.1e-4
- 偏差 / 自己的错误：第一次算最终 GPU busy 得到 6.1%（nsys 没加 --cuda-graph-trace=node），修正后 37.5%；S5 首次运行 0.94x，重复测量确认为 1.00x
- 发现：S6~S8 的尾延迟来自 147 KB 的 H2D 拷贝（走 Copy Engine），单独测试 147 KB 拷贝 p99 236 us vs 256 KB 46 us；OpenCV 线程数无关（已排除）；机制为推测
- 未实测：整步 CUDA Graph、两帧流水、真实 VLA（留作练习）；S7 中预处理变慢的原因未深究

### T11 附录（2026-11-06）✅
- 产出：`appendix/README.md`：A 命令速查（按 L0~L4 + TensorRT + 环境变量）、B 工具对比（开销均为本机实测，注明出处）、C 采集/查看流程、D 38 条常见坑（按主题汇总各章）、E 参考资料（官方文档 + doc/ 脑图中的 12 个链接及所在主题）

### T12 收尾（2026-11-06）✅
- 清理（按用户指示"全部删除"）：项目内 57 个 `.nsys-rep` / `.sqlite` / `.ncu-rep` / `.engine` / `.onnx`，2 个 > 5 MB 的 trace json（viztracer_slow、basics_trace），`05_nsight_compute/build/`，`__pycache__`；`/tmp` 下本会话创建的临时目录（含 nsys 报告和导出的 JSON）。复查：项目内无凭据字段，系统中（/opt、/usr 之外）无 nsys 报告。项目从约 1 GB 降到 46 MB
- 冒烟测试：第 00~04 章 25 个 demo 在新容器中运行（运行前备份 outputs/，运行后恢复，避免覆盖文档引用的日志）：24 个直接通过；`04/06_op_gpu_time.py` 依赖先用 nsys 采集的 sqlite（被删除后报 no such table），已加文件存在检查和提示，按 docstring 流程运行通过。第 05~08 章的 demo 本会话中都已运行
- 文档：顶层 README 目录链接到全部章节，说明了产物清理与重新生成方法；第 03、08 章补上"📎 对应笔记"；spec Q1/Q4/Q5 更新；全部 .md 的相对链接检查通过；无残留的过时状态描述
- DoD 核对（spec §7）：

| 章 | FR-1 结构 | demo 运行 | 前后对比 | 练习 | 限制说明 | tasks 记录 |
|---|---|---|---|---|---|---|
| 00 理论 | ✅ | ✅ | ✅ | 5 | ✅ | ✅ |
| 01 系统监控 | ✅ | ✅ | ✅ | 4 | ✅（tegrastats 只能在宿主机；`--apply` 未执行） | ✅ |
| 02 Python | ✅ | ✅ | ✅ | 5 | ✅ | ✅ |
| 03 torch.profiler | ✅ | ✅ | ✅ | 5 | ✅（GPU 时间不可用，privileged 下复测） | ⚠️ 平台限制 |
| 04 nsys | ✅ | ✅ | ✅ | 5 | ✅ | ✅ |
| 05 ncu | ✅ | ✅ | ✅ | **6**（比 FR-7 上限多 1） | ✅ | ✅ |
| 06 TensorRT | ✅ | ✅ | ✅ | 5 | ✅（INT8 未校准、Polygraphy 等缺失） | ✅ |
| 07 优化 | ✅ | ✅ | ✅ | 5 | ✅ | ✅ |
| 08 综合 | ✅（主体在 REPORT.md） | ✅ | ✅ | 4 | ✅ | ✅ |
| 附录 | A~E 齐全 | — | — | — | — | ✅ |

- 已知遗留：01 章 `05_power_modes.py --apply` 从未执行（会修改宿主机功耗模式，需用户同意）；各章"未实测"的练习参考思路已标注；第 08 章 S7 中预处理变慢、147 KB 拷贝尾延迟的机制未深究

### 补充：第 01 章 `05_power_modes.py --apply`（2026-11-06，经用户同意后运行）
- 结果：120W 与 MAXN 几乎无差别（2.169 vs 2.214 ms，ToyVLA b=1 是 Launch-bound）；90W/70W **没有切换成功**（jtop 仍显示 120W，事后 jtop 报告这两个模式需要重启），数据作废
- 问题：切换 nvpmodel 关掉了 jetson_clocks，脚本只恢复了 nvpmodel；jtop 的 `set_enable(True)` 对宿主机命令开启的 jetson_clocks（`uncontrolled`）无效 → 用户在宿主机上执行 `sudo jetson_clocks` 恢复；`jetson_clocks --show` 确认 CPU 2601 / GPU 1575 / EMC 4266 MHz 均 min = max，`/var/lib/nvpmodel/status` 为 MAXN，无待生效变更；恢复后 p50 2.231 ms（运行前 2.216）
- 修改：脚本保存并尝试恢复 jetson_clocks、打印恢复前后频率、跳过需要重启的模式；第 01 章 1.6、常见坑 7、附录 D.29 已更新
