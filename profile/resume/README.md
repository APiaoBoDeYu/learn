# 交接文档：在新容器中继续"模型性能分析与调优教程"

> 写给：在新容器（`brain_thor_yhy`，`--privileged`）中接手的 Claude Code 新会话。
> 写于：2026-11-06，旧容器 `brain`（镜像 `brain-service:v1.0.0_thor`，非 privileged）。
> 新会话开始时，请先完整读完本文，再读 [../plan/tasks.md](../plan/tasks.md) 的实测记录。

---

## 0. 用户偏好（必须遵守）

- **全程用简体中文回复**；代码、命令、报错保持原样。旧容器的记忆文件备份在 [claude_memory/](claude_memory/)，请写入新会话的 memory 目录（`/root/.claude/projects/-root-data-learn/memory/`，内容就是"所有回复用简体中文"）。
- 使用 `/opt/venv/bin/python`；**可以装包，但必须记录**（`requirements.txt` + `plan/env_report.md`）。
- 采用**面向规格的开发**：按 `plan/spec.md → plan.md → tasks.md` 执行，每完成一个任务就在 `tasks.md` 中更新状态并追加实测记录。
- **文档中的数字和输出必须来自本机实测**，不能编造；推测的内容要标明"预期/未实测"。工具不支持就如实写明。
- 会修改宿主机全局状态的操作（例如切换 nvpmodel 功耗模式）要**先征得用户同意**。
- 宿主机 SSH：`ssh root@10.208.40.33`（密码由用户提供，**不要写进教程文档**）。新容器有了 privileged 权限后，原则上不再需要登录宿主机。

## 1. 当前进度

| 任务 | 内容 | 状态 |
|---|---|---|
| T00 / T00' | 环境检查 | ✅ 新容器已完成（`plan/env_report.md` 新增一节） |
| T01–T04 | common、第 00~02 章 | ✅（第 02 章已补 privileged 下的 py-spy） |
| T05 | 第 03 章 torch.profiler | ⚠️ CUPTI GPU 时间戳在 privileged 下仍为 0（平台限制，已复测） |
| T06 | 第 04 章 Nsight Systems | ✅ 已补 GPU metrics、CPU 采样、拷贝由谁执行（4.8） |
| T07 | 第 05 章 Nsight Compute | ✅ |
| T08 | 第 06 章 TensorRT | ✅ |
| T09 | 第 07 章 优化手段 | ✅ |
| T10 | 第 08 章 综合实战 | ✅（检查点 4 已汇报，2026-11-06） |
| T11–T12 | 附录、收尾 | ✅ 全部完成（大产物和 nsys 报告已删除，重新运行脚本即可生成） |

> 更新于 2026-11-06（新容器会话）。各任务的实测记录见 `plan/tasks.md`。⚠️ nsys 报告包含环境变量中的凭据，见 `plan/env_report.md` 末尾。

目录：`/root/data/learn/profile/`（宿主机上是 `/data/brain/brain1/learn/profile`，新旧容器共享）。
- 顶层 [README.md](../README.md)：教程目录 + 原始大纲（第 05 章以后的大纲在这里）
- [common/](../common/)：`bench_utils`（benchmark/compare/region）、`models`（TinyCNN/TinyTransformer/ManySmallOps/ToyVLA）、`pipeline`（带 6 个低效开关的慢流水线）、`monitor`（JtopSampler/NvmlSampler）、`cupti_fix/`、`selftest.py`
- 每章：`README.md` + 编号 demo + `exercises.md` + `outputs/`（日志、图、trace）

## 2. 旧容器中做不了的事（原因）

| 事项 | 原因 | 新容器预期 |
|---|---|---|
| `ncu` 读取 GPU 计数器 | 驱动 `RmProfilingAdminOnly=1`，旧容器没有 `CAP_SYS_ADMIN`，报 `ERR_NVGPUCTRPERM` | ✅ privileged 后应可用 |
| nsys `--gpu-metrics-devices`（SM Active / Tensor Active / DRAM 带宽曲线） | 同上，报 `Insufficient privilege` | ✅ 应可用 |
| nsys CPU 采样、`--python-sampling`、`--cudabacktrace` 的调用栈 | 容器内 `perf_event_open` 失败（`nsys status -e`） | ✅ 应可用 |
| `py-spy dump/top --pid` 附加到非子进程 | `ptrace_scope=1` 且没有 `CAP_SYS_PTRACE`（当时用 `prctl(PR_SET_PTRACER_ANY)` 绕过） | ✅ 应可直接使用 |
| torch.profiler 的 GPU kernel 时间 | **CUPTI 返回的 GPU 侧记录时间戳为 0**，宿主机 root 上也一样 → 平台限制，**权限解决不了** | ❌ 大概率仍不可用，但请复测一次（§4.2） |
| HTA / TensorBoard 插件 | 依赖 GPU trace | ❌ 同上 |
| tegrastats / nvpmodel / jetson_clocks 命令 | 容器内没有这些命令 | 用 `-v /dev:/dev` 和 privileged 后也许能从宿主机路径调用，不行就继续用 jtop API |
| 切换功耗模式（`05_power_modes.py --apply`） | 会修改宿主机全局状态 | ✅ 2026-11-06 经用户同意已运行；切换会关掉 jetson_clocks，需在宿主机上 `sudo jetson_clocks` 恢复（见第 01 章 1.6）。再次运行仍需先问用户 |
| `py-spy --native` | aarch64 不支持 | ❌ |
| 拷贝/计算重叠 | Thor 上实测 H2D 拷贝与 kernel 严格交替（asyncEngineCount=2，原因不明） | 可以用 ncu/GPU metrics 进一步分析（第 04 章练习 4） |

## 3. 新容器的环境准备（T00'）

新镜像是 **`v1.3.0_thor`**（旧容器是 `v1.0.0_thor`），旧容器里 pip 装的包**不会**带过来，软件版本也可能不同。

```bash
cd /root/data/learn/profile
# 1) 版本检查：与旧环境对比（旧：torch 2.9.1 / tensorrt 10.14.1.48 / triton 3.5.0 / CUDA 13.0 / cuDNN 9.17）
/opt/venv/bin/python -c "import torch, tensorrt, triton; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), tensorrt.__version__, triton.__version__)"
# 2) 重装教程依赖（见 requirements.txt）
/opt/venv/bin/python -m pip install viztracer==1.1.1 line_profiler==5.1.0 snakeviz==2.2.2 nvtx==0.2.17 HolisticTraceAnalysis==0.5.0
/opt/venv/bin/python -m pip install "packaging<24"     # HTA 会把 packaging 升到 26.x，与 streamlit/mlflow 冲突；先确认新镜像里是否也有这两个包
# 3) 权限验证
nsys status -e                              # 期望：perf_event_open syscall available: OK
nsys profile --gpu-metrics-devices=help     # 期望：列出 Thor，不再出现 Insufficient privilege
cat /proc/driver/nvidia/params | grep Profil; grep CapEff /proc/self/status   # CapEff 应为 000001ffffffffff
ls -la /run/jtop.sock                       # jtop 服务的 socket（启动容器时需挂载）
# 4) 自测
/opt/venv/bin/python common/selftest.py     # 对比 00_theory/README.md 0.1 节的表（slow 约 11.4 ms，fast 约 3.3 ms）
```

注意事项：
- **Triton / torch.compile** 需要 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`（`common/__init__.py` 已自动设置）。新镜像里请确认是否仍然需要。
- **`common/cupti_fix/libcupti_fix.so`** 是在旧容器中编译的，会在 `cupti_fix.ensure()` 时按需重新编译（需要 gcc）。
- 如果新镜像的 torch/CUDA 版本不同，第 00~04 章的数字可能会变化：至少重跑 `common/selftest.py` 和 `00_theory/02_roofline.py`，**有明显差异时在 tasks.md 中记录**，必要时更新文档里的数字。
- 把版本、权限、安装记录写进 `plan/env_report.md`（新增一节"新容器 v1.3.0_thor"），并更新 `spec.md §8` 中的 Open Questions（Q2/Q3/Q6）。

## 4. 下一步要做的事（按顺序）

### 4.1 T07：第 05 章 Nsight Compute（首要任务）

计划见 [../plan/plan.md](../plan/plan.md) §3 和顶层 README 的大纲第 5 章。要点：
- 先用最小用例验证：`ncu --section SpeedOfLight /opt/venv/bin/python -c "import torch; x=torch.randn(1<<20,device='cuda'); (x*2).sum(); torch.cuda.synchronize()"`
- 用 Triton，或者 `torch.utils.cpp_extension.load_inline` 写 CUDA kernel，分别构造 4 类瓶颈：**Memory-bound**（逐元素）、**Register-bound**（大量局部变量，配合 `-maxrregcount` 对比）、**Control-flow-bound**（warp divergence）、**shared memory bank conflict**（加 padding 修复），每类给出修复前后的对比。第 04 章 B2 的 Triton `spin_kernel` 可以复用。
- ncu 必须加过滤：`-k regex:...`、`--launch-skip`、`--launch-count 1`、`--nvtx --nvtx-include`、指定 `--section`；锁频选项 `--clock-control`（base/none，说明它对结果的影响）。
- 报告解读：SOL、Roofline（和第 00 章的实测 Roofline 对照）、Occupancy、Memory Workload（L1/L2 命中率、bank conflict）、Warp Stall、Source 页面。
- 引用 📎 `doc/CUDA.xmind` 中的"四大瓶颈""bank""Warp divergence"等内容。
- 也可以分析慢流水线中的真实 kernel：fp32 下的 `nchwToNhwcKernel`、`Kernel2`（fp32 GEMM），fp16 下的 `nvjet_*`、`flash_fw`。

### 4.2 回补第 02~04 章中因权限缺失的内容

- **第 04 章 4.7**：GPU metrics（`--gpu-metrics-devices=all --gpu-metrics-set=?`）对慢/快流水线各采一次，给出 SM Active、Tensor Active、DRAM 带宽曲线，并与第 00 章 0.3 节的"空间维度利用率"呼应；补上 CPU 采样和 `--python-sampling` 的实测（`05_pytorch_python.sh` 中已有对应命令）。
- **第 04 章 4.5**：用 GPU metrics 或 ncu 进一步分析"拷贝与计算不重叠"的现象（练习 4 的实验）。
- **第 02 章 2.4**：说明 privileged 容器中 `py-spy dump --pid` 可以直接使用（保留 `prctl` 方案，作为非特权环境的替代）。
- **第 03 章 3.8**：在新容器中复测 CUPTI 的 GPU 时间戳（C 程序 `t6.cu` 的思路：打印 `CUpti_ActivityKernel10` 的 start/end，旧容器和宿主机上都是 0）。若仍为 0，保持现有结论，补一句"privileged 下也一样"。
- 更新 `plan/env_report.md` 的阻塞项状态，以及顶层 README 目录表中的状态。

### 4.3 T08 及之后

| 任务 | 内容 | 提示 |
|---|---|---|
| T08 第 06 章 TensorRT | ONNX 导出（TinyCNN/ToyVLA）→ trtexec FP32/FP16/INT8 → `--dumpProfile --separateProfileRun --exportLayerInfo` → nsys（`-t cuda,nvtx,tensorrt`）→ ncu，完整走一遍**三步法** | 📎 `doc/TensorRT_副本.xmind` 中有三步法的详细笔记；TensorRT 10.14 的 Python 包和 `/usr/bin/trtexec` 都可用；Thor **没有 DLA** |
| T09 第 07 章 优化手段 | 按大纲中"瓶颈 → 手段"的对照表，每项给出最小的前后对比 | 已有现成素材：fp16（6.9x GEMM）、compile/reduce-overhead（2.96x）、CUDA Graph（1.98x）、expandable_segments、去掉 `.item()`、channels_last |
| T10 第 08 章 综合实战 | 慢流水线 `PipelineConfig.slow()` → 逐个定位、逐个修复，记录每一步的收益，产出 `report_template.md` 和实测报告 | `common/selftest.py` 中每个开关的影响就是"预期答案"；最终可以叠加 compile/Graph/TRT |
| T11 附录 | 命令速查、工具对比、常见坑（汇总各章 README 中的"常见坑"）、可视化流程 | |
| T12 收尾 | 顶层 README 目录、链接检查、清理大文件（各章 `outputs/` 合计约 76 MB）、DoD 核对 | |

检查点：第 05 章完成后、第 08 章完成后向用户汇报。

## 5. 已经踩过的坑（避免重复）

- `nsys --capture-range-end` 默认是 `stop-shutdown`，会结束程序，要用 `stop`；计时必须在 `torch.cuda.profiler.stop()` **之前**完成（stop 时落盘约 1 s）。
- `nsys stats` 会复用旧的 sqlite，要加 `--force-export=true`；`gpu_gaps` 默认阈值是 500 ms。
- `--pytorch=autograd-nvtx` 下，`record_function` 会额外生成带 `op_id` 的同名区间，统计时要去重。
- Benchmark 必须刷 L2（32 MB），否则带宽会虚高；持续满载会撞功耗墙（FP16 从约 157 降到约 104 TFLOPS，jtop 显示的频率不变）。
- 容器中没有中文字体，matplotlib 图表一律用英文标注；画图前遵循 dataviz skill（先跑配色校验，再检查标签重叠）。
- `FlopCounterMode` 不认识 TransformerEncoderLayer 的 fastpath，统计前要用 `torch.backends.mha.set_fastpath_enabled(False)`，并在 GPU 上统计。
- VizTracer 的 `include_files` 会剪掉整棵子树；torch 2.9 中 `with_stack` 需要 `_ExperimentalConfig(verbose=True)`。
- CUDA Graph capture 失败后 CUDA context 会损坏，失败用例要放在子进程中运行。
- `.item()` 在 for 循环中会让 dynamo 放弃整个函数（0 个图，不报错）。

## 6. 在新会话中如何开始

在新容器中进入 `/root/data/learn`，启动 Claude Code，第一句话可以这样说：

> 请先阅读 /root/data/learn/profile/resume/README.md 和 /root/data/learn/profile/plan/tasks.md，按其中第 3 节完成新容器的环境检查（T00'），然后继续第 05 章（T07）。全程用中文。
