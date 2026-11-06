# T00 环境报告（2026-11-05）

## 硬件与系统

| 项 | 值 |
|---|---|
| 板卡 | NVIDIA Jetson AGX Thor Developer Kit（SoC tegra264） |
| JetPack / L4T | 7.2 / 38.4.0 |
| GPU | NVIDIA Thor（Blackwell GB10B），CC 11.0（sm_110），**20 个 SM** |
| GPU 规格 | 每 SM 最多 1536 线程、65536 个寄存器、228 KB shared memory；L2 32 MB；warp 32 |
| 内存 | 126 GB 统一内存（CPU/GPU 共享），EMC 最高 4266 MHz |
| CPU | 14 核 |
| 功耗模式 | `nvpmodel = MAXN`，`jetson_clocks = True`，GPU governor `performance`，频率**锁定在 1575 MHz** |
| 磁盘 | /root/data 剩余 1.1 TB |

> 结论（Q6）：容器内没有 `nvpmodel`/`jetson_clocks`/`tegrastats` 命令，但宿主机已经是 MAXN + jetson_clocks，GPU 频率固定，benchmark 结果可复现。频率状态可通过 jtop API 读取（jtop 服务的 socket 已挂进容器，root 在 `jtop` 组内）。

## 软件

| 项 | 版本 |
|---|---|
| Python | `/opt/venv/bin/python` 3.12 |
| torch / torchvision | 2.9.1 / 0.24.1（CUDA 13.0，cuDNN 9.17） |
| triton | 3.5.0（⚠️ 需要设置 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`，否则报 `Cannot find ptxas`；torch.compile 同样受影响） |
| TensorRT | 10.14.1（Python 包 + `/usr/bin/trtexec`） |
| onnx / onnxscript | 1.20.1 / 0.7.2 |
| jtop | 4.5.2 |
| pynvml | 可导入 |
| nsys / ncu | 2025.3.2 / 2025.3.1；`nsys-ui`、`ncu-ui` 也在（容器内没有显示器，需在主机上打开报告） |
| py-spy | /opt/venv/bin/py-spy |
| 其他 | numpy 1.26.3、pandas 2.1.4、matplotlib 3.8.2、opencv 4.5.5、torch2trt |

## 本次安装的包（Q1：pip 可联网 ✅）

```bash
/opt/venv/bin/python -m pip install viztracer line_profiler snakeviz nvtx HolisticTraceAnalysis
/opt/venv/bin/python -m pip install "packaging<24"     # 回滚，见下
```

| 包 | 版本 | 用途 |
|---|---|---|
| viztracer | 1.1.1 | 第 02 章 |
| line_profiler | 5.1.0 | 第 02 章 |
| snakeviz | 2.2.2 | 第 02 章（cProfile 可视化） |
| nvtx | 0.2.17 | 第 04 章 |
| HolisticTraceAnalysis | 0.5.0 | 第 03 章 |

连带安装的依赖：jupyterlab 4.6.5 一整套、plotly 7.1.0、pytest 9.2.0、pydot、objprint 等。

⚠️ **副作用及处理**：HTA 把 `packaging` 从原来的版本升级到 26.3，和已装的 `streamlit 1.31.0`、`mlflow 2.10.2`（要求 `packaging<24`）冲突，已回滚到 `packaging==23.2`。回滚后，新装的 `pytest 9.2.0` 和原有的 `twine` 会提示需要 `packaging>=24`，只影响这两个工具，HTA 能正常导入使用。
另外，`pip check` 里有一些冲突是安装前就有的，与本次无关：torch-tensorrt 与 tensorrt 版本、open3d 与 flask、cmeel-boost 与 numpy 等。

## 工具可用性实测

| 功能 | 结果 | 说明 |
|---|---|---|
| nsys 基础 trace（cuda/nvtx/osrt）+ `nsys stats` | ✅ | |
| nsys `--gpu-metrics-devices` | ❌ | `Insufficient privilege`（Q2） |
| ncu 硬件计数器 | ❌ | `ERR_NVGPUCTRPERM`（Q3） |
| trtexec | ✅ | TensorRT v101401 |
| triton / torch.compile | ✅ | 需要设置 TRITON_PTXAS_PATH |
| jtop Python API | ✅ | 可以读取 CPU/GPU/EMC/温度/功耗/nvpmodel |
| tegrastats | ❌ | 容器内没有（Q4），用 jtop API 替代 |
| nvidia-smi dmon | ✅ | |

### 🚫 阻塞项：GPU 性能计数器权限

- 现象：`ncu` 报 `ERR_NVGPUCTRPERM`；`nsys --gpu-metrics-devices` 报 `Insufficient privilege`
- 原因：驱动参数 `RmProfilingAdminOnly: 1`（只允许管理员使用 profiling），而当前容器的进程没有 `CAP_SYS_ADMIN`（CapEff=`a80425fb`，即 docker 默认的 capability 集合）
- 影响：第 05 章（ncu）全部内容；第 04 章的 GPU metrics 小节；第 06 章三步法的第三步
- 解决方案（任选其一，需要在宿主机上操作）：
  1. **推荐**：重启容器时加上 `--cap-add SYS_ADMIN`（或 `--privileged`）
  2. 在宿主机上关闭限制：`echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-prof.conf`，然后重启（Jetson 上的模块名和生效方式需要按实际情况确认）
- 在此之前：先完成不依赖计数器的章节（00–04、06 的前两步），第 05 章放到权限解决之后再做

---

# T00' 新容器 v1.3.0_thor 环境报告（2026-11-06）

容器 `brain_thor_yhy`，镜像 `brain-service:v1.3.0_thor`，`--privileged`。硬件、JetPack、功耗模式与上文相同（jtop 实测：`nvpmodel=MAXN`、`jetson_clocks=True`、GPU 1575 MHz、governor `performance`）。

## 软件版本对比

| 项 | 旧容器 v1.0.0 | 新容器 v1.3.0 |
|---|---|---|
| torch / CUDA / cuDNN | 2.9.1 / 13.0 / 9.17 | 2.9.1 / 13.0 / 9.17（相同） |
| TensorRT | 10.14.1.48 | 10.14.1.48（相同） |
| **triton** | 3.5.0 | **3.8.0**；不设 `TRITON_PTXAS_PATH` 也能编译运行（自带 `ptxas-blackwell`）。`common/__init__.py` 中的设置保留，无副作用 |
| nsys / ncu | 2025.3.2 / 2025.3.1 | 相同 |
| 驱动 | 580.00 | 580.00 |
| **jtop 客户端** | 4.5.2（镜像自带） | **镜像中没有**，需要自己安装，见下 |
| py-spy | 0.4.1 | 0.4.1 |
| packaging | 23.2（回滚后） | 镜像原有 23.0 |

## 本次安装记录

```bash
/opt/venv/bin/python -m pip install viztracer==1.1.1 line_profiler==5.1.0 snakeviz==2.2.2 nvtx==0.2.17 HolisticTraceAnalysis==0.5.0
/opt/venv/bin/python -m pip install "packaging<24"   # HTA 把 packaging 升到 26.3，与 streamlit 1.31 / mlflow 2.10 冲突 → 回滚到 23.2
/opt/venv/bin/python -m pip install "pytest<9"       # HTA 连带装的 pytest 9.2 要求 packaging>=24 → 降到 8.4.2
# jtop 客户端必须与宿主机 jtop 服务版本一致（服务端是 4.5.2；装错版本时报
#   "Mismatch version jtop service: [4.5.2] and client: [4.3.2]"）。
# PyPI 上没有 4.5.2（4.x 只到 4.3.2，然后直接是 7.x），GitHub 上也没有这个 tag，
# 只能从提交 11140a0（"v4.5.2"）安装：
git clone https://github.com/rbonghi/jetson_stats.git /tmp/jstats && git -C /tmp/jstats checkout 11140a0
/opt/venv/bin/python -m pip install /tmp/jstats      # jetson-stats 4.5.2（中途试装过 PyPI 的 4.3.2，已被覆盖）
```

`pip check` 中剩余的冲突（twine 要求 packaging>=24、open3d/flask、cmeel-boost/numpy、pyorbbecsdk2 等）在安装前就已存在，与本教程无关。

## 权限实测（旧容器的阻塞项全部解除）

| 检查 | 结果 |
|---|---|
| `grep CapEff /proc/self/status` | `000001ffffffffff`（全部 capability） |
| `RmProfilingAdminOnly` | 仍是 1，但进程有 CAP_SYS_ADMIN，不受影响 |
| `nsys status -e` | `perf_event_open syscall available: OK`，process-tree / system-wide CPU 采样均 OK（`perf_event_paranoid=2`） |
| `nsys profile --gpu-metrics-devices=help` | 列出 `0: Blackwell GB10B \| NVIDIA Thor`，不再报 Insufficient privilege |
| `ncu --section SpeedOfLight`（最小用例） | ✅ 3 个 kernel 都采到了，每个 10 个 pass。⚠️ 每个 kernel 都提示 `Data collection happened without fixed GPU frequencies`（`--clock-control` 行为见第 05 章实测） |
| `/run/jtop.sock` | 已挂载，jtop API 正常 |
| `ptrace_scope` | 仍是 1，但有 CAP_SYS_PTRACE |
| tegrastats / nvpmodel / jetson_clocks 命令 | 容器内仍然没有，继续用 jtop API |
| 宿主机 PID 命名空间 | 隔离（`ps` 看不到宿主机进程） |

## 回归测试（与旧容器对比）

| 项 | 旧容器 | 新容器 | 结论 |
|---|---|---|---|
| `common/selftest.py` slow / fast | 11.37 / 3.34 ms（3.40x） | 10.39 / 3.33 ms（3.12x） | 一致；排序 eager_small_ops > py_preprocess > sync_in_loop 不变；fp32 开关 1.21x → 1.07x，no_pin/realloc 仍约 1.0x |
| 峰值带宽（copy 256MB） | 226.4 GB/s | 239.6 GB/s | +6% |
| FP32 / TF32 / FP16 / BF16 matmul | 6.39 / 65.3 / 156.8 / 162.5 TFLOPS | 6.39 / 65.9 / 155.6 / 155.4 TFLOPS | BF16 −4%，其余几乎相同 |
| FP16 持续满载 5 s | 104.1 TFLOPS | 95.4 TFLOPS | 功耗墙，−8% |

结论：差异都在运行间波动的范围内，**第 00~04 章的数字不需要修改**。（新的 roofline 结果没有覆盖 `00_theory/outputs/`，因为文档引用的是那份。）

## 回补第 02~04 章后的工具可用性（新容器，2026-11-06）

| 功能 | 旧容器 | 新容器 | 说明 |
|---|---|---|---|
| ncu 硬件计数器 | ❌ | ✅ | 第 05 章；Thor 上没有 `dram__*` 指标，`--clock-control` 不生效 |
| nsys GPU metrics | ❌ | ✅ | `gb10b` 指标集，没有 DRAM 带宽；第 04 章 4.8 |
| nsys CPU 采样 | ❌ | ✅ | `COMPOSITE_EVENTS` + `SAMPLING_CALLCHAINS` |
| nsys `--python-sampling` | ❌ | ⚠️ 能采集，**调用栈只能在 GUI 中看**（sqlite 导出中 NVTX payload 的 `binaryData` 为空） | 第 04 章 4.7 |
| nsys `--cudabacktrace` | ❌（没有栈） | ✅ | `CUDA_CALLCHAINS`，能直接定位到 `.item()` |
| `py-spy dump --pid` 非子进程 | ❌（需要 prctl） | ✅ | 有 CAP_SYS_PTRACE；第 02 章 2.4 |
| torch.profiler GPU kernel 时间 | ❌ | ❌ | CUPTI GPU 时间戳仍全部为 0（`03_torch_profiler/cupti_timestamp_test.cu`），**权限无关，平台问题** |
| tegrastats / nvpmodel / jetson_clocks 命令 | ❌ | ❌ | 继续用 jtop API |

### ⚠️ 报告文件中的敏感信息

- **nsys**：`.nsys-rep` 和导出的 `.sqlite`（`TARGET_INFO_SYSTEM_ENV` 表）包含目标进程的**完整环境变量**。本容器的环境中有云服务的访问密钥和 token，所以各章 `outputs/` 下的 nsys 报告都含有这些凭据（已用 grep 确认字段存在，未在任何文档中写出具体的值）。
- **ncu**：`--page session` 中没有环境变量，但记录了主机名和所有网卡的 IP 地址。
- 处理：`outputs/` 已在 `.gitignore` 中；对外分享报告之前，用 `env -i PATH=$PATH HOME=$HOME python ...` 在干净的环境中重新采集；T12 收尾时决定是否删除这些报告。
