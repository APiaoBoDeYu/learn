# 附录

> 汇总第 00~08 章的实测结论。数字均来自 Jetson AGX Thor（MAXN + jetson_clocks，privileged 容器 v1.3.0_thor），出处标在括号中。

- [A. 工具速查表](#a-工具速查表)
- [B. 工具对比](#b-工具对比)
- [C. 可视化：在 Jetson 上采集、在主机上查看](#c-可视化在-jetson-上采集在主机上查看)
- [D. 常见坑（按主题汇总）](#d-常见坑按主题汇总)
- [E. 参考资料](#e-参考资料)

---

## A. 工具速查表

### 计时（第 00 章）

```python
from common import benchmark, compare
s = benchmark(fn, mode="event", warmup=10, iters=50)   # GPU 时间：CUDA Event；mode="wall" 测端到端（前后都 synchronize）
compare([s1, s2])                                      # p50/p90/p99 + 加速比
# 规则：先 warmup；计时前后同步；看分布（p50/p99）；> 32 MB 的数据才测得到 DRAM（否则是 L2）；记录功耗模式
```

### L0 系统级（第 01 章）

```bash
nvidia-smi dmon -s pucm -d 1                      # p=功耗/温度 u=利用率 c=频率 m=显存（Thor 上 mem% 始终为 0）
tegrastats --interval 100 --logfile /tmp/t.log    # 只能在宿主机上运行（容器内没有）；结束：tegrastats --stop
```
```python
from common.monitor import JtopSampler            # jtop API（容器内需要挂载 /run/jtop.sock，客户端版本要与服务端一致）
with JtopSampler(interval=0.2) as s:
    s.mark("infer"); run()
print(s.summary())                                # 按阶段的 GPU 负载、CPU 最忙核、功耗
```

### L1 Python 级（第 02 章）

```bash
python -m cProfile -o out.prof script.py && snakeviz out.prof     # 确定性，开销 +43%
kernprof -l -v script.py                                          # 行级（函数加 @profile），开销 +60%
py-spy record -r 200 -o flame.svg -- python script.py             # 采样，开销 +5%；aarch64 不支持 --native
py-spy dump --pid <PID>  /  py-spy top --pid <PID>                # attach 非子进程需要 CAP_SYS_PTRACE（privileged）或 prctl
viztracer --max_stack_depth 6 --ignore_c_function -o t.json script.py   # 时间线，开销 +5%（全量 +48%）
```

### L2 框架级（第 03 章）

```python
from torch.profiler import profile, ProfilerActivity, schedule
with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
             schedule=schedule(wait=1, warmup=2, active=5)) as p:  # Thor 上加 CUDA 也拿不到 GPU 时间
    for _ in range(8): step(); p.step()
print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
p.export_chrome_trace("trace.json")                               # https://ui.perfetto.dev
torch.cuda.memory._record_memory_history(); ...; torch.cuda.memory._dump_snapshot("snap.pickle")   # https://pytorch.org/memory_viz
torch.cuda.set_sync_debug_mode("warn")                            # 隐式同步报警
torch._dynamo.explain(fn)(*args); torch._dynamo.utils.counters["stats"]   # compile 是否真的生效
```

### L3 时间线（第 04 章）

```bash
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
     [--cuda-graph-trace=node] [--pytorch=autograd-nvtx] [--gpu-metrics-devices=all --gpu-metrics-frequency=20000] \
     [--python-sampling=true] [--cudabacktrace=sync:5000] -o out --force-overwrite true python script.py
nsys stats --force-export=true -r cuda_gpu_kern_sum,nvtx_sum,cuda_api_sum out.nsys-rep
nsys analyze -r gpu_gaps:gap=0.2:rows=5 out.nsys-rep               # gap 单位 ms，默认 500 ms，推理场景要调小
nsys export -t sqlite -o out.sqlite --force-overwrite true out.nsys-rep   # 用 SQL 自己分析（02_analyze_sqlite.py）
nsys status -e                                                     # CPU 采样需要 perf_event_open
```
```python
torch.cuda.profiler.start(); ...; torch.cuda.profiler.stop()       # 配合 --capture-range=cudaProfilerApi；计时在 stop() 之前
with common.region("infer"): ...                                   # NVTX + record_function
```

### L4 kernel 级（第 05 章）

```bash
ncu --nvtx --nvtx-include "profile/" -k regex:gemm --launch-skip 10 -c 1 \
    --section SpeedOfLight --section Occupancy --section WarpStateStats --metrics <...> \
    --import-source yes -f -o rep python script.py
ncu --profile-from-start off ...                                   # 只采 cudaProfilerStart/Stop 之间
ncu --kernel-name-base demangled -k regex:sum_functor ...          # 按完整签名（含模板参数）匹配
ncu -i rep.ncu-rep --page raw --csv --metrics gpu__time_duration.sum
ncu -i rep.ncu-rep --page source --print-source cuda,sass --csv    # 源码行级热点（编译加 -lineinfo）
nvcc -O3 -arch=sm_110 --resource-usage -c k.cu                     # 编译期看寄存器 / spill
```
Thor 的 DRAM 流量：读 = `lts__d_sectors_fill_sysmem.sum`×32 B；写 ≈ `min(lts__t_sectors_aperture_sysmem_op_write.sum, 4×..._op_write_lookup_miss.sum)`×32 B（第 05 章 5.3，[ncu_tools.py](../05_nsight_compute/ncu_tools.py)）。

### TensorRT（第 06 章）

```bash
trtexec --onnx=m.onnx --saveEngine=m.engine --profilingVerbosity=detailed [--fp16 | --int8 --fp16 | --noTF32] --skipInference
trtexec --loadEngine=m.engine --duration=3 --warmUp=500 [--useCudaGraph] [--noDataTransfers] [--useSpinWait]
trtexec --loadEngine=m.engine --dumpProfile --separateProfileRun --exportProfile=p.json --exportLayerInfo=l.json
nsys profile -t cuda,nvtx trtexec --loadEngine=m.engine --iterations=50 --duration=0 --warmUp=0   # 没有 -t tensorrt 了
ncu --nvtx --nvtx-include "regex:myelin-exec@^.stem.stem_0.Conv_myl0_2$/" trtexec --loadEngine=...  # 层名中的 / 换成 .
```

### 环境变量

| 变量 | 作用 | 出处 |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 动态 shape 下 reserved 显存 14.6 GB → 1.38 GB | 第 03、07 章 |
| `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` | 用驱动的显存池 | 第 07 章 |
| `TORCHINDUCTOR_CACHE_DIR` | inductor 的编译缓存目录（冷/热编译时间差 4~10 倍） | 第 07 章 |
| `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | v1.0.0 镜像的 triton 3.5 需要；v1.3.0（triton 3.8）不需要 | env_report |
| `CUDA_LAUNCH_BLOCKING=1` | 每个 kernel 同步执行，把 GPU 时间归回发起它的代码（慢约 3 倍，只能定性用） | 第 02 章 |

## B. 工具对比

| 工具 | 层 | 开销（本机实测） | 粒度 | 能看到 GPU 吗 | 侵入性 | 懂模型语义吗 | 查看方式 |
|---|---|---|---|---|---|---|---|
| jtop / tegrastats | L0 | 可忽略（独立服务；约 1 s 延迟） | 约 1 s | 负载、频率、功耗 | 无 | 否 | 终端 / CSV |
| nvidia-smi dmon | L0 | 可忽略 | 1 s | 负载（Thor 上 mem% 为 0） | 无 | 否 | 终端 |
| cProfile | L1 | +43% | 函数 | 否（GPU 时间算到同步点上） | 低 | 否 | snakeviz |
| line_profiler | L1 | +60% | 行 | 否 | 需要装饰器 | 否 | 终端 |
| py-spy | L1 | +5% | 采样（栈） | 否 | 无（可 attach） | 否 | 火焰图 / speedscope |
| VizTracer | L1 | +5%（过滤）~ +48%（全量） | 每次调用 | 否 | 低 | 否 | Perfetto |
| torch.profiler | L2 | 3.3~4.9 倍 | 算子 | ❌ Thor 上没有（CUPTI 时间戳为 0） | 低 | ✅ 算子、shape | Perfetto / TensorBoard |
| TRT profiler | L2 | 逐层时间多 7~34% | 层 | 层级 GPU 时间 | 无 | ✅ 层名、融合 | 终端 / JSON |
| nsys | L3 | +32% | 每个 API / kernel | ✅ 时间线、GPU metrics | 低（NVTX 可选） | 借助 NVTX | nsys-ui / sqlite |
| ncu | L4 | 约 7 s 启动 + 每个 kernel 10~40 次重放；140 个 kernel ≈ 5 分钟 | 单个 kernel | ✅ 硬件计数器 | 无 | 否（只认 kernel 名） | ncu-ui / CSV / Python |

（开销出处：第 02 章 2.1、第 03 章 3.7、第 04 章 4.1、第 05 章 5.2、第 06 章 6.3。）

**选择原则**（漏斗模型）：从开销最小、范围最大的工具开始（L0），每一层只回答一个问题，再决定往哪一层深入。不要一上来就用 ncu。

## C. 可视化：在 Jetson 上采集、在主机上查看

容器里没有显示器，GUI 工具在有图形界面的电脑上打开，**版本要和采集端一致或更新**。

| 产物 | 生成方式 | 查看方式 |
|---|---|---|
| `*.json`（Chrome trace） | torch.profiler `export_chrome_trace`、VizTracer | https://ui.perfetto.dev （浏览器，本地解析，大文件也流畅） |
| TensorBoard 目录 | `tensorboard_trace_handler` | `tensorboard --logdir ... --bind_all`；⚠️ 依赖 GPU trace，Thor 上内容不完整（第 03 章） |
| `snap.pickle` | `torch.cuda.memory._dump_snapshot` | https://pytorch.org/memory_viz |
| `*.prof` | cProfile | `snakeviz -s -H 0.0.0.0 -p 8080 x.prof`，需要映射端口 |
| `*.svg` / speedscope json | py-spy | 浏览器 / https://www.speedscope.app |
| `*.nsys-rep` | nsys | 主机上的 Nsight Systems 2025.3+；或者在 Jetson 上 `nsys export -t sqlite` 后用 SQL 分析 |
| `*.ncu-rep` | ncu | 主机上的 Nsight Compute 2025.3+；或者 `ncu -i` / `ncu_report` Python 模块 |

流程：在 Jetson 上采集（命令行）→ 先用命令行（`nsys stats`、`ncu -i --page raw`、SQL）做定量分析 → 需要看时间线或 Source 页时，把报告拷到主机上用 GUI 打开。⚠️ 拷贝 nsys 报告之前，注意其中包含的环境变量（D.6）。

## D. 常见坑（按主题汇总）

### D.1 计时与基准

1. 计时前后不同步、不做 warmup（第 00 章 0.6）。
2. 数据小于 32 MB 时测到的是 L2 带宽，而不是 DRAM（GEMV 热 L2 测出 930 GB/s，第 00 章 0.4）。
3. 不区分突发峰值和持续峰值：功耗墙会让持续 FP16 吞吐降低约 1/3，而 jtop 显示的频率不变（第 00、01 章）。
4. 单次测量就下结论：0.95~1.05 之间的结果要重复测量（第 08 章：0.94x 重复后是 1.00x）。
5. 不记录功耗模式和锁频状态，结果无法复现（第 01 章）。

### D.2 利用率指标

6. 只看 GPU 利用率：GPU 负载 97% 时 MFU 可能只有 11%（第 00 章）；GR Active 88% → SMs Active 76% → SM Issue 29% → Tensor Active 2%（第 04 章 4.8）。
7. 把 GPU load 当算力利用率：memory-bound 和 compute-bound 的负载都约 98%，只有功耗能区分（16 W vs 100 W，第 01 章）。
8. 把 `dmon` 的 mem 列当作显存占用：它是内存控制器的忙碌时间占比，在 Thor 上始终为 0（第 00、01 章）。
9. 把 CPU 100% 当成 CPU-bound：GPU 队列满时 launch 调用会自旋占满一个核（第 01 章）。
10. FLOPs 统计工具漏算融合算子（FlopCounterMode 遇到 TransformerEncoder fastpath 记为 0，第 00 章）。

### D.3 profiler 自身的影响（观察者效应）

11. profiler 的开销会改变 CPU/GPU 的速度平衡：torch.profiler 下 sync 耗时被测成约 0（第 03 章）；cProfile +43%（第 02 章）。
12. nsys 下 pageable 拷贝偶尔需要 1~6 ms，不开 nsys 时不存在（第 04 章 4.8 D）。
13. 计时包含了 `profiler.stop()`（nsys 落盘约 1 s，第 04 章 4.1）。
14. ncu 的 Duration 是冷缓存、串行化下的时间，不是真实耗时（第 05 章 5.1）。
15. TRT 逐层 profile 比真实时间多 7~34%，而且看不到层间间隙（第 06 章）。

### D.4 工具的使用细节

16. nsys `--capture-range-end` 默认会结束程序，要用 `stop`（第 04 章）。
17. nsys 默认看不到 CUDA Graph 内部的 kernel，GPU busy 会被严重低估（第 08 章：6.1% vs 37.5%），加 `--cuda-graph-trace=node`。
18. `nsys stats` 复用旧 sqlite，加 `--force-export=true`；`gpu_gaps` 默认阈值 500 ms（第 04 章）。
19. nsys 2025.3 没有 `-t tensorrt`（第 06 章）。
20. ncu 不加过滤：140 个 kernel 要 5 分钟（第 05 章）；TensorRT 层名中的 `/` 与 NVTX 过滤语法冲突（第 06 章）。
21. VizTracer 的 `include_files` 会剪掉整棵子树；torch 2.9 的 `with_stack` 需要 `_ExperimentalConfig(verbose=True)`（第 02、03 章）。
22. `torch.compile` 悄悄没生效：循环里的 `.item()` 让整个函数退回 eager（第 03 章）；`dynamic=False` 超过重编译上限后静默退回 eager（第 07 章）。
23. CUDA Graph capture 失败后 CUDA context 会损坏，失败用例要放在子进程中运行（第 04 章）。

### D.5 Thor / Jetson 平台

24. torch.profiler 拿不到 GPU kernel 时间（CUPTI 的 GPU 时间戳为 0，privileged 下也一样，第 03 章）。
25. ncu 没有 DRAM 指标，SOL 的 Memory% 对满带宽的 kernel 只显示约 17%（第 05 章 5.3）；nsys GPU metrics 也没有 DRAM 带宽（第 04 章 4.8）。
26. ncu 的 `--clock-control` 不生效，锁频靠 jetson_clocks（第 05 章）。
27. ≥256 KB 的 `cudaMemcpyAsync` 由 SM 执行，拷贝/计算不能重叠；<256 KB 走 Copy Engine，带宽低、尾延迟大（第 04 章 4.8、第 08 章）。
28. pinned memory 收益很小（0~3%），统一内存平台上不要照搬独显的经验（第 07 章）。
29. 容器内没有 tegrastats / nvpmodel / jetson_clocks；jtop 客户端版本必须与宿主机服务一致（env_report）；切换 nvpmodel 会关掉 jetson_clocks，由宿主机命令开启的 jetson_clocks 无法通过 jtop 重新开启，需要在宿主机上执行 `sudo jetson_clocks` 并用 `--show` 验证（第 01 章 1.6）。
30. GPU 计数器需要 CAP_SYS_ADMIN（`RmProfilingAdminOnly=1`），CPU 采样需要 perf_event，py-spy attach 需要 CAP_SYS_PTRACE（env_report）。
31. 只有 20 个 SM：小 batch 的 kernel 容易填不满（Tail Effect，第 05、06 章）；inductor 提示 `Not enough SMs to use max_autotune_gemm`（第 06 章）。

### D.6 优化与判断

32. 凭直觉判断瓶颈（no_pin、realloc 实测约 1.0x，第 00、08 章）。
33. 照着 ncu 规则的"预估加速"去优化：提高 Occupancy 预估 63% → 实测 0.99x；bank conflict 预估 91% → 实测 7%（第 05 章）。
34. 笔记中的经验阈值（"Occupancy < 40%""IPC < 1.5""Branch Efficiency < 90%"）在实测中多次失效（第 05 章）。
35. 把 CUDA Graph 当万能加速：对 Memory-bound、Compute-bound 无效（第 04、07 章）。
36. 修复一个瓶颈后不重新度量：瓶颈会转移（第 07、08 章）。
37. 未校准的 INT8 engine 不能用来评估精度；fp32 engine 默认是 TF32（第 06 章）。
38. ⚠️ **nsys 报告包含目标进程的完整环境变量**（sqlite 的 `TARGET_INFO_SYSTEM_ENV`），可能带有凭据；ncu 报告记录主机名和 IP。分享前检查，或者在干净的环境中采集（`env -i PATH=$PATH HOME=$HOME python ...`）（第 04 章、env_report）。

## E. 参考资料

**官方文档**
- PyTorch Profiler：https://pytorch.org/docs/stable/profiler.html ；显存快照：https://pytorch.org/docs/stable/torch_cuda_memory.html
- Holistic Trace Analysis（HTA）：https://hta.readthedocs.io
- VizTracer：https://viztracer.readthedocs.io ；py-spy：https://github.com/benfred/py-spy
- Nsight Systems User Guide：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute：Profiling Guide https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ，CLI https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html （NVTX 过滤语法见其中的 nvtx-filtering 一节）
- TensorRT Best Practices：https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
- CUDA C++ Best Practices Guide：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- jetson-stats（jtop）：https://github.com/rbonghi/jetson_stats （本机服务端 4.5.2 对应提交 `11140a0`）
- 可视化：https://ui.perfetto.dev 、https://pytorch.org/memory_viz 、https://www.speedscope.app

**[doc/](../doc/) 脑图中收录的链接**

| 脑图 · 主题 | 链接 |
|---|---|
| CUDA · 硬件架构 > SoC | https://zhuanlan.zhihu.com/p/536701808 |
| CUDA · GPU 与 CPU 之间的通讯 > PCIe | https://zhuanlan.zhihu.com/p/462191421 |
| CUDA · prefetch > GDDR6 vs DDR4 | https://zhuanlan.zhihu.com/p/83935084 ；https://forums.tomshardware.com/threads/effective-memory-clock-speed-confusions.3518637/ |
| CUDA · 显卡带宽与吞吐量 | https://blog.csdn.net/qq_51302564/article/details/119491586 ；https://www.cnblogs.com/lanlancky/p/17414299.html |
| CUDA · bank > 内存的原理（视频） | https://www.bilibili.com/video/BV1Htr8YhELV/ |
| CUDA · 性能分析 > Roofline model | https://zhuanlan.zhihu.com/p/34204282 |
| CUDA · STREAM 和 EVENT | https://zhuanlan.zhihu.com/p/699754357 ；https://youtu.be/vtR7cgYATdk |
| 程序性能分析工具 · viztracer > 火焰图 | https://blog.csdn.net/Asunqingwen/article/details/139996864 ；https://ui.perfetto.dev/ |
| 程序性能分析工具 · jtop 各字段的含义 | https://blog.csdn.net/FREEDOM_X/article/details/139979294 |
