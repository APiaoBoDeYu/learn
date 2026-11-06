"""第 05 章公用工具：脚本"自己 profile 自己"，并把 ncu 报告整理成对比表。

流程（每个 demo 都一样）：
    python 02_memory_bound.py
      ├─ 1. 计时：CUDA Event + 刷 L2，得到每个变体的真实耗时（不在 ncu 下，ncu 会让 kernel 串行化并多次 replay）
      ├─ 2. 采集：ncu --nvtx --nvtx-include "profile/" ... python 02_memory_bound.py --worker
      │        worker 先 warmup（不在 NVTX 范围内，不会被采集），再在 profile/<变体名> 范围内把每个变体各跑一次
      └─ 3. 解读：用 Nsight Compute 自带的 Python 模块 ncu_report 读取 .ncu-rep，按变体打印指标对比表

Thor 特有的换算（见 README 5.3）：
    Thor 没有独立显存，GPU 经 L2 → MC（内存控制器）访问系统内存（sysmem），ncu 里没有 dram__* 指标，
    SOL 的 "Memory Throughput" 也不是相对 DRAM 峰值。这里用 L2 的 sysmem 计数自己估算 DRAM 流量：
        读 = lts__d_sectors_fill_sysmem（L2 未命中、从内存填充进 L2 的 sector，精确）
        写 ≈ min(lts__t_sectors_aperture_sysmem_op_write, 4 x lts__t_sectors_aperture_sysmem_op_write_lookup_miss)
             前者是 L2 收到的写请求 sector 数：非合并写会把同一个 sector 写多次，直接用它会高估（实测转置 naive 达到 8 倍）；
             L2 写 miss 时按 128B 整行（4 个 sector）分配，所以"4 x 写 miss 数"是被写脏的不同 sector 数的上界。
             对本章的合并 / 非合并写 kernel，这个估计都等于逻辑写入量（见 README 5.3）。没有 DRAM 写回计数器可以直接验证。
        每个 sector 32 字节；带宽利用率以第 00 章实测的 copy 带宽为峰值。
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent))

NCU = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
_py = sorted(glob.glob("/opt/nvidia/nsight-compute/*/extras/python"))
if _py:
    sys.path.append(_py[-1])
import ncu_report  # noqa: E402

PEAK_BW_GBS = 240.0   # 第 00 章实测（copy 256MB）：旧容器 226、新容器 240 GB/s；本章在新容器中运行，取 240
SECTOR = 32

# 各 demo 都会采集的基础指标
BASE_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",              # SOL: Compute (SM) Throughput
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",  # SOL: Memory Throughput
    "lts__d_sectors_fill_sysmem.sum",                                 # Thor: 从内存读进 L2 的 sector
    "lts__t_sectors_aperture_sysmem_op_write.sum",                    # Thor: L2 收到的写 sysmem 请求（sector）
    "lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum",        # Thor: 其中 L2 未命中的
    "lts__t_sector_hit_rate.pct",                                     # L2 命中率
    "sm__warps_active.avg.pct_of_peak_sustained_active",              # Achieved Occupancy
    "sm__maximum_warps_per_active_cycle_pct",                         # Theoretical Occupancy
    "launch__registers_per_thread",
    "smsp__inst_executed.avg.per_cycle_active",                       # IPC（每个 SMSP）
    "sm__inst_executed.avg.per_cycle_active",                         # IPC（每个 SM，= 4 个 SMSP 之和）
    "smsp__inst_executed.sum",                                        # 执行的 warp 指令总数
    # FP32 运算数（画 Roofline 用；FLOP = 2*FFMA + FADD + FMUL）
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
]


def worker_mode() -> bool:
    return "--worker" in sys.argv


class nvtx_range:
    """with nvtx_range("profile"): ... —— 给 ncu 的 --nvtx-include 用。"""
    def __init__(self, name): self.name = name
    def __enter__(self): torch.cuda.nvtx.range_push(self.name)
    def __exit__(self, *a): torch.cuda.nvtx.range_pop()


def run_variants(variants: dict, warmup: int = 2):
    """worker 模式：先 warmup，再在 profile/<name> 中每个变体各跑一次。variants: {name: fn}"""
    for fn in variants.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    with nvtx_range("profile"):
        for name, fn in variants.items():
            with nvtx_range(name):
                fn()
    torch.cuda.synchronize()


def gpu_time_ms(fn, iters: int = 20, flush: bool = True) -> float:
    """CUDA Event 计时取中位数；flush=True 时每次之前写 256MB 刷掉 L2（Thor 的 L2 有 32MB）。"""
    buf = torch.empty(64 * 1024 * 1024, device="cuda")
    for _ in range(3):
        fn()
    ts = []
    for _ in range(iters):
        if flush:
            buf.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def profile(script: str, name: str, *, sections=(), metrics=(), set_=None, extra=(), script_args=(),
            nvtx_include="profile/", quiet=True) -> Path:
    """在 ncu 下以 --worker 模式运行 script，报告保存到 outputs/<name>.ncu-rep。"""
    rep = OUT / f"{name}.ncu-rep"
    cmd = [NCU, "--nvtx", "--nvtx-include", nvtx_include, "--import-source", "yes", "-f", "-o", str(rep.with_suffix(""))]
    if set_:
        cmd += ["--set", set_]
    for s in sections:
        cmd += ["--section", s]
    if metrics:
        cmd += ["--metrics", ",".join(dict.fromkeys(metrics))]
    cmd += [*extra, sys.executable, script, "--worker", *script_args]
    print("$", " ".join(c if " " not in c else repr(c) for c in cmd[:-3]), Path(script).name, "--worker", *script_args)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    import re
    passes = re.findall(r"(\d+) passes", r.stdout)
    print(f"  ncu 用时 {time.time() - t0:.1f} s，采集了 {len(passes)} 个 kernel"
          + (f"，每个 kernel replay {passes[0]} 次（pass）" if passes else ""))
    if r.returncode != 0 or not rep.exists():
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"ncu 失败（returncode={r.returncode}）")
    if not quiet:
        print(r.stdout)
    return rep


def _metric(a, n):
    m = a.metric_by_name(n)
    if m is None:
        return None
    try:
        return m.as_double()
    except Exception:
        return m.as_string()


def load(rep: Path, metrics=None) -> list[dict]:
    """读取报告：每个 kernel 一行，包含变体名（NVTX 栈的最内层）、kernel 名和指标。"""
    rows = []
    ctx = ncu_report.load_report(str(rep))
    for ri in range(ctx.num_ranges()):
        rg = ctx.range_by_idx(ri)
        for ai in range(rg.num_actions()):
            a = rg.action_by_idx(ai)
            st = a.nvtx_state()
            stack = ()
            for d in st.domains():
                stack = st.domain_by_id(d).push_pop_ranges() or stack
            row = {"variant": stack[-1] if stack else "?", "kernel": a.name(), "_action": a}
            for n in (metrics or a.metric_names()):
                row[n] = _metric(a, n)
            rows.append(row)
    return rows


def hot_lines(rep: Path, kernel_regex: str, metric: str = "Warp Stall Sampling (All Samples)", top: int = 5,
              extra_cols=()) -> str:
    """Source 页（CUDA 源码级）：kernel 需要用 -lineinfo 编译，采集时 --import-source yes。
    等价于 GUI 的 Source 页面，命令行是 ncu -i rep --page source --print-source cuda,sass --csv。"""
    import csv
    import io
    r = subprocess.run([NCU, "-i", str(rep), "--kernel-name-base", "demangled", "-k", f"regex:{kernel_regex}", "--page", "source",
                        "--print-source", "cuda,sass", "--csv"], capture_output=True, text=True)
    rows = list(csv.reader(io.StringIO(r.stdout)))
    hi = next((i for i, row in enumerate(rows) if row and row[0] == "Line No"), None)
    if hi is None:
        print(f"  （{kernel_regex}：报告中没有源码级数据，需要 -lineinfo 编译、--import-source yes、采集 SourceCounters section）")
        return ""
    head = rows[hi]
    idx = {name: head.index(name) for name in (metric, *extra_cols)}
    lines = [row for row in rows[hi + 1:] if row and row[0].strip().isdigit()]
    val = lambda row, k: float(row[idx[k]] or 0) if row[idx[k]] not in ("", "-") else 0.0  # noqa: E731
    tot = sum(val(row, metric) for row in lines) or 1
    out = [f"  {metric} 最多的 {top} 行（{kernel_regex}，合计 {tot:.0f}）："]
    for row in sorted(lines, key=lambda row: -val(row, metric))[:top]:
        ex = "".join(f"  {k}={val(row, k):.0f}" for k in extra_cols)
        out.append(f"    L{row[0]:<4s} {100 * val(row, metric) / tot:5.1f}%{ex}  {row[1].strip()[:90]}")
    text = "\n".join(out)
    print(text)
    return text


def merge_by_variant(rows: list[dict]) -> list[dict]:
    """一个变体包含多个 kernel 时（例如 torch.bmm 按 batch 拆成多次 launch）合并成一行：
    时间、流量、指令数求和，其他指标取耗时最长的那个 kernel。"""
    out: dict[str, dict] = {}
    add = ("gpu__time_duration.sum", "lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_aperture_sysmem_op_write.sum",
           "lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum", "smsp__inst_executed.sum")
    for r in rows:
        v = r["variant"]
        if v not in out:
            out[v] = dict(r, _n=1, _max=r["gpu__time_duration.sum"])
            continue
        o = out[v]
        o["_n"] += 1
        if r["gpu__time_duration.sum"] > o["_max"]:
            keep = {k: o[k] for k in add if k in o}
            o.update(r); o.update(keep); o["_max"] = r["gpu__time_duration.sum"]
        for k in add:
            if k in r and r[k] is not None:
                o[k] = (o.get(k) or 0) + r[k]
    for o in out.values():
        if o["_n"] > 1:
            o["kernel"] = f"{o['kernel']} x{o['_n']}"
    return list(out.values())


def sysmem_read_bytes(r: dict) -> float:
    return (r.get("lts__d_sectors_fill_sysmem.sum") or 0) * SECTOR


def sysmem_write_bytes(r: dict) -> float:
    wr = r.get("lts__t_sectors_aperture_sysmem_op_write.sum") or 0
    miss = r.get("lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum")
    return (min(wr, 4 * miss) if miss is not None else wr) * SECTOR


def sysmem_bytes(r: dict) -> float:
    return sysmem_read_bytes(r) + sysmem_write_bytes(r)


def sysmem_gbs(r: dict) -> float:
    return sysmem_bytes(r) / r["gpu__time_duration.sum"]   # bytes / ns = GB/s


def table(rows: list[dict], cols: list[tuple], title: str = "") -> str:
    """cols: [(表头, 取值函数或指标名, 格式)]"""
    head = [c[0] for c in cols]
    body = []
    for r in rows:
        line = []
        for _, get, fmt in cols:
            v = get(r) if callable(get) else r.get(get)
            line.append("n/a" if v is None else (format(v, fmt) if not isinstance(v, str) else v))
        body.append(line)
    w = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    lines = ([f"\n{title}"] if title else []) + ["  ".join(h.rjust(n) if i else h.ljust(n) for i, (h, n) in enumerate(zip(head, w))),
                                                   "-" * (sum(w) + 2 * len(w))]
    for b in body:
        lines.append("  ".join(str(x).rjust(n) if i else str(x).ljust(n) for i, (x, n) in enumerate(zip(b, w))))
    text = "\n".join(lines)
    print(text)
    return text


# 常用列
COL_VARIANT = ("variant", "variant", "")
COL_TIME = ("ncu_us", lambda r: r["gpu__time_duration.sum"] / 1e3, ".1f")
COL_SM = ("SM%", "sm__throughput.avg.pct_of_peak_sustained_elapsed", ".1f")
COL_MEM = ("Mem%(ncu)", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", ".1f")
COL_DRAM_MB = ("DRAM_MB", lambda r: sysmem_bytes(r) / 1e6, ".1f")
COL_DRAM_RD = ("DRAM_rd_MB", lambda r: sysmem_read_bytes(r) / 1e6, ".1f")
COL_DRAM_WR = ("DRAM_wr_MB", lambda r: sysmem_write_bytes(r) / 1e6, ".1f")
COL_L2_WR_REQ = ("L2_wr_req_MB", lambda r: (r.get("lts__t_sectors_aperture_sysmem_op_write.sum") or 0) * SECTOR / 1e6, ".0f")
COL_DRAM_GBS = ("DRAM_GB/s", sysmem_gbs, ".1f")
COL_DRAM_PCT = ("DRAM%(实测峰值)", lambda r: 100 * sysmem_gbs(r) / PEAK_BW_GBS, ".0f")
COL_L2HIT = ("L2hit%", "lts__t_sector_hit_rate.pct", ".1f")
COL_OCC = ("Occ%", "sm__warps_active.avg.pct_of_peak_sustained_active", ".1f")
COL_TOCC = ("理论Occ%", "sm__maximum_warps_per_active_cycle_pct", ".1f")
COL_REGS = ("regs", "launch__registers_per_thread", ".0f")
COL_IPC = ("IPC(SM)", "sm__inst_executed.avg.per_cycle_active", ".2f")


def rules(r: dict, min_speedup: float = 5.0) -> list[str]:
    """取出 ncu 规则引擎给出的优化建议（OPT），按预估加速排序。"""
    import re
    out = []
    for d in r["_action"].rule_results_as_dicts():
        msg = d.get("rule_message") or {}
        if msg.get("type") != 2:          # 2 = OPT（优化建议），1 = INF，3 = WRN
            continue
        v = (d.get("speedup_estimation") or {}).get("speedup", 0.0)
        text = re.sub(r"@\w+:([^:@]+)(:[^@]*)?@", r"\1", msg.get("message", "")).replace("\n", " ")
        out.append((v, d.get("rule_identifier", "?"), msg.get("title") or "", text))
    out.sort(key=lambda x: -x[0])
    return [f"[{k}{'/' + t if t else ''}] 预估加速 {v:.1f}%: {m[:260]}" for v, k, t, m in out if v >= min_speedup]


class Tee:
    """把 print 同时写到 outputs/<name>.log。"""
    def __init__(self, name):
        self.f = open(OUT / f"{name}.log", "w")
        self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s)
    def flush(self):
        self.out.flush(); self.f.flush()


def tee(name: str):
    if not worker_mode():
        sys.stdout = Tee(name)
