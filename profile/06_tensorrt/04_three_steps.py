"""06-04 三步法：TRT Profiler（哪层慢）→ nsys（为什么慢）→ ncu（这个 kernel 为什么慢）

    /opt/venv/bin/python 06_tensorrt/04_three_steps.py      （需要先运行 02、03）

第 1 步：读 03 的逐层 profile，找出最慢的层。
第 2 步：nsys profile -t cuda,nvtx trtexec --loadEngine=...
    - nsys 2025.3 已经没有 `-t tensorrt`（会报 Illegal --trace argument），TensorRT 自己会发 NVTX：
        domain "TensorRT"    : ExecutionContext::enqueueV3、{ForeignNode[...]}（整张 Myelin 图对 TRT 来说只是一层）
        domain "myelin-exec" : Myelin 内部每一层一个 push/pop range，名字与逐层 profile 一致（需要 --profilingVerbosity=detailed）
    - 层 → kernel：NVTX range（CPU 时间）内发起的 cudaLaunchKernel/cuLaunchKernel → correlationId → GPU 上的 kernel
    - 统计每次推理的 GPU busy、kernel 间隙；对比 --useCudaGraph
第 3 步：ncu --nvtx --nvtx-include "regex:myelin-exec@^<层名，/ 换成 .>$/" trtexec ...，只采这一层的 kernel
"""
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "05_nsight_compute"))
import ncu_tools as NT  # noqa: E402  第 05 章的 ncu 报告读取工具（含 Thor 的 DRAM 换算）

lines = []


def p(s=""):
    print(s); lines.append(s)


def step1(model, prec):
    prof = [x for x in json.load(open(OUT / f"03_{model}_{prec}_profile.json")) if "name" in x]
    prof.sort(key=lambda x: -x["averageMs"])
    p(f"[第 1 步] {model} {prec} 逐层 profile 前 3：" + "；".join(f"{x['name']} {x['averageMs']:.4f} ms ({x['percentage']:.1f}%)" for x in prof[:3]))
    return prof


def nsys(model, prec, graph=False):
    tag = f"04_{model}_{prec}{'_graph' if graph else ''}"
    rep = OUT / f"{tag}.nsys-rep"
    cmd = ["nsys", "profile", "-t", "cuda,nvtx", "-o", str(rep.with_suffix("")), "--force-overwrite", "true"]
    if graph:
        cmd += ["--cuda-graph-trace=node"]
    cmd += ["trtexec", f"--loadEngine={OUT / f'{model}_{prec}.engine'}", "--iterations=50", "--duration=0", "--warmUp=0"]
    if graph:
        cmd += ["--useCudaGraph"]
    subprocess.run(cmd, capture_output=True)
    db = rep.with_suffix(".sqlite")
    subprocess.run(["nsys", "export", "-t", "sqlite", "-o", str(db), "--force-overwrite", "true", str(rep)], capture_output=True)
    return sqlite3.connect(db)


def union(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


def step2(model, prec, prof):
    res = {}
    for graph in (False, True):
        c = nsys(model, prec, graph)
        name = lambda col: f"COALESCE({col}, (SELECT value FROM StringIds WHERE id = textId))"  # noqa: E731
        enq = c.execute(f"""SELECT start, end, globalTid FROM NVTX_EVENTS WHERE eventType = 59 AND
                            {name('text')} = 'ExecutionContext::enqueueV3' ORDER BY start""").fetchall()
        kern = c.execute("""SELECT k.start, k.end, k.correlationId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                            JOIN StringIds s ON s.id = k.shortName ORDER BY k.start""").fetchall()
        # 每次推理：从第 i 次 enqueue 开始，到第 i+1 次 enqueue 开始之前，GPU 上执行的 kernel
        # （CUDA Graph 模式下 enqueue 只在捕获时调用一次，改为按 graph launch 切分）
        if graph:
            enq = c.execute("""SELECT r.start, r.end, r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId
                               WHERE s.value LIKE 'cudaGraphLaunch%' OR s.value LIKE 'cuGraphLaunch%' ORDER BY r.start""").fetchall()
        launch_t = dict(c.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
        per = []
        for i, (s, e, _) in enumerate(enq[5:-1], start=5):            # 跳过前 5 次（首次运行、捕获）
            nxt = enq[i + 1][0]
            ks = [k for k in kern if s <= launch_t.get(k[2], 0) < nxt] if not graph else \
                 [k for k in kern if s <= launch_t.get(k[2], k[0]) < nxt]
            if not ks:
                continue
            span = ks[-1][1] - ks[0][0]
            busy = union([(k[0], k[1]) for k in ks])
            per.append((len(ks), span, busy, e - s))
        n = len(per)
        avg = [sum(x[j] for x in per) / n for j in range(4)]
        res[graph] = avg
        p(f"[第 2 步] {model} {prec} {'--useCudaGraph' if graph else '默认       '}：每次推理 {avg[0]:.0f} 个 kernel，"
          f"GPU 跨度 {avg[1] / 1e3:.1f} us，其中 kernel 执行（并集）{avg[2] / 1e3:.1f} us（{100 * avg[2] / avg[1]:.0f}%），"
          f"间隙 {(avg[1] - avg[2]) / 1e3:.1f} us；CPU 侧 {'enqueueV3' if not graph else 'cudaGraphLaunch'} {avg[3] / 1e3:.1f} us（{n} 次平均）")
        if graph:
            continue
        # 层 → kernel（只看默认模式）
        layers = c.execute(f"""SELECT start, end, globalTid, {name('text')} FROM NVTX_EVENTS
                               WHERE eventType = 59 AND domainId = (SELECT domainId FROM NVTX_EVENTS WHERE eventType = 75
                               AND {name('text')} = 'myelin-exec') ORDER BY start""").fetchall()
        launches = c.execute("""SELECT r.start, r.globalTid, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                                JOIN StringIds s ON s.id = r.nameId WHERE s.value LIKE '%LaunchKernel%' ORDER BY r.start""").fetchall()
        k_by_corr = {k[2]: k for k in kern}
        layer_k = defaultdict(list)
        j = 0
        for ls, le, tid, lname in layers:
            for (ts, ttid, corr) in launches:
                if ls <= ts < le and ttid == tid and corr in k_by_corr:
                    layer_k[lname].append(k_by_corr[corr])
        p("  层 → kernel（nsys 测得的 GPU 时间 vs 第 1 步 TRT profiler 的时间）：")
        p(f"    {'层':<40s} {'TRT_ms':>7s} {'nsys_ms':>8s} {'kernel 数/次':>10s}  kernel")
        for x in prof[:5]:
            ks = layer_k.get(x["name"], [])
            n_inf = max(1, len([1 for l in layers if l[3] == x["name"]]))
            t = sum(k[1] - k[0] for k in ks) / n_inf / 1e6
            names = sorted({k[3] for k in ks})
            p(f"    {x['name'][:40]:<40s} {x['averageMs']:7.4f} {t:8.4f} {len(ks) / n_inf:10.1f}  {', '.join(n[:60] for n in names)[:120]}")
    p(f"  → CUDA Graph 让每次推理的 GPU 跨度从 {res[False][1] / 1e3:.1f} us 降到 {res[True][1] / 1e3:.1f} us"
      f"（{res[False][1] / res[True][1]:.2f}x）")
    return res


def step3(model, prec, layer, compare_layer=None):
    rep = OUT / f"04_{model}_{prec}_ncu.ncu-rep"
    # 层名里本身含有 "/"，和 NVTX 过滤语法中表示嵌套的 "/" 冲突（"myelin-exec@/stem/stem_0/Conv_myl0_2/" 匹配不到）。
    # 改用正则形式 "regex:<domain>@<正则>/"，并把层名中的 "/" 换成正则的 "."
    rx = lambda n: "regex:myelin-exec@^" + n.replace("/", ".") + "$/"  # noqa: E731
    incl = [rx(layer)] + ([rx(compare_layer)] if compare_layer else [])
    cmd = [NT.NCU, "--nvtx"]
    for x in incl:
        cmd += ["--nvtx-include", x]
    cmd += ["-c", str(len(incl) * 2), "-f", "-o", str(rep.with_suffix("")),
            "--section", "SpeedOfLight", "--section", "Occupancy", "--section", "LaunchStats", "--section", "WarpStateStats",
            "--section", "ComputeWorkloadAnalysis", "--section", "MemoryWorkloadAnalysis",
            "--metrics", ",".join(NT.BASE_METRICS + ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                                                      "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active",
                                                      "launch__grid_size", "launch__block_size"]),
            "trtexec", f"--loadEngine={OUT / f'{model}_{prec}.engine'}", "--iterations=2", "--duration=0", "--warmUp=0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not rep.exists():
        p(r.stdout[-2000:]); return
    rows = NT.load(rep)
    for r_ in rows:
        st = r_["_action"].nvtx_state()
        for d in st.domains():                       # 用 myelin-exec domain 的最内层 range（= TRT 层名）作为变体名
            dm = st.domain_by_id(d)
            if dm.name() == "myelin-exec" and dm.push_pop_ranges():
                r_["variant"] = dm.push_pop_ranges()[-1]
        r_["variant"] = r_["variant"][:34]
        r_["kernel"] = r_["kernel"][:44]
    p(f"[第 3 步] ncu --nvtx " + " ".join(f'--nvtx-include "{x}"' for x in incl) + "：")
    lines.append(NT.table(rows, [NT.COL_VARIANT, ("kernel", "kernel", ""), NT.COL_TIME, NT.COL_SM, NT.COL_DRAM_GBS, NT.COL_DRAM_PCT,
                    ("tensor%", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", ".1f"),
                    ("tc%", "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active", ".1f"),
                    ("grid", "launch__grid_size", ".0f"), NT.COL_OCC, NT.COL_REGS]))
    for r_ in rows[:1]:
        for s in NT.rules(r_, min_speedup=10)[:3]:
            p("    - " + s)
    return rows


def main():
    for model, prec in (("tinycnn_b8", "fp16"), ("toyvla_b1", "fp16")):
        p(f"\n================ {model} {prec} ================")
        prof = step1(model, prec)
        step2(model, prec, prof)
        if model == "tinycnn_b8":
            rows = step3(model, prec, prof[0]["name"], compare_layer="/layers/layers_2/conv2/Conv_myl0_19")
            if rows:
                p("\n".join(f"    {r['variant']}: {r['kernel']}" for r in rows))
    (OUT / "04_three_steps.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
