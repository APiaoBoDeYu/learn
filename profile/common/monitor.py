"""后台系统指标采样器：在推理/benchmark 运行时，于后台线程定期采样系统指标，结束后导出 CSV 或做汇总。

- JtopSampler : 基于 jetson-stats（jtop）的 Python API。字段包括 CPU 各核负载、GPU 负载/频率、EMC、功耗、温度
- NvmlSampler : 基于 NVML（pynvml），即 nvidia-smi 背后的库，跨平台（x86 独显也能用）

用法：
    with JtopSampler(interval=0.1) as s:
        run_workload()
    s.to_csv("outputs/x.csv"); print(s.summary())
    s.mark("phase-name")   # 在时间线上打标记，方便把采样结果和负载阶段对应起来
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path


class _BaseSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.records: list[dict] = []
        self.marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.t0 = 0.0
        self._phase = ""

    # 子类实现
    def _open(self): ...
    def _close(self): ...
    def _sample(self) -> dict: raise NotImplementedError

    def mark(self, phase: str) -> None:
        """标记从现在起进入某个阶段，之后的采样都会带上 phase 字段。"""
        self._phase = phase
        self.marks.append((time.perf_counter() - self.t0, phase))

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter() - self.t0
            try:
                rec = self._sample()
            except Exception as e:  # noqa: BLE001
                rec = {"error": repr(e)}
            rec = {"t": round(t, 3), "phase": self._phase, **rec}
            self.records.append(rec)
            dt = self.interval - (time.perf_counter() - self.t0 - t)
            if dt > 0:
                self._stop.wait(dt)

    def start(self):
        self._open()
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join()
        self._close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def to_csv(self, path: str | Path) -> None:
        if not self.records:
            return
        keys = list(dict.fromkeys(k for r in self.records for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.records)

    def summary(self, fields: list[str] | None = None) -> str:
        """按 phase 分组，计算每个字段的平均值。"""
        phases = list(dict.fromkeys(r["phase"] for r in self.records))
        fields = fields or [k for k, v in self.records[-1].items()
                            if isinstance(v, (int, float)) and k != "t"]
        lines = [f"{'phase':<22s}" + "".join(f"{f:>12s}" for f in fields)]
        for ph in phases:
            rs = [r for r in self.records if r["phase"] == ph]
            row = f"{ph or '(none)':<22s}"
            for f in fields:
                vals = [r[f] for r in rs if isinstance(r.get(f), (int, float))]
                row += f"{(sum(vals) / len(vals)) if vals else float('nan'):12.1f}"
            lines.append(row)
        return "\n".join(lines)


class JtopSampler(_BaseSampler):
    """jtop 采样。注意：jtop 服务端自身有刷新周期，interval 太小会拿到重复的值。"""

    def _open(self):
        from jtop import jtop
        self._j = jtop(interval=max(self.interval, 0.1))
        self._j.start()

    def _close(self):
        self._j.close()

    def _sample(self) -> dict:
        j = self._j
        st = j.stats
        cores = [st[k] for k in st if k.startswith("CPU") and isinstance(st[k], (int, float))]
        gpu = next(iter(j.gpu.values()))
        rail = j.power["rail"]
        return {
            "gpu_load": st.get("GPU", 0.0),                       # %，时间维度利用率
            "gpu_freq_mhz": gpu["freq"]["cur"] // 1000,
            "emc_load": st.get("EMC", 0),                         # %，内存控制器利用率
            "cpu_avg": sum(cores) / max(len(cores), 1),           # 所有核的平均负载 %
            "cpu_max": max(cores) if cores else 0,                # 最忙那个核的负载 %（Python 单线程瓶颈看这个）
            "ram_gb": round(j.memory["RAM"]["used"] / 1024 ** 2, 2),
            "temp_gpu": st.get("Temp gpu"),
            "p_gpu_w": rail.get("VDD_GPU", {}).get("power", 0) / 1000,
            "p_cpu_soc_w": rail.get("VDD_CPU_SOC_MSS", {}).get("power", 0) / 1000,
            "p_tot_w": j.power["tot"]["power"] / 1000,
        }


class NvmlSampler(_BaseSampler):
    """NVML 采样（nvidia-smi 用的就是 NVML）。在 Thor 上只有 utilization 等部分字段可用。"""

    def _open(self):
        import pynvml
        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _close(self):
        self._nv.nvmlShutdown()

    def _sample(self) -> dict:
        u = self._nv.nvmlDeviceGetUtilizationRates(self._h)
        return {"sm_util": u.gpu, "mem_util": u.memory}
