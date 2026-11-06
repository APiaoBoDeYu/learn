"""01-02 nvidia-smi / NVML：在 Thor 上哪些字段可用，dmon 怎么用。

1. 探测 NVML 常用查询接口在本机是否支持（独显上全部可用，Jetson 上很多返回 Not Supported）
2. 在后台运行 `nvidia-smi dmon -s u`，同时依次跑"空闲 → 拷贝 → GEMM"三个阶段，解析输出

运行：/opt/venv/bin/python 01_system_monitor/02_nvidia_smi.py
"""
import subprocess
import time

import pynvml
import torch


def probe_nvml():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    print(f"driver {pynvml.nvmlSystemGetDriverVersion()}  device {pynvml.nvmlDeviceGetName(h)}")
    probes = {
        "utilization (gpu/mem %)": lambda: pynvml.nvmlDeviceGetUtilizationRates(h),
        "memory info": lambda: pynvml.nvmlDeviceGetMemoryInfo(h),
        "temperature": lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        "power usage (mW)": lambda: pynvml.nvmlDeviceGetPowerUsage(h),
        "SM clock (MHz)": lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        "mem clock (MHz)": lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
        "throttle reasons": lambda: hex(pynvml.nvmlDeviceGetCurrentClocksEventReasons(h)),
        "util samples (history)": lambda: len(pynvml.nvmlDeviceGetSamples(h, pynvml.NVML_GPU_UTILIZATION_SAMPLES, 0)[1]),
        "running processes": lambda: len(pynvml.nvmlDeviceGetComputeRunningProcesses(h)),
        "PCIe throughput": lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES),
    }
    print(f"\n{'NVML query':<28s} result")
    for name, fn in probes.items():
        try:
            v = fn()
            if hasattr(v, "gpu"):
                v = f"gpu={v.gpu}% mem={v.memory}%"
            elif hasattr(v, "total"):
                v = f"used={v.used / 1e9:.1f}GB total={v.total / 1e9:.1f}GB"
            print(f"{name:<28s} ✅ {v}")
        except pynvml.NVMLError as e:
            print(f"{name:<28s} ❌ {e}")
    pynvml.nvmlShutdown()


def dmon_during_workload():
    print("\n=== nvidia-smi dmon -s u（每秒一行），负载：空闲 2s → 拷贝 4s → FP16 GEMM 4s ===")
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", "11"],
                         stdout=subprocess.PIPE, text=True)
    x = torch.empty(64 * 1024 * 1024, device="cuda")
    y = torch.empty_like(x)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    t0 = time.time()
    phases = []
    for name, dur, fn in [("idle", 2, None), ("copy", 4, lambda: y.copy_(x)), ("gemm", 4, lambda: a @ a)]:
        phases.append((round(time.time() - t0, 1), name))
        t = time.time()
        while time.time() - t < dur:
            if fn:
                fn()
            else:
                time.sleep(0.05)
        torch.cuda.synchronize()
    out = p.communicate()[0]
    print(out)
    print("阶段起始时间(s):", phases)


if __name__ == "__main__":
    probe_nvml()
    dmon_during_workload()
