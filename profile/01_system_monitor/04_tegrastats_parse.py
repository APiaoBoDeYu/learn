"""01-04 解析 tegrastats 日志。

tegrastats 是 Jetson 自带的命令行监控工具（容器内没有，需在宿主机上运行）：
    sudo tegrastats --interval 100 --logfile /tmp/tegra.log     # 100ms 采样，写入文件
    sudo tegrastats --stop
本脚本把它的单行文本解析成结构化数据，便于和推理延迟日志对齐、画图。

运行：/opt/venv/bin/python 01_system_monitor/04_tegrastats_parse.py [日志路径]
默认解析 outputs/tegrastats_host_sample.log（在宿主机上采集的真实样例）。
"""
import re
import sys
from pathlib import Path

RE_TIME = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)")
RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU \[([^\]]*)\]")
RE_EMC = re.compile(r"EMC_FREQ (\d+)%(?:@(\d+))?")
RE_GR3D = re.compile(r"GR3D_FREQ (?:(\d+)%)?@\[?([\d,]+)\]?")
RE_TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")
RE_RAIL = re.compile(r"(VDD_\w+|VIN_\w+) (\d+)mW/(\d+)mW")


def parse_line(line: str) -> dict:
    d: dict = {}
    if m := RE_TIME.search(line):
        d["time"] = m.group(1)
    if m := RE_RAM.search(line):
        d["ram_used_mb"], d["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    if m := RE_CPU.search(line):
        cores = []
        for c in m.group(1).split(","):
            mm = re.match(r"(\d+)%@(\d+)", c)
            cores.append((int(mm.group(1)), int(mm.group(2))) if mm else (None, None))  # off 的核为 None
        d["cpu_load"] = [c[0] for c in cores]
        d["cpu_freq_mhz"] = cores[0][1]
    if m := RE_EMC.search(line):
        d["emc_load"], d["emc_freq_mhz"] = int(m.group(1)), int(m.group(2) or 0)
    if m := RE_GR3D.search(line):
        # Thor 上格式为 "GR3D_FREQ @[1574,1574,1574]"：每个 GPC 一个频率，没有负载百分比
        d["gpu_load"] = int(m.group(1)) if m.group(1) else None
        d["gpu_freq_mhz"] = [int(x) for x in m.group(2).split(",")]
    d["temps"] = {k: float(v) for k, v in RE_TEMP.findall(line)}
    d["rails_mw"] = {k: (int(cur), int(avg)) for k, cur, avg in RE_RAIL.findall(line)}
    return d


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs" / "tegrastats_host_sample.log"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    print(f"解析 {path}（{len(lines)} 行）\n")
    for ln in lines:
        d = parse_line(ln)
        busy = [x for x in d["cpu_load"] if x is not None]
        print(f"{d['time']}  RAM {d['ram_used_mb']}/{d['ram_total_mb']}MB  "
              f"CPU max {max(busy)}% avg {sum(busy) / len(busy):.1f}% @{d['cpu_freq_mhz']}MHz  "
              f"EMC {d['emc_load']}%@{d['emc_freq_mhz']}  GPU load={d['gpu_load']} freq={d['gpu_freq_mhz']}  "
              f"gpu {d['temps'].get('gpu')}C  "
              + "  ".join(f"{k} {c / 1000:.1f}W(avg {a / 1000:.1f})" for k, (c, a) in d["rails_mw"].items()))


if __name__ == "__main__":
    main()
