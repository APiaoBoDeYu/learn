"""01-01 jtop Python API：读取 Jetson 的所有系统信息（相当于 jtop 界面各页签的内容）。

运行：/opt/venv/bin/python 01_system_monitor/01_jtop_snapshot.py
说明：容器内的 jtop 通过挂载进来的 /run/jtop.sock 连接宿主机上的 jtop 服务。
"""
from jtop import jtop


def main():
    with jtop() as j:
        print("=== INFO 页：板卡与软件版本 ===")
        hw = j.board["hardware"]
        print(f"Model    : {hw.get('Model')}")
        print(f"SoC      : {hw.get('SoC')}   CUDA Arch: {hw.get('CUDA Arch BIN')}")
        print(f"L4T      : {hw.get('L4T')}   JetPack: {hw.get('Jetpack')}")
        libs = j.board.get("libraries", {})
        print("Libraries:", {k: v for k, v in libs.items() if v})

        print("\n=== CTRL 页：功耗模式与锁频 ===")
        print(f"nvpmodel      : {j.nvpmodel}    (可选模式: {[str(m) for m in j.nvpmodel.models]})")
        print(f"jetson_clocks : {j.jetson_clocks}")

        print("\n=== GPU 页 ===")
        for name, g in j.gpu.items():
            f = g["freq"]
            print(f"{name}: load={g['status']['load']}%  freq={f['cur'] // 1000} MHz "
                  f"(min {f['min'] // 1000} / max {f['max'] // 1000})  governor={f['governor']}")

        print("\n=== CPU 页 ===")
        cores = j.cpu["cpu"]
        c0 = cores[0]
        print(f"{len(cores)} cores, governor={c0['governor']}, freq={c0['freq']['cur'] // 1000} MHz")
        busy = [round(100 - c["idle"], 1) for c in cores if c["online"]]
        print("per-core busy %:", busy)

        print("\n=== MEM 页 ===")
        ram = j.memory["RAM"]
        print(f"RAM  used {ram['used'] / 1024 ** 2:.1f} / {ram['tot'] / 1024 ** 2:.1f} GB  "
              f"(cached {ram['cached'] / 1024 ** 2:.1f} GB)  ← Jetson 统一内存：CPU 和 GPU 共用这一份")
        emc = j.memory["EMC"]
        print(f"EMC  freq {emc['cur'] // 1000} MHz (max {emc['max'] // 1000})  load={j.stats.get('EMC')}%")
        print(f"SWAP {j.memory['SWAP']['used']} / {j.memory['SWAP']['tot']}")

        print("\n=== 功耗（右下角 Power 区域）===")
        for rail, v in j.power["rail"].items():
            print(f"{rail:<18s} {v['power'] / 1000:6.2f} W   (avg {v['avg'] / 1000:6.2f} W)")
        print(f"{'TOTAL':<18s} {j.power['tot']['power'] / 1000:6.2f} W")

        print("\n=== 温度 ===")
        print({k: round(v["temp"], 1) for k, v in j.temperature.items() if v.get("online") and v["temp"] > -100})

        print("\n=== ENG 页：硬件加速引擎 ===")
        for group, engs in j.engine.items():
            for name, e in engs.items():
                state = f"{e['cur'] // 1000} MHz" if e["online"] else "OFF"
                print(f"{group:<6s} {name:<32s} {state}")

        print("\n=== 进程（ALL 页的进程表）===")
        for p in j.processes[:8]:
            # [pid, user, gpu, type, priority, state, cpu%, mem(KB), gpu_mem(KB), name]
            print(f"pid={p[0]:<8} {p[9]:<20s} cpu={p[6]:5.1f}%  mem={p[7] / 1024:8.1f}MB  gpu_mem={p[8] / 1024:8.1f}MB")


if __name__ == "__main__":
    main()
