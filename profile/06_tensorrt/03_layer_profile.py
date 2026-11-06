"""06-03 TensorRT 内置 profiler（三步法第 1 步："哪层慢？"）+ Engine Inspector（每层用了哪个 kernel、什么格式）

    /opt/venv/bin/python 06_tensorrt/03_layer_profile.py         （需要先运行 02_trtexec.py 生成 engine）

    trtexec --loadEngine=x.engine --dumpProfile --separateProfileRun \
            --exportProfile=x_profile.json --exportLayerInfo=x_layers.json
    - --dumpProfile：逐层计时（每层前后插入同步 + event，所以有额外开销）
    - --separateProfileRun：性能数字来自不挂 profiler 的那一轮，逐层数字来自单独的一轮 → 两者互不干扰
    - --exportLayerInfo：Engine Inspector 的输出，需要构建时 --profilingVerbosity=detailed，否则只有层名
"""
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = ["tinycnn_b8", "toyvla_b1"]
PRECS = ["fp32", "fp16", "int8"]


def profile(model, prec):
    eng = OUT / f"{model}_{prec}.engine"
    pj, lj = OUT / f"03_{model}_{prec}_profile.json", OUT / f"03_{model}_{prec}_layers.json"
    r = subprocess.run(["trtexec", f"--loadEngine={eng}", "--dumpProfile", "--separateProfileRun", f"--exportProfile={pj}",
                        f"--exportLayerInfo={lj}", "--duration=2", "--warmUp=500"], capture_output=True, text=True)
    gpu = float(re.search(r"\] GPU Compute Time: .*?median = ([\d.]+) ms", r.stdout).group(1))
    prof = [x for x in json.load(open(pj)) if "name" in x]
    layers = {l["Name"]: l for l in json.load(open(lj))["Layers"]}
    return gpu, prof, layers


def fmt_io(l, key):
    return ",".join(x.get("Format/Datatype", "?").split(" ")[0] for x in l.get(key, [])[:1])


def main():
    lines = []

    def p(s=""):
        print(s); lines.append(s)

    for model in MODELS:
        for prec in PRECS:
            gpu, prof, layers = profile(model, prec)
            tot = sum(x["averageMs"] for x in prof)
            p(f"\n######## {model} {prec}：{len(prof)} 个 TRT 层；逐层时间之和 {tot:.3f} ms；"
              f"不挂 profiler 时 GPU Compute Time 中位数 {gpu:.3f} ms")
            p(f"  {'avg_ms':>7s} {'%':>5s}  {'层（TRT）':<44s} {'LayerType':<12s} {'in→out':<12s} {'#ONNX':>5s}  tactic / kernel")
            for x in sorted(prof, key=lambda x: -x["averageMs"])[:8]:
                l = layers.get(x["name"], {})
                n_onnx = l.get("Metadata", "").count("[ONNX Layer:")
                p(f"  {x['averageMs']:7.4f} {x['percentage']:5.1f}  {x['name'][:44]:<44s} {l.get('LayerType', '?')[:12]:<12s} "
                  f"{(fmt_io(l, 'Inputs') + '→' + fmt_io(l, 'Outputs'))[:12]:<12s} {n_onnx:5d}  {l.get('TacticName', '?')[:70]}")
            by_type = defaultdict(float)
            for x in prof:
                by_type[layers.get(x["name"], {}).get("LayerType", "?")] += x["averageMs"]
            p("  按 LayerType 汇总: " + "，".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])[:7]))
            refmt = [x for x in prof if re.search(r"reformat|Reformat|CastConc|copy|Copy", x["name"] + layers.get(x["name"], {}).get("LayerType", ""))]
            p(f"  格式/精度转换类的层（名字或类型含 Reformat/Cast/Copy）: {len(refmt)} 个，"
              f"合计 {sum(x['averageMs'] for x in refmt):.4f} ms（{100 * sum(x['averageMs'] for x in refmt) / tot:.1f}%）")
    (OUT / "03_layer_profile.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
