"""08-03 每个阶段的推理输出与 S0（fp32 eager）的最大绝对误差（同一帧输入；ToyVLA 输出范围 [-1, 1]）

    /opt/venv/bin/python 08_case_study/03_accuracy.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case  # noqa: E402

lines = []
ref = None
for name, mk in case.STAGES:
    pipe = mk()
    for i in range(3):
        pipe.step(i)
    with torch.no_grad():
        out = pipe.infer(pipe.preprocess(pipe.capture(0))).float().clone()
    ref = out if ref is None else ref
    lines.append(f"{name:<34s} max|out - S0| = {(out - ref).abs().max().item():.2e}")
    print(lines[-1])
    del pipe
(case.OUT / "03_accuracy.log").write_text("\n".join(lines) + "\n")
