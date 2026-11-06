"""06-01 导出 ONNX：TinyCNN（batch=8）和 ToyVLA（batch=1），并保存 PyTorch 参考输出。

    /opt/venv/bin/python 06_tensorrt/01_export_onnx.py

产物（outputs/）：
    tinycnn_b8.onnx / toyvla_b1.onnx      模型（opset 17，静态 shape）
    *_ref.npz                             固定的输入和 PyTorch fp32 输出（后面比较 TensorRT 的精度）
说明：
    - 用 torch.onnx.export(dynamo=False)（TorchScript 导出器）。新镜像中没有 onnxscript，dynamo 导出器不可用。
    - 导出前关闭 TransformerEncoderLayer 的 fastpath（torch.backends.mha.set_fastpath_enabled(False)），
      否则 eval + no_grad 下会走 _transformer_encoder_layer_fwd 这个融合算子，导出器不认识。
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TinyCNN, ToyVLA  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)


def build(name):
    """固定随机种子构造模型和输入，05_compare.py 会用同样的方式重建，保证权重一致。"""
    torch.manual_seed(0)
    if name == "tinycnn_b8":
        m = TinyCNN().cuda().eval()
        inputs = (torch.randn(8, 3, 224, 224, device="cuda"),)
        names = ["image"]
    else:
        m = ToyVLA().cuda().eval()
        inputs = (torch.randn(1, 3, 224, 224, device="cuda"), torch.randint(0, 32000, (1, 32), device="cuda"))
        names = ["image", "tokens"]
    return m, inputs, names


def main():
    torch.backends.mha.set_fastpath_enabled(False)
    for name in ("tinycnn_b8", "toyvla_b1"):
        m, inputs, names = build(name)
        with torch.no_grad():
            ref = m(*inputs)
        path = OUT / f"{name}.onnx"
        t0 = time.time()
        torch.onnx.export(m, inputs, str(path), input_names=names, output_names=["out"], opset_version=17, dynamo=False)
        g = onnx.load(str(path)).graph
        ops = Counter(n.op_type for n in g.node)
        np.savez(OUT / f"{name}_ref.npz", **{n: x.cpu().numpy() for n, x in zip(names, inputs)}, out=ref.cpu().numpy())
        print(f"[{name}] {path.name}  {path.stat().st_size / 1e6:.1f} MB  导出 {time.time() - t0:.1f} s  "
              f"输出 {tuple(ref.shape)}  ONNX 节点 {len(g.node)} 个")
        print("    算子:", ", ".join(f"{k}x{v}" for k, v in ops.most_common(12)))


if __name__ == "__main__":
    main()
