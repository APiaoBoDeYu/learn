"""Chapter 6: comprehensive benchmark across TorchScript / torch.compile /
hand-written Triton / ONNX+TensorRT, all on the same small transformer-style
block (Linear -> GELU -> Linear -> residual -> LayerNorm).

This intentionally reuses the MLP from Chapter 4 and the fused LayerNorm
kernel from Chapter 3 so every number here traces back to code you've
already read. Run: python3 06_benchmark_all.py
"""
from common import device_info, check_close, bench_ms

import subprocess
import time
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


# --- Model under test -------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        out = self.fc2(h)
        return self.ln(out + x)


# --- Path 4: hand-written Triton fused LayerNorm, rest in eager -------------
@triton.jit
def layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, row_stride, n_cols, eps,
                      BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row_ptr = x_ptr + row_idx * row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * weight + bias
    out_row_ptr = out_ptr + row_idx * row_stride
    tl.store(out_row_ptr + col_offsets, y, mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    layernorm_kernel[(n_rows,)](x, weight, bias, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


class BlockWithTritonLN(nn.Module):
    """Same math as Block, but LayerNorm is our hand-written fused Triton kernel."""

    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = torch.nn.functional.gelu(self.fc1(x))
        out = self.fc2(h)
        return triton_layernorm(out + x, self.weight, self.bias)


def run_trtexec_benchmark(onnx_path: str, batch: int) -> float | None:
    """Build a TensorRT engine from ONNX with trtexec and parse its own
    reported mean GPU latency. Returns None if the step fails (reported, not faked)."""
    engine_path = f"/tmp/block_bs{batch}.engine"
    build_cmd = [
        "trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
        f"--shapes=x:{batch}x1024", "--fp16",
    ]
    build = subprocess.run(build_cmd, capture_output=True, text=True, timeout=300)
    if build.returncode != 0:
        print("trtexec build failed:")
        print(build.stdout[-2000:])
        print(build.stderr[-2000:])
        return None

    run_cmd = [
        "trtexec", f"--loadEngine={engine_path}",
        f"--shapes=x:{batch}x1024", "--fp16",
    ]
    run = subprocess.run(run_cmd, capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        print("trtexec run failed:")
        print(run.stderr[-2000:])
        return None

    # trtexec prints a line like: "GPU Compute Time: min = .. ms, max = .. ms, mean = X ms, ..."
    for line in run.stdout.splitlines():
        if "GPU Compute Time" in line and "mean" in line:
            for part in line.split(","):
                part = part.strip()
                if part.startswith("mean"):
                    return float(part.split("=")[1].strip().split()[0])
    print("Could not parse trtexec output; last 2000 chars:")
    print(run.stdout[-2000:])
    return None


if __name__ == "__main__":
    device_info()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    dim, hidden, batch = 1024, 4096, 256
    model = Block(dim, hidden).cuda().eval()
    x = torch.randn(batch, dim, device="cuda")

    results = {}

    with torch.no_grad():
        # --- 1. Eager baseline ---
        out_ref = model(x)
        results["eager"] = bench_ms(lambda: model(x))

        # --- 2. TorchScript (script, not trace, so control flow is preserved) ---
        scripted = torch.jit.script(model)
        out_ts = scripted(x)
        check_close(out_ts, out_ref, "torchscript")
        results["torchscript"] = bench_ms(lambda: scripted(x))

        # --- 3a. torch.compile default ---
        torch._dynamo.reset()
        compiled_default = torch.compile(model, mode="default")
        out_c1 = compiled_default(x)
        check_close(out_c1, out_ref, "torch.compile default")
        results["compile_default"] = bench_ms(lambda: compiled_default(x))

        # --- 3b. torch.compile max-autotune ---
        torch._dynamo.reset()
        compiled_max = torch.compile(model, mode="max-autotune")
        out_c2 = compiled_max(x)
        check_close(out_c2, out_ref, "torch.compile max-autotune")
        results["compile_max_autotune"] = bench_ms(lambda: compiled_max(x))

        # --- 4. Hand-written Triton kernel for LayerNorm, rest eager ---
        model_triton = BlockWithTritonLN(dim, hidden).cuda().eval()
        model_triton.fc1.load_state_dict(model.fc1.state_dict())
        model_triton.fc2.load_state_dict(model.fc2.state_dict())
        model_triton.weight.data.copy_(model.ln.weight.data)
        model_triton.bias.data.copy_(model.ln.bias.data)
        out_triton = model_triton(x)
        check_close(out_triton, out_ref, "hand-written Triton LN")
        results["manual_triton_ln"] = bench_ms(lambda: model_triton(x))

        # --- 5. ONNX export + TensorRT engine via trtexec -------------------
        onnx_path = "/tmp/block.onnx"
        torch.onnx.export(
            model, (x,), onnx_path,
            input_names=["x"], output_names=["out"],
            dynamic_axes={"x": {0: "batch"}, "out": {0: "batch"}},
            opset_version=17,
            dynamo=False,  # legacy TorchScript-tracing exporter; avoids the
                           # onnxscript dependency the new torch.export-based
                           # exporter (dynamo=True, default since torch 2.x) needs
        )
        trt_ms = run_trtexec_benchmark(onnx_path, batch)
        if trt_ms is not None:
            results["onnx_tensorrt_fp16"] = trt_ms

    print("\n=== Summary (batch size =", batch, ", dim =", dim, ") ===")
    print(f"{'path':<28}{'latency (ms)':>14}{'speedup vs eager':>20}")
    baseline = results["eager"]
    for name, ms in results.items():
        print(f"{name:<28}{ms:>14.4f}{baseline / ms:>19.2f}x")
