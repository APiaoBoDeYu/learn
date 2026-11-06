"""Chapter 5, Example 3: using a hand-written Triton kernel inside torch.compile.

This is where Triton and torch.compile meet most directly: you write the
kernel yourself (same vector_add kernel from Chapter 2), wrap it as a
`torch.library` custom op so Dynamo/AOTAutograd can trace it as an opaque
node, and torch.compile happily fuses the rest of the graph around it.
Run: python3 05_custom_triton_in_compile.py
"""
from common import device_info, check_close, bench_ms

import torch
import triton
import triton.language as tl


@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(out_ptr + offsets, tl.maximum(z, 0.0), mask=mask)


# torch.library.triton_op / wrap_triton (added in PyTorch 2.5+) tell Dynamo
# "trace this whole call as one op, don't try to peek inside the Triton
# kernel" -- similar in spirit to registering a custom TensorRT plugin so the
# builder treats it as an opaque, fused unit in the graph.
@torch.library.triton_op("tutorial::add_relu", mutates_args={})
def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    torch.library.wrap_triton(add_relu_kernel)[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelWithCustomKernel(torch.nn.Module):
    def forward(self, x, y):
        z = add_relu(x, y)  # our hand-written Triton kernel
        return z * 2 - 1     # ordinary ops Inductor will fuse around it


if __name__ == "__main__":
    device_info()

    x = torch.randn(2**16, device="cuda")
    y = torch.randn(2**16, device="cuda")

    ref = torch.relu(x + y) * 2 - 1
    out_eager = add_relu(x, y) * 2 - 1
    check_close(out_eager, ref, "custom_triton_op_eager")

    model = ModelWithCustomKernel()
    compiled_model = torch.compile(model, fullgraph=True)
    out_compiled = compiled_model(x, y)
    check_close(out_compiled, ref, "custom_triton_op_compiled_fullgraph")

    t_eager = bench_ms(lambda: model(x, y))
    t_compiled = bench_ms(lambda: compiled_model(x, y))
    print(f"eager (custom kernel + 2 ops)    : {t_eager:.4f} ms")
    print(f"compiled (kernel fused with rest): {t_compiled:.4f} ms")
    print("\nfullgraph=True succeeded: Dynamo traced straight through our")
    print("hand-written Triton kernel without a graph break.")
