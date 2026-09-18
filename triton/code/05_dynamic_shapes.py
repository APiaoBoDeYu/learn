"""Chapter 5, Example 2: dynamic shapes and recompilation.

torch.compile specializes compiled code to the input shapes it has *seen*,
guarded by shape checks. A new shape triggers a recompile unless you tell
Dynamo up front that a dimension is dynamic. This is the JIT counterpart to
TensorRT's optimization profiles (min/opt/max shape ranges chosen ahead of
time at engine-build time).
Run: TORCH_LOGS=recompiles python3 05_dynamic_shapes.py
"""
from common import device_info

import torch
import torch.nn as nn


if __name__ == "__main__":
    device_info()

    model = nn.Linear(128, 128).cuda().eval()
    compiled = torch.compile(model)

    print("=== Static shapes: a new batch size triggers a recompile each time ===")
    with torch.no_grad():
        for batch in [8, 16, 32]:
            x = torch.randn(batch, 128, device="cuda")
            compiled(x)  # first call per shape compiles; watch TORCH_LOGS=recompiles output
    print("(compare recompile log lines above across the three batch sizes)\n")

    print("=== torch._dynamo.mark_dynamic: mark batch dim dynamic up front ===")
    torch._dynamo.reset()
    model2 = nn.Linear(128, 128).cuda().eval()
    compiled2 = torch.compile(model2)
    with torch.no_grad():
        x0 = torch.randn(8, 128, device="cuda")
        torch._dynamo.mark_dynamic(x0, 0)  # dim 0 (batch) is dynamic
        compiled2(x0)  # this compile handles a *range* of batch sizes
        for batch in [16, 32, 64]:
            x = torch.randn(batch, 128, device="cuda")
            compiled2(x)  # should NOT trigger a new recompile
    print("(with mark_dynamic, no further recompiles should appear above)")
