"""Chapter 5, Example 1: observing graph breaks.

torch.jit.trace would silently bake in a Python-level branch's *taken* path
forever. torch.compile instead splits ("breaks") the graph at points it can't
trace through, falls back to eager for that piece, and resumes compiling
after it -- the whole thing still runs correctly, just with less of it fused.
Run: python3 05_graph_break_demo.py
"""
from common import device_info

import torch


def has_graph_break(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    # .item() forces a device->host sync and a concrete Python int, which
    # Dynamo cannot trace symbolically -> this causes a graph break.
    if y.sum().item() > 0:
        z = y + 1
    else:
        z = y - 1
    return z * 2


def no_graph_break(x: torch.Tensor) -> torch.Tensor:
    y = x * 2
    # torch.where keeps everything as tensor ops, so Dynamo can trace the
    # whole function into a single graph -- no break needed.
    z = torch.where(y.sum() > 0, y + 1, y - 1)
    return z * 2


if __name__ == "__main__":
    device_info()

    x = torch.randn(1024, device="cuda")

    print("=== torch._dynamo.explain on a function WITH a graph break ===")
    explanation = torch._dynamo.explain(has_graph_break)(x)
    print(explanation)

    print("\n=== Same, but fullgraph=True raises instead of silently breaking ===")
    try:
        compiled_strict = torch.compile(has_graph_break, fullgraph=True)
        compiled_strict(x)
    except Exception as e:
        print(f"Raised as expected: {type(e).__name__}: {str(e)[:200]}")

    print("\n=== torch._dynamo.explain on the rewritten, break-free function ===")
    explanation2 = torch._dynamo.explain(no_graph_break)(x)
    print(explanation2)

    print("\n=== fullgraph=True succeeds on the rewritten version ===")
    compiled_ok = torch.compile(no_graph_break, fullgraph=True)
    out = compiled_ok(x)
    print("ran fine, output shape:", out.shape)
