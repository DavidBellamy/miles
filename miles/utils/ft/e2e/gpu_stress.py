"""Standalone GPU stress workload for E2E fault injection.

Saturates all visible GPUs with continuous matmul operations.
Runs until killed or --duration timeout is reached.
"""
from __future__ import annotations

import argparse
import time


def _stress_loop(duration: float) -> None:
    import torch

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices available")

    size = 4096
    tensors = []
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        tensors.append((a, b, device))

    start = time.monotonic()
    while time.monotonic() - start < duration:
        for a, b, _ in tensors:
            torch.mm(a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU stress workload")
    parser.add_argument("--duration", type=float, default=3600.0, help="Max duration in seconds")
    args = parser.parse_args()
    _stress_loop(duration=args.duration)


if __name__ == "__main__":
    main()
