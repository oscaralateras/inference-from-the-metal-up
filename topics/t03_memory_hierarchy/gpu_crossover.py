"""GPU companion for T3 Experiment 3 — the memory-bound → compute-bound crossover.

Runs the same weight×batch sweep as `bench.c`'s Experiment (c), but on a GPU — where the ridge
point is high enough that single-vector decode (B=1) is genuinely memory-bandwidth-bound. A CPU
cannot show this (its ridge point is too low; see the lab note), so this piece lives on a GPU.

How to run (no repo needed):
  1. Open Google Colab, Runtime → Change runtime type → T4 GPU.
  2. Paste this file's contents into a cell and run it (torch is pre-installed on Colab).
  3. Save the printed CSV as results/crossover_gpu.csv — plot.py picks it up automatically.

Per batch size B it reports throughput (GFLOP/s), HBM bandwidth used (GB/s), and latency (ms/step).
At B=1, bandwidth-used sits near the card's HBM peak (memory-bound); at large B, throughput nears
the compute peak while bandwidth-used falls (compute-bound).
"""

from __future__ import annotations

import statistics

import torch

M = K = 8192  # weights W are M×K; fp16 → 128 MB, far larger than any GPU cache
DTYPE = torch.float16
TRIALS = 30
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def bench(w: torch.Tensor, batch: int) -> tuple[float, float, float]:
    """Return (GFLOP/s, GB/s used, ms/step) for W × X with X of shape (K, batch)."""
    x = torch.randn(K, batch, device="cuda", dtype=DTYPE)
    for _ in range(5):  # warm up cuBLAS (kernel selection / autotune)
        _ = w @ x
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(TRIALS):
        start.record()
        _ = w @ x
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # ms → s
    t = statistics.median(times)

    flops = 2 * M * K * batch
    bytes_moved = (M * K + K * batch + M * batch) * 2  # W + X + C, fp16
    return flops / t / 1e9, bytes_moved / t / 1e9, t * 1e3


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU — set Runtime > Change runtime type > T4 GPU")
    print("device:", torch.cuda.get_device_name(0))
    w = torch.randn(M, K, device="cuda", dtype=DTYPE)
    print("experiment,variant,size,metric,value")
    for b in BATCHES:
        gflops, gbps, ms = bench(w, b)
        print(f"crossover_gpu,batched,{b},gflop_per_s,{gflops:.1f}")
        print(f"crossover_gpu,bandwidth,{b},gb_per_s,{gbps:.1f}")
        print(f"crossover_gpu,latency,{b},ms_per_step,{ms:.3f}")


if __name__ == "__main__":
    main()
