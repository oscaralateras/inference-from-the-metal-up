"""Experiment B — the pipeline bubble, predicted vs measured.

Pipeline parallelism splits a model by *depth*: stage 0 owns layers 1-8, stage 1 owns 9-16, and so
on. Items flow through like an assembly line. The structural cost is the **bubble**: while the
first item is still in stage 0, every later stage sits idle, and while the last item drains, every
earlier stage sits idle. With P stages and M microbatches in flight, the theoretical efficiency is

    efficiency = M / (M + P - 1)

which is 4/7 = 57% at M=4, P=4, and 94% at M=64. The bubble is paid down only by having many
microbatches in flight — and that is the whole inference story: **decode batches are small by
construction** (T3), so the thing that fills a pipeline is exactly the thing decode does not have.

This builds a real P-stage pipeline out of worker threads and a queue per seam, sweeps M, and
compares measured efficiency against the prediction. Then it repeats the sweep with one stage made
deliberately 2x slower, because a pipeline runs at the speed of its slowest stage no matter how
many microbatches you feed it — which is why layer partitioning has to be balanced.

    uv run python topics/t05_parallelism/pipeline.py
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import torch
from model import MODEL_ID, TransformerBlock, load_block
from results_io import append_rows
from workload import assert_pinned, pin_single_thread

N_STAGES = 4

# Transformer MLP blocks owned by each pipeline stage — this is what makes the experiment
# *pipeline parallelism over a model* rather than an assembly-line analogy. Stage i owns
# LAYERS_PER_STAGE consecutive layers, exactly as PP partitions a real model by depth.
LAYERS_PER_STAGE = 6

# One microbatch is one sequence of this length. Small on purpose: this is the decode regime,
# where microbatches are small by construction (T3) — precisely why the bubble bites.
SEQ_PER_MICROBATCH = 64

# Microbatch counts to sweep. Spans the regime where the bubble dominates (M < P) through to
# where it is nearly amortised (M >> P).
MICROBATCH_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

# Relative cost of each stage, in units of LAYERS_PER_STAGE. Balanced is the reference;
# imbalanced gives stage 0 twice as many layers, modelling an uneven layer split across devices.
LAYOUTS: dict[str, tuple[int, ...]] = {
    "balanced": (1, 1, 1, 1),
    "imbalanced": (2, 1, 1, 1),
}

SENTINEL = -1


@dataclass(frozen=True)
class PipelineRun:
    """One (microbatches, efficiency) observation for one stage layout."""

    layout: str
    microbatches: int
    wall_seconds: float
    efficiency: float
    predicted_efficiency: float


def predicted_efficiency(weights: tuple[int, ...], microbatches: int) -> float:
    """Efficiency a perfect pipeline would achieve: useful work / (stages x wall).

    A linear pipeline with per-stage costs `weights` and M items has an ideal wall time of

        fill-and-drain (one item through every stage) + (M - 1) x the slowest stage

    because after the first item, the line emits one item per slowest-stage period. For equal
    weights this reduces exactly to the textbook `M / (M + P - 1)`.
    """
    total = sum(weights)
    slowest = max(weights)
    ideal_wall = total + (microbatches - 1) * slowest
    return (microbatches * total) / (len(weights) * ideal_wall)


def build_stages(weights: tuple[int, ...], block: TransformerBlock) -> list[list[TransformerBlock]]:
    """Give each stage its own list of layers — stage i owns weights[i] * LAYERS_PER_STAGE of them.

    Each stage holds its own copy so no two stages share a weight tensor: in real PP the stages
    live on different devices and genuinely hold disjoint layers.
    """
    return [[block for _ in range(w * LAYERS_PER_STAGE)] for w in weights]


def run_pipeline(
    weights: tuple[int, ...], microbatches: int, block: TransformerBlock
) -> tuple[float, float]:
    """Push `microbatches` activations through a real threaded pipeline of transformer layers.

    Each stage is a thread with an inbound queue. It runs its layers over the incoming hidden
    state and forwards the result to the next stage — the activation hand-off of real pipeline
    parallelism. Returns (wall_seconds, checksum).
    """
    n_stages = len(weights)
    queues: list[queue.Queue[torch.Tensor | int]] = [queue.Queue() for _ in range(n_stages + 1)]
    stages = build_stages(weights, block)
    checksums = [0.0] * n_stages

    def stage(idx: int) -> None:
        layers = stages[idx]
        local = 0.0
        while True:
            item = queues[idx].get()
            if isinstance(item, int) and item == SENTINEL:
                queues[idx + 1].put(SENTINEL)  # propagate shutdown down the line
                break
            h = item
            assert isinstance(h, torch.Tensor)
            for layer in layers:
                h = layer.forward(h)
            local += float(h.sum().item())
            queues[idx + 1].put(h)
        checksums[idx] = local

    threads = [threading.Thread(target=stage, args=(i,)) for i in range(n_stages)]
    for t in threads:
        t.start()

    inputs = [
        torch.randn(1, SEQ_PER_MICROBATCH, block.hidden, generator=torch.Generator().manual_seed(m))
        for m in range(microbatches)
    ]

    t0 = time.perf_counter()
    for h in inputs:
        queues[0].put(h)
    queues[0].put(SENTINEL)

    received = 0
    while received < microbatches:
        item = queues[n_stages].get()
        if not (isinstance(item, int) and item == SENTINEL):
            received += 1
    wall = time.perf_counter() - t0

    for t in threads:
        t.join()
    return wall, sum(checksums)


def calibrate_stage_seconds(block: TransformerBlock, n_layers: int, reps: int = 15) -> float:
    """Median seconds for one stage's layers on one microbatch, measured solo on one thread."""
    h0 = torch.randn(
        1, SEQ_PER_MICROBATCH, block.hidden, generator=torch.Generator().manual_seed(7)
    )
    for _ in range(3):  # warm-up
        h = h0
        for _ in range(n_layers):
            h = block.forward(h)
    samples = []
    for _ in range(reps):
        h = h0
        t0 = time.perf_counter()
        for _ in range(n_layers):
            h = block.forward(h)
        float(h.sum().item())
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def sweep(
    layout: str, weights: tuple[int, ...], block: TransformerBlock, layer_seconds: float
) -> list[PipelineRun]:
    """Sweep microbatch count for one stage layout."""
    runs: list[PipelineRun] = []
    for m in MICROBATCH_COUNTS:
        wall, _ = run_pipeline(weights, m, block)
        # Useful work = every stage's layers for every item, at the calibrated solo rate.
        useful = m * sum(weights) * LAYERS_PER_STAGE * layer_seconds
        efficiency = useful / (len(weights) * wall)
        runs.append(PipelineRun(layout, m, wall, efficiency, predicted_efficiency(weights, m)))
    return runs


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Experiment B - the pipeline bubble")
    parser.add_argument("--model", default=MODEL_ID, help="HF repo id for the transformer block")
    args = parser.parse_args()

    pin_single_thread()
    assert_pinned()

    block = load_block(0, args.model)
    layer_s = calibrate_stage_seconds(block, 1)
    stage_ms = layer_s * LAYERS_PER_STAGE * 1e3
    print(
        f"{N_STAGES} stages x {LAYERS_PER_STAGE} transformer layers, "
        f"{SEQ_PER_MICROBATCH} tokens/microbatch"
    )
    print(f"one layer: {layer_s * 1e3:.2f} ms  ->  ~{stage_ms:.1f} ms per stage")
    print("pinned to 1 torch thread\n")

    rows: list[dict[str, object]] = []
    for layout, weights in LAYOUTS.items():
        ceiling = predicted_efficiency(weights, 10**6)
        print(f"--- {layout}  weights={weights}  asymptotic ceiling {ceiling:.3f} ---")
        print(f"{'M':>4} {'wall (s)':>10} {'measured':>10} {'predicted':>10} {'gap':>8}")
        for run in sweep(layout, weights, block, layer_s):
            gap = run.efficiency - run.predicted_efficiency
            print(
                f"{run.microbatches:>4} {run.wall_seconds:>10.3f} "
                f"{run.efficiency:>10.3f} {run.predicted_efficiency:>10.3f} {gap:>+8.3f}"
            )
            for metric, value in (
                ("efficiency", run.efficiency),
                ("predicted_efficiency", run.predicted_efficiency),
                ("wall_seconds", run.wall_seconds),
            ):
                rows.append(
                    {
                        "experiment": "pipeline_bubble",
                        "variant": layout,
                        "workers": run.microbatches,
                        "metric": metric,
                        "value": f"{value:.6f}",
                    }
                )
        print()

    append_rows(rows)


if __name__ == "__main__":
    _main()
