"""Stage 3 — does the microbenchmark's model survive contact with real work?

    uv run python -m topics.t09_interconnects.tp_matmul --backend nccl --world-sizes 1,2,4

`measure.py` fits `t = alpha + n/beta` to an all-reduce running on its own, with nothing else on
the GPU and the same buffer every time. That is the cleanest possible setting and therefore the
most flattering one. A collective inside a real tensor-parallel layer is a different animal: it
follows a matmul that has just left its output in L2, it contends with that matmul's memory
traffic, and it is a barrier every rank must reach.

So this stage runs the actual thing — a **row-parallel down-projection**, the second of the two
all-reduces a TP transformer block performs — and asks whether the model predicts its comms cost.
Band (4) is scored here, and it is the only band that can genuinely fail for an interesting
reason.

The shape is Qwen2.5-7B's MLP down-projection (K=18944, N=3584), the same shape T7 and T8 use, so
the three topics describe one model rather than three. Splitting is along K, which is what makes
it row-parallel: each rank owns a slice of the contraction dimension, computes a **partial sum**
over the full output, and the partials must be added before the next RMSNorm can run. That
addition is the all-reduce, and it is why the payload is `M x N` and does not depend on K at all —
sharding wider does not shrink the message, only the matmul.

World size 1 is the control: identical code path, no collective, so the comparison isolates the
communication rather than comparing two different programs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar
from arch_common.timing import time_op
from topics.t09_interconnects.measure import CSV_PATH
from topics.t09_interconnects.model import DEFAULT_HIDDEN, RingCost, allreduce_bytes
from topics.t09_interconnects.predict import TP_MODEL_TOLERANCE
from topics.t09_interconnects.topology import check_declared

# Qwen2.5-7B's MLP down-projection, matching T7's shapes module.
DEFAULT_INTERMEDIATE = 18944

# The decode batches to walk. Kept short: this stage is the expensive one and every point is a
# full spawn across every world size.
DEFAULT_BATCHES = (1, 8, 32, 128)

CALLS_PER_TIMING = 16


def _worker(rank: int, world: int, cfg: dict, out_path: str) -> None:
    backend = cfg["backend"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(cfg["port"]))

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")

    if world > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world)

    dtype = torch.bfloat16 if backend == "nccl" else torch.float32
    hidden, inter = cfg["hidden"], cfg["intermediate"]
    if inter % world:
        raise ValueError(f"intermediate {inter} does not divide across {world} ranks")
    k_local = inter // world

    # Each rank owns a K-slice. Weights are synthetic: the byte budget and the collective's size
    # depend on the shape, not the values, exactly as in T7 and T8.
    gen = torch.Generator(device="cpu").manual_seed(9 + rank)
    w = torch.randn(hidden, k_local, generator=gen).to(device=device, dtype=dtype)

    out: dict[str, dict[str, float]] = {}
    for batch in cfg["batches"]:
        x = torch.randn(batch, k_local, generator=gen).to(device=device, dtype=dtype)
        partial = torch.empty(batch, hidden, device=device, dtype=dtype)

        def matmul(x=x, w=w, partial=partial) -> None:
            torch.matmul(x, w.t(), out=partial)

        def full(x=x, w=w, partial=partial) -> None:
            torch.matmul(x, w.t(), out=partial)
            if world > 1:
                dist.all_reduce(partial)

        if world > 1:
            dist.barrier()
        matmul_ms = time_op(matmul, device, inner=CALLS_PER_TIMING)
        if world > 1:
            dist.barrier()
        full_ms = time_op(full, device, inner=CALLS_PER_TIMING)

        # The collective's cost as the layer actually experiences it: what the step costs with it
        # minus what the same step costs without it. Timing the all-reduce alone would measure it
        # in isolation again, which is precisely the thing this stage exists to stop doing.
        #
        # Pinned to exactly zero at world 1, where the two timed closures are the same code and
        # their difference is nothing but timing noise. Reporting that noise as a communication
        # cost for a run that performs no communication would be a small lie in the control row.
        comms_us = max(0.0, (full_ms - matmul_ms) * 1e3) if world > 1 else 0.0

        if world > 1:
            stats = torch.tensor([matmul_ms, full_ms, comms_us], device=device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            matmul_ms, full_ms, comms_us = (float(v) for v in stats)

        out[str(batch)] = {
            "matmul_us": matmul_ms * 1e3,
            "full_us": full_ms * 1e3,
            "comms_us": comms_us,
            "comms_share": comms_us / (full_ms * 1e3) if full_ms > 0 else 0.0,
        }
        del x, partial

    if rank == 0:
        Path(out_path).write_text(json.dumps({"world": world, "batches": out}))

    if world > 1:
        dist.destroy_process_group()


def run_world(world: int, cfg: dict) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world, cfg, out_path), nprocs=world, join=True
    )
    return json.loads(Path(out_path).read_text())


def _load_fit(world: int) -> RingCost | None:
    """Recover `measure.py`'s fitted model for this world size, if it has been run."""
    if not CSV_PATH.exists():
        return None
    rows = read_rows(CSV_PATH)
    try:
        return RingCost(
            world=world,
            alpha_us=scalar(rows, "fit", f"world{world}", "alpha_us"),
            beta_gbps=scalar(rows, "fit", f"world{world}", "beta_gbps"),
            r_squared=scalar(rows, "fit", f"world{world}", "fit_r_squared"),
            n_points=0,
        )
    except KeyError:
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-sizes", default="1,2,4")
    parser.add_argument("--port", type=int, default=29517)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    worlds = [int(w) for w in args.world_sizes.split(",")]
    session = load_profile().session_id if args.backend == "nccl" else "rehearsal"
    cfg = {
        "backend": args.backend,
        "port": args.port,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "batches": list(DEFAULT_BATCHES),
    }

    print(f"T9 stage 3 — row-parallel down-projection, K={args.intermediate}, N={args.hidden}\n")

    baseline: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for world in worlds:
        if args.backend == "nccl" and world > 1 and not args.skip_gate:
            check_declared(world)

        result = run_world(world, cfg)
        print(f"world {world}:")
        for batch, stats in sorted(result["batches"].items(), key=lambda kv: int(kv[0])):
            if world == 1:
                baseline[batch] = stats["full_us"]
            speedup = baseline.get(batch, 0.0) / stats["full_us"] if stats["full_us"] else 0.0
            print(
                f"  batch {int(batch):>3}: matmul {stats['matmul_us']:>8.1f} us  "
                f"step {stats['full_us']:>8.1f} us  comms {stats['comms_us']:>7.1f} us "
                f"({stats['comms_share']:>5.1%})  speedup {speedup:.2f}x"
            )
            for metric in ("matmul_us", "full_us", "comms_us", "comms_share"):
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "tp_matmul",
                        "variant": f"world{world}",
                        "x": int(batch),
                        "metric": metric,
                        "value": stats[metric],
                    }
                )
            if speedup:
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "tp_matmul",
                        "variant": f"world{world}",
                        "x": int(batch),
                        "metric": "tp_speedup",
                        "value": speedup,
                    }
                )

        fit = _load_fit(world)
        if fit and world > 1:
            print(f"  band (4) against the fitted model (alpha {fit.alpha_us:.2f} us):")
            for batch, stats in sorted(result["batches"].items(), key=lambda kv: int(kv[0])):
                predicted = fit.time_us(allreduce_bytes(int(batch), args.hidden))
                # `time_op` already divides by `inner`, so this is per-call, as the fit is.
                measured = stats["comms_us"]
                ratio = measured / predicted if predicted else 0.0
                ok = 1 / TP_MODEL_TOLERANCE <= ratio <= TP_MODEL_TOLERANCE
                print(
                    f"    batch {int(batch):>3}: predicted {predicted:>7.2f} us  "
                    f"measured {measured:>7.2f} us  {ratio:>5.2f}x  "
                    f"{'WITHIN' if ok else 'OUTSIDE'}"
                )
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "tp_model_check",
                        "variant": f"world{world}",
                        "x": int(batch),
                        "metric": "measured_over_predicted",
                        "value": ratio,
                    }
                )

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    _main()
