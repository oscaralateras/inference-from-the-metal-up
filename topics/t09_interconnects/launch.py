"""Is alpha host launch overhead? Amortise the launches away and see what moves.

    uv run python -m topics.t09_interconnects.launch --world-sizes 2,4
    uv run python -m topics.t09_interconnects.launch --world-sizes 2,4 --graphs

The main result says a decode-sized all-reduce costs ~34 us of which the ring's hops explain almost
none, and attributes the rest to launch and synchronisation. That attribution was an **inference**
— it rested on alpha being flat across world size plus NCCL reporting a single channel. Neither
observes launch cost. This module measures it.

**The amortisation sweep, and why it is the primary measurement.** Host-side dispatch is a cost
paid once per call *on the CPU*, and it overlaps with the GPU executing the previous call. So
issuing N collectives back to back inside one timed window hides all but the first one's dispatch:
if alpha is host launch cost, per-call time must fall as N rises, and flatten once the queue stays
ahead of the device. If alpha is device-side — protocol synchronisation, flag exchange, the LL
protocol's own handshake — then batching the launches changes nothing at all, because each call
still has to happen on the GPU in sequence.

This reframed the topic's own headline mid-flight, and the reason is worth recording. `measure.py`
already times with 16 back-to-back calls per window, so the alpha it reports is **already** an
amortised-launch number. Host dispatch had therefore already been ruled out before this module was
written, and the lab note's phrase "launch and synchronisation" was carrying an implication about
the launch half that the measurement did not support. This sweep tests it directly instead of
inferring, and it can falsify the claim in either direction.

**CUDA graph replay is the secondary check.** Capturing the collective and replaying it removes the
per-call dispatch entirely rather than merely overlapping it, which is a stronger intervention than
the sweep. It is second rather than first because it deadlocks easily: `torch.cuda.graph` defaults
to `capture_error_mode="global"`, which treats CUDA work from any other thread as illegal, and
NCCL's watchdog thread does exactly that. The first attempt hung two ranks for eleven minutes at
0% GPU utilisation before it was killed. `thread_local` is the fix, and the whole stage is guarded
so a repeat cannot cost another session.

Deliberately **not** an Nsight capture. `nsys` is not present in every pod image, and a timeline
would still need interpreting — "this region is launch" is a judgement call, whereas "batching the
launches changed nothing" is a measurement.
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
from arch_common.results_io import append_rows
from arch_common.timing import time_op
from topics.t09_interconnects.measure import CSV_PATH
from topics.t09_interconnects.model import (
    ALLREDUCES_PER_LAYER,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    allreduce_bytes,
)

# The decode batches to probe. Small: this is about the latency floor, and above the crossover the
# question does not arise because the call is dominated by bytes.
DEFAULT_BATCHES = (1, 8, 32)

# How many collectives share one timed window. At 1 the host pays dispatch for every call with
# nothing to hide it behind; by 64 the queue is far enough ahead of the device that dispatch is
# entirely overlapped. The shape of per-call time across this sweep is the measurement.
INNER_SWEEP = (1, 2, 4, 8, 16, 32, 64)

# Collectives per token under tensor parallelism — the chain a serving engine captures as one graph.
CHAIN_LENGTH = DEFAULT_LAYERS * ALLREDUCES_PER_LAYER


def _capture(fn, device: torch.device) -> torch.cuda.CUDAGraph:
    """Record `fn` into a CUDA graph on a side stream, then return the replayable graph.

    Two things here are load bearing rather than hygiene, and the first attempt at this module
    deadlocked two ranks for eleven minutes by getting the second one wrong:

    * **The warmup.** A first collective inside capture tries to build the communicator, which
      allocates and synchronises, and capture rejects both.
    * **`capture_error_mode="thread_local"`.** The default is `"global"`, which treats CUDA work
      issued from *any* thread during capture as an error. NCCL runs a watchdog thread that does
      exactly that, so capture and watchdog block on each other and neither times out. Scoping the
      check to the capturing thread is what makes NCCL capturable at all.
    """
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        fn()
    return graph


def _worker(rank: int, world: int, cfg: dict, out_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(cfg["port"]))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    out: dict[str, dict[str, float]] = {}
    for batch in cfg["batches"]:
        nbytes = allreduce_bytes(batch, cfg["hidden"])
        buf = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=device)

        def one(buf=buf) -> None:
            dist.all_reduce(buf)

        # The amortisation sweep. `time_op` divides by `inner`, so every entry is a per-call cost
        # and they are directly comparable: a falling curve is dispatch being hidden, a flat one is
        # a cost the device pays whatever the host does.
        per_call: dict[str, float] = {}
        for inner in cfg["inner_sweep"]:
            dist.barrier()
            us = time_op(one, device, inner=inner) * 1e3
            reduced = torch.tensor([us], device=device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
            per_call[str(inner)] = float(reduced.item())

        lo, hi = per_call[str(cfg["inner_sweep"][0])], per_call[str(cfg["inner_sweep"][-1])]
        stats = {
            "percall_single_us": lo,
            "percall_batched_us": hi,
            # > 1 means batching the launches made each call cheaper, i.e. there was host dispatch
            # to hide. ~1 means alpha is device-side and "launch overhead" is the wrong label.
            "amortisation_ratio": lo / hi if hi else 0.0,
        }

        if cfg["graphs"]:
            dist.barrier()
            graph_one = _capture(one, device)
            graphed = (
                time_op(lambda g=graph_one: g.replay(), device, inner=cfg["inner_sweep"][-1]) * 1e3
            )
            reduced = torch.tensor([graphed], device=device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
            graphed = float(reduced.item())
            stats["graphed_us"] = graphed
            stats["graph_speedup"] = hi / graphed if graphed else 0.0

        out[str(batch)] = {"per_call": per_call, **stats}  # type: ignore[dict-item]

        del buf
        torch.cuda.empty_cache()

    if rank == 0:
        Path(out_path).write_text(json.dumps({"world": world, "batches": out}))

    dist.destroy_process_group()


def run_world(world: int, cfg: dict) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world, cfg, out_path), nprocs=world, join=True
    )
    return json.loads(Path(out_path).read_text())


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", default="2,4")
    parser.add_argument("--port", type=int, default=29523)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument(
        "--graphs",
        action="store_true",
        help="also capture and replay the collective. Off by default: NCCL inside CUDA graph "
        "capture deadlocks if anything is slightly wrong, and the sweep alone answers the question",
    )
    args = parser.parse_args()

    session = load_profile().session_id
    rows: list[dict[str, object]] = []

    print("T9 — is alpha host launch overhead? Amortise the launches and see what moves.\n")
    print(f"per-call microseconds, by how many calls share one timed window {INNER_SWEEP}\n")

    for world in [int(w) for w in args.world_sizes.split(",")]:
        result = run_world(
            world,
            {
                "port": args.port,
                "hidden": args.hidden,
                "batches": list(DEFAULT_BATCHES),
                "inner_sweep": list(INNER_SWEEP),
                "graphs": args.graphs,
            },
        )
        print(f"world {world}:")
        for batch, s in sorted(result["batches"].items(), key=lambda kv: int(kv[0])):
            curve = "  ".join(f"{int(i):>2}:{s['per_call'][str(i)]:>6.2f}" for i in INNER_SWEEP)
            line = f"  batch {int(batch):>3}: {curve}   ratio {s['amortisation_ratio']:.2f}x"
            if "graph_speedup" in s:
                line += f"   graphed {s['graphed_us']:.2f} us ({s['graph_speedup']:.2f}x)"
            print(line)

            for inner in INNER_SWEEP:
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "launch_amortisation",
                        "variant": f"world{world}_b{int(batch)}",
                        "x": int(inner),
                        "metric": "percall_us",
                        "value": s["per_call"][str(inner)],
                    }
                )
            for metric, value in s.items():
                if metric == "per_call":
                    continue
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "launch",
                        "variant": f"world{world}",
                        "x": int(batch),
                        "metric": metric,
                        "value": value,
                    }
                )

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    _main()
