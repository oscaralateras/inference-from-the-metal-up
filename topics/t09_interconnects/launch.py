"""Is alpha launch overhead? Test it by removing the launches.

    uv run python -m topics.t09_interconnects.launch --backend nccl --world-sizes 2,4

The main result says a decode-sized all-reduce costs ~35 us of which the ring's hops explain 1.5%,
and attributes the rest to launch and synchronisation. That attribution was an **inference** — it
rested on alpha being flat across world size plus NCCL reporting a single channel. Neither observes
launch cost. This module measures it, by the most direct means available:

**Capture the collective into a CUDA graph and replay it.** A graph replay submits the whole
recorded sequence with one launch instead of one launch per operation, so whatever part of alpha is
per-call dispatch cost disappears; whatever part is genuinely on the wire or in the kernel does not.
The prediction is therefore sharp and falsifiable in both directions:

* if alpha is launch-bound, graph replay collapses it
* if alpha is ring latency or protocol synchronisation, graph replay changes almost nothing

This is a **causal** test rather than a profiler reading, which is why it is preferred here: it
intervenes on the suspected mechanism instead of inspecting a timeline and inferring from it. It is
also the same instrument T6 used to find CUDA graphs worth 15-36% of a decode step, and the one T11
is built around — so a result here is directly comparable with both.

The chain is timed as well as the single call. A decode step fires 56 collectives back to back, and
capturing the whole chain into one graph is what a serving engine actually does; measuring only a
single captured call would understate what the technique buys.

Deliberately **not** an Nsight capture. `nsys` is not present in every pod image, and a timeline
would still need interpreting — "this region is launch" is a judgement call, whereas "removing the
launches removed 30 us" is a measurement.
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

# Collectives per token under tensor parallelism — the chain a serving engine captures as one graph.
CHAIN_LENGTH = DEFAULT_LAYERS * ALLREDUCES_PER_LAYER

CALLS_PER_TIMING = 16


def _capture(fn, device: torch.device) -> torch.cuda.CUDAGraph:
    """Record `fn` into a CUDA graph on a side stream, then return the replayable graph.

    NCCL collectives are capturable, but only from a non-default stream and only after the
    communicator has been warmed up — a first call inside capture tries to build the communicator,
    which allocates and synchronises, and capture rejects both. The warmup below is therefore load
    bearing rather than hygiene.
    """
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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

        def chain(buf=buf) -> None:
            for _ in range(CHAIN_LENGTH):
                dist.all_reduce(buf)

        dist.barrier()
        eager_us = time_op(one, device, inner=CALLS_PER_TIMING) * 1e3

        dist.barrier()
        graph_one = _capture(one, device)
        # Bound as a default argument rather than captured, like every other closure here: a bare
        # `lambda: graph_one.replay()` closes over the loop variable, so it would replay whichever
        # graph the loop happened to be holding when it ran rather than this batch's.
        graphed_us = time_op(lambda g=graph_one: g.replay(), device, inner=CALLS_PER_TIMING) * 1e3

        dist.barrier()
        eager_chain_us = time_op(chain, device, inner=1) * 1e3

        dist.barrier()
        graph_chain = _capture(chain, device)
        graphed_chain_us = time_op(lambda g=graph_chain: g.replay(), device, inner=1) * 1e3

        stats = torch.tensor(
            [eager_us, graphed_us, eager_chain_us, graphed_chain_us],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        eager_us, graphed_us, eager_chain_us, graphed_chain_us = (float(v) for v in stats)

        out[str(batch)] = {
            "eager_us": eager_us,
            "graphed_us": graphed_us,
            "launch_us": max(0.0, eager_us - graphed_us),
            "launch_share": max(0.0, eager_us - graphed_us) / eager_us if eager_us else 0.0,
            "eager_chain_us": eager_chain_us,
            "graphed_chain_us": graphed_chain_us,
            "chain_speedup": eager_chain_us / graphed_chain_us if graphed_chain_us else 0.0,
        }

        # Not `del`ed: the graphs are rebound next iteration and freed then. Deleting them here
        # would leave the names unbound, which is what the default-arg binding above exists to
        # avoid depending on.
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
        "--backend", choices=("nccl",), default="nccl", help="CUDA graphs need CUDA"
    )
    args = parser.parse_args()

    session = load_profile().session_id
    rows: list[dict[str, object]] = []

    print("T9 — is alpha launch overhead? Capture the collective and see what survives.\n")
    print(f"chain length {CHAIN_LENGTH} collectives (= one token under TP)\n")

    for world in [int(w) for w in args.world_sizes.split(",")]:
        result = run_world(
            world,
            {"port": args.port, "hidden": args.hidden, "batches": list(DEFAULT_BATCHES)},
        )
        print(f"world {world}:")
        for batch, s in sorted(result["batches"].items(), key=lambda kv: int(kv[0])):
            print(
                f"  batch {int(batch):>3}: eager {s['eager_us']:>7.2f} us  "
                f"graphed {s['graphed_us']:>7.2f} us  "
                f"launch {s['launch_us']:>7.2f} us ({s['launch_share']:>5.1%})   "
                f"| chain {s['eager_chain_us']:>8.1f} -> {s['graphed_chain_us']:>8.1f} us "
                f"({s['chain_speedup']:.2f}x)"
            )
            for metric, value in s.items():
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
