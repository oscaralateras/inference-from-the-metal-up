"""Stages 1 and 2 — the collective's cost curve, and where decode sits on it.

    # rehearsal, any machine, no GPU:
    uv run python -m topics.t09_interconnects.measure --backend gloo --world-sizes 2,4

    # the real thing, on the pod:
    uv run python -m topics.t09_interconnects.measure --backend nccl --world-sizes 2,4

The harness is backend-agnostic, exactly as T5's is: `--backend gloo` runs the identical code over
CPU processes so every path except the NCCL calls themselves can be exercised before the meter
starts. The CPU numbers are rehearsal and are never published — gloo over loopback has a different
alpha and no ring at all.

What it does, in order:

1. **Topology gate.** Declared (`nvidia-smi topo -m`) then empirical (a 256 MB all-reduce). Aborts
   on either. This runs before anything is measured because its whole job is to prevent a PCIe
   node from producing publishable-looking numbers.
2. **The sweep.** All-reduce from 1 KB to 1 GB at each world size, timed with CUDA events, in
   steady state. Fit `t = alpha + n/beta`.
3. **The decode operating points.** Evaluate the fit at the payload sizes a real decode step
   actually generates, and report what fraction of each call is fixed cost.

Then it scores the four pre-registered bands and writes everything to `results/interconnect.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from arch_common.timing import time_op
from topics.t09_interconnects.model import (
    BATCH_WALK,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    allreduce_bytes,
    bus_gbps,
    comms_per_token_us,
    fit_ring_cost,
    predicted_tp_speedup,
    sweep_sizes,
)
from topics.t09_interconnects.predict import (
    ALPHA_SCALING_TOLERANCE,
    MIN_DECODE_ALPHA_SHARE,
    MIN_SHARE_OF_LINK_SPEC,
    NVLINK_GBPS_PER_LINK,
    build_prediction,
    t6_budget,
)
from topics.t09_interconnects.topology import (
    SMOKE_TEST_BYTES,
    check_declared,
    check_empirical,
    format_topology,
    read_topo,
)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "interconnect.csv"
TOPOLOGY_PATH = RESULTS_DIR / "topology.txt"

# Back-to-back collectives inside one timing window, for the same reason T8 batches its launches:
# a small all-reduce takes single-digit microseconds, so timing one in isolation reports the
# dispatch path rather than the collective. It is also the faithful setup — a TP decode step fires
# 56 of these in sequence and never pays for one alone.
CALLS_PER_TIMING = 16

# Collectives are barriers, so the cost of one is the cost to its *slowest* rank. Timing rank 0
# alone would report whichever rank happened to be luckiest.
REDUCE_ACROSS_RANKS = "max"


@dataclass(frozen=True)
class SweepResult:
    """One world size's worth of measurements, gathered on rank 0."""

    world: int
    sizes: list[int]
    times_us: list[float]
    smoke_bus_gbps: float
    nvlink_width: int
    device_name: str


def _time_allreduce(buf: torch.Tensor, device: torch.device, *, inner: int) -> float:
    """Microseconds for one all-reduce of `buf`, taken as the slowest rank's time.

    `dist.barrier()` before the timed window so ranks that arrived early do not have their wait
    folded into the measurement — without it the first timed call absorbs the spread in process
    start-up and reads several milliseconds high.
    """
    dist.barrier()
    ms = time_op(lambda: dist.all_reduce(buf), device, inner=inner)
    us = torch.tensor([ms * 1e3], device=device, dtype=torch.float64)
    dist.all_reduce(us, op=dist.ReduceOp.MAX)
    return float(us.item())


def _worker(rank: int, world: int, cfg: dict, out_path: str) -> None:
    backend = cfg["backend"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(cfg["port"]))

    # set_device BEFORE init_process_group. NCCL binds a rank to a device at init, and leaving it
    # to the default puts every rank on cuda:0 — which either deadlocks or silently serialises,
    # and the silent case would report a number that looks like a very slow interconnect.
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world)

    dtype = torch.bfloat16 if backend == "nccl" else torch.float32
    element = torch.finfo(dtype).bits // 8

    # The empirical half of the topology gate, first and on its own, so a bad node is abandoned
    # before it has consumed a sweep's worth of rented minutes.
    smoke = torch.ones(SMOKE_TEST_BYTES // element, dtype=dtype, device=device)
    smoke_us = _time_allreduce(smoke, device, inner=1)
    smoke_bus = bus_gbps(SMOKE_TEST_BYTES, world, smoke_us)
    del smoke
    if backend == "nccl":
        torch.cuda.empty_cache()

    sizes: list[int] = []
    times: list[float] = []
    for nbytes in cfg["sizes"]:
        numel = max(1, nbytes // element)
        buf = torch.ones(numel, dtype=dtype, device=device)
        # Large messages take milliseconds, so batching launches inside them wastes wall clock on
        # a metered box and buys nothing — the launch cost is already negligible against them.
        inner = CALLS_PER_TIMING if nbytes <= 1024 * 1024 else 1
        times.append(_time_allreduce(buf, device, inner=inner))
        sizes.append(numel * element)
        del buf
        if backend == "nccl":
            torch.cuda.empty_cache()

    if rank == 0:
        topo = read_topo() if backend == "nccl" else None
        Path(out_path).write_text(
            json.dumps(
                {
                    "world": world,
                    "sizes": sizes,
                    "times_us": times,
                    "smoke_bus_gbps": smoke_bus,
                    "nvlink_width": topo.nvlink_width(world) if topo else 0,
                    "device_name": (
                        torch.cuda.get_device_name(device) if backend == "nccl" else "cpu"
                    ),
                }
            )
        )

    dist.destroy_process_group()


def run_world(world: int, cfg: dict) -> SweepResult:
    """Spawn `world` ranks, sweep, and return rank 0's observations."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world, cfg, out_path), nprocs=world, join=True
    )
    return SweepResult(**json.loads(Path(out_path).read_text()))


def _rows(session: str, result: SweepResult, fit, prediction) -> list[dict[str, object]]:
    """Long-format rows: the sweep, the fit, and the decode operating points."""
    world = result.world
    rows: list[dict[str, object]] = []

    for nbytes, us in zip(result.sizes, result.times_us, strict=True):
        rows.append(
            {
                "session_id": session,
                "experiment": "sweep",
                "variant": f"world{world}",
                "x": nbytes,
                "metric": "allreduce_us",
                "value": us,
            }
        )
        rows.append(
            {
                "session_id": session,
                "experiment": "sweep",
                "variant": f"world{world}",
                "x": nbytes,
                "metric": "bus_gbps",
                "value": bus_gbps(nbytes, world, us),
            }
        )

    for metric, value in (
        ("alpha_us", fit.alpha_us),
        ("alpha_step_us", fit.alpha_step_us),
        ("beta_gbps", fit.beta_gbps),
        ("fit_r_squared", fit.r_squared),
        ("crossover_bytes", fit.crossover_bytes()),
    ):
        rows.append(
            {
                "session_id": session,
                "experiment": "fit",
                "variant": f"world{world}",
                "x": 0,
                "metric": metric,
                "value": value,
            }
        )

    weight_share, _, step_ms = prediction
    for batch in BATCH_WALK:
        nbytes = allreduce_bytes(batch, DEFAULT_HIDDEN)
        per_token = comms_per_token_us(fit, batch, layers=DEFAULT_LAYERS, hidden=DEFAULT_HIDDEN)
        for metric, value in (
            ("allreduce_us", fit.time_us(nbytes)),
            ("alpha_share", fit.alpha_share(nbytes)),
            ("comms_us_per_token", per_token),
            ("tp_speedup", predicted_tp_speedup(weight_share, world, step_ms, per_token)),
        ):
            rows.append(
                {
                    "session_id": session,
                    "experiment": "decode",
                    "variant": f"world{world}",
                    "x": batch,
                    "metric": metric,
                    "value": value,
                }
            )

    return rows


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-sizes", default="2,4")
    parser.add_argument("--port", type=int, default=29509)
    parser.add_argument("--max-bytes", type=int, default=1024**3)
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="rehearsal only — never pass this on a node whose numbers will be published",
    )
    args = parser.parse_args()

    worlds = [int(w) for w in args.world_sizes.split(",")]
    if any(w < 2 for w in worlds):
        raise SystemExit("a collective needs world >= 2; world 1 is tp_matmul.py's control")

    sizes = sweep_sizes(max_bytes=args.max_bytes)
    session = load_profile().session_id if args.backend == "nccl" else "rehearsal"
    weight_share, tokens_per_sec, step_ms = t6_budget()
    prediction = build_prediction(session_id=session)

    print(f"T9 — {args.backend}, worlds {worlds}, {len(sizes)} message sizes\n")

    fits = {}
    for world in worlds:
        if args.backend == "nccl" and not args.skip_gate:
            topo = check_declared(world)
            TOPOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOPOLOGY_PATH.write_text(format_topology(topo, world) + "\n")
            print(
                f"topology gate (declared) PASSED for world={world}: NV{topo.nvlink_width(world)}"
            )

        result = run_world(world, {"backend": args.backend, "port": args.port, "sizes": sizes})

        if args.backend == "nccl" and not args.skip_gate:
            check_empirical(result.smoke_bus_gbps, world)
            print(
                f"topology gate (empirical) PASSED for world={world}: "
                f"{result.smoke_bus_gbps:,.1f} GB/s bus on a "
                f"{SMOKE_TEST_BYTES / 1024**2:.0f} MB all-reduce"
            )

        fit = fit_ring_cost(world, list(zip(result.sizes, result.times_us, strict=True)))
        fits[world] = (fit, result)

        print(
            f"\nworld {world}: alpha {fit.alpha_us:.2f} us "
            f"({fit.alpha_step_us:.2f} us/hop), beta {fit.beta_gbps:,.1f} GB/s, "
            f"R^2 {fit.r_squared:.4f}"
        )
        print(f"  crossover at {fit.crossover_bytes() / 1024:,.0f} KB")
        for batch in (1, 8, 32, 128):
            nbytes = allreduce_bytes(batch, DEFAULT_HIDDEN)
            per_token = comms_per_token_us(fit, batch)
            speedup = predicted_tp_speedup(weight_share, world, step_ms, per_token)
            print(
                f"  decode batch {batch:>3}: {nbytes:>9,} B  "
                f"{fit.time_us(nbytes):>7.2f} us  alpha {fit.alpha_share(nbytes):>6.1%}  "
                f"-> {speedup:.2f}x"
            )

        append_rows(CSV_PATH, _rows(session, result, fit, (weight_share, tokens_per_sec, step_ms)))

    _score_bands(fits, prediction)


def _score_bands(fits: dict, prediction) -> None:
    """Report every pre-registered band, WITHIN or OUTSIDE. Failures are results."""
    print("\n" + "=" * 78)
    print("pre-registered bands")
    print("=" * 78)

    if 2 in fits and 4 in fits:
        ratio = fits[4][0].alpha_us / fits[2][0].alpha_us
        expected = prediction.alpha_scaling_2_to_4
        ok = abs(ratio - expected) / expected <= ALPHA_SCALING_TOLERANCE
        print(
            f"(1) alpha(4)/alpha(2): predicted {expected:.1f}, measured {ratio:.2f}  "
            f"{'WITHIN' if ok else 'OUTSIDE'}"
        )
    else:
        print("(1) alpha scaling: SKIPPED (needs both world 2 and world 4)")

    for world, (fit, result) in sorted(fits.items()):
        if result.nvlink_width:
            spec = result.nvlink_width * NVLINK_GBPS_PER_LINK
            share = fit.beta_gbps / spec
            ok = share >= MIN_SHARE_OF_LINK_SPEC
            print(
                f"(2) world {world} beta {fit.beta_gbps:,.0f} vs NV{result.nvlink_width} spec "
                f"{spec:,.0f} GB/s = {share:.1%}  {'WITHIN' if ok else 'OUTSIDE'}"
            )
        else:
            print(f"(2) world {world}: SKIPPED (no NVLink width reported)")

        share = fit.alpha_share(allreduce_bytes(1, DEFAULT_HIDDEN))
        ok = share >= MIN_DECODE_ALPHA_SHARE
        print(
            f"(3) world {world} batch-1 all-reduce is {share:.1%} fixed cost  "
            f"{'WITHIN' if ok else 'OUTSIDE'}"
        )

    print("(4) TP-matmul agreement: run tp_matmul.py")


if __name__ == "__main__":
    _main()
