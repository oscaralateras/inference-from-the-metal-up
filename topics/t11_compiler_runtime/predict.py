"""Phase 0 — commit to the crossover before renting anything.

    uv run python -m topics.t11_compiler_runtime.predict --write     # laptop, no GPU

Two mechanisms, two regimes, one boundary between them. Everything below is derived from byte
counts and from ceilings **other topics already measured** — T7's bandwidth and T9's per-call fixed
cost — so the prediction is arithmetic rather than intuition, and the run can only confirm it or
embarrass it.

    1. fusion is invisible at batch 1        the chain is 7 KB; nothing there is bandwidth-bound
    2. graph capture is large at batch 1     launches are the entire cost
    3. the crossover exists and is findable  <- the finding
    4. the fused chain reaches most of T7's memory roof
    5. measured fusion tracks the byte model within T9's tolerance

Band 3 is why the topic exists. "Fusion helps sometimes" is an observation; "here is the batch size
where the dominant mechanism flips, and here is why it sits where it does" is a result — and it is
the same species of finding as T7's ridge point and T9's crossover, on the axis neither of them
plots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arch_common.gpu import load_profile
from topics.t11_compiler_runtime.chain import (
    CHAIN_OPS,
    DEFAULT_HIDDEN,
    fused_bytes,
    fusion_ceiling,
    fusion_crossover_batch,
    traffic_seconds,
    unfused_bytes,
)

RESULTS_DIR = Path(__file__).parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# The batch sizes swept. Spans four decades on purpose: the whole claim is that the answer changes
# across it, so a sweep that stopped at 32 would report one regime and call it the conclusion.
BATCHES = (1, 8, 32, 128, 512, 2048)

# --------------------------------------------------------------------------------------------
# Pre-registered bands.
# --------------------------------------------------------------------------------------------

# (1) At batch 1 the whole chain is 7 KB — about 4 nanoseconds of traffic on an A100. Fusion
# removes traffic. There is no traffic to remove.
MAX_FUSION_SPEEDUP_AT_BATCH_1 = 1.1

# (2) The same chain is five kernel launches, and T9 measured a *collective's* fixed cost at ~34 us
# on this class of hardware. Plain kernels are cheaper than collectives, but the chain is small
# enough that launches should still be nearly all of it.
MIN_GRAPH_SPEEDUP_AT_BATCH_1 = 2.0

# (3) The headline. Derived below from T7's measured bandwidth and an assumed per-launch cost, so
# the band is a range rather than a point: the launch cost is the term this repo has not yet
# measured for a plain kernel, and T11 measures it.
CROSSOVER_BATCH_RANGE = (64, 512)

# (4) A fused chain is a pure streaming kernel with no reuse, which is the easiest possible case
# for the memory system. Below 70% of T7's roof the fuser's output is the story.
MIN_SHARE_OF_MEMORY_ROOF = 0.70

# (5) Same tolerance T9 scored its model against, and for the same reason: a two-term model of a
# real kernel earns an order-of-magnitude check, not a tight one.
FUSION_MODEL_TOLERANCE = 1.5

# Assumed only for sizing the crossover. T9 measured 34 us for an NCCL collective, which includes
# rendezvous a plain kernel does not pay; single-digit microseconds is the right order for a bare
# launch, and the measurement replaces this everywhere it matters.
ASSUMED_LAUNCH_US = 5.0


@dataclass(frozen=True)
class Prediction:
    """Everything committed to before the run, with each term's provenance."""

    hidden: int
    chain_ops: tuple[str, ...]
    batches: tuple[int, ...]
    fusion_ceiling: float
    unfused_bytes_by_batch: dict[str, int]
    fused_bytes_by_batch: dict[str, int]
    peak_bandwidth_gbps: float
    assumed_launch_us: float
    predicted_crossover_batch: float
    predicted_fused_us_by_batch: dict[str, float]
    predicted_unfused_us_by_batch: dict[str, float]
    bands: dict[str, object] = field(default_factory=dict)
    session_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def build_prediction(peak_gbps: float, session_id: str = "") -> Prediction:
    """Assemble the pre-registered prediction from byte counts and one measured ceiling."""
    unfused = {str(b): unfused_bytes(b, DEFAULT_HIDDEN) for b in BATCHES}
    fused = {str(b): fused_bytes(b, DEFAULT_HIDDEN) for b in BATCHES}

    return Prediction(
        hidden=DEFAULT_HIDDEN,
        chain_ops=CHAIN_OPS,
        batches=BATCHES,
        # Independent of batch and of hardware: it is 2*k*n over 3*n, i.e. ops against boundaries.
        fusion_ceiling=fusion_ceiling(1, DEFAULT_HIDDEN),
        unfused_bytes_by_batch=unfused,
        fused_bytes_by_batch=fused,
        peak_bandwidth_gbps=peak_gbps,
        assumed_launch_us=ASSUMED_LAUNCH_US,
        predicted_crossover_batch=fusion_crossover_batch(peak_gbps, ASSUMED_LAUNCH_US),
        predicted_fused_us_by_batch={
            b: traffic_seconds(n, peak_gbps) * 1e6 for b, n in fused.items()
        },
        predicted_unfused_us_by_batch={
            b: traffic_seconds(n, peak_gbps) * 1e6 for b, n in unfused.items()
        },
        bands={
            "max_fusion_speedup_at_batch_1": MAX_FUSION_SPEEDUP_AT_BATCH_1,
            "min_graph_speedup_at_batch_1": MIN_GRAPH_SPEEDUP_AT_BATCH_1,
            "crossover_batch_range": list(CROSSOVER_BATCH_RANGE),
            "min_share_of_memory_roof": MIN_SHARE_OF_MEMORY_ROOF,
            "fusion_model_tolerance": FUSION_MODEL_TOLERANCE,
        },
        session_id=session_id,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.0,
        help="GB/s to predict against. Defaults to the session's measured probe; pass a value to "
        "run this on a laptop before any hardware exists",
    )
    parser.add_argument("--write", action="store_true", help="register to results/predictions.json")
    args = parser.parse_args()

    peak = args.bandwidth
    session = ""
    if not peak:
        profile = load_profile()
        peak, session = profile.peak_bandwidth_gbps, profile.session_id

    p = build_prediction(peak, session_id=session)

    print("T11 — pre-registered bands, filed before the chain is ever run\n")
    print(f"chain              {' -> '.join(p.chain_ops)}")
    print(f"hidden             {p.hidden}")
    nops = len(p.chain_ops)
    print(f"fusion ceiling     {p.fusion_ceiling:.2f}x   ({nops} ops -> 3 boundary tensors)")
    print(f"bandwidth          {p.peak_bandwidth_gbps:,.1f} GB/s   (T7's measure, not a spec)\n")

    print(f"{'batch':>7} {'unfused B':>12} {'fused B':>10} {'unfused us':>11} {'fused us':>9}")
    for b in p.batches:
        print(
            f"{b:>7} {p.unfused_bytes_by_batch[str(b)]:>12,} "
            f"{p.fused_bytes_by_batch[str(b)]:>10,} "
            f"{p.predicted_unfused_us_by_batch[str(b)]:>11.3f} "
            f"{p.predicted_fused_us_by_batch[str(b)]:>9.3f}"
        )

    print(
        f"\nAt batch 1 the unfused chain is {p.predicted_unfused_us_by_batch['1']:.3f} us of "
        f"traffic. A kernel launch is assumed at {p.assumed_launch_us:.1f} us."
    )
    print(f"=> predicted crossover at batch {p.predicted_crossover_batch:,.0f}\n")

    print("bands:")
    print(f"  (1) fusion speedup at batch 1        <= {MAX_FUSION_SPEEDUP_AT_BATCH_1}x")
    print(f"  (2) graph speedup at batch 1         >= {MIN_GRAPH_SPEEDUP_AT_BATCH_1}x")
    print(
        f"  (3) crossover batch in               {CROSSOVER_BATCH_RANGE}   <- the one that matters"
    )
    print(f"  (4) fused chain vs T7's memory roof  >= {MIN_SHARE_OF_MEMORY_ROOF:.0%}")
    print(f"  (5) measured fusion vs byte model    within {FUSION_MODEL_TOLERANCE}x")

    if args.write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_PATH.write_text(p.to_json())
        print(f"\nregistered -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    _main()
