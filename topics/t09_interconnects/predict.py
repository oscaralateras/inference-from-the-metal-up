"""Phase 0 — commit to the numbers before renting anything.

    uv run python -m topics.t09_interconnects.predict          # runs anywhere, no GPU

Everything here is registered to `results/predictions.json` and judged later, pass or fail. The
point of doing it first is that the run can then only confirm or embarrass it — a band invented
after seeing the data is not a band.

Three of the four predictions are **structural**: they are statements about the shape of the cost
curve that follow from the ring algorithm alone, so they can be committed to without knowing a
single thing about the hardware. That is what makes them worth pre-registering.

    1. alpha scales as 2(N-1)          ->  alpha(4) / alpha(2) = 3.0
    2. large messages saturate the link ->  bus bandwidth >= 70% of the fabric's spec
    3. decode is alpha-bound            ->  >= 90% of a batch-1 all-reduce is fixed cost
    4. the fitted model predicts a real TP matmul's comms within 1.5x

The fourth is the one that could genuinely go either way, and it is the one that matters: a
two-term model fitted to an isolated microbenchmark is only interesting if it survives contact
with a collective embedded in real work.

The absolute tokens/sec figures below additionally need a value for alpha, which is exactly what
has not been measured yet. They are computed from an **assumed** alpha, labelled as assumed
wherever they appear, and recomputed from the measured one in `measure.py`. They are here to size
the experiment, not to be quoted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arch_common.results_io import read_rows, scalar
from topics.t09_interconnects.model import (
    ALLREDUCES_PER_LAYER,
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    RingCost,
    allreduce_bytes,
    comms_per_token_us,
    predicted_tp_speedup,
    prefill_allreduce_bytes,
    ring_hops,
)

RESULTS_DIR = Path(__file__).parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# T6's error budget, read at runtime rather than restated, so the two topics cannot drift.
T6_CSV = Path(__file__).resolve().parent.parent / "t06_perf_reasoning" / "results" / "perf.csv"

# T5's TP operating point, so the note can put both regimes on one axis. Not a guess: these are
# the flags in T5's own reproduce block.
T5_BATCH = 16
T5_SEQ = 512
T5_WORLD = 4

# The world sizes this topic measures. 1 is not a collective and is present only as the TP-matmul
# control; the fit needs 2 and 4, and 4 is what makes prediction (1) testable at all.
WORLD_SIZES = (2, 4)

# --------------------------------------------------------------------------------------------
# Pre-registered bands. Reported WITHIN/OUTSIDE either way — T8 failed two of three and was a
# better note for it.
# --------------------------------------------------------------------------------------------

# (1) Ring latency is 2(N-1) dependent hops, so the fixed cost of going 2 -> 4 GPUs should triple.
# The tolerance is wide because NCCL does not always choose a ring: it switches to tree algorithms
# for small payloads on some topologies, which would break this prediction *for a reason worth
# reporting* rather than through noise.
ALPHA_SCALING_TOLERANCE = 0.30

# (2) A large all-reduce should saturate the fabric. Scored against the NVLink generation the node
# actually reports rather than a number typed here, so the band travels to whatever gets rented.
MIN_SHARE_OF_LINK_SPEC = 0.70

# (3) The headline claim. If a batch-1 decode all-reduce is not overwhelmingly fixed cost, the
# central argument of this topic is wrong and the note has to say so.
MIN_DECODE_ALPHA_SHARE = 0.90

# (4) A microbenchmark's model, judged against a collective doing real work.
TP_MODEL_TOLERANCE = 1.5

# Assumed only for sizing. NCCL small-message latency on NVLink is single-digit microseconds; the
# measured value replaces this everywhere it matters.
ASSUMED_ALPHA_US = 8.0
ASSUMED_BETA_GBPS = 480.0

# Per-GPU NVLink bandwidth by bond width, GB/s, unidirectional. NV12 is the A100 SXM4 configuration
# T5 measured on: 12 links x 25 GB/s each direction = 300 GB/s each way. Used only to give band (2)
# a denominator; every measured number in this topic is measured.
NVLINK_GBPS_PER_LINK = 25.0


@dataclass(frozen=True)
class Prediction:
    """Everything committed to before the run, with each term's provenance."""

    hidden: int
    layers: int
    allreduces_per_token: int
    decode_bytes_by_batch: dict[str, int]
    t5_prefill_bytes: int
    decode_vs_prefill_ratio: float
    weight_share: float
    t6_tokens_per_sec: float
    t6_step_ms: float
    alpha_scaling_2_to_4: float
    assumed_alpha_us: float
    assumed_beta_gbps: float
    assumed_comms_us_per_token: dict[str, float]
    assumed_tp_speedup: dict[str, float]
    bands: dict[str, float] = field(default_factory=dict)
    session_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def t6_budget() -> tuple[float, float, float]:
    """Read `(weight_share, tokens_per_sec, step_ms)` out of T6's measured error budget.

    Raises rather than defaulting. T9's whole end-to-end claim is built on T6's decomposition, and
    substituting a plausible 0.74 would be precisely the kind of convenient number this repo
    exists not to publish.
    """
    if not T6_CSV.exists():
        raise FileNotFoundError(
            f"no T6 results at {T6_CSV} — T9's TP prediction is built from T6's error budget "
            "(the weight share that tensor parallelism divides, and the step it divides into). "
            "Run T6 first."
        )
    rows = read_rows(T6_CSV)
    weights_ms = scalar(rows, "decomposition", "weights", "step_time_ms")
    measured_ms = scalar(rows, "decomposition", "measured", "step_time_ms")
    return weights_ms / measured_ms, 1000.0 / measured_ms, measured_ms


def build_prediction(session_id: str = "") -> Prediction:
    """Assemble the pre-registered prediction. Pure arithmetic — no hardware touched."""
    weight_share, tokens_per_sec, step_ms = t6_budget()

    assumed = {
        world: RingCost(
            world=world,
            alpha_us=ASSUMED_ALPHA_US * ring_hops(world) / ring_hops(2),
            beta_gbps=ASSUMED_BETA_GBPS,
            r_squared=1.0,
            n_points=0,
        )
        for world in WORLD_SIZES
    }

    decode_bytes = {str(b): allreduce_bytes(b, DEFAULT_HIDDEN) for b in (1, 8, 32, 128)}
    t5_bytes = prefill_allreduce_bytes(T5_BATCH, T5_SEQ, DEFAULT_HIDDEN)

    comms = {
        str(world): comms_per_token_us(cost, 1, layers=DEFAULT_LAYERS, hidden=DEFAULT_HIDDEN)
        for world, cost in assumed.items()
    }
    speedups = {
        str(world): predicted_tp_speedup(weight_share, world, step_ms, comms[str(world)])
        for world in WORLD_SIZES
    }

    return Prediction(
        hidden=DEFAULT_HIDDEN,
        layers=DEFAULT_LAYERS,
        allreduces_per_token=DEFAULT_LAYERS * ALLREDUCES_PER_LAYER,
        decode_bytes_by_batch=decode_bytes,
        t5_prefill_bytes=t5_bytes,
        decode_vs_prefill_ratio=t5_bytes / decode_bytes["1"],
        weight_share=weight_share,
        t6_tokens_per_sec=tokens_per_sec,
        t6_step_ms=step_ms,
        alpha_scaling_2_to_4=ring_hops(4) / ring_hops(2),
        assumed_alpha_us=ASSUMED_ALPHA_US,
        assumed_beta_gbps=ASSUMED_BETA_GBPS,
        assumed_comms_us_per_token=comms,
        assumed_tp_speedup=speedups,
        bands={
            "alpha_scaling_tolerance": ALPHA_SCALING_TOLERANCE,
            "min_share_of_link_spec": MIN_SHARE_OF_LINK_SPEC,
            "min_decode_alpha_share": MIN_DECODE_ALPHA_SHARE,
            "tp_model_tolerance": TP_MODEL_TOLERANCE,
        },
        session_id=session_id,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="register the prediction to results/predictions.json",
    )
    args = parser.parse_args()

    p = build_prediction()

    print("T9 — pre-registered prediction (no GPU touched)\n")
    print(f"model            Qwen2.5-7B, hidden {p.hidden}, {p.layers} layers")
    print(f"all-reduces/token {p.allreduces_per_token}  ({ALLREDUCES_PER_LAYER} per layer)")
    print("\nWhere the payload actually lands:")
    for batch, nbytes in p.decode_bytes_by_batch.items():
        print(f"  decode, batch {batch:>3}   {nbytes:>12,} B")
    print(f"  T5 prefill 16x512  {p.t5_prefill_bytes:>12,} B")
    print(f"  -> T5's collective is {p.decode_vs_prefill_ratio:,.0f}x decode's at batch 1")

    print(f"\nFrom T6: weight share {p.weight_share:.1%}, step {p.t6_step_ms:.2f} ms, ")
    print(f"         {p.t6_tokens_per_sec:.1f} tok/s")

    print(
        f"\nAssuming alpha = {p.assumed_alpha_us:.0f} us/call at world=2 (ASSUMED, not measured):"
    )
    for world in WORLD_SIZES:
        print(
            f"  world {world}: comms {p.assumed_comms_us_per_token[str(world)]:>7.1f} us/token"
            f"   ->  {p.assumed_tp_speedup[str(world)]:.2f}x"
        )

    print("\nPre-registered bands:")
    print(
        f"  (1) alpha(4)/alpha(2) = {p.alpha_scaling_2_to_4:.1f} +/- {ALPHA_SCALING_TOLERANCE:.0%}"
    )
    print(f"  (2) bus bandwidth >= {MIN_SHARE_OF_LINK_SPEC:.0%} of the node's NVLink spec")
    print(f"  (3) batch-1 all-reduce is >= {MIN_DECODE_ALPHA_SHARE:.0%} fixed cost")
    print(f"  (4) fitted model predicts TP-matmul comms within {TP_MODEL_TOLERANCE:.1f}x")

    if args.write:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_PATH.write_text(p.to_json())
        print(f"\nregistered -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    _main()
