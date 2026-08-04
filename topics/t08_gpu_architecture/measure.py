"""Does moving 3.9x fewer bytes make decode 3.9x faster? Predict first, then measure.

The prediction is assembled entirely from earlier topics — nothing here is a fresh guess:

* **T1** supplies the quantiser and the granularity that keeps cosine at 0.99.
* **`pack.py`** supplies the byte ratio, computed from the tensors actually allocated.
* **T6** supplies the share of a real decode step that is weight traffic (76.9% when it ran).
* **T5** supplies Amdahl, which turns those two into an end-to-end number that is *much* smaller
  than the kernel-level one.
* **T7** supplies the memory roof the result is scored against.

    kernel speedup    ~ bytes(bf16) / bytes(int4)                        ~ 3.9x
    end-to-end        ~ 1 / (weight_share / kernel_speedup + rest)       ~ 2.3x

The gap between those two is the finding. Everything above is printed before a single kernel runs,
so the run can only confirm or embarrass it.

    python measure.py                       # full, on the GPU pod
    python measure.py --skip-kernel         # prediction only; runs anywhere, no Triton needed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar
from arch_common.timing import time_op
from topics.t01_number_representation.metrics import cosine_similarity
from topics.t07_roofline.shapes import DEFAULT_HIDDEN, DEFAULT_INTERMEDIATE
from topics.t08_gpu_architecture.kernel import int4_gemv
from topics.t08_gpu_architecture.pack import DEFAULT_GROUP_SIZE, PackedWeight, quantise_and_pack

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "int4.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# T6's error budget, read rather than restated so the two topics cannot drift apart.
T6_CSV = Path(__file__).resolve().parent.parent / "t06_perf_reasoning" / "results" / "perf.csv"

BASELINE_DTYPE = torch.bfloat16
BASELINE_BYTES_PER_PARAM = 2.0

# ---------------------------------------------------------------------------------------------
# Pre-registered bands. Committed before the run; reported WITHIN/OUTSIDE either way. A miss that
# gets explained is a better lab note than a hit that does not.
# ---------------------------------------------------------------------------------------------

# The kernel cannot exceed the byte ratio — that is the roof. It can fall short of it, through
# unpacking instructions, redundant scale loads and imperfect latency hiding. 75% is the line
# below which the shortfall is the story rather than a footnote.
MIN_KERNEL_SHARE_OF_BYTE_RATIO = 0.75

# T1 measured cosine 0.99 for int4 per-group on a real weight. The kernel must not degrade that;
# if it does, the packing or the scale indexing is wrong, not the quantisation.
MIN_COSINE = 0.99

# Amdahl's prediction is a model, not an identity — it assumes the non-weight 23% is untouched by
# quantisation, which is approximately but not exactly true.
END_TO_END_TOLERANCE = 0.25


@dataclass(frozen=True)
class Prediction:
    """Everything committed to before the run, with the provenance of each term."""

    bytes_per_param_bf16: float
    bytes_per_param_int4: float
    byte_ratio: float
    weight_share: float
    predicted_end_to_end: float
    t6_tokens_per_sec: float
    predicted_tokens_per_sec: float
    # Stamped from the hardware profile so a guard can prove the prediction was registered against
    # the same silicon it is later judged on. Empty under --skip-kernel, where there is no GPU and
    # therefore no session — the guard treats that as "not yet run" rather than as a failure.
    session_id: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2) + "\n"


def _t6_weight_share() -> tuple[float, float]:
    """Read the weight fraction of a decode step, and the measured tok/s, out of T6's results.

    Returns `(weight_share, tokens_per_sec)`. Raises if T6 has not been run — the prediction is
    not meaningful without it, and inventing a plausible 0.77 would be exactly the kind of
    convenient number this repo exists not to publish.
    """
    if not T6_CSV.exists():
        raise FileNotFoundError(
            f"no T6 results at {T6_CSV} — T8's end-to-end prediction is built from T6's error "
            "budget. Run T6 first, or pass --skip-end-to-end."
        )
    rows = read_rows(T6_CSV)
    weights_ms = scalar(rows, "decomposition", "weights", "step_time_ms")
    measured_ms = scalar(rows, "decomposition", "measured", "step_time_ms")
    return weights_ms / measured_ms, 1000.0 / measured_ms


def predict(
    pw: PackedWeight,
    *,
    weight_share: float,
    t6_tokens_per_sec: float,
    session_id: str = "",
) -> Prediction:
    """Assemble the pre-registered prediction from the packed layout and T6's budget.

    Amdahl, exactly as plotted in T5: the fraction you sped up divides by the speedup, the
    fraction you did not is unchanged, and the total is the reciprocal of the sum. It is the
    reason a 3.9x kernel does not buy 3.9x tokens.
    """
    byte_ratio = BASELINE_BYTES_PER_PARAM / pw.bytes_per_param
    end_to_end = 1.0 / (weight_share / byte_ratio + (1.0 - weight_share))
    return Prediction(
        bytes_per_param_bf16=BASELINE_BYTES_PER_PARAM,
        bytes_per_param_int4=pw.bytes_per_param,
        byte_ratio=byte_ratio,
        weight_share=weight_share,
        predicted_end_to_end=end_to_end,
        t6_tokens_per_sec=t6_tokens_per_sec,
        predicted_tokens_per_sec=t6_tokens_per_sec * end_to_end,
        session_id=session_id,
    )


def benchmark_baseline(
    weight: torch.Tensor, x: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """`torch.matmul` on the bf16 weight — what a decode step does today."""
    w = weight.to(BASELINE_DTYPE)
    xv = x.to(BASELINE_DTYPE)
    ms = time_op(lambda: w @ xv, device)
    moved = w.numel() * w.element_size()
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def benchmark_int4(pw: PackedWeight, x: torch.Tensor, device: torch.device) -> dict[str, float]:
    """The fused kernel. Bytes counted off the packed tensors, scales included."""
    ms = time_op(lambda: int4_gemv(pw, x), device)
    moved = pw.bytes_stored
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def _verdict(value: float, lo: float, hi: float) -> str:
    return "WITHIN" if lo <= value <= hi else "OUTSIDE"


def main() -> None:  # noqa: PLR0915 - a linear script; splitting it would obscure the order
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="print the prediction and exit — no GPU or Triton required",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # The decode MLP up-projection: T7's `decode_mlp_up`, the shape that sat at AI 1.0 and 0.5% of
    # peak. Synthetic weights, as in T7 — the byte budget depends on the shape, not the values.
    n, k = args.intermediate, args.hidden
    weight = torch.randn(n, k, device=device, dtype=torch.float32) * 0.02
    x = torch.randn(k, device=device, dtype=torch.float32)

    pw = quantise_and_pack(weight, args.group_size)
    weight_share, t6_tps = _t6_weight_share()

    # The profile is loaded *before* the prediction is written, not after the kernel runs, so the
    # session stamped into predictions.json is provably the one the kernel is about to be measured
    # on. Registering a prediction and then discovering which GPU you are on would defeat the point.
    profile = None if args.skip_kernel else load_profile()
    pred = predict(
        pw,
        weight_share=weight_share,
        t6_tokens_per_sec=t6_tps,
        session_id="" if profile is None else profile.session_id,
    )

    print(f"shape        (N={n}, K={k}), group {args.group_size}\n")
    print("PRE-REGISTERED, from earlier topics:")
    print(f"  bytes/param  {pred.bytes_per_param_bf16:.3f} bf16 -> {pred.bytes_per_param_int4:.3f}")
    print(f"  byte ratio   {pred.byte_ratio:.2f}x           <- the kernel's ceiling")
    print(f"  weight share {pred.weight_share:.1%}            <- T6 error budget")
    print(f"  end-to-end   {pred.predicted_end_to_end:.2f}x           <- Amdahl (T5)")
    print(
        f"  decode       {pred.t6_tokens_per_sec:.1f} -> "
        f"{pred.predicted_tokens_per_sec:.1f} tok/s\n"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(pred.to_json())

    if profile is None:
        print(f"prediction written to {PREDICTIONS_PATH}; skipping the kernel")
        return

    print(f"{profile.device_name}  |  measured roof {profile.peak_bandwidth_gbps:,.1f} GB/s\n")

    baseline = benchmark_baseline(weight, x, device)
    fused = benchmark_int4(pw, x, device)

    reference = (weight @ x).to(torch.float32)
    measured_y = int4_gemv(pw, x)
    cosine = cosine_similarity(
        reference.cpu().numpy().astype("float32"),
        measured_y.cpu().numpy().astype("float32"),
    )

    speedup = baseline["ms"] / fused["ms"]
    share_of_ratio = speedup / pred.byte_ratio
    end_to_end = 1.0 / (weight_share / speedup + (1.0 - weight_share))

    print(f"{'kernel':<14} {'ms':>9} {'GB/s':>10} {'% of roof':>11}")
    print("-" * 48)
    for name, r in (("bf16 torch", baseline), ("int4 fused", fused)):
        print(
            f"{name:<14} {r['ms']:>9.3f} {r['gbps']:>10,.1f} "
            f"{r['gbps'] / profile.peak_bandwidth_gbps:>10.1%}"
        )

    print(
        f"\n  kernel speedup   {speedup:.2f}x "
        f"({share_of_ratio:.0%} of the {pred.byte_ratio:.2f}x byte ratio)"
    )
    print(f"  cosine vs fp32   {cosine:.4f}")
    print(
        f"  implied decode   {end_to_end:.2f}x -> {pred.t6_tokens_per_sec * end_to_end:.1f} tok/s"
    )

    rows: list[dict[str, object]] = []
    for variant, r in (("bf16_torch", baseline), ("int4_fused", fused)):
        rows.extend(
            {
                "session_id": profile.session_id,
                "experiment": "gemv",
                "variant": variant,
                "x": 1,
                "metric": metric,
                "value": value,
            }
            for metric, value in r.items()
        )
    rows.extend(
        {
            "session_id": profile.session_id,
            "experiment": "summary",
            "variant": "int4_fused",
            "x": 0,
            "metric": metric,
            "value": value,
        }
        for metric, value in {
            "kernel_speedup": speedup,
            "byte_ratio": pred.byte_ratio,
            "cosine": cosine,
            "end_to_end_speedup": end_to_end,
            "predicted_end_to_end": pred.predicted_end_to_end,
        }.items()
    )
    append_rows(CSV_PATH, rows)

    lo = pred.predicted_end_to_end * (1 - END_TO_END_TOLERANCE)
    hi = pred.predicted_end_to_end * (1 + END_TO_END_TOLERANCE)
    print("\npre-registered bands:")
    print(
        f"  kernel >= {MIN_KERNEL_SHARE_OF_BYTE_RATIO:.0%} of byte ratio  -> "
        f"{_verdict(share_of_ratio, MIN_KERNEL_SHARE_OF_BYTE_RATIO, 1.0)} ({share_of_ratio:.0%})"
    )
    print(
        f"  cosine >= {MIN_COSINE}                  -> "
        f"{_verdict(cosine, MIN_COSINE, 1.0)} ({cosine:.4f})"
    )
    print(
        f"  end-to-end within +/-{END_TO_END_TOLERANCE:.0%}       -> "
        f"{_verdict(end_to_end, lo, hi)} ({end_to_end:.2f}x vs {pred.predicted_end_to_end:.2f}x)"
    )


if __name__ == "__main__":
    main()
