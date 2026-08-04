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
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar
from arch_common.timing import time_op
from topics.t01_number_representation.metrics import cosine_similarity
from topics.t07_roofline.shapes import DEFAULT_HIDDEN, DEFAULT_INTERMEDIATE
from topics.t08_gpu_architecture.kernel import bf16_gemv, int4_gemv, int8_gemv
from topics.t08_gpu_architecture.pack import (
    DEFAULT_GROUP_SIZE,
    PackedWeight,
    quantise_and_pack,
    quantise_int8,
)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "int4.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# T6's error budget, read rather than restated so the two topics cannot drift apart.
T6_CSV = Path(__file__).resolve().parent.parent / "t06_perf_reasoning" / "results" / "perf.csv"

BASELINE_DTYPE = torch.bfloat16
BASELINE_BYTES_PER_PARAM = 2.0

# Back-to-back launches inside one timing window. A decode step launches hundreds of these GEMVs in
# sequence and never pays for one in isolation, so steady-state is both the faithful measurement and
# the one that does not report the dispatch path instead of the kernel. Applied identically to both
# variants — the comparison would be meaningless otherwise.
LAUNCHES_PER_TIMING = 16

# Whole-benchmark repeats. `time_op` already takes a median across its own iterations, but Triton's
# autotuner re-searches in every fresh process and does not always land on the same config, which on
# the development GPU produced a 3.10x-3.53x spread across otherwise identical runs. One run is
# therefore a sample of the autotuner as much as of the kernel. Repeating in-process and reporting
# the median with its spread is the difference between a number and a reproducible number.
DEFAULT_REPEATS = 5

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


# How far the rotating weight pool must exceed L2 before a "streaming" measurement is honest.
# Timing one weight repeatedly leaves it resident in L2 and measures cache bandwidth: on a 4090
# (75.5 MB L2) the 35 MB int4 weight fits entirely, and the bf16 baseline was measured at 112% of
# the HBM roof — impossible, and the tell that the number was wrong. An A100's 40 MB L2 swallows
# the int4 weight too, so this is not a quirk of the development GPU.
L2_HEADROOM = 4


def _pool_size(device: torch.device, smallest_bytes: int) -> int:
    """How many distinct weights to rotate through so even the smallest never stays cached.

    Sized against the *int4* footprint because it is the smaller of the two and therefore the
    easier one to accidentally serve from L2. This also happens to be the physically faithful
    setup: a real decode step reads every layer's weights once and reuses none of them within a
    token, so rotating distinct tensors is what the hardware actually sees.
    """
    if device.type != "cuda":
        return 2
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    return max(2, -(-L2_HEADROOM * l2_bytes // smallest_bytes))


def _rotating(count: int) -> Callable[[], int]:
    """Round-robin index generator, so each timed iteration touches a different tensor."""
    counter = itertools.count()
    return lambda: next(counter) % count


def benchmark_baseline(
    weights: list[torch.Tensor], x: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """`torch.matmul` over a rotating pool of bf16 weights — the *practical* baseline.

    cuBLAS, in other words: what a decode step actually runs today. Useful for "is this worth
    shipping", and the wrong control for "does cutting bytes help", because it differs from the
    int4 kernel in two ways at once — the data format and roughly fifteen points of roof that
    separate hand-written Triton from NVIDIA's hand-tuned assembly. See `benchmark_bf16_triton`.
    """
    pool = [w.to(BASELINE_DTYPE) for w in weights]
    xv = x.to(BASELINE_DTYPE)
    nxt = _rotating(len(pool))
    ms = time_op(lambda: pool[nxt()] @ xv, device, inner=LAUNCHES_PER_TIMING)
    moved = pool[0].numel() * pool[0].element_size()
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def benchmark_bf16_triton(
    weights: list[torch.Tensor], x: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """The same GEMV in Triton over bf16 weights — the *controlled* baseline.

    Same author, same framework, same tiling, same reduction, same reused output buffer. Only the
    data format differs, so the ratio against the int4 kernel isolates the byte reduction and
    nothing else. This is the comparison the pre-registered band was always about.
    """
    pool = [w.to(BASELINE_DTYPE) for w in weights]
    nxt = _rotating(len(pool))
    out = torch.empty(pool[0].shape[0], device=x.device, dtype=torch.float32)
    ms = time_op(lambda: bf16_gemv(pool[nxt()], x, out=out), device, inner=LAUNCHES_PER_TIMING)
    moved = pool[0].numel() * pool[0].element_size()
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def benchmark_int4(
    packed: list[PackedWeight], x: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """The fused kernel, over the same rotating pool. Bytes counted off the packed tensors."""
    nxt = _rotating(len(packed))
    out = torch.empty(packed[0].n, device=x.device, dtype=torch.float32)
    ms = time_op(lambda: int4_gemv(packed[nxt()], x, out=out), device, inner=LAUNCHES_PER_TIMING)
    moved = packed[0].bytes_stored
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def benchmark_int8(
    packed: list[PackedWeight], x: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """The int8 kernel — the arithmetic control.

    Half the byte reduction of int4 and half the work per byte. If work per byte is what limits the
    int4 kernel, this must reach a *higher* fraction of the memory roof despite moving more bytes.
    """
    nxt = _rotating(len(packed))
    out = torch.empty(packed[0].n, device=x.device, dtype=torch.float32)
    ms = time_op(lambda: int8_gemv(packed[nxt()], x, out=out), device, inner=LAUNCHES_PER_TIMING)
    moved = packed[0].bytes_stored
    return {"ms": ms, "gbps": moved / (ms * 1e-3) / 1e9, "bytes": float(moved)}


def _verdict(value: float, lo: float, hi: float) -> str:
    return "WITHIN" if lo <= value <= hi else "OUTSIDE"


def repeat_median(
    measure: Callable[[], dict[str, float]], repeats: int
) -> tuple[dict[str, float], float, float]:
    """Run a benchmark `repeats` times; return the median run plus the min and max of its speed.

    The *median run* is returned whole rather than a per-metric median, so `ms`, `gbps` and `bytes`
    stay mutually consistent — a synthetic row assembled from three different runs' medians would
    report a GB/s that no single measurement ever produced.
    """
    runs = sorted((measure() for _ in range(repeats)), key=lambda r: r["ms"])
    speeds = [r["gbps"] for r in runs]
    return runs[len(runs) // 2], min(speeds), max(speeds)


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
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="whole-benchmark repeats; the median is reported with its spread",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="distinct weights to rotate through (0 = size it from the GPU's L2)",
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
    pool_size = args.layers or _pool_size(device, pw.bytes_stored)

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

    l2_mb = (
        torch.cuda.get_device_properties(device).L2_cache_size / 1e6 if device.type == "cuda" else 0
    )
    print(f"{profile.device_name}  |  measured roof {profile.peak_bandwidth_gbps:,.1f} GB/s")
    print(
        f"rotating {pool_size} distinct weights "
        f"({pool_size * pw.bytes_stored / 1e6:,.0f} MB int4 against {l2_mb:,.0f} MB of L2)\n"
    )

    # Distinct tensors, not copies — a decode step reads every layer once and reuses none of them,
    # and one tensor timed repeatedly would sit in L2 and report cache bandwidth as if it were HBM.
    pool = [weight] + [
        torch.randn(n, k, device=device, dtype=torch.float32) * 0.02 for _ in range(pool_size - 1)
    ]
    packed_pool = [pw] + [quantise_and_pack(w, args.group_size) for w in pool[1:]]
    int8_pool = [quantise_int8(w, args.group_size) for w in pool]

    baseline, base_lo, base_hi = repeat_median(
        lambda: benchmark_baseline(pool, x, device), args.repeats
    )
    triton_bf16, tri_lo, tri_hi = repeat_median(
        lambda: benchmark_bf16_triton(pool, x, device), args.repeats
    )
    fused, fused_lo, fused_hi = repeat_median(
        lambda: benchmark_int4(packed_pool, x, device), args.repeats
    )
    int8, int8_lo, int8_hi = repeat_median(
        lambda: benchmark_int8(int8_pool, x, device), args.repeats
    )

    reference = (weight @ x).to(torch.float32)
    measured_y = int4_gemv(pw, x)
    cosine = cosine_similarity(
        reference.cpu().numpy().astype("float32"),
        measured_y.cpu().numpy().astype("float32"),
    )

    # The band is scored against the *controlled* baseline: same framework, same author, only the
    # data format differs. Scoring against cuBLAS would charge quantisation for the distance
    # between hand-written Triton and NVIDIA's assembly, which is not what the topic measures.
    speedup = triton_bf16["ms"] / fused["ms"]
    speedup_vs_cublas = baseline["ms"] / fused["ms"]
    share_of_ratio = speedup / pred.byte_ratio
    end_to_end = 1.0 / (weight_share / speedup + (1.0 - weight_share))

    print(f"median of {args.repeats} runs\n")
    print(f"{'kernel':<14} {'ms':>9} {'GB/s':>10} {'% of roof':>11} {'spread GB/s':>20}")
    print("-" * 70)
    for name, r, lo, hi in (
        ("bf16 cuBLAS", baseline, base_lo, base_hi),
        ("bf16 Triton", triton_bf16, tri_lo, tri_hi),
        ("int8 Triton", int8, int8_lo, int8_hi),
        ("int4 Triton", fused, fused_lo, fused_hi),
    ):
        print(
            f"{name:<14} {r['ms']:>9.3f} {r['gbps']:>10,.1f} "
            f"{r['gbps'] / profile.peak_bandwidth_gbps:>10.1%} {lo:>10,.0f}-{hi:<9,.0f}"
        )

    print(
        f"\n  kernel speedup   {speedup:.2f}x "
        f"({share_of_ratio:.0%} of the {pred.byte_ratio:.2f}x byte ratio)"
    )
    print(f"  vs cuBLAS        {speedup_vs_cublas:.2f}x  (practical, not the band)")
    print(
        f"  int8 control     {triton_bf16['ms'] / int8['ms']:.2f}x at "
        f"{BASELINE_BYTES_PER_PARAM / int8_pool[0].bytes_per_param:.2f}x fewer bytes "
        f"— half the work per byte"
    )
    print(f"  cosine vs fp32   {cosine:.4f}")
    print(
        f"  implied decode   {end_to_end:.2f}x -> {pred.t6_tokens_per_sec * end_to_end:.1f} tok/s"
    )

    rows: list[dict[str, object]] = []
    for variant, r in (
        ("bf16_cublas", baseline),
        ("bf16_triton", triton_bf16),
        ("int8_fused", int8),
        ("int4_fused", fused),
    ):
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
            "speedup_vs_cublas": speedup_vs_cublas,
            "int8_speedup": triton_bf16["ms"] / int8["ms"],
            "int8_byte_ratio": BASELINE_BYTES_PER_PARAM / int8_pool[0].bytes_per_param,
            # Bounds on the *controlled* ratio, so they share a denominator with kernel_speedup.
            # Computing them against cuBLAS while the headline used the Triton control produced a
            # spread that did not contain its own median — visibly wrong only if you looked.
            "kernel_speedup_min": triton_bf16["ms"] / (fused["ms"] * fused_hi / fused["gbps"]),
            "kernel_speedup_max": triton_bf16["ms"] / (fused["ms"] * fused_lo / fused["gbps"]),
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
