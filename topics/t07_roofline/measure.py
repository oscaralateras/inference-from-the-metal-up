"""Place real inference matmuls on this GPU's roofline.

Two experiments:

* **regimes**    - the four dominant transformer matmuls in both prefill and decode form, each
                   benchmarked in isolation on synthetic tensors of the right shape.
* **batch_walk** - the decode MLP projection with M rising from 1 to 256, showing the roofline
                   point climb from hard memory-bound toward the ridge.

The ceilings are **not** re-measured here. They come from `results/hardware.json`, which T6 reads
too, so both topics draw on exactly the same silicon in the same thermal state. Re-probing would
produce a slightly different roof and the two lab notes would quietly disagree.

    python measure.py --device cuda
    python measure.py --device cpu --prefill-tokens 128    # rehearsal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from arch_common.gpu import format_tflops, gemm_tflops, load_profile
from arch_common.results_io import append_rows
from topics.t07_roofline.shapes import (
    DEFAULT_HIDDEN,
    DEFAULT_INTERMEDIATE,
    DEFAULT_PREFILL_TOKENS,
    GemmShape,
    batch_walk_shapes,
    inference_shapes,
)

CSV_PATH = Path(__file__).parent / "results" / "roofline.csv"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Pre-registered: a benchmark that cannot reach this share of the session's measured peak on a
# large compute-bound GEMM indicates a methodology problem, not a slow GPU. Reported either way.
MIN_PREFILL_SHARE_OF_PEAK = 0.70


def benchmark(
    shape: GemmShape, device: torch.device, dtype: torch.dtype, peak_tflops: float
) -> dict[str, float]:
    """Measure one shape and derive its position on the roofline.

    `flops_per_byte` is analytic (the minimum traffic the shape requires); `achieved_tflops` is
    measured. Plotting the measured y against the analytic x is what a roofline *is* — the vertical
    distance to the roof is the performance the kernel did not claim.
    """
    bytes_per_element = torch.finfo(dtype).bits // 8
    achieved = gemm_tflops(shape.m, shape.n, shape.k, device, dtype)
    intensity = shape.arithmetic_intensity(bytes_per_element)

    return {
        "flops_per_byte": intensity,
        "achieved_tflops": achieved,
        "share_of_peak": achieved / peak_tflops,
        # Bytes/second implied by the measured rate at this shape's intensity. For a memory-bound
        # kernel this should approach the bandwidth ceiling; for a compute-bound one it will sit
        # far below it, which is the definition of being compute-bound.
        "implied_gbps": achieved * 1e12 / intensity / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--prefill-tokens", type=int, default=DEFAULT_PREFILL_TOKENS)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    profile = load_profile()

    print(f"{profile.device_name} ({args.dtype})")
    print(f"  measured peak   {format_tflops(profile.peak_tflops):>8} TFLOP/s")
    print(f"  measured bw     {profile.peak_bandwidth_gbps:>8,.1f} GB/s")
    print(f"  ridge point     {profile.ridge_point:>8,.2f} FLOPs/byte\n")

    rows: list[dict[str, object]] = []

    def record(experiment: str, shape: GemmShape, x: int, metrics: dict[str, float]) -> None:
        rows.extend(
            {
                "session_id": profile.session_id,
                "experiment": experiment,
                "variant": shape.name,
                "x": x,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        )

    print(f"{'shape':<20} {'regime':<9} {'AI':>10} {'TFLOP/s':>10} {'% peak':>8} {'bound by':>12}")
    print("-" * 76)
    regimes = inference_shapes(args.hidden, args.intermediate, args.prefill_tokens)
    for shape in regimes:
        metrics = benchmark(shape, device, dtype, profile.peak_tflops)
        record("regimes", shape, shape.m, metrics)
        bound = "compute" if metrics["flops_per_byte"] > profile.ridge_point else "memory"
        print(
            f"{shape.name:<20} {shape.regime:<9} {metrics['flops_per_byte']:>10,.1f} "
            f"{format_tflops(metrics['achieved_tflops']):>10} {metrics['share_of_peak']:>7.1%} "
            f"{bound:>12}"
        )

    print(f"\n{'batch (M)':>10} {'AI':>10} {'TFLOP/s':>10} {'% of ridge':>12}")
    print("-" * 46)
    for shape in batch_walk_shapes(args.hidden, args.intermediate):
        metrics = benchmark(shape, device, dtype, profile.peak_tflops)
        record("batch_walk", shape, shape.m, metrics)
        print(
            f"{shape.m:>10} {metrics['flops_per_byte']:>10,.1f} "
            f"{format_tflops(metrics['achieved_tflops']):>10} "
            f"{metrics['flops_per_byte'] / profile.ridge_point:>11.1%}"
        )

    append_rows(CSV_PATH, rows)

    best = max(
        float(str(row["value"]))
        for row in rows
        if row["experiment"] == "regimes"
        and row["metric"] == "share_of_peak"
        and str(row["variant"]).startswith("prefill")
    )
    verdict = "WITHIN" if best >= MIN_PREFILL_SHARE_OF_PEAK else "OUTSIDE"
    print(
        f"\npre-registered band: best prefill shape >= {MIN_PREFILL_SHARE_OF_PEAK:.0%} of "
        f"measured peak  ->  {verdict} ({best:.1%})"
    )


if __name__ == "__main__":
    main()
