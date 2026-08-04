"""One figure: T7's roofline, with the decode point moved.

The argument is spatial, so the plot is T7's axes reused rather than a new chart. bf16 decode sits
where T7 left it — far left, hard against the memory roof. int4 decode sits up and to the right,
because dividing the bytes both raises arithmetic intensity and raises achieved FLOP/s. Neither
point crosses the roof, and that is the point of drawing the roof.

    python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arch_common.gpu import HardwareProfile, load_profile  # noqa: E402
from arch_common.results_io import read_rows, scalar  # noqa: E402
from topics.t08_gpu_architecture.measure import CSV_PATH  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# The arrow runs between the two *Triton* points, which is the controlled comparison: same author,
# same framework, only the data format differs. cuBLAS is plotted too, unconnected, because it is
# the practical reference rather than part of the experiment.
COLOURS = {
    "bf16_cublas": "#7f7f7f",
    "bf16_triton": "#d62728",
    "int4_fused": "#2ca02c",
}
LABELS = {
    "bf16_cublas": "decode, bf16 (cuBLAS — reference)",
    "bf16_triton": "decode, bf16 (Triton — control)",
    "int4_fused": "decode, int4 (fused Triton)",
}
MARKERS = {"bf16_cublas": "s", "bf16_triton": "o", "int4_fused": "o"}


def _roof(profile: HardwareProfile, intensities: list[float]) -> list[float]:
    """min(bandwidth x AI, peak compute) — the diagonal and the ceiling, whichever binds."""
    return [
        min(profile.peak_bandwidth_gbps * 1e9 * ai, profile.peak_tflops * 1e12) / 1e12
        for ai in intensities
    ]


def plot_moved_point(rows: list[dict[str, str]], profile: HardwareProfile) -> Path:
    """Both decode variants under the measured roof."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    grid = [10 ** (i / 20) for i in range(-40, 81)]
    ax.plot(grid, _roof(profile, grid), color="black", lw=2, label="roofline (measured ceilings)")
    ax.axvline(
        profile.ridge_point,
        color="grey",
        ls=":",
        lw=1.5,
        label=f"ridge point ({profile.ridge_point:,.0f} FLOPs/byte)",
    )

    points: list[tuple[float, float]] = []
    for variant in ("bf16_cublas", "bf16_triton", "int4_fused"):
        ms = scalar(rows, "gemv", variant, "ms")
        moved = scalar(rows, "gemv", variant, "bytes")
        # FLOPs are identical for both variants — same maths, different storage. Only the byte
        # count changes, which is exactly why the point moves horizontally as well as vertically.
        flops = profile_shape_flops(rows)
        intensity = flops / moved
        tflops = flops / (ms * 1e-3) / 1e12
        if variant != "bf16_cublas":
            points.append((intensity, tflops))
        ax.scatter(
            intensity,
            tflops,
            s=140,
            zorder=5,
            marker=MARKERS[variant],
            color=COLOURS[variant],
            edgecolor="white",
            linewidth=1.2,
            label=LABELS[variant],
        )

    (x0, y0), (x1, y1) = points
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#555555", "shrinkA": 9, "shrinkB": 9},
    )
    # Offset below-right of the connecting arrow rather than centred on it: at these coordinates
    # the arrow runs diagonally through the midpoint, and centred text lands on top of it.
    ax.text(
        (x0 * x1) ** 0.5 * 1.35,
        (y0 * y1) ** 0.5 * 0.72,
        f"{x1 / x0:.1f}× fewer bytes",
        fontsize=9,
        color="#555555",
        va="top",
        ha="left",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("achieved throughput (TFLOP/s)")
    ax.set_title(
        f"Quantisation moves decode along the memory roof — it does not lift it\n"
        f"{profile.device_name}, decode MLP up-projection (M=1)",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = RESULTS_DIR / "int4_roofline.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def profile_shape_flops(rows: list[dict[str, str]]) -> float:
    """FLOPs for the benchmarked GEMV, derived from the bf16 variant's byte count.

    bf16 stores 2 bytes per weight and a GEMV does 2 FLOPs per weight, so the FLOP count is simply
    the bf16 byte count. Deriving it here keeps the plot from needing the shape passed in, and
    means a change to the benchmarked shape cannot leave the plot silently mislabelled.
    """
    return scalar(rows, "gemv", "bf16_triton", "bytes")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"no results at {CSV_PATH} — run `python measure.py` first")
    rows = read_rows(CSV_PATH)
    profile = load_profile()
    print(f"wrote {plot_moved_point(rows, profile)}")


if __name__ == "__main__":
    main()
