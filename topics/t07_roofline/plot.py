"""Two figures: the roofline itself, and the batch walk across it.

python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from arch_common.gpu import HardwareProfile, load_profile  # noqa: E402
from arch_common.results_io import read_rows, select  # noqa: E402
from topics.t07_roofline.measure import CSV_PATH  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
REGIME_COLOURS = {"prefill": "#1f77b4", "decode": "#d62728"}


def _roof(profile: HardwareProfile, intensities: list[float]) -> list[float]:
    """The roof at each intensity: bandwidth-limited on the left, compute-limited on the right.

    `min(bandwidth * AI, peak_compute)` — the diagonal and the ceiling, whichever binds first.
    """
    return [
        min(profile.peak_bandwidth_gbps * 1e9 * ai, profile.peak_tflops * 1e12) / 1e12
        for ai in intensities
    ]


def _ridge_label(ridge: float) -> str:
    """A GPU's ridge is ~170; a CPU's is ~0.05. One fixed precision cannot render both."""
    return f"{ridge:,.2f}" if ridge < 10 else f"{ridge:,.0f}"


def _draw_roof(ax: Axes, profile: HardwareProfile) -> None:
    grid = [10 ** (i / 20) for i in range(-40, 81)]
    ax.plot(grid, _roof(profile, grid), color="black", lw=2, label="roofline (measured ceilings)")
    ax.axvline(
        profile.ridge_point,
        color="grey",
        ls=":",
        lw=1.5,
        label=f"ridge point ({_ridge_label(profile.ridge_point)} FLOPs/byte)",
    )


def plot_roofline(rows: list[dict[str, str]], profile: HardwareProfile) -> Path:
    """Prefill and decode shapes placed under the roof."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_roof(ax, profile)

    variants = sorted({r["variant"] for r in rows if r["experiment"] == "regimes"})
    for variant in variants:
        intensity = select(rows, "regimes", variant, "flops_per_byte")[0][1]
        achieved = select(rows, "regimes", variant, "achieved_tflops")[0][1]
        regime = "prefill" if variant.startswith("prefill") else "decode"
        ax.scatter(
            intensity,
            achieved,
            s=90,
            zorder=3,
            color=REGIME_COLOURS[regime],
            edgecolor="white",
            linewidth=1,
        )
        ax.annotate(
            variant.replace("_", " "),
            (intensity, achieved),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("achieved throughput (TFLOP/s)")
    ax.set_title(f"Inference matmuls on the roofline — {profile.device_name}, {profile.dtype}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    path = RESULTS_DIR / "roofline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_batch_walk(rows: list[dict[str, str]], profile: HardwareProfile) -> Path:
    """The decode point climbing toward the ridge as batch size rises."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_roof(ax, profile)

    points = []
    for variant in sorted({r["variant"] for r in rows if r["experiment"] == "batch_walk"}):
        intensity = select(rows, "batch_walk", variant, "flops_per_byte")[0][1]
        achieved = select(rows, "batch_walk", variant, "achieved_tflops")[0][1]
        batch = int(variant.rsplit("_", 1)[1])
        points.append((batch, intensity, achieved))
    points.sort()

    ax.plot(
        [p[1] for p in points],
        [p[2] for p in points],
        "-o",
        color="#d62728",
        markersize=6,
        zorder=3,
        label="decode MLP projection, batch 1 -> 256",
    )
    for batch, intensity, achieved in points:
        if batch in (1, 8, 64, 256):
            ax.annotate(
                f"B={batch}",
                (intensity, achieved),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("achieved throughput (TFLOP/s)")
    ax.set_title(f"Batching walks decode toward the ridge — {profile.device_name}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    path = RESULTS_DIR / "batch_walk.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    rows = read_rows(CSV_PATH)
    profile = load_profile()
    for path in (plot_roofline(rows, profile), plot_batch_walk(rows, profile)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
