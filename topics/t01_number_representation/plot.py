"""Plots for the quantisation study — writes two PNGs into results/.

Uses the Okabe-Ito colourblind-safe categorical palette (assigned to granularities in a fixed
order, never cycled), recessive axes/grid, and direct legends, so the figures read as designed
rather than default-matplotlib.

  - quality_vs_bits.png : two panels — output SQNR (dB) and output cosine — vs bit-width, one line
    per granularity. The standard quant-fidelity view of how granularity x bit-width drives impact.
  - weight_outliers.png : the |weight| distribution with the int4 per-tensor step marked — the
    mechanism behind the per-tensor collapse.

    uv run python topics/t01_number_representation/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from probe import GRANULARITIES, Result, run_sweep

RESULTS_DIR = Path(__file__).parent / "results"

# Okabe-Ito colourblind-safe palette, assigned to granularities in fixed order (never cycled).
GRAN_COLOR: dict[str, str] = {
    "per_tensor": "#D55E00",  # vermillion
    "per_channel": "#0072B2",  # blue
    "per_group": "#009E73",  # bluish green
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _style(ax: Axes) -> None:
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_xaxis()  # 8 bits (left) -> 2 bits (right): more compression rightward
    ax.set_xlabel("bit-width")


def plot_quality_vs_bits(results: list[Result]) -> Path:
    """Two panels — output SQNR (dB) and output cosine — vs bit-width, one line per granularity."""
    fig, (ax_sqnr, ax_cos) = plt.subplots(1, 2, figsize=(11, 4.6))
    for gran in GRANULARITIES:
        rows = sorted((r for r in results if r.granularity == gran), key=lambda r: r.n_bits)
        bits = [r.n_bits for r in rows]
        color = GRAN_COLOR[gran]
        ax_sqnr.plot(
            bits, [r.out_sqnr for r in rows], marker="o", ms=6, lw=2, color=color, label=gran
        )
        ax_cos.plot(
            bits, [r.out_cos for r in rows], marker="o", ms=6, lw=2, color=color, label=gran
        )
    ax_sqnr.set_ylabel("output SQNR (dB) — higher is better")
    ax_cos.set_ylabel("output cosine similarity — higher is better")
    ax_cos.set_ylim(0, 1.02)
    for ax in (ax_sqnr, ax_cos):
        _style(ax)
    ax_sqnr.legend(title="granularity", frameon=False)
    fig.suptitle(
        "Quantisation impact vs bit-width & granularity\nSmolLM2-135M gate_proj weight", fontsize=13
    )
    fig.tight_layout()
    path = RESULTS_DIR / "quality_vs_bits.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weight_outliers(w: npt.NDArray[np.float32]) -> Path:
    """Histogram of |weight| with the int4 per-tensor step and the max marked."""
    absw = np.abs(w).reshape(-1)
    max_abs = float(absw.max())
    int4_step = max_abs / (2 ** (4 - 1) - 1)  # per-tensor int4 step = max / 7

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(absw, bins=120, color="#56B4E9", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(
        int4_step, color="#D55E00", lw=2, ls="--", label=f"int4 per-tensor step ≈ {int4_step:.2f}"
    )
    ax.axvline(
        max_abs, color="0.2", lw=1.5, ls=":", label=f"max|w| = {max_abs:.2f} (sets the scale)"
    )
    ax.set_xlabel("|weight|")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Why per-tensor int4 collapses:\nmost weights fall below a single quant step")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = RESULTS_DIR / "weight_outliers.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results, w = run_sweep()
    p1 = plot_quality_vs_bits(results)
    p2 = plot_weight_outliers(w)
    print(f"\nwrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
