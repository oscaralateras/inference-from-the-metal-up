"""Plots for the quantisation study — writes two PNGs into results/.

- quality_vs_bits.png : output SQNR vs bit-width, one line per granularity. The degradation
  curve and the granularity gap, at a glance.
- weight_outliers.png : the |weight| distribution with the int4 per-tensor step marked — shows
  *why* per-tensor collapses (most weights fall below a single quantisation step).

  uv run python topics/t01_number_representation/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from load_weights import load_linear_weight
from probe import GRANULARITIES, Result, run_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def plot_quality_vs_bits(results: list[Result]) -> Path:
    """Line chart: output SQNR (dB) vs bit-width, one line per granularity."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for gran in GRANULARITIES:
        rows = sorted((r for r in results if r.granularity == gran), key=lambda r: r.n_bits)
        ax.plot([r.n_bits for r in rows], [r.out_sqnr for r in rows], marker="o", label=gran)
    ax.set_xlabel("bit-width")
    ax.set_ylabel("output SQNR (dB) — higher is better")
    ax.set_title("Quantisation quality vs bit-width\nSmolLM2-135M gate_proj weight")
    ax.legend(title="granularity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = RESULTS_DIR / "quality_vs_bits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weight_outliers(w: npt.NDArray[np.float32]) -> Path:
    """Histogram of |weight|, with the int4 per-tensor step and the max marked."""
    absw = np.abs(w).reshape(-1)
    max_abs = float(absw.max())
    int4_step = max_abs / (2 ** (4 - 1) - 1)  # per-tensor int4 step = max / 7

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(absw, bins=120)
    ax.set_yscale("log")
    ax.axvline(
        int4_step, color="red", linestyle="--", label=f"int4 per-tensor step ≈ {int4_step:.2f}"
    )
    ax.axvline(
        max_abs, color="black", linestyle=":", label=f"max|w| = {max_abs:.2f} (sets the scale)"
    )
    ax.set_xlabel("|weight|")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Why per-tensor int4 collapses\nmost weights fall below a single quant step")
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / "weight_outliers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_sweep()
    w = load_linear_weight()
    p1 = plot_quality_vs_bits(results)
    p2 = plot_weight_outliers(w)
    print(f"\nwrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
