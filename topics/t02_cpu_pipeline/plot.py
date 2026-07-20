"""Plots for the T2 CPU-pipeline study — reads results/pipeline.csv, writes two PNGs.

Same house style as T1: Okabe-Ito colourblind-safe palette, recessive axes/grid, direct
value labels. Two figures, each chosen for the job:

  - pipeline_costs.png    : three small-multiple panels (branch, ILP, SIMD), each with its
    OWN y-axis, showing the pipeline-hostile vs pipeline-friendly variant in ns per
    operation. Small multiples (not one shared axis) because the three effects live at very
    different magnitudes — a shared axis would crush the small ones and lie about the data.
  - pipeline_speedups.png : one bar per experiment = how many times faster the
    pipeline-friendly variant is. Normalises away the differing units -> the 30-second
    headline of the whole artefact.

Run (writes into results/):
    uv run python topics/t02_cpu_pipeline/plot.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: the canonical run happens on a display-less x86 box

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "pipeline.csv"

# Each experiment: (csv key, hostile variant, friendly variant, panel title).
# The "hostile" variant fights the pipeline; the "friendly" one works with it.
EXPERIMENTS: list[tuple[str, str, str, str]] = [
    ("branch", "unsorted", "sorted", "Branch prediction"),
    ("ilp", "dependent", "independent", "Instruction-level parallelism"),
    ("simd", "scalar", "vector", "SIMD (vectorisation)"),
]

# Okabe-Ito palette (same fixed choices as T1). Colour carries meaning consistently:
# vermillion = pipeline-hostile (slow), green = pipeline-friendly (fast).
C_HOSTILE = "#D55E00"  # vermillion
C_FRIENDLY = "#009E73"  # bluish green
C_SPEEDUP = "#0072B2"  # blue — the headline speedup bars

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


def load() -> dict[tuple[str, str], float]:
    """Read pipeline.csv into {(experiment, variant): ns_per_elem}."""
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found — run `make run` (ideally on the Linux x86 box) first."
        )
    rows: dict[tuple[str, str], float] = {}
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            rows[(row["experiment"], row["variant"])] = float(row["ns_per_elem"])
    return rows


def _label_bars(ax: Axes, xs: list[float], values: list[float]) -> None:
    """Direct value label above each bar (3 significant figures)."""
    for x, v in zip(xs, values, strict=True):
        ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9, color="0.2")


def plot_costs(data: dict[tuple[str, str], float]) -> Path:
    """Three panels — hostile vs friendly variant per experiment, ns per operation."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3))
    for ax, (exp, hostile, friendly, title) in zip(axes, EXPERIMENTS, strict=True):
        vals = [data[(exp, hostile)], data[(exp, friendly)]]
        ax.bar([0, 1], vals, width=0.62, color=[C_HOSTILE, C_FRIENDLY], zorder=3)
        _label_bars(ax, [0, 1], vals)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([hostile, friendly])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.18)  # headroom for the value labels
        ax.grid(True, axis="y", color="0.88", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ns per operation — lower is better")

    legend_handles = [
        Patch(facecolor=C_HOSTILE, label="pipeline-hostile"),
        Patch(facecolor=C_FRIENDLY, label="pipeline-friendly"),
    ]
    fig.legend(handles=legend_handles, frameon=False, ncol=2, loc="lower center")
    fig.suptitle(
        "Working with the CPU pipeline vs against it\nsame work each pair — only the "
        "pipeline behaviour changes",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))  # leave room for the bottom legend
    path = RESULTS_DIR / "pipeline_costs.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_speedups(data: dict[tuple[str, str], float]) -> Path:
    """One horizontal bar per experiment = friendly/hostile speedup factor."""
    titles = [title for _, _, _, title in EXPERIMENTS]
    speedups = [data[(exp, h)] / data[(exp, f)] for exp, h, f, _ in EXPERIMENTS]

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ys = list(range(len(titles)))
    ax.barh(ys, speedups, height=0.6, color=C_SPEEDUP, zorder=3)
    ax.axvline(1.0, color="0.5", linewidth=1.2, linestyle="--", zorder=2)  # 1x = no gain
    ax.text(1.0, len(titles) - 0.5, "1× (no gain)", color="0.4", fontsize=9, va="center")

    for y, s in zip(ys, speedups, strict=True):
        ax.text(s, y, f"  {s:.1f}×", va="center", ha="left", fontsize=12, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(titles)
    ax.invert_yaxis()  # first experiment on top
    ax.set_xlim(0, max(speedups) * 1.18)
    ax.set_xlabel("speedup, pipeline-friendly vs pipeline-hostile (×) — higher is a bigger win")
    ax.set_title("How much each CPU-pipeline feature is worth")
    ax.grid(True, axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = RESULTS_DIR / "pipeline_speedups.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load()
    p1 = plot_costs(data)
    p2 = plot_speedups(data)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
