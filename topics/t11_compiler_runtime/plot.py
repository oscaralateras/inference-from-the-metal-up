"""Figures for T11. The first one is the topic.

    uv run python -m topics.t11_compiler_runtime.plot

`mechanism_crossover.png` puts both speedups on one axis against batch size, so the point where
they cross is the picture rather than a number in a table. A reader who looks only at that figure
should come away knowing that the right optimisation depends on the batch size, and roughly where
the answer changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arch_common.results_io import read_rows, select  # noqa: E402
from topics.t11_compiler_runtime.chain import CHAIN_OPS  # noqa: E402
from topics.t11_compiler_runtime.measure import CSV_PATH  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def plot_crossover(rows: list[dict[str, str]]) -> None:
    """Fusion's speedup and graph capture's, against batch. Where they cross is the finding."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    series = (
        ("fusion_speedup", "#1f77b4", "fusion — removes bytes"),
        ("graph_speedup", "#d62728", "graph capture — removes launches"),
        ("combined_speedup", "#2ca02c", "both"),
        ("byte_model_speedup", "#888888", "fusion's ceiling, from byte counts alone"),
    )
    for metric, colour, label in series:
        points = select(rows, "mechanism", "chain", metric)
        if not points:
            continue
        style = ":" if metric == "byte_model_speedup" else "o-"
        ax.plot([b for b, _ in points], [v for _, v in points], style, color=colour, label=label)

    crossover = select(rows, "crossover", "chain", "crossover_batch")
    if crossover and crossover[0][1] > 0:
        batch = crossover[0][1]
        ax.axvline(batch, color="k", ls="--", lw=1, alpha=0.6)
        ax.annotate(
            f" crossover\n batch {batch:,.0f}",
            xy=(batch, ax.get_ylim()[1]),
            va="top",
            fontsize=9,
        )

    ax.axhline(1.0, color="#555555", lw=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("speedup vs eager")
    ax.set_title(
        "Two mechanisms, two regimes — the right optimisation depends which side you are on\n"
        f"chain: {' -> '.join(CHAIN_OPS)}",
        fontsize=11,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mechanism_crossover.png", dpi=150)
    plt.close(fig)


def plot_launches(rows: list[dict[str, str]]) -> bool:
    """Kernel launches per call, by mode. The mechanism evidence, independent of any timer.

    Fusion has to show up as fewer kernels at the same batch size and graph capture as one launch
    regardless — if the timings move and these do not, the speedup is coming from somewhere the
    note has not identified.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plotted = False

    for mode, colour in (
        ("eager", "#1f77b4"),
        ("compile", "#ff7f0e"),
        ("graph", "#d62728"),
        ("compile_graph", "#2ca02c"),
    ):
        points = [(b, v) for b, v in select(rows, "modes", mode, "kernel_launches") if v >= 0]
        if not points:
            continue
        plotted = True
        ax.plot([b for b, _ in points], [v for _, v in points], "o-", color=colour, label=mode)

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("device kernels per call")
    ax.set_title("What each mechanism actually does to the launch count", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "launch_count.png", dpi=150)
    plt.close(fig)
    return True


def _main() -> None:
    rows = read_rows(CSV_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_crossover(rows)
    written = 1 + int(plot_launches(rows))
    print(f"wrote {written} figures -> {RESULTS_DIR}")


if __name__ == "__main__":
    _main()
