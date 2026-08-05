"""Figures for T9. The first one carries the whole argument.

    uv run python -m topics.t09_interconnects.plot

`allreduce_cost.png` is the topic in one image: the measured cost curve with the fitted two-term
model over it, the crossover marked, and — the point of the whole exercise — decode's operating
points and T5's prefill point plotted on the same axis, four orders of magnitude apart. A reader
who looks only at that figure should come away with the finding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arch_common.results_io import read_rows, scalar, select  # noqa: E402
from topics.t09_interconnects.measure import CSV_PATH  # noqa: E402
from topics.t09_interconnects.model import (  # noqa: E402
    DEFAULT_HIDDEN,
    RingCost,
    allreduce_bytes,
    prefill_allreduce_bytes,
)
from topics.t09_interconnects.predict import T5_BATCH, T5_SEQ  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
KB = 1024.0


def _worlds(rows: list[dict[str, str]]) -> list[int]:
    return sorted(
        {
            int(r["variant"].removeprefix("world"))
            for r in rows
            if r["experiment"] == "sweep" and r["variant"].startswith("world")
        }
    )


def _fit(rows: list[dict[str, str]], world: int) -> RingCost:
    v = f"world{world}"
    return RingCost(
        world=world,
        alpha_us=scalar(rows, "fit", v, "alpha_us"),
        beta_gbps=scalar(rows, "fit", v, "beta_gbps"),
        r_squared=scalar(rows, "fit", v, "fit_r_squared"),
        n_points=0,
    )


def plot_cost_curve(rows: list[dict[str, str]]) -> None:
    """Measured all-reduce time vs message size, with the fit, the corner, and where work lands."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colours = {2: "#1f77b4", 4: "#d62728"}

    for world in _worlds(rows):
        points = select(rows, "sweep", f"world{world}", "allreduce_us")
        if not points:
            continue
        xs = [n / KB for n, _ in points]
        ys = [t for _, t in points]
        colour = colours.get(world, "#555555")
        ax.plot(xs, ys, "o", ms=4, color=colour, label=f"{world} GPUs (measured)")

        fit = _fit(rows, world)
        model = [fit.time_us(int(n)) for n, _ in points]
        ax.plot(xs, model, "-", lw=1.3, color=colour, alpha=0.75)
        ax.axhline(fit.alpha_us, color=colour, ls=":", lw=1, alpha=0.6)
        ax.annotate(
            f"α = {fit.alpha_us:.1f} µs",
            xy=(xs[0], fit.alpha_us),
            xytext=(4, 5),
            textcoords="offset points",
            color=colour,
            fontsize=9,
        )
        ax.axvline(fit.crossover_bytes() / KB, color=colour, ls="--", lw=1, alpha=0.45)

    # Where the work actually is. Labels are placed in axes-fraction coordinates on the y axis
    # (`get_xaxis_transform`) so they sit inside the plot regardless of how the log limits land —
    # positioning them at a data-space y clipped them off the bottom of the figure.
    for batch in (1, 32, 128):
        x = allreduce_bytes(batch, DEFAULT_HIDDEN) / KB
        ax.axvline(x, color="#2ca02c", lw=1, alpha=0.55)
        ax.text(
            x,
            0.03,
            f" decode b{batch}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color="#2ca02c",
        )

    t5 = prefill_allreduce_bytes(T5_BATCH, T5_SEQ, DEFAULT_HIDDEN) / KB
    ax.axvline(t5, color="#9467bd", lw=1.4, alpha=0.85)
    ax.text(
        t5,
        0.03,
        f" T5 prefill {T5_BATCH}×{T5_SEQ}",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=8,
        color="#9467bd",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("all-reduce payload (KB)")
    ax.set_ylabel("time per call (µs)")
    ax.set_title(
        "One collective, two cost regimes — and inference runs in the flat one\n"
        "dotted: fixed cost α   dashed: crossover where moving bytes starts to dominate",
        fontsize=11,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "allreduce_cost.png", dpi=150)
    plt.close(fig)


def plot_bus_bandwidth(rows: list[dict[str, str]]) -> None:
    """Achieved bus bandwidth vs size — the same data, in the units a spec sheet uses."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for world in _worlds(rows):
        points = select(rows, "sweep", f"world{world}", "bus_gbps")
        if not points:
            continue
        ax.plot(
            [n / KB for n, _ in points], [v for _, v in points], "o-", ms=4, label=f"{world} GPUs"
        )

    for batch in (1, 128):
        ax.axvline(allreduce_bytes(batch, DEFAULT_HIDDEN) / KB, color="#2ca02c", lw=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("all-reduce payload (KB)")
    ax.set_ylabel("bus bandwidth (GB/s)")
    ax.set_title(
        "The link's headline bandwidth is only available to messages decode never sends",
        fontsize=11,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bus_bandwidth.png", dpi=150)
    plt.close(fig)


def plot_decode_tax(rows: list[dict[str, str]]) -> None:
    """Predicted TP speedup vs batch — the decision the topic exists to inform."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for world in _worlds(rows):
        points = select(rows, "decode", f"world{world}", "tp_speedup")
        if not points:
            continue
        ax.plot([b for b, _ in points], [v for _, v in points], "o-", ms=5, label=f"TP{world}")
        ax.axhline(world, ls=":", lw=1, alpha=0.4)

    measured = [
        (b, v)
        for world in _worlds(rows)
        for b, v in select(rows, "tp_matmul", f"world{world}", "tp_speedup")
    ]
    if measured:
        ax.plot(
            [b for b, _ in measured],
            [v for _, v in measured],
            "x",
            ms=8,
            color="k",
            label="measured (TP matmul)",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode batch size")
    ax.set_ylabel("speedup vs 1 GPU")
    ax.set_title("Sharding buys least exactly where latency matters most", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "decode_tax.png", dpi=150)
    plt.close(fig)


def _main() -> None:
    rows = read_rows(CSV_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_cost_curve(rows)
    plot_bus_bandwidth(rows)
    plot_decode_tax(rows)
    print(f"wrote 3 figures -> {RESULTS_DIR}")


if __name__ == "__main__":
    _main()
