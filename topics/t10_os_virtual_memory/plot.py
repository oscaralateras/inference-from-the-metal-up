"""Figures for T10. The first one is the topic.

    uv run python -m topics.t10_os_virtual_memory.plot

`load_paths.png` is the finding in one image: four bars, each split into what the load call
reported and what it deferred. The loader that looks fastest and the loader that is fastest are not
the same bar, and the picture says so without a caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arch_common.results_io import read_rows, select  # noqa: E402
from topics.t10_os_virtual_memory.measure import CSV_PATH  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# Ordered so the two cache states sit next to each other for each loader, which is the comparison
# a reader makes first.
PATH_ORDER = ("read_cold", "mmap_cold", "read_warm", "mmap_warm")


def _value(rows: list[dict[str, str]], variant: str, metric: str) -> float:
    points = select(rows, "load", variant, metric)
    return points[0][1] if points else 0.0


def plot_load_paths(rows: list[dict[str, str]]) -> None:
    """Load time and deferred time, stacked. The gap between the two is the whole point."""
    present = [v for v in PATH_ORDER if select(rows, "load", v, "load_seconds")]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    load = [_value(rows, v, "load_seconds") for v in present]
    touch = [_value(rows, v, "first_touch_seconds") for v in present]

    ax.bar(present, load, color="#1f77b4", label="reported by the load call")
    ax.bar(
        present,
        touch,
        bottom=load,
        color="#d62728",
        label="deferred — paid by the first pass that touches the pages",
    )

    for i in range(len(present)):
        total = load[i] + touch[i]
        share = touch[i] / total if total else 0.0
        ax.text(i, total, f"  {share:.0%} deferred", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("seconds")
    ax.set_title(
        "The loader that looks fastest and the loader that is fastest are not the same bar\n"
        "mmap returns before it has read anything; the bill arrives at the first touch",
        fontsize=11,
    )
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "load_paths.png", dpi=150)
    plt.close(fig)


def plot_h2d(rows: list[dict[str, str]]) -> bool:
    """Pinned vs pageable H2D against transfer size, with the two-copy prediction over it."""
    pinned = select(rows, "h2d", "transfer", "pinned_gbps")
    if not pinned:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, style, colour, label in (
        ("pinned_gbps", "o-", "#1f77b4", "pinned — DMA reads it directly"),
        ("pageable_gbps", "o-", "#d62728", "pageable — staged through a hidden pinned buffer"),
        (
            "serial_bound_gbps",
            ":",
            "#555555",
            "serial two-copy bound — measured beats it via overlap",
        ),
        ("memcpy_gbps", "--", "#2ca02c", "the staging copy alone (host to host)"),
    ):
        points = select(rows, "h2d", "transfer", metric)
        if points:
            ax.plot(
                [n / 1024**2 for n, _ in points],
                [v for _, v in points],
                style,
                color=colour,
                label=label,
                ms=4,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("transfer size (MiB)")
    ax.set_ylabel("GB/s")
    ax.set_title(
        "Pageable host memory cannot be a DMA source, so every byte is copied twice",
        fontsize=11,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "h2d_bandwidth.png", dpi=150)
    plt.close(fig)
    return True


def _main() -> None:
    rows = read_rows(CSV_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_load_paths(rows)
    written = 1 + int(plot_h2d(rows))
    print(f"wrote {written} figures -> {RESULTS_DIR}")


if __name__ == "__main__":
    _main()
