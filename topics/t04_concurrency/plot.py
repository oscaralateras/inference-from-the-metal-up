"""Plots for the T4 concurrency study — reads results/concurrency.csv, writes four PNGs.

Same house style as T1–T3: Okabe-Ito colourblind-safe palette, recessive axes/grid, direct value
labels, headless Agg. One figure per experiment; each plots only if its rows are present.

  - false_sharing.png : two bars — adjacent (shared line) vs padded (own line), ns per increment.
  - race.png          : two bars — % of updates lost, racy (load+store) vs atomic (fetch_add).
  - contention.png    : throughput vs thread count for mutex / atomic / sharded (log-log).
  - scheduler.png     : throughput vs thread count for a global-locked queue vs sharded dispatch.

Run (writes into results/):
    uv run python topics/t04_concurrency/plot.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: canonical numbers come from a display-less x86 box

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "concurrency.csv"

# Okabe-Ito palette. Colour carries meaning: vermillion = the costly/contended variant, green = the
# cheap/scalable one, sky = the in-between (lock-free but still contended).
VERMILLION = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"

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


def load() -> dict[str, list[tuple[str, int, float]]]:
    """Read concurrency.csv into {experiment: [(variant, threads, value), ...]}."""
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found — run `make run` on a multicore x86 box first.")
    rows: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            rows[row["experiment"]].append(
                (row["variant"], int(row["threads"]), float(row["value"]))
            )
    return rows


def _two_bar(
    data: dict[str, list[tuple[str, int, float]]],
    experiment: str,
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    ylabel: str,
    title: str,
    fname: str,
    fmt: str = "{:.2f}",
) -> Path:
    """A two-bar figure (left = costly variant, right = cheap variant), with direct labels."""
    by = {variant: val for variant, _, val in data[experiment]}
    vals = [by[left[0]], by[right[0]]]
    colours = [left[1], right[1]]
    labels = [left[2], right[2]]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar([0, 1], vals, width=0.6, color=colours, zorder=3)
    for x, v in zip([0, 1], vals, strict=True):
        ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color="0.2")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    ax.set_title(title)
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _scaling(
    data: dict[str, list[tuple[str, int, float]]],
    experiment: str,
    series: list[tuple[str, str, str]],
    title: str,
    fname: str,
) -> Path:
    """A throughput-vs-thread-count line figure (log-log), one line per synchronisation strategy."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for variant, colour, label in series:
        pts = sorted((t, v) for var, t, v in data[experiment] if var == variant)
        xs = [t for t, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=colour, label=label, zorder=3)

    all_threads = sorted({t for _, t, _ in data[experiment]})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_threads)
    ax.set_xticklabels([str(t) for t in all_threads])
    ax.set_xlabel("threads (workers, log scale)")
    ax.set_ylabel("throughput (M dispatches/sec, log) — higher is better")
    ax.set_title(title)
    ax.grid(True, which="both", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load()
    written: list[Path] = []

    if "false_sharing" in data:
        threads = data["false_sharing"][0][1]
        written.append(
            _two_bar(
                data,
                "false_sharing",
                ("adjacent", VERMILLION, "adjacent\n(shared cache line)"),
                ("padded", GREEN, "padded\n(own cache line)"),
                "ns per increment — lower is faster",
                f"False sharing: same work, {threads} workers, layout is the only change",
                "false_sharing.png",
            )
        )
    if "race" in data:
        threads = data["race"][0][1]
        written.append(
            _two_bar(
                data,
                "race",
                ("racy", VERMILLION, "racy\n(load + store)"),
                ("atomic", GREEN, "atomic\n(fetch_add)"),
                "% of updates lost — lower is correct",
                f"The race: {threads} workers bump one shared counter",
                "race.png",
                fmt="{:.1f}%",
            )
        )
    if "contention" in data:
        written.append(
            _scaling(
                data,
                "contention",
                [
                    ("mutex", VERMILLION, "mutex (one lock)"),
                    ("atomic", SKY, "atomic (one hot line)"),
                    ("sharded", GREEN, "sharded (own line each)"),
                ],
                "Token dispatch: the lock serialises, sharding scales",
                "contention.png",
            )
        )
    if "scheduler" in data:
        written.append(
            _scaling(
                data,
                "scheduler",
                [
                    ("global_lock", VERMILLION, "global-locked queue"),
                    ("sharded", GREEN, "sharded per-worker queues"),
                ],
                "The scheduler: a central lock bottlenecks, sharded dispatch scales",
                "scheduler.png",
            )
        )

    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
