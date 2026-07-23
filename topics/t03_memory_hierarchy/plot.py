"""Plots for the T3 memory-hierarchy study — reads results/*.csv, writes PNGs.

Same house style as T1/T2: Okabe-Ito colourblind-safe palette, recessive axes/grid, direct
value labels, headless Agg. Figures, each matched to its experiment:

  - memory_bandwidth.png : the bandwidth *staircase* — GB/s vs working-set size on a log axis,
    with the machine's real L1/L2/L3 boundaries marked. One series, so no legend; the shape is
    the message (fast in cache, a cliff down to DRAM).
  - tiling_gflops.png    : grouped bars — naive / ikj / blocked GFLOP/s at each matrix size.
    Shows the naive triple-loop collapsing as the matrix grows while the cache-friendly kernels
    hold flat. Same arithmetic in every bar; only the memory-access pattern differs. (naive is
    O(N^3) and only run up to 1024, so its bars are absent at the larger sizes.)
  - crossover.png        : TWO panels sharing the batch axis (never a dual-axis chart) —
    throughput (GFLOP/s, rises then plateaus) beside per-request latency (ms, keeps rising).
    The CPU version: shows the batching *mechanism*, but the CPU's low ridge point means small
    batches are overhead/compute-limited, not memory-bandwidth-limited (see the lab note).
  - crossover_gpu.png    : the same sweep on a Tesla T4 (optional; only if crossover_gpu.csv
    exists). On a GPU the ridge point is high, so B=1 genuinely saturates HBM bandwidth
    (memory-bound decode) before the throughput climbs to the compute-bound plateau.

Run (writes into results/):
    uv run python topics/t03_memory_hierarchy/plot.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: canonical numbers come from a display-less x86 box

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "memory.csv"
GPU_CSV_PATH = RESULTS_DIR / "crossover_gpu.csv"

# Okabe-Ito palette (same fixed choices as T1/T2). Colour carries meaning consistently.
BLUE = "#0072B2"  # the measured quantity (bandwidth / throughput)
GREEN = "#009E73"  # bluish green — the best / cache-friendly variant
VERMILLION = "#D55E00"  # the slow / costly variant (naive matmul; latency; bandwidth-used)
SKY = "#56B4E9"  # the middle variant (ikj)
GREY = "0.45"

# Tiling: three methods in fixed order, coloured bad -> good so colour reads as a verdict.
TILING_METHODS: list[tuple[str, str, str]] = [
    ("naive", VERMILLION, "naive ijk (cache-thrashing)"),
    ("ikj", SKY, "loop-reordered ikj (streams)"),
    ("blocked", GREEN, "cache-blocked (tiled)"),
]

T4_HBM_PEAK_GBPS = 320.0  # Tesla T4 datasheet HBM bandwidth — the memory ceiling on that card.

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


def _fmt_bytes(n: float) -> str:
    """Human-readable byte size for axis ticks: 4096 -> '4K', 33554432 -> '32M'."""
    n = int(n)
    for unit, suffix in ((1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")):
        if n >= unit:
            return f"{n // unit}{suffix}"
    return str(n)


def load() -> dict[str, list[tuple[str, int, float]]]:
    """Read memory.csv into {experiment: [(variant, size, value), ...]}."""
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found — run `make run` (ideally on the Linux x86 box) first."
        )
    rows: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            rows[row["experiment"]].append((row["variant"], int(row["size"]), float(row["value"])))
    return rows


def load_gpu() -> dict[str, list[tuple[int, float]]] | None:
    """Read the optional GPU sweep into {variant: [(batch, value), ...]}; None if absent."""
    if not GPU_CSV_PATH.exists():
        return None
    rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with GPU_CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            rows[row["variant"]].append((int(row["size"]), float(row["value"])))
    return rows


def plot_bandwidth(data: dict[str, list[tuple[str, int, float]]]) -> Path:
    """The bandwidth staircase, with the machine's real cache boundaries marked."""
    pts = sorted((size, val) for _, size, val in data["bandwidth"])
    xs = [size for size, _ in pts]
    ys = [val for _, val in pts]
    caches = {v: size for v, size, _ in data.get("cache", [])}  # {'L1': 32768, ...}

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=BLUE, zorder=3)

    ymax = max(ys) * 1.14
    for name in ("L1", "L2", "L3"):
        size = caches.get(name)
        if size:
            ax.axvline(size, color=GREY, linewidth=1.0, linestyle="--", zorder=2)
            ax.text(
                size,
                ymax * 0.985,
                f" {name}\n {_fmt_bytes(size)}",
                color=GREY,
                fontsize=9,
                va="top",
                ha="left",
            )

    # Name the four regimes across the top (geometric midpoints read centred on a log axis).
    l1, l2, l3 = caches.get("L1", 0), caches.get("L2", 0), caches.get("L3", 0)
    xmin, xmax = min(xs), max(xs)
    for label, a, b in (("in L1", xmin, l1), ("L2", l1, l2), ("L3", l2, l3), ("DRAM", l3, xmax)):
        if a and b and b > a:
            ax.text(
                math.sqrt(a * b),
                ymax * 0.62,
                label,
                color=GREY,
                fontsize=10,
                ha="center",
                va="center",
                style="italic",
            )

    dram_bw = ys[-1]
    ax.annotate(
        f"DRAM plateau ≈ {dram_bw:.0f} GB/s\n(the decode bandwidth ceiling)",
        xy=(xs[-1], dram_bw),
        xytext=(xs[-1] * 0.18, dram_bw + max(ys) * 0.22),
        fontsize=9.5,
        color="0.2",
        arrowprops={"arrowstyle": "->", "color": "0.4", "linewidth": 1.1},
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _fmt_bytes(v)))
    ax.set_xlabel("working-set size (bytes, log scale)")
    ax.set_ylabel("streaming read bandwidth (GB/s) — higher is faster")
    ax.set_ylim(0, ymax)
    ax.set_title(
        "Memory bandwidth falls as the working set outgrows each cache\n"
        "sequential read; the steps land on this machine's real L1/L2/L3 sizes"
    )
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = RESULTS_DIR / "memory_bandwidth.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tiling(data: dict[str, list[tuple[str, int, float]]]) -> Path:
    """Grouped bars: naive vs ikj vs blocked GFLOP/s at each matrix size.

    naive is only measured up to 1024 (O(N^3), too slow above), so its bars are simply absent at
    the larger sizes — that gap is part of the story, not a rendering bug.
    """
    by_key = {(v, size): val for v, size, val in data["tiling"]}
    sizes = sorted({size for _, size, _ in data["tiling"]})
    l3 = {v: s for v, s, _ in data.get("cache", [])}.get("L3", 0)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    n_methods = len(TILING_METHODS)
    bar_w = 0.8 / n_methods
    for m, (variant, colour, _label) in enumerate(TILING_METHODS):
        offset = (m - (n_methods - 1) / 2) * bar_w
        xs = [i + offset for i, s in enumerate(sizes) if (variant, s) in by_key]
        vals = [by_key[(variant, s)] for s in sizes if (variant, s) in by_key]
        ax.bar(xs, vals, width=bar_w * 0.92, color=colour, zorder=3)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color="0.25")

    # Mark where a matrix (N*N*4 bytes) outgrows the L3 — the regime blocking is meant to help.
    if l3:
        spill_n = math.sqrt(l3 / 4)  # N at which one N×N float matrix == L3 size
        boundary = next((i - 0.5 for i, s in enumerate(sizes) if s > spill_n), None)
        if boundary is not None:
            ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle=":", zorder=1)
            ax.text(
                boundary,
                ax.get_ylim()[1] * 0.02,
                "  matrices spill L3 →",
                color=GREY,
                fontsize=9,
                ha="left",
                va="bottom",
                style="italic",
            )

    ax.set_xticks(list(range(len(sizes))))
    ax.set_xticklabels([f"{s}×{s}\n{s * s * 4 / (1 << 20):g} MB" for s in sizes])
    ax.set_xlabel("matrix size (memory per matrix) — 'naive' skipped above 1024 (O(N³))")
    ax.set_ylabel("GFLOP/s — higher is faster")
    ax.set_ylim(0, max(val for _, _, val in data["tiling"]) * 1.18)
    ax.set_title(
        "Same matrix multiply, three memory-access patterns\n"
        "identical arithmetic and identical result — only cache behaviour differs"
    )
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=c, label=lbl) for _, c, lbl in TILING_METHODS],
        frameon=False,
        loc="upper right",
    )

    fig.tight_layout()
    path = RESULTS_DIR / "tiling_gflops.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _batch_axis(ax: Axes, batches: list[int], label: str = "batch size (log scale)") -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel(label)
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_crossover(data: dict[str, list[tuple[str, int, float]]]) -> Path:
    """CPU: two panels sharing the batch axis — throughput (plateaus) vs latency (keeps rising)."""
    tp = sorted((b, v) for variant, b, v in data["crossover"] if variant == "batched")
    lat = sorted((b, v) for variant, b, v in data["crossover"] if variant == "latency")
    batches = [b for b, _ in tp]
    gflops = [v for _, v in tp]
    ms = [v for _, v in lat]
    plateau = max(gflops)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax1.plot(batches, gflops, marker="o", markersize=5, linewidth=2, color=BLUE, zorder=3)
    ax1.axhline(plateau, color=GREEN, linewidth=1.1, linestyle="--", zorder=2)
    ax1.text(
        batches[0],
        plateau,
        f" compute ceiling ≈ {plateau:.1f} GFLOP/s",
        color=GREEN,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    ax1.set_ylabel("throughput (GFLOP/s) — higher is more tokens/sec")
    ax1.set_ylim(0, plateau * 1.25)
    ax1.set_title("Throughput climbs, then plateaus\n(each weight reused across the batch)")
    _batch_axis(ax1, batches, "batch size (tokens together, log scale)")

    ax2.plot(batches, ms, marker="o", markersize=5, linewidth=2, color=VERMILLION, zorder=3)
    ax2.set_ylabel("latency per step (ms) — lower is snappier")
    ax2.set_ylim(0, max(ms) * 1.15)
    ax2.set_title("But latency keeps rising\n(a wider batch does more work per step)")
    _batch_axis(ax2, batches, "batch size (tokens together, log scale)")

    fig.suptitle(
        "CPU: batching trades latency for throughput\n"
        "(mechanism only — this CPU's low ridge point means small B is overhead-limited, not "
        "bandwidth-limited)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = RESULTS_DIR / "crossover.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_crossover_gpu(gpu: dict[str, list[tuple[int, float]]]) -> Path:
    """GPU: throughput (log-log, plateaus) beside bandwidth-used (B=1 saturates HBM)."""
    tp = sorted(gpu["batched"])
    bw = sorted(gpu["bandwidth"])
    batches = [b for b, _ in tp]
    gflops = [v for _, v in tp]
    gbps = [v for _, v in bw]
    plateau = max(gflops)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # Panel A — throughput rises ~100x from decode (B=1) to the compute-bound plateau.
    ax1.plot(batches, gflops, marker="o", markersize=5, linewidth=2, color=BLUE, zorder=3)
    ax1.set_yscale("log")
    ax1.axhline(plateau, color=GREEN, linewidth=1.1, linestyle="--", zorder=2)
    ax1.text(
        batches[0],
        plateau,
        f" compute ceiling ≈ {plateau / 1000:.0f} TFLOP/s",
        color=GREEN,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    ax1.annotate(
        "B=1 decode\n(memory-bound)",
        xy=(batches[0], gflops[0]),
        xytext=(batches[0] * 1.5, gflops[0] * 4),
        fontsize=9,
        color="0.2",
        arrowprops={"arrowstyle": "->", "color": "0.4", "linewidth": 1.1},
    )
    ax1.set_ylabel("throughput (GFLOP/s, log) — higher is more tokens/sec")
    ax1.set_title(f"Throughput: {plateau / gflops[0]:.0f}× from decode to batched")
    _batch_axis(ax1, batches, "batch size (tokens together, log scale)")

    # Panel B — bandwidth used: B=1 saturates HBM (memory-bound), large B leaves it idle.
    ax2.plot(batches, gbps, marker="o", markersize=5, linewidth=2, color=VERMILLION, zorder=3)
    ax2.axhline(T4_HBM_PEAK_GBPS, color=GREY, linewidth=1.1, linestyle="--", zorder=2)
    ax2.text(
        batches[0],
        T4_HBM_PEAK_GBPS,
        f" HBM peak ≈ {T4_HBM_PEAK_GBPS:.0f} GB/s",
        color=GREY,
        fontsize=9,
        va="bottom",
        ha="left",
    )
    ax2.set_ylabel("memory bandwidth used (GB/s)")
    ax2.set_ylim(0, T4_HBM_PEAK_GBPS * 1.12)
    ax2.set_title(
        "B=1 nearly saturates HBM (memory-bound);\nbig batch leaves it idle (compute-bound)"
    )
    _batch_axis(ax2, batches, "batch size (tokens together, log scale)")

    fig.suptitle(
        "Tesla T4 GPU — the crossover the CPU can't show: B=1 decode is memory-bandwidth-bound",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = RESULTS_DIR / "crossover_gpu.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load()
    written = [plot_bandwidth(data), plot_tiling(data), plot_crossover(data)]
    gpu = load_gpu()
    if gpu is not None:
        written.append(plot_crossover_gpu(gpu))
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
