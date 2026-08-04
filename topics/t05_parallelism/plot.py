"""Figures for T5 — reads results/parallelism.csv, writes results/*.png.

Same house style as the earlier topics: Okabe-Ito colourblind-safe palette assigned in a fixed
order (never cycled), recessive axes and grid, direct labelling in preference to legends where it
fits, so the figures read as designed rather than as default matplotlib.

  amdahl_calibration.png — measured speedup curves for four injected serial fractions, with the
      fitted Amdahl curve over each. The test of the method: does the fit recover what was put in?
  amdahl_t4.png         — the same estimator pointed at T4's contention curves, where the serial
      fraction was unknown. Shows sharded fitting cleanly and mutex/atomic falling outside the
      model's domain entirely.
  pipeline_bubble.png   — measured vs predicted pipeline efficiency against microbatch count, for
      a balanced and a deliberately imbalanced stage layout.
  strategies_throughput.png — the five strategies' throughput against world size.
  strategies_cost.png   — what throughput hides: bytes communicated per step, and weight bytes
      each rank must hold. The axis on which DP and SP lose.

    uv run python topics/t05_parallelism/plot.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from results_io import read_rows

RESULTS_DIR = Path(__file__).parent / "results"

# Okabe-Ito, assigned in fixed order so a strategy keeps its colour across every figure.
COLOR: dict[str, str] = {
    "dp": "#0072B2",  # blue
    "tp": "#D55E00",  # vermillion
    "pp": "#009E73",  # bluish green
    "sp": "#CC79A7",  # reddish purple
    "ep": "#E69F00",  # orange
    "balanced": "#0072B2",
    "imbalanced": "#D55E00",
    "mutex": "#D55E00",
    "atomic": "#E69F00",
    "sharded": "#009E73",
}
LABEL = {
    "dp": "data",
    "tp": "tensor",
    "pp": "pipeline",
    "sp": "sequence",
    "ep": "expert",
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
        "axes.edgecolor": "#666666",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "figure.autolayout": True,
    }
)


def _series(experiment: str, metric: str) -> dict[str, dict[int, float]]:
    """{variant: {workers: value}} for one experiment/metric, skipping summary rows."""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r in read_rows(experiment):
        if r["metric"] != metric:
            continue
        workers = int(r["workers"])
        if workers == 0:  # summary row (a fitted constant), not a per-worker observation
            continue
        out[r["variant"]][workers] = float(r["value"])
    return out


def _summary(experiment: str, metric: str) -> dict[str, float]:
    """{variant: value} for the workers==0 summary rows."""
    return {
        r["variant"]: float(r["value"])
        for r in read_rows(experiment)
        if r["metric"] == metric and int(r["workers"]) == 0
    }


def _amdahl(n: float, p: float) -> float:
    return 1.0 / ((1.0 - p) + p / n)


def _style_worker_axis(ax: Axes, workers: list[int]) -> None:
    ax.set_xlabel("workers")
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers])


# ---- figures --------------------------------------------------------------------------------


def plot_amdahl_calibration() -> None:
    curves = _series("amdahl_calibration", "speedup")
    fits = _summary("amdahl_calibration", "recovered_p")
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    palette = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]

    for i, variant in enumerate(sorted(curves, key=lambda v: -float(v.rsplit("_", 1)[1]))):
        pts = curves[variant]
        workers = sorted(pts)
        injected_serial = float(variant.rsplit("_", 1)[1])
        p_true = 1.0 - injected_serial
        p_hat = fits.get(variant, p_true)
        c = palette[i % len(palette)]

        ax.plot(workers, [pts[w] for w in workers], "o", color=c, markersize=5, zorder=3)
        fine = [w / 4 for w in range(4, max(workers) * 4 + 1)]
        ax.plot(fine, [_amdahl(n, p_hat) for n in fine], "-", color=c, linewidth=1.6, alpha=0.9)
        ax.annotate(
            f"injected p={p_true:.2f} → fit {p_hat:.3f}",
            xy=(max(workers), pts[max(workers)]),
            xytext=(4, 0),
            textcoords="offset points",
            color=c,
            fontsize=9,
            va="center",
        )

    workers_all = sorted({w for pts in curves.values() for w in pts})
    ax.plot(workers_all, workers_all, ":", color="#999999", linewidth=1, label="ideal (linear)")
    _style_worker_axis(ax, workers_all)
    ax.set_ylabel("speedup vs 1 worker")
    ax.set_title("Amdahl's law, measured backwards: does the fit recover the injected fraction?")
    ax.set_xlim(0.8, max(workers_all) * 1.55)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(RESULTS_DIR / "amdahl_calibration.png", bbox_inches="tight")
    plt.close(fig)


def plot_amdahl_t4() -> None:
    curves = _series("amdahl_t4", "speedup")
    fits = _summary("amdahl_t4", "recovered_p")
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for variant in ("sharded", "atomic", "mutex"):
        if variant not in curves:
            continue
        pts = curves[variant]
        workers = sorted(pts)
        c = COLOR[variant]
        p_hat = fits.get(variant, 0.0)
        ax.plot(workers, [pts[w] for w in workers], "o-", color=c, linewidth=1.8, markersize=5)
        note = "outside Amdahl's domain" if p_hat < 0 else f"fits p = {p_hat:.3f}"
        ax.annotate(
            f"{variant} — {note}",
            xy=(max(workers), pts[max(workers)]),
            xytext=(5, 0),
            textcoords="offset points",
            color=c,
            fontsize=9,
            va="center",
        )

    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
    ax.annotate(
        "Amdahl's floor: 1.0x\n(the model assumes coordination is free)",
        xy=(1.05, 1.0),
        xytext=(0, 12),
        textcoords="offset points",
        fontsize=8.5,
        color="#666666",
    )
    workers_all = sorted({w for pts in curves.values() for w in pts})
    _style_worker_axis(ax, workers_all)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs 1 thread (log)")
    ax.set_title("The same estimator on T4's contention curves")
    ax.set_xlim(0.8, max(workers_all) * 1.9)
    fig.savefig(RESULTS_DIR / "amdahl_t4.png", bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_bubble() -> None:
    """The bubble law, with the one GPU measurement that tests it plotted on top.

    Deliberately does NOT plot the CPU microbatch sweep. That sweep runs, but on a machine where
    BLAS matmul does not parallelise across threads its efficiency numbers are depressed by the
    hardware rather than by the bubble — plotting them would put a number on the chart that
    measures the wrong thing. What is plotted instead is the exact prediction, plus the pipeline
    efficiency actually measured on 4 GPUs, which is the honest test of the law.
    """
    predicted = _series("pipeline_bubble", "predicted_efficiency")
    if not predicted:
        return

    exp = _strategy_experiment()
    pp = _series(exp, "tokens_per_s").get("pp", {}) if exp else {}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for layout, style in (("balanced", "-"), ("imbalanced", "--")):
        if layout not in predicted:
            continue
        ms = sorted(predicted[layout])
        ax.plot(
            ms,
            [predicted[layout][m] for m in ms],
            style,
            color=COLOR[layout],
            linewidth=2,
            label=f"{layout} stages — predicted",
        )
        ax.annotate(
            f"{layout}\nceiling {predicted[layout][max(ms)]:.2f}",
            xy=(max(ms), predicted[layout][max(ms)]),
            xytext=(6, 0),
            textcoords="offset points",
            color=COLOR[layout],
            fontsize=8.5,
            va="center",
        )

    # The GPU test: PP ran with P=4 stages and M=8 microbatches. Its measured efficiency is
    # speedup / stages, directly comparable to the predicted curve at M=8.
    if pp and 1 in pp and 4 in pp:
        m_used, stages = 8, 4
        measured_eff = (pp[4] / pp[1]) / stages
        ax.plot(
            [m_used],
            [measured_eff],
            "o",
            color="#000000",
            markersize=9,
            zorder=5,
            label=f"measured on 4x A100 (M={m_used})",
        )
        pred = predicted["balanced"][m_used] if m_used in predicted.get("balanced", {}) else None
        if pred:
            ax.annotate(
                f"measured {measured_eff:.2f}\n= {100 * measured_eff / pred:.0f}% of the\n"
                f"bubble ceiling ({pred:.2f})",
                xy=(m_used, measured_eff),
                xytext=(-14, -58),
                textcoords="offset points",
                fontsize=8.5,
                ha="right",
                arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 1},
            )

    ms_all = sorted(predicted.get("balanced", predicted[next(iter(predicted))]))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms_all)
    ax.set_xticklabels([str(m) for m in ms_all])
    ax.set_xlabel("microbatches in flight (M)")
    ax.set_ylabel("pipeline efficiency")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.9, max(ms_all) * 1.9)
    ax.set_title("The pipeline bubble: efficiency is bought with microbatches, not bandwidth")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(RESULTS_DIR / "pipeline_bubble.png", bbox_inches="tight")
    plt.close(fig)


def _strategy_experiment() -> str | None:
    for candidate in ("strategies_nccl", "strategies_gloo"):
        if read_rows(candidate):
            return candidate
    return None


def plot_strategies_throughput() -> None:
    """Scaling, not absolute throughput — the only axis on which all five are comparable.

    Absolute tokens/s would be dishonest here: expert parallelism runs an MoE layer (one expert
    per token) rather than a dense block, so its raw numbers are an order of magnitude larger and
    say nothing about the *decomposition*. Normalising each strategy to its own single-device
    throughput removes the workload difference and leaves exactly the question being asked: how
    well does this way of splitting the model convert extra devices into speed?
    """
    exp = _strategy_experiment()
    if not exp:
        return
    curves = _series(exp, "tokens_per_s")
    comms = _series(exp, "comms_bytes_per_step")
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    workers_all = sorted({w for pts in curves.values() for w in pts})
    ax.plot(workers_all, workers_all, ":", color="#999999", linewidth=1.2, label="ideal (linear)")

    for variant, pts in sorted(curves.items()):
        key = variant.split("_")[0]
        ws = sorted(pts)
        base = pts[min(ws)]
        speedup = [pts[w] / base for w in ws]
        skewed = variant.endswith("skewed")
        name = LABEL.get(key, key) + (" (skewed routing)" if skewed else "")
        mb = comms.get(variant, {}).get(max(ws), 0) / 1e6
        ax.plot(
            ws,
            speedup,
            "o--" if skewed else "o-",
            color=COLOR.get(key, "#666666"),
            linewidth=1.8,
            markersize=5,
        )
        ax.annotate(
            f"{name}  {speedup[-1]:.2f}x   [{mb:,.0f} MB/step]",
            xy=(max(ws), speedup[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=COLOR.get(key, "#666666"),
            fontsize=8.5,
            va="center",
        )

    _style_worker_axis(ax, workers_all)
    ax.set_ylabel("speedup vs 1 device")
    ax.set_title("Five ways to split a transformer — scaling, and what each pays to communicate")
    ax.set_xlim(0.8, max(workers_all) * 2.15)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(RESULTS_DIR / "strategies_throughput.png", bbox_inches="tight")
    plt.close(fig)


def plot_expert_imbalance() -> None:
    """EP under uniform vs skewed routing — identical hardware, FLOPs and communication."""
    exp = _strategy_experiment()
    if not exp:
        return
    curves = _series(exp, "tokens_per_s")
    loads = _series(exp, "load_factor")
    if "ep_uniform" not in curves or "ep_skewed" not in curves:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for variant, color, label in (
        ("ep_uniform", "#009E73", "uniform routing"),
        ("ep_skewed", "#D55E00", "skewed routing"),
    ):
        ws = sorted(curves[variant])
        ax.plot(
            ws,
            [curves[variant][w] / 1e6 for w in ws],
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
        )
        lf = loads.get(variant, {}).get(max(ws), 1.0)
        ax.annotate(
            f"{label}\nload factor {lf:.2f}x",
            xy=(max(ws), curves[variant][max(ws)] / 1e6),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )

    ws = sorted(curves["ep_uniform"])
    lost = 1 - curves["ep_skewed"][max(ws)] / curves["ep_uniform"][max(ws)]
    ax.set_title(
        f"Expert parallelism: routing decides throughput, not hardware\n"
        f"{lost:.0%} of throughput lost to imbalance at {max(ws)} devices — "
        "same FLOPs, same bytes moved"
    )
    _style_worker_axis(ax, ws)
    ax.set_ylabel("throughput (million tokens/s)")
    ax.set_xlim(0.8, max(ws) * 1.7)
    fig.savefig(RESULTS_DIR / "expert_imbalance.png", bbox_inches="tight")
    plt.close(fig)


def plot_strategies_cost() -> None:
    """What throughput hides: communication volume, and how much of the model each rank holds."""
    exp = _strategy_experiment()
    if not exp:
        return
    comms = _series(exp, "comms_bytes_per_step")
    weights = _series(exp, "weight_bytes_per_rank")
    if not comms:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    workers_all = sorted({w for pts in comms.values() for w in pts})

    # A log axis cannot show zero, and substituting a small non-zero value would put a number on
    # the chart that was never measured. Data parallelism genuinely communicates NOTHING at
    # inference time, so it is stated in words instead of drawn as a fake line.
    zero_comms = [
        LABEL.get(v.split("_")[0], v) for v, pts in sorted(comms.items()) if max(pts.values()) == 0
    ]
    for variant, pts in sorted(comms.items()):
        key = variant.split("_")[0]
        ws = sorted(pts)
        if max(pts.values()) == 0:
            continue
        axes[0].plot(
            ws,
            [pts[w] / 1e6 for w in ws],
            "o-",
            color=COLOR.get(key, "#666666"),
            linewidth=1.8,
            markersize=5,
            label=LABEL.get(key, key),
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MB communicated per step (log)")
    axes[0].set_title("Communication volume")
    if zero_comms:
        axes[0].annotate(
            f"{', '.join(zero_comms)} parallel: exactly 0 bytes\n(no collectives at inference — "
            "cannot be drawn on a log axis)",
            xy=(0.03, 0.06),
            xycoords="axes fraction",
            fontsize=8.5,
            color=COLOR["dp"],
        )
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    _style_worker_axis(axes[0], workers_all)

    # TP and PP divide the footprint identically, so one would hide the other. Dash the second.
    seen: set[tuple[float, ...]] = set()
    for variant, pts in sorted(weights.items()):
        key = variant.split("_")[0]
        ws = sorted(pts)
        signature = tuple(pts[w] for w in ws)
        style = "--" if signature in seen else "-"
        seen.add(signature)
        axes[1].plot(
            ws,
            [pts[w] / 1e6 for w in ws],
            "o",
            linestyle=style,
            color=COLOR.get(key, "#666666"),
            linewidth=1.8,
            markersize=5,
            label=LABEL.get(key, key) + (" (overlaps)" if style == "--" else ""),
        )
    axes[1].set_ylabel("model bytes held per rank (MB)")
    axes[1].set_title("Memory per rank — DP and SP replicate")
    axes[1].legend(frameon=False, fontsize=9)
    _style_worker_axis(axes[1], workers_all)

    fig.suptitle("The two costs throughput alone hides", y=1.02)
    fig.savefig(RESULTS_DIR / "strategies_cost.png", bbox_inches="tight")
    plt.close(fig)


def _main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (
        plot_amdahl_calibration,
        plot_amdahl_t4,
        plot_pipeline_bubble,
        plot_strategies_throughput,
        plot_expert_imbalance,
        plot_strategies_cost,
    ):
        fn()
    written = sorted(p.name for p in RESULTS_DIR.glob("*.png"))
    print(f"wrote {len(written)} figures -> {RESULTS_DIR}")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    _main()
