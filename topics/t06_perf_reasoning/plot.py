"""Three figures: the error budget, the serving tradeoff, and the KV cache decay.

python plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arch_common.gpu import load_profile  # noqa: E402
from arch_common.results_io import read_rows, select  # noqa: E402
from topics.t06_perf_reasoning.measure import CSV_PATH  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
TERM_ORDER = ("weights", "kv_cache", "activations", "launch_overhead", "unexplained")
TERM_COLOURS = {
    "weights": "#1f77b4",
    "kv_cache": "#ff7f0e",
    "activations": "#2ca02c",
    "launch_overhead": "#9467bd",
    "unexplained": "#7f7f7f",
}


def plot_error_budget(rows: list[dict[str, str]], device_name: str) -> Path:
    """Stacked bar: where every millisecond of a decode step actually goes.

    The headline figure. A single stacked bar next to the measured total makes the residual
    visible as a quantity rather than a caveat in prose.
    """
    terms = {
        r["variant"]: float(r["value"])
        for r in rows
        if r["experiment"] == "decomposition" and r["metric"] == "step_time_ms"
    }
    measured = terms.pop("measured")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bottom = 0.0
    for name in TERM_ORDER:
        height = terms.get(name, 0.0)
        ax.bar(
            0,
            height,
            bottom=bottom,
            width=0.55,
            color=TERM_COLOURS[name],
            label=f"{name.replace('_', ' ')} — {height:.2f} ms ({height / measured:.0%})",
        )
        bottom += height
    ax.bar(1, measured, width=0.55, color="#333333", label=f"measured — {measured:.2f} ms")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["attributed", "measured"])
    ax.set_ylabel("per-token decode time (ms)")
    ax.set_title(f"Where a decode step goes — {device_name}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    path = RESULTS_DIR / "error_budget.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_serving_tradeoff(rows: list[dict[str, str]], device_name: str) -> Path:
    """Throughput against tail latency across batch sizes — the real serving decision.

    Batching buys throughput and charges for it in p99. Plotting the two against each other shows
    the knee directly, which a pair of separate curves against batch size does not.
    """
    throughput = dict(select(rows, "batching", "measured", "tokens_per_sec"))
    p99 = dict(select(rows, "batching", "measured", "latency_p99_ms"))
    batches = sorted(throughput)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([p99[b] for b in batches], [throughput[b] for b in batches], "-o", color="#1f77b4")
    for batch in batches:
        ax.annotate(
            f"B={batch:g}",
            (p99[batch], throughput[batch]),
            textcoords="offset points",
            xytext=(7, -4),
            fontsize=8,
        )

    ax.set_xlabel("per-token p99 latency (ms)")
    ax.set_ylabel("throughput (tokens / sec)")
    ax.set_title(f"Batching buys throughput with tail latency — {device_name}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = RESULTS_DIR / "serving_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_context_decay(rows: list[dict[str, str]], device_name: str) -> Path:
    """Tokens/sec against context length — the KV cache term, made visible.

    Most people's mental model has decode speed as a constant. It is not: the KV cache grows with
    every token generated, so the same model measurably slows down the more it has already said.
    """
    points = select(rows, "context", "measured", "tokens_per_sec")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([p[0] for p in points], [p[1] for p in points], "-o", color="#ff7f0e")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("throughput (tokens / sec)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Decode slows as the KV cache grows — {device_name}, batch 1")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    path = RESULTS_DIR / "context_decay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    rows = read_rows(CSV_PATH)
    device_name = load_profile().device_name
    for path in (
        plot_error_budget(rows, device_name),
        plot_serving_tradeoff(rows, device_name),
        plot_context_decay(rows, device_name),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
