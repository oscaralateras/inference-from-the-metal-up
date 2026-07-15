"""Sweep: how does quantisation error depend on bit-width and granularity?

Loads one real LLM weight tensor, then quantises -> dequantises it under every combination
of bit-width (int8, int4) and granularity (per-tensor, per-channel, per-group), measures the
error four ways, and prints a results table. This is the experiment; step 5 plots it.

    uv run python topics/t01_number_representation/probe.py
"""

from __future__ import annotations

import itertools

import numpy as np
from load_weights import load_linear_weight
from metrics import all_metrics
from quantise import Granularity, dequantise, quantise_symmetric

BIT_WIDTHS = [8, 4]
GRANULARITIES: list[Granularity] = ["per_tensor", "per_channel", "per_group"]
GROUP_SIZE = 64  # must divide in_features (576 for SmolLM2 gate_proj); 576 / 64 = 9


def run_sweep() -> None:
    w = load_linear_weight()  # real SmolLM2 gate_proj weight, float32
    print(f"weight {w.shape}   max|w| {np.abs(w).max():.3f}   mean|w| {np.abs(w).mean():.3f}\n")

    header = (
        f"{'n_bits':>6} {'granularity':>12} {'mse':>10} {'max_abs':>9} {'cosine':>9} {'snr_dB':>7}"
    )
    print(header)
    print("-" * len(header))

    for n_bits, gran in itertools.product(BIT_WIDTHS, GRANULARITIES):
        q, scale = quantise_symmetric(w, n_bits=n_bits, granularity=gran, group_size=GROUP_SIZE)
        w_hat = dequantise(q, scale)
        m = all_metrics(w, w_hat)
        print(
            f"{n_bits:>6} {gran:>12} {m['mse']:>10.2e} {m['max_abs_error']:>9.4f} "
            f"{m['cosine_similarity']:>9.5f} {m['snr_db']:>7.2f}"
        )


if __name__ == "__main__":
    run_sweep()
