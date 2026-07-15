"""Sweep: how does quantisation error depend on bit-width and granularity?

Loads one real LLM weight tensor, then quantises -> dequantises it under every combination of
bit-width (int8 down to int2) and granularity (per-tensor, per-channel, per-group), measuring
fidelity two ways:

  - weight fidelity : SQNR of the weights themselves.
  - output fidelity : SQNR and cosine of the layer output W @ x vs W_hat @ x on a random input
    x — the error inference actually experiences (weight error is only a proxy for this).

    uv run python topics/t01_number_representation/probe.py
"""

from __future__ import annotations

import itertools

import numpy as np
from load_weights import load_linear_weight
from metrics import cosine_similarity, sqnr_db
from quantise import Granularity, dequantise, quantise_symmetric

BIT_WIDTHS = [8, 7, 6, 5, 4, 3, 2]
GRANULARITIES: list[Granularity] = ["per_tensor", "per_channel", "per_group"]
GROUP_SIZE = 64  # must divide in_features (576 for SmolLM2 gate_proj); 576 / 64 = 9
N_SAMPLES = 512  # random input columns for the output-fidelity forward pass
SEED = 0


def run_sweep() -> None:
    w = load_linear_weight()  # real SmolLM2 gate_proj weight, float32
    in_features = w.shape[1]

    # Fixed random input batch x of shape (in_features, N). We compare the true layer output
    # W @ x against the quantised output W_hat @ x — the error inference actually sees. Random
    # Gaussian x is a proxy for real activations (see the lab-note caveat).
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((in_features, N_SAMPLES)).astype(np.float32)
    out = w @ x

    print(f"weight {w.shape}   max|w| {np.abs(w).max():.3f}   mean|w| {np.abs(w).mean():.3f}\n")
    header = f"{'n_bits':>6} {'granularity':>12} {'w_sqnr':>8} {'out_sqnr':>9} {'out_cos':>9}"
    print(header)
    print("-" * len(header))

    for n_bits, gran in itertools.product(BIT_WIDTHS, GRANULARITIES):
        q, scale = quantise_symmetric(w, n_bits=n_bits, granularity=gran, group_size=GROUP_SIZE)
        w_hat = dequantise(q, scale)
        out_hat = w_hat @ x
        print(
            f"{n_bits:>6} {gran:>12} "
            f"{sqnr_db(w, w_hat):>8.2f} {sqnr_db(out, out_hat):>9.2f} "
            f"{cosine_similarity(out, out_hat):>9.5f}"
        )


if __name__ == "__main__":
    run_sweep()
