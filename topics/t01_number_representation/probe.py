"""Sweep: how does quantisation error depend on bit-width and granularity?

Loads one real LLM weight tensor, then quantises -> dequantises it under every combination of
bit-width (int8 down to int2) and granularity (per-tensor, per-channel, per-group), measuring
fidelity two ways:

  - weight fidelity : SQNR of the weights themselves.
  - output fidelity : SQNR, relative error, and cosine of the layer output W @ x vs W_hat @ x on
    a random input x — the error inference actually experiences (weight error is only a proxy).

`run_sweep` returns (results, weight) so the plotter can reuse both without re-loading.

    uv run python topics/t01_number_representation/probe.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from load_weights import load_linear_weight
from metrics import FloatArray, cosine_similarity, relative_error, sqnr_db
from quantise import Granularity, dequantise, quantise_symmetric

BIT_WIDTHS = [8, 7, 6, 5, 4, 3, 2]
GRANULARITIES: list[Granularity] = ["per_tensor", "per_channel", "per_group"]
GROUP_SIZE = 64  # must divide in_features (576 for SmolLM2 gate_proj); 576 / 64 = 9
N_SAMPLES = 512  # random input columns for the output-fidelity forward pass
SEED = 0


@dataclass
class Result:
    n_bits: int
    granularity: Granularity
    w_sqnr: float  # weight fidelity (dB)
    out_sqnr: float  # output fidelity (dB)
    out_relerr: float  # output relative L2 error (fraction; x100 for %)
    out_cos: float  # output direction


def run_sweep() -> tuple[list[Result], FloatArray]:
    w = load_linear_weight()  # real SmolLM2 gate_proj weight, float32
    in_features = w.shape[1]

    # Fixed random input batch x of shape (in_features, N). Compare true output W @ x against
    # quantised output W_hat @ x — the error inference actually sees. Random Gaussian x is a
    # proxy for real activations (see the lab-note caveat).
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((in_features, N_SAMPLES)).astype(np.float32)
    out = w @ x

    print(f"weight {w.shape}   max|w| {np.abs(w).max():.3f}   mean|w| {np.abs(w).mean():.3f}\n")
    header = (
        f"{'n_bits':>6} {'granularity':>12} {'w_sqnr':>8} "
        f"{'out_sqnr':>9} {'out_err%':>9} {'out_cos':>9}"
    )
    print(header)
    print("-" * len(header))

    results: list[Result] = []
    for n_bits, gran in itertools.product(BIT_WIDTHS, GRANULARITIES):
        q, scale = quantise_symmetric(w, n_bits=n_bits, granularity=gran, group_size=GROUP_SIZE)
        w_hat = dequantise(q, scale)
        out_hat = w_hat @ x
        r = Result(
            n_bits=n_bits,
            granularity=gran,
            w_sqnr=sqnr_db(w, w_hat),
            out_sqnr=sqnr_db(out, out_hat),
            out_relerr=relative_error(out, out_hat),
            out_cos=cosine_similarity(out, out_hat),
        )
        results.append(r)
        print(
            f"{r.n_bits:>6} {r.granularity:>12} {r.w_sqnr:>8.2f} "
            f"{r.out_sqnr:>9.2f} {r.out_relerr * 100:>9.2f} {r.out_cos:>9.5f}"
        )
    return results, w


if __name__ == "__main__":
    run_sweep()
