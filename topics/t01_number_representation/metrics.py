"""Gold-standard error metrics for the quantisation experiment.

Two generic metrics, each comparing a `reference` array against an `approx` of the same shape:

  - sqnr_db          : signal-to-quantisation-noise ratio in dB (the standard fidelity number).
  - cosine_similarity: 1.0 = identical direction (does the vector still point the same way).

They are applied to BOTH the weights (weight fidelity) and the layer output W @ x (output
fidelity — what inference actually experiences). Raw MSE and max-abs-error were dropped: MSE is
just the un-normalised SQNR numerator (redundant), and max-abs is a misleading worst-case number
dominated by a single outlier.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def sqnr_db(reference: FloatArray, approx: FloatArray) -> float:
    """Signal-to-quantisation-noise ratio in decibels. Higher is better.

    signal = the reference; noise = (reference - approx). Compare their powers (mean of
    squares): 10 * log10(signal_power / noise_power). Scale-invariant, so it's comparable
    across tensors of different magnitude.
    """
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean((reference - approx) ** 2))
    if noise_power == 0.0:
        return float("inf")  # perfect reconstruction — no noise
    return 10.0 * float(np.log10(signal_power / noise_power))


def cosine_similarity(a: FloatArray, b: FloatArray) -> float:
    """Cosine similarity of the flattened arrays. 1.0 = identical direction.

    Flatten to vectors and measure the angle between them. On layer outputs this is the
    closest cheap proxy for "does the layer still produce the same result".
    """
    va = a.reshape(-1)
    vb = b.reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0  # two zero vectors: treat as identical
    return float(np.dot(va, vb) / denom)


def _selftest() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print("identical :", f"sqnr={sqnr_db(a, a)}  cosine={cosine_similarity(a, a):.5f}")
    b = np.array([1.1, 1.9, 3.0], dtype=np.float32)
    print("perturbed :", f"sqnr={sqnr_db(a, b):.2f}dB  cosine={cosine_similarity(a, b):.5f}")


if __name__ == "__main__":
    _selftest()
