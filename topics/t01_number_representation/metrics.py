"""Gold-standard error metrics for the quantisation experiment.

Three generic metrics, each comparing a `reference` array against an `approx` of the same shape.
There are only two independent axes here — error *magnitude* and error *direction* — so:

  - sqnr_db          : magnitude, in dB (the field-standard fidelity number; good for plots).
  - relative_error   : magnitude, as a fraction (the human-readable form; 0.11 => "~11% off").
  - cosine_similarity: direction (1.0 = the vector still points the same way).

sqnr_db and relative_error are the same axis in two units (relerr = 10 ** (-sqnr_db / 20)); both
are reported because one is standard and one is intuitive. Raw MSE was dropped: it's that same
magnitude axis but scale-dependent (not comparable across tensors), and max-abs was a misleading
worst-case number dominated by a single outlier.

They are applied to BOTH the weights (weight fidelity) and the layer output W @ x (output
fidelity — what inference actually experiences).
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


def relative_error(reference: FloatArray, approx: FloatArray) -> float:
    """Relative L2 error: ||reference - approx|| / ||reference||. Lower is better.

    Reads directly as "what fraction of the signal is error" (x100 for a percentage;
    0.11 => the output is ~11% off). Same magnitude axis as sqnr_db, in human units.
    """
    ref = reference.reshape(-1)
    err = (reference - approx).reshape(-1)
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return 0.0  # a zero reference has no signal to be wrong about
    return float(np.linalg.norm(err) / denom)


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
    print("identical :", f"sqnr={sqnr_db(a, a)}  relerr={relative_error(a, a):.4f}")
    b = np.array([1.1, 1.9, 3.0], dtype=np.float32)
    print(
        "perturbed :",
        f"sqnr={sqnr_db(a, b):.2f}dB  relerr={relative_error(a, b):.4f}  "
        f"cosine={cosine_similarity(a, b):.5f}",
    )


if __name__ == "__main__":
    _selftest()
