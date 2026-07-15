"""Error metrics for the quantisation experiment.

Given the original weights `w` and their dequantised reconstruction `w_hat`, these
quantify how much information quantisation destroyed. Each takes two float32 arrays of the
same shape and returns a single float — the numbers the T1 artefact reports and plots.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def mse(w: FloatArray, w_hat: FloatArray) -> float:
    """Mean squared error: average of (w - w_hat)^2. Lower is better.

    Squaring punishes big errors far more than small ones, so MSE is dominated by the
    weights that got hit the hardest.
    """
    return float(np.mean((w - w_hat) ** 2))


def max_abs_error(w: FloatArray, w_hat: FloatArray) -> float:
    """The single worst error: max |w - w_hat|. Lower is better.

    How far off is the most damaged weight.
    """
    return float(np.max(np.abs(w - w_hat)))


def cosine_similarity(w: FloatArray, w_hat: FloatArray) -> float:
    """Cosine similarity of the flattened weights. 1.0 = identical direction.

    Flatten both to vectors and measure the angle between them. This matters because a
    linear layer computes dot products: if the *direction* of the weights survives, the
    layer's outputs stay close even when individual magnitudes drift.
    """
    a = w.reshape(-1)
    b = w_hat.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0  # two zero vectors: treat as identical
    return float(np.dot(a, b) / denom)


def snr_db(w: FloatArray, w_hat: FloatArray) -> float:
    """Signal-to-noise ratio in decibels. Higher is better.

    Signal = the real weights; noise = the quantisation error (w - w_hat). Compare their
    powers (mean of squares): 10 * log10(signal_power / noise_power). Higher dB means the
    real signal dwarfs the quantisation noise.
    """
    signal_power = float(np.mean(w**2))
    noise_power = float(np.mean((w - w_hat) ** 2))
    if noise_power == 0.0:
        return float("inf")  # perfect reconstruction - no noise
    return 10.0 * float(np.log10(signal_power / noise_power))


def all_metrics(w: FloatArray, w_hat: FloatArray) -> dict[str, float]:
    """Convenience: compute all four at once, keyed by name."""
    return {
        "mse": mse(w, w_hat),
        "max_abs_error": max_abs_error(w, w_hat),
        "cosine_similarity": cosine_similarity(w, w_hat),
        "snr_db": snr_db(w, w_hat),
    }


def _selftest() -> None:
    w = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print("identical :", all_metrics(w, w))  # mse 0, cosine 1, snr inf
    w_hat = np.array([[1.1, 1.9, 3.0]], dtype=np.float32)  # small errors
    print("perturbed :", all_metrics(w, w_hat))


if __name__ == "__main__":
    _selftest()
