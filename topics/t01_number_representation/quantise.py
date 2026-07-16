"""Symmetric integer quantisation of weight tensors, from scratch

Implements quantise/dequantise for signed n-bit *symmetric* quantisation at three granularities
(per tensor, per channel and per group), plus a hand checkable self-test.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

# A closed set of allowed granularity values. Using Literal means pyright flags a typo
# like "per_chanel" at check time, instead of it blowing up at runtime.
Granularity = Literal["per_tensor", "per_channel", "per_group"]

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int8]


def quantise_symmetric(
    w: FloatArray,
    n_bits: int,
    granularity: Granularity = "per_tensor",
    group_size: int = 128,
) -> tuple[IntArray, FloatArray]:
    """Symmetrically quantise a 2D weight matrix to signed n-bit integers.

    Args:
        w: weight matrix, shape (out_features, in_features). Each row = one output channel.
        n_bits: bit-width of the integer grid (8 for int8)
        granularity: how many scales to use:
            - "per_tensor": one scale for the whole matrix.
            - "per_channel": one scale per row(output channel)
            - "per_group": one scale per group of "group_size" within each row
        group_size: columns per group (only used for "per_group")

    Returns:
        (q, scale) where q = integer codes (same shape as w, int8) and scale is
        broadcastable to w's shape, so dequantise is just q * scale.
    """
    # int8 storage requires 2 <= n_bits <= 8: below 2 the grid collapses (qmax 0 -> all zeros),
    # above 8 the codes overflow int8 silently.
    if not 2 <= n_bits <= 8:
        raise ValueError(f"n_bits must be in [2, 8] for int8 storage, got {n_bits}")

    #  Biggest integer the signed grid holds, e.g. int8 -> 2^7 - 1 = 127.
    qmax = 2 ** (n_bits - 1) - 1

    # ---1) find max|w| at requested granularity
    if granularity == "per_tensor":
        # One number for the whole matrix
        max_abs = np.abs(w).max()
    elif granularity == "per_channel":
        # One number per row. Keep dims -> shape (rows, 1) so it is broadcastable over columns
        max_abs = np.abs(w).max(axis=1, keepdims=True)
    elif granularity == "per_group":
        rows, cols = w.shape
        if group_size < 1 or cols % group_size != 0:
            raise ValueError(
                f"group_size must be >= 1 and divide in_features {cols}, got {group_size}"
            )
        n_groups = cols // group_size
        # Split each row into groups and take max|w| per group -> (rows, n_groups), then
        # repeat each group's value across its own columns -> back to (rows, cols)
        grouped = np.abs(w).reshape(rows, n_groups, group_size).max(axis=2)
        max_abs = np.repeat(grouped, group_size, axis=1)
    else:
        raise ValueError(f"unknown granularity: {granularity!r}")

    # scale = max|w| / qmax. Clamp max_abs off zero to avoid divide-by-zero on an all-zero
    # tensor/row/group (real weights won't be, but production code guards it).
    scale = np.asarray(np.maximum(max_abs, 1e-8) / qmax, dtype=np.float32)

    # ---2) snap to nearest grid point, clamp into range, store as ints ---
    q = np.round(w / scale)  # nearest integer code
    q = np.clip(q, -qmax, qmax)  # anything past the edge saturates
    q = q.astype(np.int8)
    return q, scale


def dequantise(q: IntArray, scale: FloatArray) -> FloatArray:
    """Reconstruct approximate float weights: w_hat = q * scale.

    Broadcasting handles all three granularities uniformly, because scale was
    returned broadcastable to q's shape.
    """
    return np.asarray(q * scale, dtype=np.float32)


def _selftest() -> None:
    # Tiny hand checkable example
    w = np.array([[0.5, -0.8, 0.02, 0.0]], dtype=np.float32)
    q, scale = quantise_symmetric(w, n_bits=8, granularity="per_tensor")
    w_hat = dequantise(q, scale)
    print("w      =", w)
    print("scale  =", float(scale), "  (expect ~0.8/127 = 0.0063)")
    print("q      =", q, "  (expect 0.5->79, -0.8->-127, 0.02->3, 0.0->0)")
    print("w_hat  =", w_hat)
    print("abs err=", np.abs(w - w_hat))


if __name__ == "__main__":
    _selftest()
