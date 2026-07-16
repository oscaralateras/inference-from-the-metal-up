"""Unit tests for the T1 quantiser and metrics.

These assert the behaviour the `_selftest` demos only printed — so a regression fails CI instead
of a human having to eyeball the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from metrics import cosine_similarity, relative_error, sqnr_db
from quantise import dequantise, quantise_symmetric


def test_quantise_hand_computed() -> None:
    # max|w| = 0.8, int8 symmetric -> scale = 0.8/127; the by-hand integer codes.
    w = np.array([[0.5, -0.8, 0.02, 0.0]], dtype=np.float32)
    q, scale = quantise_symmetric(w, n_bits=8, granularity="per_tensor")
    assert q.tolist() == [[79, -127, 3, 0]]
    assert float(scale) == pytest.approx(0.8 / 127, rel=1e-4)


def test_dequantise_roundtrip_close() -> None:
    w = np.array([[0.5, -0.8, 0.02, 0.0]], dtype=np.float32)
    w_hat = dequantise(*quantise_symmetric(w, n_bits=8, granularity="per_tensor"))
    assert float(np.max(np.abs(w - w_hat))) < 0.01


def test_per_channel_beats_per_tensor_on_outlier() -> None:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((8, 32)) * 0.1).astype(np.float32)
    w[0] *= 50.0  # one outlier channel would dominate a single per-tensor scale
    wt = dequantise(*quantise_symmetric(w, 4, "per_tensor"))
    wc = dequantise(*quantise_symmetric(w, 4, "per_channel"))
    assert sqnr_db(w, wc) > sqnr_db(w, wt)


def test_n_bits_out_of_range_raises() -> None:
    w = np.zeros((4, 8), dtype=np.float32)
    for bad in (1, 9, 16):
        with pytest.raises(ValueError):
            quantise_symmetric(w, bad, "per_tensor")


def test_per_group_requires_valid_group_size() -> None:
    w = np.zeros((4, 30), dtype=np.float32)
    for bad in (0, 64):  # 0 invalid; 64 does not divide 30
        with pytest.raises(ValueError):
            quantise_symmetric(w, 8, "per_group", group_size=bad)


def test_metrics_on_identical_inputs() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert sqnr_db(a, a) == float("inf")
    assert relative_error(a, a) == 0.0
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-4)


def test_cosine_ignores_scale_but_relerr_does_not() -> None:
    a = np.array([3.0, 4.0], dtype=np.float32)
    b = a * 2.0  # same direction, double magnitude
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)
    assert relative_error(a, b) == pytest.approx(1.0, abs=1e-4)  # 100% error
