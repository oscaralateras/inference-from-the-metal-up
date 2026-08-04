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


def test_lab_note_numbers_match_the_results() -> None:
    """Every headline figure in the write-up must be recomputable from the results file.

    This topic originally persisted nothing — its table was transcribed by hand from stdout, so a
    re-run that changed the numbers would have left the note quietly wrong with nothing to catch
    it. T5 shipped with a guard like this after exactly that happened; T6 and T7 have one too.
    """
    import csv
    from pathlib import Path

    import pytest

    results = Path(__file__).parent / "results" / "quantisation.csv"
    note = Path(__file__).parent / "README.md"
    if not results.exists():
        pytest.skip("run probe.py to generate results/quantisation.csv")

    with results.open() as f:
        rows = list(csv.DictReader(f))

    def value(variant: str, n_bits: int, metric: str) -> float:
        hits = [
            float(r["value"])
            for r in rows
            if r["variant"] == variant and int(r["x"]) == n_bits and r["metric"] == metric
        ]
        assert len(hits) == 1, f"expected one {variant}/{n_bits}/{metric}, found {len(hits)}"
        return hits[0]

    text = note.read_text()
    claims = {
        "int4 per-tensor error": f"{value('per_tensor', 4, 'output_relative_error') * 100:.1f}",
        "int4 per-tensor cosine": f"{value('per_tensor', 4, 'output_cosine'):.3f}",
        "int4 per-group error": f"{value('per_group', 4, 'output_relative_error') * 100:.1f}",
        "int4 per-group cosine": f"{value('per_group', 4, 'output_cosine'):.4f}",
        "int4 per-tensor SQNR": f"{value('per_tensor', 4, 'output_sqnr_db'):.1f}",
    }
    for label, quoted in claims.items():
        assert quoted in text, f"README no longer quotes the measured {label}: {quoted}"
