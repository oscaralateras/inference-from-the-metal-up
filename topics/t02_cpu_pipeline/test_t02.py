"""Directional integration test for T2.

Timings aren't deterministic, but the *direction* of each effect and the checksum invariant
are — so we assert those against the committed canonical CSV. This guards against shipping a
broken benchmark (e.g. the branch silently getting `cmov`-ed away, collapsing to no effect).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV = Path(__file__).parent / "results" / "pipeline.csv"


def _load() -> dict[tuple[str, str], dict[str, float]]:
    if not CSV.exists():
        pytest.skip(f"{CSV} not present — run `make run` first")
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with CSV.open() as f:
        for r in csv.DictReader(f):
            rows[(r["experiment"], r["variant"])] = {
                "ns": float(r["ns_per_elem"]),
                "checksum": float(r["checksum"]),
            }
    return rows


def test_all_variants_present_and_sane() -> None:
    rows = _load()
    expected = {
        ("branch", "unsorted"),
        ("branch", "sorted"),
        ("ilp", "dependent"),
        ("ilp", "independent"),
        ("simd", "scalar"),
        ("simd", "vector"),
    }
    assert expected <= set(rows)
    for v in rows.values():
        assert 0.0 < v["ns"] < 1e6  # positive, finite, sane magnitude


def test_pipeline_friendly_beats_hostile() -> None:
    """The whole point: the pipeline-friendly variant is measurably faster."""
    rows = _load()
    assert rows[("branch", "unsorted")]["ns"] > rows[("branch", "sorted")]["ns"]
    assert rows[("ilp", "dependent")]["ns"] > rows[("ilp", "independent")]["ns"]
    assert rows[("simd", "scalar")]["ns"] > rows[("simd", "vector")]["ns"]


def test_branch_checksums_match() -> None:
    """Sorting only reorders the same values, so the two sums must be identical."""
    rows = _load()
    assert rows[("branch", "unsorted")]["checksum"] == rows[("branch", "sorted")]["checksum"]
