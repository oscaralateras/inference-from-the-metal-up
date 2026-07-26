"""Directional integration test for T4.

Exact timings are hardware-specific, but the *direction* of each effect is robust and
machine-independent, so we assert those against the committed CSV. Skips until the canonical run on
a multicore x86 box has produced results/concurrency.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV = Path(__file__).parent / "results" / "concurrency.csv"


def _rows(experiment: str) -> list[tuple[str, int, float]]:
    if not CSV.exists():
        pytest.skip(f"{CSV} not present — run `make run` on a multicore x86 box first")
    out: list[tuple[str, int, float]] = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["experiment"] == experiment:
                out.append((r["variant"], int(r["threads"]), float(r["value"])))
    return out


def _at_max_threads(experiment: str) -> dict[str, float]:
    rows = _rows(experiment)
    max_t = max(t for _, t, _ in rows)
    return {variant: val for variant, t, val in rows if t == max_t}


# --- (a) false sharing ------------------------------------------------------
def test_false_sharing_costs_time() -> None:
    by = {variant: val for variant, _, val in _rows("false_sharing")}
    assert by["adjacent"] > by["padded"]  # sharing a cache line is slower than not


# --- (b) the race -----------------------------------------------------------
def test_atomic_is_exact_racy_loses_updates() -> None:
    by = {variant: val for variant, _, val in _rows("race")}
    assert by["atomic"] == 0.0  # fetch_add is indivisible: no update is ever lost
    assert by["racy"] > 0.0  # a non-atomic read-modify-write drops updates


# --- (c) contention ---------------------------------------------------------
def test_sharding_beats_locking_and_contention_at_scale() -> None:
    at = _at_max_threads("contention")
    assert at["sharded"] > at["mutex"]  # the lock serialises; sharding scales
    assert at["sharded"] > at["atomic"]  # and beats the one contended hot cache line


# --- (d) the scheduler ------------------------------------------------------
def test_sharded_scheduler_beats_global_lock_at_scale() -> None:
    at = _at_max_threads("scheduler")
    assert at["sharded"] > at["global_lock"]  # a central lock bottlenecks the dispatcher
