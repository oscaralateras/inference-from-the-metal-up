"""Directional integration test for T3.

Exact numbers are hardware-specific, but the *direction* of each effect is robust and
machine-independent, so we assert those against the committed CSV. Skips until the canonical
run on the x86 box has produced results/memory.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV = Path(__file__).parent / "results" / "memory.csv"


def _rows(experiment: str) -> list[tuple[str, int, float]]:
    if not CSV.exists():
        pytest.skip(f"{CSV} not present — run `make run` on the x86 box first")
    out: list[tuple[str, int, float]] = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["experiment"] == experiment:
                out.append((r["variant"], int(r["size"]), float(r["value"])))
    return out


# --- (a) bandwidth hierarchy ------------------------------------------------
def test_bandwidth_drops_from_cache_to_dram() -> None:
    pts = sorted((size, val) for _, size, val in _rows("bandwidth"))
    assert len(pts) >= 5
    for _, gbps in pts:
        assert 0.0 < gbps < 1e5
    l1_gbps, dram_gbps = pts[0][1], pts[-1][1]  # smallest set (L1) vs largest (DRAM)
    assert l1_gbps > dram_gbps * 1.5


# --- (b) tiling matmul ------------------------------------------------------
def test_tiling_reorder_crushes_naive() -> None:
    # The headline: fixing the memory-access pattern (ikj) is worth an order of magnitude.
    # naive is only measured up to 1024, so compare at the largest size where both exist.
    rows = _rows("tiling")
    naive = {size: val for variant, size, val in rows if variant == "naive"}
    ikj = {size: val for variant, size, val in rows if variant == "ikj"}
    n = max(naive)
    assert ikj[n] > naive[n] * 3  # in practice ~20x at N=1024


def test_tiling_naive_collapses_as_matrix_grows() -> None:
    # naive gets WORSE with size (cache-thrashing intensifies); the friendly kernels stay flat.
    naive = sorted((size, val) for variant, size, val in _rows("tiling") if variant == "naive")
    assert naive[-1][1] < naive[0][1]


def test_tiling_blocked_at_least_matches_ikj() -> None:
    # Same maths, cache-friendlier or equal — blocked never meaningfully loses to ikj.
    rows = _rows("tiling")
    ikj = {size: val for variant, size, val in rows if variant == "ikj"}
    blocked = {size: val for variant, size, val in rows if variant == "blocked"}
    for n in ikj:
        assert blocked[n] >= ikj[n] * 0.9


# --- (c) memory-bound vs compute-bound --------------------------------------
def test_crossover_batching_raises_throughput() -> None:
    pts = sorted((size, val) for variant, size, val in _rows("crossover") if variant == "batched")
    assert len(pts) >= 3
    batch1_gflops, maxbatch_gflops = pts[0][1], pts[-1][1]
    # batching (large B, compute-bound) is much faster per-op than B=1 (memory-bound decode)
    assert maxbatch_gflops > batch1_gflops * 1.5


def test_crossover_latency_rises_with_batch() -> None:
    # batching isn't free: a larger batch does more work per step, so per-request latency rises
    pts = sorted((size, val) for variant, size, val in _rows("crossover") if variant == "latency")
    assert len(pts) >= 3
    batch1_ms, maxbatch_ms = pts[0][1], pts[-1][1]
    assert maxbatch_ms > batch1_ms
