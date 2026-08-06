"""T10 and T11 answer different questions — from each other, and from T6. Enforced, not promised.

The three are close enough to collide. All concern overhead rather than arithmetic, all sweep a
size, and two of them measure something a serving stack would call "compile it and it gets faster".
The split is deliberate and it is the reason all three exist:

* **T10 owns the cold path.** Storage to HBM, once per model load, over PCIe. Seconds, gigabytes,
  page faults. Nothing in it happens again after start-up.
* **T11 owns the hot path.** Inside HBM, every step, forever. Microseconds, kilobytes, kernel
  launches. Nothing in it happens before the model is resident.
* **T6 owns the whole engine.** Wall-clock on a real vLLM step, with both of T11's mechanisms
  bundled into one number — which is exactly what T11 exists to unbundle.

T5 taught this the hard way: its lab note and its CSV disagreed until a test compared them. A claim
that is not enforced by a test drifts.
"""

from __future__ import annotations

import pytest

from arch_common.results_io import read_rows
from topics.t06_perf_reasoning.measure import CSV_PATH as T6_CSV
from topics.t10_os_virtual_memory.measure import CSV_PATH as T10_CSV
from topics.t10_os_virtual_memory.pipeline import model_bytes
from topics.t11_compiler_runtime.chain import DEFAULT_HIDDEN, activation_bytes
from topics.t11_compiler_runtime.measure import CSV_PATH as T11_CSV

# Metric vocabularies, disjoint by construction — each topic reports in its own domain's units.
T10_METRICS = {
    "load_seconds",
    "first_touch_seconds",
    "total_seconds",
    "deferred_share",
    "faults_per_page",
    "load_gbps",
    "pinned_gbps",
    "pageable_gbps",
    "memcpy_gbps",
    "pinned_over_pageable",
    "predicted_pageable_gbps",
    "stage_gbps",
    "stage_seconds",
    "cold_start_seconds",
    "tokens_foregone",
}
T11_METRICS = {
    "latency_us",
    "kernel_launches",
    "fusion_speedup",
    "graph_speedup",
    "combined_speedup",
    "fused_gbps",
    "byte_model_speedup",
    "crossover_batch",
}
T6_METRICS = {
    "tokens_per_sec",
    "step_time_ms",
    "request_latency_p50_ms",
    "request_latency_p99_ms",
    "littles_law_concurrency",
    "effective_bandwidth_gbps",
}

# The two topics' payloads, and the gap between them. This is the same argument that separates T5
# from T9 — one regime each, enforced by a ratio rather than described in prose.
MIN_PAYLOAD_RATIO = 100_000


def test_the_three_vocabularies_do_not_overlap() -> None:
    """The declared split, checked against itself before any results exist."""
    assert not (T10_METRICS & T11_METRICS)
    assert not (T10_METRICS & T6_METRICS)
    assert not (T11_METRICS & T6_METRICS)


def test_the_payloads_are_orders_of_magnitude_apart() -> None:
    """T10 moves the whole model once; T11 moves one activation, twice per op, every step.

    15.2 GB against 7 KB is a factor of two million. They are not the same problem, they do not
    cross the same link, and an optimisation for one does nothing for the other.
    """
    ratio = model_bytes() / activation_bytes(1, DEFAULT_HIDDEN)
    assert ratio > MIN_PAYLOAD_RATIO


def test_t10_reports_only_cold_path_metrics() -> None:
    if not T10_CSV.exists():
        pytest.skip("T10 has not been run")
    metrics = {r["metric"] for r in read_rows(T10_CSV)}
    assert not (metrics & T11_METRICS), f"T10 emitted hot-path metrics: {metrics & T11_METRICS}"
    assert metrics <= T10_METRICS, f"T10 emitted undeclared metrics: {metrics - T10_METRICS}"


def test_t11_reports_only_hot_path_metrics() -> None:
    if not T11_CSV.exists():
        pytest.skip("T11 has not been run")
    metrics = {r["metric"] for r in read_rows(T11_CSV)}
    assert not (metrics & T10_METRICS), f"T11 emitted cold-path metrics: {metrics & T10_METRICS}"
    assert metrics <= T11_METRICS, f"T11 emitted undeclared metrics: {metrics - T11_METRICS}"


def test_t11_does_not_restate_t6s_engine_level_numbers() -> None:
    """T6 already owns the bundled, whole-engine result. T11's job is to take it apart.

    If T11 ever emitted a step time or a tokens/sec it would be re-running T6 with a smaller model,
    which is the exact trap the topic was redesigned to avoid.
    """
    if not T11_CSV.exists():
        pytest.skip("T11 has not been run")
    metrics = {r["metric"] for r in read_rows(T11_CSV)}
    assert not (metrics & T6_METRICS), f"T11 emitted engine-level metrics: {metrics & T6_METRICS}"


def test_neither_topic_publishes_rehearsal_numbers_alongside_measured_ones() -> None:
    """A CPU rehearsal is for debugging off the clock; its numbers are never a result.

    Both topics stamp `session_id = "rehearsal"` when they run without a GPU. A results file that
    mixes those with a real session is a file describing two machines, and the note built from it
    would quote whichever row it happened to read.
    """
    for name, csv_path in (("T10", T10_CSV), ("T11", T11_CSV)):
        if not csv_path.exists():
            continue
        sessions = {r["session_id"] for r in read_rows(csv_path)}
        assert len(sessions) == 1, f"{name} results mix sessions: {sessions}"


def test_t10_and_t11_were_measured_in_the_same_session_as_t6() -> None:
    """All three quote T6's step time or its ceilings, so all three must share one probe.

    Skipped until every file exists and none is a rehearsal — which is the state the repo is in
    before the GPU session, and the state this test is written to catch afterwards.
    """
    paths = {"T6": T6_CSV, "T10": T10_CSV, "T11": T11_CSV}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("not all three topics have been run")

    sessions = {name: {r["session_id"] for r in read_rows(p)} for name, p in paths.items()}
    if any(s == {"rehearsal"} for s in sessions.values()):
        pytest.skip("at least one topic holds rehearsal numbers, not measured ones")

    assert len(set(map(frozenset, sessions.values()))) == 1, (
        f"topics were measured against different sessions: {sessions}"
    )
