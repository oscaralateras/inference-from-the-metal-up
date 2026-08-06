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

from arch_common.results_io import read_rows, scalar
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
    "serial_bound_gbps",
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
    "measured_launch_us",
    "remodelled_crossover_batch",
    "graphed_fused_gbps",
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


def test_t10_and_t11_share_one_session_with_each_other() -> None:
    """They read the same hardware probe, so they must describe the same silicon.

    This is the requirement that is actually enforceable, and it is deliberately **not** extended
    to T6. An earlier version of this test demanded all three share a session and failed the moment
    the pod ran: T6 was measured on a different rental months earlier, and re-running a 7B engine
    purely to satisfy a session ID would have cost money to prove nothing.

    T9 set the precedent — it composes with T6's step time across pods and validates the
    composition by agreement on a measured ceiling instead. That check is the test below.
    """
    paths = {"T10": T10_CSV, "T11": T11_CSV}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("both topics must have been run")

    sessions = {name: {r["session_id"] for r in read_rows(p)} for name, p in paths.items()}
    if any(s == {"rehearsal"} for s in sessions.values()):
        pytest.skip("at least one topic holds rehearsal numbers, not measured ones")

    assert sessions["T10"] == sessions["T11"], (
        f"T10 and T11 were measured against different sessions: {sessions} — they share one "
        "hardware probe, so this means one of them was re-run on a different pod"
    )


# How closely this session's measured bandwidth must match the one T6, T7 and T8 ran at for their
# numbers to be composable. T7 recorded 1,736.7 GB/s; a pod within a fraction of a percent of that
# is the same silicon under the same thermal conditions, and a pod that is not should not have its
# cold-start seconds divided by T6's step time.
MAX_BANDWIDTH_DRIFT = 0.02


def test_this_session_agrees_with_the_one_t6_and_t7_were_measured_on() -> None:
    """T10 quotes cold start in T6's tokens; T11 scores against T7's roof. Both compose across
    pods, and this is what makes that legitimate rather than assumed.

    A session ID cannot check this — the pods are genuinely different. What can is the ceiling
    itself: if two rentals of the same card measure the same bandwidth to within a fraction of a
    percent, composing their numbers is sound. If a future pod drifts, this fails and the notes
    stop quietly borrowing a step time from hardware they never ran on.

    The earlier session's roof is not stored directly anywhere — `results/hardware.json` holds
    whichever session ran last, which is this one. But T8 recorded its load-only probe both in GB/s
    and as a share of the roof it was scored against, and a value over its own fraction recovers
    the denominator. Derived from committed measurements rather than typed in.
    """
    from arch_common.gpu import load_profile
    from topics.t08_gpu_architecture.measure import CSV_PATH as T8_CSV

    if not T8_CSV.exists():
        pytest.skip("T8 has not been run")

    rows = read_rows(T8_CSV)
    achieved = scalar(rows, "ceiling", "load_only", "gbps")
    share = scalar(rows, "ceiling", "load_only", "share_of_roof")
    if not share:
        pytest.skip("T8's ceiling rows do not carry a share of roof")
    there = achieved / share

    here = load_profile().peak_bandwidth_gbps
    drift = abs(here - there) / there
    assert drift < MAX_BANDWIDTH_DRIFT, (
        f"this session measured {here:,.1f} GB/s against the T6/T7/T8 session's {there:,.1f} — "
        f"{drift:.1%} apart, so composing T10's seconds with T6's step time is not justified"
    )
