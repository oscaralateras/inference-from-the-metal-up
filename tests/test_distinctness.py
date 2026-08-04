"""T6 and T7 answer different questions. This makes that a CI failure rather than a promise.

The two topics are close enough to collide — both concern decode, both need the same hardware
ceilings, both sweep batch size. The split is deliberate:

* **T6 owns the time domain.** Whole model, real KV cache, wall-clock: tokens/sec, latency
  percentiles, a per-token error budget in milliseconds.
* **T7 owns the shape domain.** Isolated matmuls on synthetic tensors: FLOPs per byte, achieved
  TFLOP/s, position relative to the ridge. No clock on either axis, no weights ever loaded.

A claim that is not enforced by a test drifts. T5 taught that directly — its lab note and its CSV
disagreed until a test was added that compared them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arch_common.results_io import read_rows
from topics.t06_perf_reasoning.measure import CSV_PATH as T6_CSV
from topics.t07_roofline.measure import CSV_PATH as T7_CSV

# Metric vocabularies. Disjoint by construction — each topic reports in its own domain's units.
T6_METRICS = {
    "tokens_per_sec",
    "step_time_ms",
    "request_latency_p50_ms",
    "request_latency_p99_ms",
    "littles_law_concurrency",
    "effective_bandwidth_gbps",
}
T7_METRICS = {
    "flops_per_byte",
    "achieved_tflops",
    "share_of_peak",
    "implied_gbps",
}


def test_the_two_metric_vocabularies_do_not_overlap() -> None:
    """The declared split, checked against itself before any results exist."""
    assert not (T6_METRICS & T7_METRICS)


def test_t6_reports_only_time_domain_metrics() -> None:
    if not T6_CSV.exists():
        pytest.skip("T6 has not been run in this session")
    metrics = {r["metric"] for r in read_rows(T6_CSV)}
    assert not (metrics & T7_METRICS), f"T6 emitted shape-domain metrics: {metrics & T7_METRICS}"
    assert metrics <= T6_METRICS, f"T6 emitted undeclared metrics: {metrics - T6_METRICS}"


def test_t7_reports_only_shape_domain_metrics() -> None:
    if not T7_CSV.exists():
        pytest.skip("T7 has not been run in this session")
    metrics = {r["metric"] for r in read_rows(T7_CSV)}
    assert not (metrics & T6_METRICS), f"T7 emitted time-domain metrics: {metrics & T6_METRICS}"
    assert metrics <= T7_METRICS, f"T7 emitted undeclared metrics: {metrics - T7_METRICS}"


def test_both_topics_were_measured_in_the_same_session() -> None:
    """Both cite the same measured ceilings, so both must come from the same probe.

    Split across two pod sessions the numbers would be internally consistent but mutually wrong —
    two lab notes quoting different bandwidths for "the same" GPU, with nothing to flag it.
    """
    if not (T6_CSV.exists() and T7_CSV.exists()):
        pytest.skip("both topics must have been run")

    t6_sessions = {r["session_id"] for r in read_rows(T6_CSV)}
    t7_sessions = {r["session_id"] for r in read_rows(T7_CSV)}

    assert len(t6_sessions) == 1, f"T6 results mix sessions: {t6_sessions}"
    assert len(t7_sessions) == 1, f"T7 results mix sessions: {t7_sessions}"
    assert t6_sessions == t7_sessions, (
        f"T6 measured against session {t6_sessions} but T7 against {t7_sessions} — "
        "re-run both against one hardware probe"
    )


def test_the_prediction_was_registered_before_the_measurement() -> None:
    """Predict-then-measure, verified rather than asserted.

    A prediction computed after seeing the measurement is not a prediction. The prediction file
    must exist, and must be pinned to the same session as the results it will be judged against.
    """
    predictions = Path("topics/t06_perf_reasoning/results/predictions.json")
    if not (predictions.exists() and T6_CSV.exists()):
        pytest.skip("T6 has not been run in this session")

    registered = json.loads(predictions.read_text())["session_id"]
    measured = {r["session_id"] for r in read_rows(T6_CSV)}
    assert measured == {registered}, (
        f"prediction registered against session {registered} but results are from {measured}"
    )
