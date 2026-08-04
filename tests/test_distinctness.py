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

from arch_common.results_io import read_rows, scalar, select
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


# The weight classes a decode step reads, split by the shape of the matmul that reads them.
# Wide-N projections (MLP 18944, LM head 152064) behave like T7's `decode_mlp_up`; narrow-N
# attention projections (3584, and 512 for K/V under grouped-query attention) behave like its
# `decode_qkv_proj`. RMSNorm parameters are excluded — 200,704 of 7.07B, far below the resolution
# of a two-bucket model.
WIDE_SHARE_OF_WEIGHTS = 0.884
CROSS_CHECK_TOLERANCE = 0.15


def test_the_two_topics_agree_on_effective_decode_bandwidth() -> None:
    """The corroboration between T6 and T7, enforced rather than asserted in prose.

    T7 measures isolated synthetic GEMVs and finds decode kernel bandwidth is shape-dependent.
    T6 measures a whole model's wall-clock and infers a single effective bandwidth. Weighting T7's
    two kernel classes by their share of the weights a decode step reads must reproduce T6's
    number — they are the same physical quantity reached from opposite directions, so a
    disagreement means one of the two topics is wrong.

    The tolerance is deliberately loose (15%). The observed agreement is far tighter, but a
    two-bucket model of a transformer's matmuls does not *deserve* tight agreement, and a test
    that encoded the luck would fail on the next model or GPU for no good reason.
    """
    if not (T6_CSV.exists() and T7_CSV.exists()):
        pytest.skip("both topics must have been run")

    t6 = read_rows(T6_CSV)
    t7 = read_rows(T7_CSV)

    measured = scalar(t6, "decomposition", "effective_bandwidth", "effective_bandwidth_gbps")
    wide = select(t7, "regimes", "decode_mlp_up", "implied_gbps")[0][1]
    narrow = select(t7, "regimes", "decode_qkv_proj", "implied_gbps")[0][1]

    predicted = WIDE_SHARE_OF_WEIGHTS * wide + (1 - WIDE_SHARE_OF_WEIGHTS) * narrow
    disagreement = abs(predicted - measured) / measured

    assert disagreement < CROSS_CHECK_TOLERANCE, (
        f"T7's kernels predict {predicted:,.0f} GB/s but T6 measured {measured:,.0f} GB/s "
        f"({disagreement:.1%} apart) — one of the two topics is wrong"
    )


def test_t6_effective_bandwidth_lies_between_t7_kernel_bandwidths() -> None:
    """A weaker check that holds regardless of the weighting: the whole is between its parts."""
    if not (T6_CSV.exists() and T7_CSV.exists()):
        pytest.skip("both topics must have been run")

    t7 = read_rows(T7_CSV)
    measured = scalar(
        read_rows(T6_CSV), "decomposition", "effective_bandwidth", "effective_bandwidth_gbps"
    )
    kernels = [
        select(t7, "regimes", name, "implied_gbps")[0][1]
        for name in ("decode_mlp_up", "decode_qkv_proj")
    ]
    assert min(kernels) < measured < max(kernels)


# Headline numbers quoted in each lab note, paired with how to recompute them from the results.
# T5 shipped with a guard like this because its note and its CSV had silently disagreed; every
# topic that quotes numbers in prose needs one.
def _t6_claims(rows: list[dict[str, str]]) -> dict[str, str]:
    step = scalar(rows, "decode", "measured", "step_time_ms")
    weights = scalar(rows, "decomposition", "weights", "step_time_ms")
    unexplained = scalar(rows, "decomposition", "unexplained", "step_time_ms")
    bandwidth = scalar(rows, "decomposition", "effective_bandwidth", "effective_bandwidth_gbps")
    return {
        "decode throughput": f"{scalar(rows, 'decode', 'measured', 'tokens_per_sec'):.1f}",
        "step time": f"{step:.2f}",
        "effective bandwidth": f"{bandwidth:,.0f}",
        "weights share": f"{weights / step:.1%}",
        "unexplained share": f"{unexplained / step:.1%}",
    }


def _t7_claims(rows: list[dict[str, str]]) -> dict[str, str]:
    def one(variant: str, metric: str) -> float:
        return select(rows, "regimes", variant, metric)[0][1]

    return {
        "prefill share of peak": f"{one('prefill_mlp_up', 'share_of_peak'):.1%}",
        "decode achieved tflops": f"{one('decode_mlp_up', 'achieved_tflops'):.2f}",
        "decode implied bandwidth": f"{one('decode_mlp_up', 'implied_gbps'):,.0f}",
    }


@pytest.mark.parametrize(
    ("csv_path", "note", "claims"),
    [
        (T6_CSV, Path("topics/t06_perf_reasoning/README.md"), _t6_claims),
        (T7_CSV, Path("topics/t07_roofline/README.md"), _t7_claims),
    ],
)
def test_lab_note_numbers_match_the_results(csv_path, note, claims) -> None:
    """Every headline figure quoted in prose must still be recomputable from the CSV.

    Write-ups drift: a re-run changes the data, the note keeps the old number, and nothing fails.
    This is the cheapest possible defence against a lab note that quietly stops being true.
    """
    if not (csv_path.exists() and note.exists()):
        pytest.skip("topic has not been run in this session")

    text = note.read_text()
    for label, value in claims(read_rows(csv_path)).items():
        assert value in text, f"{note.name} no longer quotes the measured {label}: {value}"
