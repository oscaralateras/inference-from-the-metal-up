"""Guards on T8's claims — the same discipline T5 learned the hard way and T6/T7 inherited.

T8 is the first topic that claims a *win*, which makes it the first topic with an incentive to
drift. Three specific ways it could:

1. **The prediction stops being a prediction.** Register it after seeing the measurement and the
   whole pre-registration framing is theatre. Enforced by pinning the prediction to the session
   its results came from.
2. **The lab note stops matching the CSV.** T5 shipped with its note and its results disagreeing;
   T1, T6 and T7 all carry a guard against it now. T8 does too.
3. **The result quietly breaks physics.** A memory-bound kernel cannot beat its own byte ratio.
   If it appears to, the timing is wrong or the byte accounting is — either way it is a bug being
   reported as a triumph, which is the worst failure mode this repo has.

These run without a GPU. They skip when T8 has not been measured, so the Mac stays green, and they
bite the moment results exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from arch_common.results_io import read_rows, scalar
from topics.t08_gpu_architecture.measure import (
    BASELINE_BYTES_PER_PARAM,
    MIN_COSINE,
    PREDICTIONS_PATH,
)
from topics.t08_gpu_architecture.measure import (
    CSV_PATH as T8_CSV,
)
from topics.t08_gpu_architecture.pack import DEFAULT_GROUP_SIZE, N_BITS

NOTE = Path(__file__).resolve().parent.parent / "topics" / "t08_gpu_architecture" / "README.md"

# T8 reports in the byte domain — the deliberate consequence of the topic's argument that FLOP/s
# is the wrong unit for a decode kernel. Declared here so an accidental `achieved_tflops` creeping
# into the CSV is a failure rather than a silent regime confusion.
T8_METRICS = {
    "ms",
    "gbps",
    "bytes",
    "kernel_speedup",
    "byte_ratio",
    "cosine",
    "end_to_end_speedup",
    "predicted_end_to_end",
    "kernel_speedup_min",
    "kernel_speedup_max",
    "speedup_vs_cublas",
    "int8_speedup",
    "int8_byte_ratio",
    "share_of_roof",
}


def _skip_unless_measured() -> list[dict[str, str]]:
    if not T8_CSV.exists():
        pytest.skip("T8 has not been run in this session")
    return read_rows(T8_CSV)


def test_byte_ratio_matches_the_packing_arithmetic() -> None:
    """3.88x is derived, not chosen. Recompute it from the format constants alone.

    Runs with no results at all — this is the claim the whole topic is built on, so it should not
    depend on anyone having rented a GPU.
    """
    bytes_per_param = (N_BITS + 16 / DEFAULT_GROUP_SIZE) / 8
    assert BASELINE_BYTES_PER_PARAM / bytes_per_param == pytest.approx(3.88, abs=0.01)


def test_the_prediction_was_registered_against_the_session_it_is_judged_on() -> None:
    """Predict-then-measure, verified rather than promised.

    Mirrors T6's guard. The prediction file is written before the kernel runs and stamped with the
    session from the hardware profile, so a prediction quietly regenerated after a disappointing
    run would point at a different session and fail here.
    """
    rows = _skip_unless_measured()
    if not PREDICTIONS_PATH.exists():
        pytest.fail("results exist but predictions.json does not — the prediction was not filed")

    registered = json.loads(PREDICTIONS_PATH.read_text())["session_id"]
    measured = {r["session_id"] for r in rows}
    assert registered, "prediction has no session_id — it was written by --skip-kernel, not a run"
    assert measured == {registered}, (
        f"prediction registered against session {registered} but results are from {measured}"
    )


def test_t8_reports_only_declared_metrics() -> None:
    rows = _skip_unless_measured()
    metrics = {r["metric"] for r in rows}
    assert metrics <= T8_METRICS, f"T8 emitted undeclared metrics: {metrics - T8_METRICS}"


def test_t8_does_not_report_in_the_compute_domain() -> None:
    """A decode GEMV has no compute story. Reporting TFLOP/s here would undo T7's whole lesson."""
    rows = _skip_unless_measured()
    metrics = {r["metric"] for r in rows}
    assert not (metrics & {"achieved_tflops", "share_of_peak", "flops_per_byte"}), (
        "T8 emitted compute-domain metrics — it is scored in bytes against the memory roof"
    )


def test_the_load_only_ceiling_bounds_the_full_kernel() -> None:
    """The full kernel cannot out-run the same access pattern with the arithmetic removed.

    `probe_ceiling` measures loads only. Anything the full kernel appears to gain over that is a
    measurement error, not a fast kernel — and it is the probe that localised T8's shortfall to the
    harness, so it is worth keeping honest.
    """
    rows = _skip_unless_measured()
    ceiling = [r for r in rows if r["experiment"] == "ceiling"]
    if not ceiling:
        pytest.skip("probe_ceiling has not been run in this session")

    load_only = scalar(rows, "ceiling", "load_only", "gbps")
    full = scalar(rows, "gemv", "int4_fused", "gbps")
    assert full <= load_only * 1.05, (
        f"full kernel {full:,.0f} GB/s exceeds the load-only ceiling {load_only:,.0f} GB/s — "
        "removing work cannot make a kernel slower; check the timing"
    )


def test_the_kernel_cannot_beat_its_own_byte_ratio() -> None:
    """The physical ceiling. Exceeding it means the measurement is wrong, not the kernel fast.

    Both variants do identical arithmetic; only the bytes differ. So in a purely bandwidth-bound
    regime the speedup is bounded above by the traffic ratio. A small overshoot is possible if the
    baseline is not perfectly bandwidth-bound, hence the 5% margin — but a large one is a bug in
    the timing or in the byte accounting.
    """
    rows = _skip_unless_measured()
    speedup = scalar(rows, "summary", "int4_fused", "kernel_speedup")
    byte_ratio = scalar(rows, "summary", "int4_fused", "byte_ratio")
    assert speedup <= byte_ratio * 1.05, (
        f"kernel reported {speedup:.2f}x against a {byte_ratio:.2f}x byte ratio — "
        "a bandwidth-bound kernel cannot exceed its traffic reduction; check the timing"
    )


def test_end_to_end_is_smaller_than_the_kernel_speedup() -> None:
    """Amdahl, as an invariant. The whole is always less improved than the part."""
    rows = _skip_unless_measured()
    kernel = scalar(rows, "summary", "int4_fused", "kernel_speedup")
    end_to_end = scalar(rows, "summary", "int4_fused", "end_to_end_speedup")
    assert end_to_end < kernel, (
        f"end-to-end {end_to_end:.2f}x >= kernel {kernel:.2f}x — impossible while any part of the "
        "step is untouched by quantisation"
    )


def test_accuracy_floor_held() -> None:
    """int4 per-group must hold T1's cosine. Below it, the win is being bought with correctness."""
    rows = _skip_unless_measured()
    cosine = scalar(rows, "summary", "int4_fused", "cosine")
    assert cosine >= MIN_COSINE, (
        f"cosine {cosine:.4f} below T1's int4-per-group floor of {MIN_COSINE} — "
        "the speedup is not free, which changes the finding"
    )


def _speedups_quoted_in_the_note() -> set[float]:
    """Every `N.NNx` figure in the lab note — the form all four speedup claims are written in."""
    return {float(m) for m in re.findall(r"\b(\d+\.\d\d)x\b", NOTE.read_text())}


def test_lab_note_headline_numbers_come_from_the_results() -> None:
    """Every speedup quoted in the note must exist in the CSV, to two decimal places.

    T5's note and its CSV disagreed for a week because the table was transcribed from stdout by
    hand. This is the cheap guard that makes that impossible: a re-run that changes the numbers
    fails CI until the note is updated with them.

    Only *measured* speedups are checked. The note also quotes the predicted 3.88x and 2.33x,
    which are in the results too (`byte_ratio`, `predicted_end_to_end`), so both survive.
    """
    rows = _skip_unless_measured()
    if "___" in NOTE.read_text():
        pytest.skip("lab note still has placeholders — fill it in after the run")

    available = {
        round(scalar(rows, "summary", "int4_fused", metric), 2)
        for metric in ("kernel_speedup", "byte_ratio", "end_to_end_speedup", "predicted_end_to_end")
    }
    unsupported = {q for q in _speedups_quoted_in_the_note() if q not in available}
    assert not unsupported, (
        f"lab note quotes speedups {sorted(unsupported)} that are not in the results "
        f"{sorted(available)} — re-run or correct the note"
    )
