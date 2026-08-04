"""Tests for the shared measurement layer. If timing is wrong, every topic's numbers are wrong."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest
import torch

from arch_common.gpu import HardwareProfile, load_profile, measure_peak_bandwidth
from arch_common.results_io import FIELDS, append_rows, read_rows, scalar, select
from arch_common.timing import time_op

CPU = torch.device("cpu")


def _row(experiment: str, variant: str, x: float, metric: str, value: float) -> dict[str, object]:
    return {
        "session_id": "s1",
        "experiment": experiment,
        "variant": variant,
        "x": x,
        "metric": metric,
        "value": value,
    }


# -- timing ---------------------------------------------------------------------------------


def test_time_op_measures_real_elapsed_time() -> None:
    """A 20 ms sleep must report ~20 ms. Guards against a timer that measures nothing."""
    ms = time_op(lambda: time.sleep(0.02), CPU, warmup=1, iters=3)
    assert 15.0 < ms < 60.0


def test_time_op_returns_the_median_not_the_mean() -> None:
    """One pathological sample must not move the reported number."""
    calls = {"n": 0}

    def occasionally_slow() -> None:
        calls["n"] += 1
        time.sleep(0.05 if calls["n"] == 1 else 0.001)

    ms = time_op(occasionally_slow, CPU, warmup=0, iters=9)
    assert ms < 10.0


def test_time_op_rejects_nonsense_arguments() -> None:
    with pytest.raises(ValueError, match="iters"):
        time_op(lambda: None, CPU, iters=0)


# -- hardware profile -----------------------------------------------------------------------


def test_bandwidth_probe_counts_both_the_read_and_the_write() -> None:
    """A copy touches each element twice. Forgetting that halves the reported bandwidth."""
    gbps = measure_peak_bandwidth(CPU, torch.float32)
    assert gbps > 0.5, "a plausible machine moves more than 0.5 GB/s"


def test_ridge_point_is_compute_over_bandwidth() -> None:
    profile = HardwareProfile(
        session_id="t",
        device="cuda",
        device_name="test",
        dtype="bfloat16",
        peak_bandwidth_gbps=2000.0,
        peak_tflops=300.0,
    )
    assert profile.ridge_point == pytest.approx(150.0)


def test_missing_profile_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="arch_common.probe"):
        load_profile(tmp_path / "absent.json")


# -- results io -----------------------------------------------------------------------------


def test_rerunning_one_variant_leaves_its_siblings_alone(tmp_path: Path) -> None:
    """The T5 data-loss bug, pinned. Keying on experiment alone silently deleted sibling rows."""
    csv_path = tmp_path / "r.csv"
    append_rows(csv_path, [_row("e", "a", 1, "m", 1.0), _row("e", "b", 1, "m", 2.0)])
    append_rows(csv_path, [_row("e", "a", 1, "m", 9.0)])

    rows = read_rows(csv_path)
    assert scalar(rows, "e", "a", "m") == 9.0
    assert scalar(rows, "e", "b", "m") == 2.0


def test_writes_are_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    rows = [_row("e", "a", 1, "m", 1.0)]
    append_rows(csv_path, rows)
    append_rows(csv_path, rows)
    assert len(read_rows(csv_path)) == 1


def test_an_empty_result_set_is_rejected(tmp_path: Path) -> None:
    """A run that produced no rows failed. Writing nothing silently would hide that."""
    with pytest.raises(ValueError, match="empty result set"):
        append_rows(tmp_path / "r.csv", [])


def test_rows_missing_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        append_rows(tmp_path / "r.csv", [{"experiment": "e", "variant": "a"}])


def test_header_matches_the_declared_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    append_rows(csv_path, [_row("e", "a", 1, "m", 1.0)])
    with csv_path.open() as f:
        assert tuple(next(csv.reader(f))) == FIELDS


def test_select_returns_a_sorted_curve(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    append_rows(csv_path, [_row("e", "a", x, "m", float(x)) for x in (8, 1, 4)])
    assert select(read_rows(csv_path), "e", "a", "m") == [(1.0, 1.0), (4.0, 4.0), (8.0, 8.0)]


def test_scalar_refuses_an_ambiguous_lookup(tmp_path: Path) -> None:
    csv_path = tmp_path / "r.csv"
    append_rows(csv_path, [_row("e", "a", 1, "m", 1.0), _row("e", "a", 2, "m", 2.0)])
    with pytest.raises(KeyError, match="exactly one"):
        scalar(read_rows(csv_path), "e", "a", "m")


def test_inner_reports_per_call_time_not_total() -> None:
    """`inner` amortises launch cost across N calls and must divide, not accumulate.

    A missing division here would silently multiply every T8 timing by 16 and report a kernel
    sixteen times slower than it is — the kind of arithmetic slip that produces a confident,
    catastrophically wrong number rather than an error.
    """
    single = time_op(lambda: time.sleep(0.01), CPU, warmup=0, iters=3)
    batched = time_op(lambda: time.sleep(0.01), CPU, warmup=0, iters=3, inner=4)
    assert batched == pytest.approx(single, rel=0.5)


def test_inner_must_be_positive() -> None:
    with pytest.raises(ValueError, match="inner >= 1"):
        time_op(lambda: None, CPU, inner=0)
