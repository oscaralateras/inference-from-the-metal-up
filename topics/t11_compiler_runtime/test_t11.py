"""Unit tests for T11. All run on any machine — no GPU, no CUDA graphs, no compiler required.

The measurement needs a GPU, so what is tested here is everything it *decides with*: the byte
arithmetic that predicts fusion's ceiling, the derivation that places the crossover, the chain
itself (which runs identically on CPU), and the crossover locator.

The derivation tests pin the model as it was *registered*, which is not the same as pinning it as
correct — the measurement refuted its chain-length scaling, and the lab-note tests at the bottom of
this file pin that refutation so it cannot be quietly softened later. Both are kept deliberately: a
repo that edits its model to match the data afterwards has no pre-registration at all.
"""

from __future__ import annotations

import pytest
import torch

from arch_common.results_io import read_rows, select
from topics.t11_compiler_runtime.chain import (
    BOUNDARY_READS,
    BOUNDARY_WRITES,
    CHAIN_OPS,
    DEFAULT_HIDDEN,
    activation_bytes,
    bandwidth_bound_batch,
    decode_chain,
    fused_bytes,
    fusion_ceiling,
    fusion_crossover_batch,
    make_inputs,
    traffic_seconds,
    unfused_bytes,
)
from topics.t11_compiler_runtime.measure import crossover_batch
from topics.t11_compiler_runtime.modes import (
    MODES,
    assert_fusion_is_not_secretly_graphs,
    build_callable,
    count_kernel_launches,
)

CPU = torch.device("cpu")

# T7's measured A100 bandwidth, used as a fixed input so these tests assert the arithmetic rather
# than whatever machine happens to run them.
A100_GBPS = 1736.7


# ---------------------------------------------------------------------------------------------
# byte arithmetic
# ---------------------------------------------------------------------------------------------


def test_activation_bytes_is_the_decode_tensor_every_other_topic_quotes() -> None:
    """Batch 1 at hidden 3584 in bf16 is 7,168 B — the same payload T9's all-reduce carries."""
    assert activation_bytes(1, DEFAULT_HIDDEN) == 7168


def test_activation_bytes_rejects_a_zero_batch() -> None:
    with pytest.raises(ValueError, match="positive"):
        activation_bytes(0, DEFAULT_HIDDEN)


def test_unfused_traffic_is_two_round_trips_per_op() -> None:
    """Each unfused op reads its input and writes its output. Nothing subtler than that."""
    assert unfused_bytes(1, DEFAULT_HIDDEN, ops=3) == 2 * 3 * 7168


def test_fused_traffic_is_the_boundary_tensors_only() -> None:
    """A perfect fuser touches HBM for the chain's inputs and its output, and never in between."""
    assert fused_bytes(1, DEFAULT_HIDDEN) == (BOUNDARY_READS + BOUNDARY_WRITES) * 7168


def test_fusion_ceiling_is_independent_of_batch_and_hardware() -> None:
    """It is 2k/3 — a property of the chain's shape, so it cannot be tuned into or out of."""
    expected = 2 * len(CHAIN_OPS) / (BOUNDARY_READS + BOUNDARY_WRITES)
    for batch in (1, 32, 2048):
        assert fusion_ceiling(batch, DEFAULT_HIDDEN) == pytest.approx(expected)


def test_traffic_seconds_rejects_a_zero_bandwidth() -> None:
    with pytest.raises(ValueError, match="positive"):
        traffic_seconds(1024, 0.0)


def test_the_batch_1_chain_is_nanoseconds_of_traffic() -> None:
    """Band 1's justification, as arithmetic: there is nothing here for fusion to remove.

    41 nanoseconds of traffic against a kernel launch measured in microseconds is three orders of
    magnitude of headroom for launch overhead to hide in.
    """
    us = traffic_seconds(unfused_bytes(1, DEFAULT_HIDDEN), A100_GBPS) * 1e6
    assert us < 0.1


# ---------------------------------------------------------------------------------------------
# the crossover derivation — the topic's headline, and the thing that was wrong once
# ---------------------------------------------------------------------------------------------


def test_the_bandwidth_bound_threshold_does_not_depend_on_chain_length() -> None:
    """Here `ops` genuinely does cancel: it multiplies traffic and launches equally."""
    assert bandwidth_bound_batch(A100_GBPS, 5.0) == pytest.approx(
        A100_GBPS * 1e9 * 5e-6 / (4 * DEFAULT_HIDDEN)
    )


def test_the_fusion_crossover_does_depend_on_chain_length() -> None:
    """The correction. Fusion removes 2k-3 round trips; capture removes k launches.

    An earlier version of this model cancelled `ops` here and predicted a crossover 3.5x too high.
    A longer chain gives fusion more to remove, so it crosses over *earlier*.
    """
    five = fusion_crossover_batch(A100_GBPS, 5.0, ops=5)
    two = fusion_crossover_batch(A100_GBPS, 5.0, ops=2)

    assert five < two
    assert two / five == pytest.approx((2 * 5 - 3) / (2 * 2 - 3))


def test_the_crossover_lands_inside_the_pre_registered_band() -> None:
    """The band has to contain the model's own prediction, or it was chosen to fail."""
    from topics.t11_compiler_runtime.predict import ASSUMED_LAUNCH_US, CROSSOVER_BATCH_RANGE

    predicted = fusion_crossover_batch(A100_GBPS, ASSUMED_LAUNCH_US)
    lo, hi = CROSSOVER_BATCH_RANGE
    assert lo <= predicted <= hi


def test_a_chain_too_short_to_win_has_no_crossover_rather_than_a_large_one() -> None:
    """With one op, fusion removes nothing — 2*1 round trips against 3 boundary tensors.

    Returning a huge number here would be worse than raising: it would look like a prediction.
    """
    with pytest.raises(ValueError, match="never overtakes"):
        fusion_crossover_batch(A100_GBPS, 5.0, ops=1)


def test_a_faster_launch_moves_the_crossover_earlier() -> None:
    """Cheaper launches mean less for capture to remove, so fusion wins sooner."""
    assert fusion_crossover_batch(A100_GBPS, 1.0) < fusion_crossover_batch(A100_GBPS, 10.0)


# ---------------------------------------------------------------------------------------------
# locating the crossover in measured data
# ---------------------------------------------------------------------------------------------


def test_crossover_is_interpolated_in_log_space() -> None:
    """The sweep doubles, so linear interpolation would bias every answer toward the upper point."""
    batches = [1, 8, 64, 512]
    fusion = [0.5, 0.8, 1.5, 3.0]
    graphs = [3.0, 2.0, 1.5, 1.0]

    # Fusion overtakes exactly at the third point, where the two are equal.
    assert crossover_batch(batches, fusion, graphs) == pytest.approx(64.0)


def test_crossover_returns_zero_when_fusion_never_overtakes() -> None:
    """A real answer the note must be able to report, not a crash."""
    assert crossover_batch([1, 8, 64], [0.5, 0.6, 0.7], [3.0, 2.5, 2.0]) == 0.0


def test_crossover_lands_between_its_bracketing_points() -> None:
    batches = [32, 128]
    result = crossover_batch(batches, [0.9, 1.4], [1.1, 1.0])
    assert 32 < result < 128


# ---------------------------------------------------------------------------------------------
# the chain itself, which runs identically on CPU
# ---------------------------------------------------------------------------------------------


def test_the_chain_preserves_shape() -> None:
    inputs = make_inputs(4, 64, CPU, torch.float32)
    out = decode_chain(inputs.hidden_state, inputs.residual, inputs.gate, inputs.weight)
    assert out.shape == inputs.hidden_state.shape


def test_truncating_the_chain_changes_the_result() -> None:
    """If two lengths agreed, the truncation would not be exercising what the control claims."""
    inputs = make_inputs(4, 64, CPU, torch.float32)
    args = (inputs.hidden_state, inputs.residual, inputs.gate, inputs.weight)

    outputs = [decode_chain(*args, ops=k) for k in range(1, len(CHAIN_OPS) + 1)]
    for shorter, longer in zip(outputs, outputs[1:], strict=False):
        assert not torch.allclose(shorter, longer)


def test_the_chain_rejects_an_out_of_range_length() -> None:
    inputs = make_inputs(2, 64, CPU, torch.float32)
    args = (inputs.hidden_state, inputs.residual, inputs.gate, inputs.weight)

    for bad in (0, len(CHAIN_OPS) + 1):
        with pytest.raises(ValueError, match="ops must be"):
            decode_chain(*args, ops=bad)


def test_the_chain_is_numerically_finite() -> None:
    """RMSNorm divides; a chain that quietly produced NaN would still time beautifully."""
    inputs = make_inputs(8, 128, CPU, torch.float32)
    out = decode_chain(inputs.hidden_state, inputs.residual, inputs.gate, inputs.weight)
    assert torch.isfinite(out).all()


def test_every_mode_gets_identical_inputs() -> None:
    """The 2x2 compares execution strategies, so anything else differing would confound it."""
    a = make_inputs(4, 64, CPU, torch.float32)
    b = make_inputs(4, 64, CPU, torch.float32)
    assert torch.equal(a.hidden_state, b.hidden_state)
    assert torch.equal(a.weight, b.weight)


# ---------------------------------------------------------------------------------------------
# the guard that keeps the 2x2 a 2x2
# ---------------------------------------------------------------------------------------------


def test_inductor_is_not_configured_to_apply_cuda_graphs() -> None:
    """If it were, the `compile` cell would contain both mechanisms and every attribution is wrong.

    This is the failure this repo cares most about: plausible numbers, invalid conclusion, nothing
    raising. So it is a test rather than a comment.
    """
    assert_fusion_is_not_secretly_graphs()


def test_the_eager_callable_runs_without_a_gpu() -> None:
    inputs = make_inputs(2, 64, CPU, torch.float32)
    fn = build_callable(inputs, compiled=False, ops=len(CHAIN_OPS))
    assert fn().shape == inputs.hidden_state.shape


def test_launch_counting_reports_unavailable_rather_than_zero_on_cpu() -> None:
    """Zero launches would be a plausible-looking lie; -1 cannot be plotted by accident."""
    inputs = make_inputs(2, 64, CPU, torch.float32)
    fn = build_callable(inputs, compiled=False, ops=len(CHAIN_OPS))
    assert count_kernel_launches(fn, CPU) == -1


def test_the_four_modes_are_the_two_by_two() -> None:
    """Two mechanisms crossed, which is the entire experimental design."""
    assert set(MODES) == {"eager", "compile", "graph", "compile_graph"}
    assert sum(m.startswith("compile") for m in MODES) == 2
    assert sum(m.endswith("graph") for m in MODES) == 2


# ---------------------------------------------------------------------------------------------
# The lab note, checked against the data it claims to describe
# ---------------------------------------------------------------------------------------------


def _results() -> list[dict[str, str]]:
    from topics.t11_compiler_runtime.measure import CSV_PATH

    if not CSV_PATH.exists():
        pytest.skip("T11 has not been run in this session")
    rows = read_rows(CSV_PATH)
    if {r["session_id"] for r in rows} == {"rehearsal"}:
        pytest.skip("T11 holds rehearsal numbers, not measured ones")
    return rows


def _at(rows: list[dict[str, str]], exp: str, variant: str, metric: str, x: int) -> float:
    hits = [v for xx, v in select(rows, exp, variant, metric) if int(xx) == x]
    assert len(hits) == 1, f"expected one {exp}/{variant}/{metric} at x={x}, found {len(hits)}"
    return hits[0]


# (batch, fusion speedup, graph speedup, combined) — the note's headline table.
QUOTED_SPEEDUPS = [
    (1, 1.48, 5.50, 16.4),
    (8, 1.88, 5.01, 20.2),
    (32, 1.85, 3.91, 19.4),
    (128, 1.85, 3.50, 19.0),
    (512, 1.84, 2.18, 11.9),
    (2048, 2.72, 1.07, 4.7),
]


@pytest.mark.parametrize(("batch", "fusion", "graphs", "both"), QUOTED_SPEEDUPS)
def test_lab_note_speedup_table(batch: int, fusion: float, graphs: float, both: float) -> None:
    rows = _results()
    assert _at(rows, "mechanism", "chain", "fusion_speedup", batch) == pytest.approx(
        fusion, rel=0.01
    )
    assert _at(rows, "mechanism", "chain", "graph_speedup", batch) == pytest.approx(
        graphs, rel=0.01
    )
    assert _at(rows, "mechanism", "chain", "combined_speedup", batch) == pytest.approx(
        both, rel=0.01
    )


def test_lab_note_the_mechanisms_swap_dominance() -> None:
    """The topic's whole claim: capture leads at batch 1, fusion leads at 2048."""
    rows = _results()
    assert _at(rows, "mechanism", "chain", "graph_speedup", 1) > _at(
        rows, "mechanism", "chain", "fusion_speedup", 1
    )
    assert _at(rows, "mechanism", "chain", "fusion_speedup", 2048) > _at(
        rows, "mechanism", "chain", "graph_speedup", 2048
    )


def test_lab_note_crossover_and_the_remodelled_prediction() -> None:
    """The note quotes a crossover at 648, a measured launch cost of 23.15 us, and a remodelled
    prediction of 801 — 24% out, against the 3.7x the assumed 5 us input was out by."""
    rows = _results()
    cross = _at(rows, "crossover", "chain", "crossover_batch", 5)
    launch = _at(rows, "crossover", "chain", "measured_launch_us", 5)
    remodelled = _at(rows, "crossover", "chain", "remodelled_crossover_batch", 5)

    assert cross == pytest.approx(648, rel=0.01)
    assert launch == pytest.approx(23.15, rel=0.01)
    assert remodelled == pytest.approx(801, rel=0.01)
    # The remodelled prediction must beat the originally registered one, or the note's claim that
    # "the structure survived, the input did not" is not supported.
    from topics.t11_compiler_runtime.predict import build_prediction

    registered = build_prediction(1737.1).predicted_crossover_batch
    assert abs(remodelled - cross) / cross < abs(registered - cross) / cross


def test_lab_note_the_chain_length_control_refutes_the_scaling_prediction() -> None:
    """The most important failure in the topic, pinned so it cannot be quietly softened.

    The model predicted the crossover moves by (2*5-3)/(2*k-3) as the chain shortens: 2.33x at
    3 ops and 7.00x at 2. Measured: 0.97x and 1.16x. Refuted, not marginally.
    """
    rows = _results()
    five = _at(rows, "crossover", "chain", "crossover_batch", 5)
    three = _at(rows, "crossover", "chain_ops3", "crossover_batch", 3)
    two = _at(rows, "crossover", "chain_ops2", "crossover_batch", 2)

    assert three == pytest.approx(631, rel=0.01)
    assert two == pytest.approx(748, rel=0.01)

    for measured, ops in ((three, 3), (two, 2)):
        predicted_shift = (2 * 5 - 3) / (2 * ops - 3)
        actual_shift = measured / five
        assert actual_shift < predicted_shift / 2, (
            f"at {ops} ops the model predicted a {predicted_shift:.2f}x shift and the note reports "
            f"the measured {actual_shift:.2f}x as a refutation — this test keeps that true"
        )

    # And the empirical claim that replaced it: the crossover is insensitive to chain length.
    assert max(five, three, two) / min(five, three, two) < 1.25


def test_lab_note_band_4_fails_because_the_five_op_chain_does_not_fully_fuse() -> None:
    """The kernel counts are the evidence, and they need no timer to read."""
    rows = _results()

    assert _at(rows, "modes", "eager", "kernel_launches", 2048) == 11
    assert _at(rows, "modes", "compile", "kernel_launches", 2048) == 2, (
        "the 5-op chain emits TWO kernels — the note's explanation for band 4 depends on this"
    )
    for variant in ("compile_ops2", "compile_ops3"):
        assert _at(rows, "modes", variant, "kernel_launches", 2048) == 1, (
            f"{variant} should fuse to a single kernel"
        )


def test_lab_note_fused_bandwidth_by_chain_length() -> None:
    """91.7% / 80.1% / 47.9% of roof at 2, 3 and 5 ops — the shorter chains would pass band 4."""
    from topics.t11_compiler_runtime.predict import MIN_SHARE_OF_MEMORY_ROOF

    rows = _results()
    roof = 1737.1
    shares = {}
    for suffix, ops in (("_ops2", 2), ("_ops3", 3), ("", 5)):
        us = _at(rows, "modes", f"compile_graph{suffix}", "latency_us", 2048)
        shares[ops] = fused_bytes(2048, DEFAULT_HIDDEN) / (us * 1e-6) / 1e9 / roof

    assert shares[2] == pytest.approx(0.917, abs=0.01)
    assert shares[3] == pytest.approx(0.801, abs=0.01)
    assert shares[5] == pytest.approx(0.479, abs=0.01)

    assert shares[2] >= MIN_SHARE_OF_MEMORY_ROOF and shares[3] >= MIN_SHARE_OF_MEMORY_ROOF
    assert shares[5] < MIN_SHARE_OF_MEMORY_ROOF


def test_lab_note_band_4_as_registered_and_as_diagnosed() -> None:
    """The band stays failed as registered; the diagnostic is reported beside it, not instead."""
    rows = _results()
    roof = 1737.1
    registered = _at(rows, "mechanism", "chain", "fused_gbps", 2048) / roof
    diagnosed = _at(rows, "mechanism", "chain", "graphed_fused_gbps", 2048) / roof

    assert registered == pytest.approx(0.276, abs=0.005)
    assert diagnosed == pytest.approx(0.479, abs=0.005)
    assert diagnosed > registered


def test_lab_note_compile_is_host_bound() -> None:
    """The note's caveat, as data: 2,048x the work costs `compile` less time, not more."""
    rows = _results()
    assert _at(rows, "modes", "compile", "latency_us", 2048) <= _at(
        rows, "modes", "compile", "latency_us", 1
    )


BATCH_SWEEP = (1, 8, 32, 128, 512, 2048)


def test_lab_note_removing_the_host_cost_collapses_the_crossover() -> None:
    """The note reports a registered crossover of 648 and a host-free one of 15.8.

    `compile` pays Dynamo's guard cost on every call and `graph` does not, so the registered
    comparison charges fusion for something that is not fusion. Comparing the two graph-captured
    columns removes it from both sides. If that ever stopped mattering, the note's central caveat
    about batch 648 would be overstated and this test should fail.
    """
    rows = _results()
    batches: list[int] = list(BATCH_SWEEP)
    capture = [_at(rows, "modes", "eager", "latency_us", b) for b in batches]
    unfused = [_at(rows, "modes", "graph", "latency_us", b) for b in batches]
    fused = [_at(rows, "modes", "compile_graph", "latency_us", b) for b in batches]
    registered = [_at(rows, "mechanism", "chain", "fusion_speedup", b) for b in batches]

    graphs = [c / u for c, u in zip(capture, unfused, strict=True)]
    host_free = [u / f for u, f in zip(unfused, fused, strict=True)]

    assert crossover_batch(batches, registered, graphs) == pytest.approx(648, rel=0.02)
    assert crossover_batch(batches, host_free, graphs) == pytest.approx(15.8, rel=0.05)


def test_lab_note_the_all_fusing_chain_reaches_one_kernel() -> None:
    """The whole second control rests on this: five ops, one kernel, at every batch."""
    rows = _results()
    for batch in BATCH_SWEEP:
        assert _at(rows, "modes", "compile_fusing", "kernel_launches", batch) == 1
        assert _at(rows, "modes", "eager_fusing", "kernel_launches", batch) == 15


def test_lab_note_fusion_completeness_moves_the_crossover_far_more_than_length() -> None:
    """The note's claim: 6.5x from fusion completeness against 1.16x from halving the chain.

    The rotary and all-fusing chains have the same op count and the same byte model, so the only
    thing separating their crossovers is whether the chain collapses to one kernel.
    """
    rows = _results()
    fusing = [v for _, v in select(rows, "crossover", "chain_fusing", "crossover_batch")][0]
    rotary = [v for _, v in select(rows, "crossover", "chain", "crossover_batch")][0]
    two_op = [v for _, v in select(rows, "crossover", "chain_ops2", "crossover_batch")][0]

    assert fusing == pytest.approx(102, rel=0.05)
    assert rotary / fusing > 4.0, "the note quotes a 4.6-6.5x collapse"
    assert two_op / rotary == pytest.approx(1.16, rel=0.05), "chain length barely moved it"


def test_lab_note_compile_emits_three_kernels_at_batch_one() -> None:
    """The note used to say `compile` runs 2 kernels. At batch 1 Inductor emits 3."""
    rows = _results()
    assert _at(rows, "modes", "compile", "kernel_launches", 1) == 3
    assert _at(rows, "modes", "compile", "kernel_launches", 2048) == 2
    assert _at(rows, "modes", "eager", "kernel_launches", 1) == 11
