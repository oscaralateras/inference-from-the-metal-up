"""Unit tests for T11. All run on any machine — no GPU, no CUDA graphs, no compiler required.

The measurement needs a GPU, so what is tested here is everything it *decides with*: the byte
arithmetic that predicts fusion's ceiling, the derivation that places the crossover, the chain
itself (which runs identically on CPU), and the crossover locator.

The derivation tests are the important ones, because the crossover formula is the topic's headline
and it was wrong once already: an earlier version claimed the crossover does not depend on chain
length, having cancelled `ops` from an equation where it does not cancel. Fusion removes `2k-3`
round trips while capture removes `k` launches, and those scale differently. The tests below pin
that down so the claim cannot silently regress to the wrong one.
"""

from __future__ import annotations

import pytest
import torch

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
