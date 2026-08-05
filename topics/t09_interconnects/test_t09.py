"""Unit tests for T9. All run on any machine — no GPU, no collectives.

The distributed code cannot be unit-tested without hardware, so what is tested here is everything
the distributed code *decides with*: the cost model, the fit, and the topology gate. The gate in
particular is tested against a captured PCIe matrix, because the one thing it must do is fail, and
a gate that has never been observed failing is not a gate.
"""

from __future__ import annotations

import pytest

from topics.t09_interconnects.model import (
    DEFAULT_HIDDEN,
    RingCost,
    allreduce_bytes,
    bus_factor,
    bus_gbps,
    comms_per_token_us,
    fit_ring_cost,
    predicted_tp_speedup,
    prefill_allreduce_bytes,
    ring_hops,
    sweep_sizes,
)
from topics.t09_interconnects.topology import (
    MIN_NVLINK_BUS_GBPS,
    check_declared,
    check_empirical,
    parse_topo,
)

# A real 4x A100 SXM node — the topology T5 measured on.
NVLINK_MATRIX = """\t GPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV12\tNV12\tNV12\t0-31\t0
GPU1\tNV12\t X \tNV12\tNV12\t0-31\t0
GPU2\tNV12\tNV12\t X \tNV12\t32-63\t1
GPU3\tNV12\tNV12\tNV12\t X \t32-63\t1
"""

# The node this topic must refuse: GPUs on separate PCIe root complexes, no NVLink anywhere.
PCIE_MATRIX = """\t GPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tSYS\t0-31\t0
GPU1\tSYS\t X \t32-63\t1
"""

# The subtle one: NVLink between two of four, PCIe for the rest. Passes at world 2, must fail at 4.
MIXED_MATRIX = """\t GPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity
GPU0\t X \tNV4\tSYS\tSYS\t0-31
GPU1\tNV4\t X \tSYS\tSYS\t0-31
GPU2\tSYS\tSYS\t X \tNV4\t32-63
GPU3\tSYS\tSYS\tNV4\t X \t32-63
"""


# ---------------------------------------------------------------------------------------------
# message sizes — the arithmetic the whole topic turns on
# ---------------------------------------------------------------------------------------------


def test_decode_allreduce_does_not_depend_on_sequence_length() -> None:
    """Decode emits one token per sequence, so the payload is batch x hidden and nothing else."""
    assert allreduce_bytes(1, 3584) == 3584 * 2
    assert allreduce_bytes(32, 3584) == 32 * 3584 * 2


def test_t5_operating_point_is_four_orders_of_magnitude_larger() -> None:
    """The premise of the topic, as a test rather than a claim in prose."""
    decode = allreduce_bytes(1, DEFAULT_HIDDEN)
    prefill = prefill_allreduce_bytes(16, 512, DEFAULT_HIDDEN)
    assert prefill / decode == pytest.approx(16 * 512)
    assert prefill / decode > 8000


@pytest.mark.parametrize("bad", [(0, 3584, 2), (1, 0, 2), (1, 3584, 0)])
def test_message_size_rejects_nonsense(bad: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        allreduce_bytes(*bad)


# ---------------------------------------------------------------------------------------------
# the ring model
# ---------------------------------------------------------------------------------------------


def test_ring_hops_and_bus_factor_match_the_algorithm() -> None:
    assert ring_hops(2) == 2
    assert ring_hops(4) == 6
    assert bus_factor(2) == pytest.approx(1.0)
    assert bus_factor(4) == pytest.approx(1.5)
    assert bus_factor(8) == pytest.approx(1.75)


def test_bus_factor_approaches_two_but_never_reaches_it() -> None:
    assert bus_factor(1024) < 2.0
    assert bus_factor(1024) > 1.99


def test_bus_bandwidth_is_zero_for_a_world_of_one() -> None:
    """No collective happens, so reporting a bandwidth would be inventing one."""
    assert bus_gbps(1024, 1, 10.0) == 0.0


def test_alpha_scaling_prediction_is_three() -> None:
    """Band (1), asserted against the algorithm rather than against a remembered number."""
    assert ring_hops(4) / ring_hops(2) == 3.0


# ---------------------------------------------------------------------------------------------
# the fit — the estimator has to recover what was put in
# ---------------------------------------------------------------------------------------------


def test_fit_recovers_known_alpha_and_beta() -> None:
    """Synthesise a curve from a known (alpha, beta), fit it, and check both come back.

    The same move T5 made with Amdahl: an estimator you have not tested against a known answer is
    a number generator, not a measurement.
    """
    truth = RingCost(world=4, alpha_us=7.5, beta_gbps=480.0, r_squared=1.0, n_points=0)
    points = [(n, truth.time_us(n)) for n in sweep_sizes(1024, 1024**3)]

    fit = fit_ring_cost(4, points)

    assert fit.alpha_us == pytest.approx(7.5, rel=0.02)
    assert fit.beta_gbps == pytest.approx(480.0, rel=0.02)
    assert fit.r_squared > 0.999


def test_fit_reports_a_bad_r_squared_when_the_curve_is_not_two_term() -> None:
    """A link that changes protocol mid-sweep must not come back as a confident straight line.

    NCCL really does this — it switches algorithm between tree and ring by message size — so this
    is the realistic way the two-term model stops describing the hardware, and R^2 is the only
    thing standing between that and a published bandwidth that averages two different regimes.
    """
    slow = RingCost(world=4, alpha_us=7.5, beta_gbps=120.0, r_squared=1.0, n_points=0)
    fast = RingCost(world=4, alpha_us=7.5, beta_gbps=480.0, r_squared=1.0, n_points=0)
    switch = 64 * 1024**2
    points = [(n, (slow if n < switch else fast).time_us(n)) for n in sweep_sizes()]

    fit = fit_ring_cost(4, points)

    assert fit.r_squared < 0.999


def test_fit_refuses_a_sweep_that_never_left_the_latency_floor() -> None:
    """Flat data has no bandwidth in it, and reporting one would be fabrication."""
    with pytest.raises(ValueError, match="beta is unconstrained"):
        fit_ring_cost(2, [(n, 8.0) for n in (1024, 2048, 4096, 8192)])


def test_fit_refuses_a_sweep_with_no_floor() -> None:
    """Alpha is the topic's headline number; it must come from data, not from extrapolation."""
    with pytest.raises(ValueError, match="alpha is unconstrained"):
        fit_ring_cost(2, [(n, n / 1e6) for n in (8 * 1024**2, 64 * 1024**2, 512 * 1024**2)])


def test_fit_refuses_a_ramp_that_does_not_rise() -> None:
    """A decreasing ramp is a broken measurement, not a negative bandwidth."""
    with pytest.raises(ValueError, match="non-positive slope"):
        fit_ring_cost(
            2,
            [(1024, 8.0), (8 * 1024**2, 400.0), (64 * 1024**2, 300.0), (512 * 1024**2, 100.0)],
        )


def test_fit_needs_two_points_and_a_real_collective() -> None:
    with pytest.raises(ValueError, match="world >= 2"):
        fit_ring_cost(1, [(1024, 8.0), (2048, 9.0)])
    with pytest.raises(ValueError, match=">= 2 points"):
        fit_ring_cost(2, [(1024, 8.0)])


def test_alpha_step_divides_out_the_ring_length() -> None:
    """Per-hop latency is what the ring model says should be constant across world sizes."""
    two = RingCost(world=2, alpha_us=8.0, beta_gbps=480.0, r_squared=1.0, n_points=0)
    four = RingCost(world=4, alpha_us=24.0, beta_gbps=480.0, r_squared=1.0, n_points=0)
    assert two.alpha_step_us == pytest.approx(4.0)
    assert four.alpha_step_us == pytest.approx(4.0)


# ---------------------------------------------------------------------------------------------
# the two regimes
# ---------------------------------------------------------------------------------------------


def test_crossover_is_where_the_two_terms_are_equal() -> None:
    cost = RingCost(world=4, alpha_us=7.5, beta_gbps=480.0, r_squared=1.0, n_points=0)
    n = cost.crossover_bytes()
    assert cost.alpha_share(int(n)) == pytest.approx(0.5, rel=1e-3)


def test_decode_is_alpha_bound_and_prefill_is_not() -> None:
    """Band (3), and the central claim of the topic, on a plausible fitted model."""
    cost = RingCost(world=4, alpha_us=7.5, beta_gbps=480.0, r_squared=1.0, n_points=0)
    assert cost.alpha_share(allreduce_bytes(1, DEFAULT_HIDDEN)) > 0.99
    assert cost.alpha_share(prefill_allreduce_bytes(16, 512, DEFAULT_HIDDEN)) < 0.05


def test_comms_per_token_amortises_across_the_batch() -> None:
    """Fixed cost per call is paid once per step, so per-token it falls as the batch grows."""
    cost = RingCost(world=4, alpha_us=7.5, beta_gbps=480.0, r_squared=1.0, n_points=0)
    assert comms_per_token_us(cost, 32) < comms_per_token_us(cost, 1)


def test_comms_per_token_counts_two_collectives_per_layer() -> None:
    cost = RingCost(world=2, alpha_us=10.0, beta_gbps=1e9, r_squared=1.0, n_points=0)
    # beta enormous, so the moving term vanishes and only 28*2 alphas remain.
    assert comms_per_token_us(cost, 1, layers=28) == pytest.approx(28 * 2 * 10.0, rel=1e-3)


# ---------------------------------------------------------------------------------------------
# Amdahl with a comms penalty
# ---------------------------------------------------------------------------------------------


def test_free_comms_and_all_weights_gives_perfect_scaling() -> None:
    """The model's upper bound, which is the thing it is allowed to reach and never exceed."""
    assert predicted_tp_speedup(1.0, 4, 10.0, 0.0) == pytest.approx(4.0)


def test_comms_only_ever_costs() -> None:
    free = predicted_tp_speedup(0.74, 4, 11.0, 0.0)
    taxed = predicted_tp_speedup(0.74, 4, 11.0, 500.0)
    assert taxed < free


def test_speedup_can_fall_below_one_when_comms_dominates() -> None:
    """Sharding a small model over a slow link should be *worse* — the model must allow that."""
    assert predicted_tp_speedup(0.74, 4, 1.0, 5000.0) < 1.0


def test_speedup_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        predicted_tp_speedup(1.5, 4, 10.0, 0.0)
    with pytest.raises(ValueError):
        predicted_tp_speedup(0.7, 4, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------------------------


def test_sweep_is_strictly_increasing_and_within_bounds() -> None:
    sizes = sweep_sizes()
    assert sizes == sorted(set(sizes))
    assert sizes[0] >= 1024
    assert sizes[-1] <= 1024**3
    assert all(n % 4 == 0 for n in sizes)


def test_sweep_spans_both_regimes() -> None:
    """It has to contain decode's payload and something far past the crossover, or it fits noise."""
    sizes = sweep_sizes()
    assert min(sizes) <= allreduce_bytes(1, DEFAULT_HIDDEN)
    assert max(sizes) >= 100 * 1024**2


# ---------------------------------------------------------------------------------------------
# the topology gate — the test that matters is the one where it fails
# ---------------------------------------------------------------------------------------------


def test_parses_a_real_nvlink_matrix() -> None:
    topo = parse_topo(NVLINK_MATRIX)
    assert topo.link(0, 3) == "NV12"
    assert topo.link(3, 0) == "NV12"
    assert topo.non_nvlink_pairs(4) == []
    assert topo.nvlink_width(4) == 12


def test_gate_rejects_a_pcie_node() -> None:
    """The failure this whole module exists for."""
    with pytest.raises(RuntimeError, match="topology gate FAILED"):
        check_declared(2, parse_topo(PCIE_MATRIX))


def test_gate_rejects_a_partially_connected_node_at_the_world_size_that_needs_it() -> None:
    """Passing at world 2 and failing at world 4 is exactly the trap on a rented box."""
    topo = parse_topo(MIXED_MATRIX)
    check_declared(2, topo)
    with pytest.raises(RuntimeError, match="GPU0-GPU2=SYS"):
        check_declared(4, topo)


def test_nvlink_width_takes_the_narrowest_hop() -> None:
    """A ring runs at its weakest link, so the summary must not advertise the best one."""
    assert parse_topo(MIXED_MATRIX).nvlink_width(2) == 4


def test_empirical_gate_catches_a_link_that_declared_well_and_ran_badly() -> None:
    check_empirical(MIN_NVLINK_BUS_GBPS + 1.0, 4)
    with pytest.raises(RuntimeError, match="empirically"):
        check_empirical(MIN_NVLINK_BUS_GBPS - 1.0, 4)


def test_gate_ignores_the_affinity_columns() -> None:
    """`0-31` in a CPU-affinity column must never be read as an interconnect class."""
    topo = parse_topo(NVLINK_MATRIX)
    assert set(topo.matrix.values()) == {"NV12"}
