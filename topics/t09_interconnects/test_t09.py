"""Unit tests for T9. All run on any machine — no GPU, no collectives.

The distributed code cannot be unit-tested without hardware, so what is tested here is everything
the distributed code *decides with*: the cost model, the fit, and the topology gate. The gate in
particular is tested against a captured PCIe matrix, because the one thing it must do is fail, and
a gate that has never been observed failing is not a gate.
"""

from __future__ import annotations

import pytest

from arch_common.results_io import read_rows, scalar, select
from topics.t09_interconnects.model import (
    DEFAULT_HIDDEN,
    RingCost,
    allreduce_bytes,
    bus_factor,
    bus_gbps,
    comms_per_token_us,
    fit_latency_budget,
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

# Captured verbatim from the RunPod 4x A100 SXM node this topic ran on, ANSI escape and all.
# nvidia-smi underlines the header row even when its output is piped, and with that escape left in
# the header is skipped and the GPU0 *data* row is mistaken for it -- leaving only GPU0's pairs in
# the matrix. On this node that passed the gate at world 2 and failed it at world 4, which reads as
# a partially-connected box. The regression is kept as real captured output rather than a
# hand-written fixture, because a hand-written one is exactly what missed this the first time.
REAL_POD_MATRIX = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tNIC1\tNIC2\tNIC3\tNIC4\tNIC5\tNIC6\tNIC7"
    "\tNIC8\tNIC9\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV12\tNV12\tNV12\tPXB\tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS"
    "\t0-63\t0\t\tN/A\n"
    "GPU1\tNV12\t X \tNV12\tNV12\tPXB\tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS"
    "\t0-63\t0\t\tN/A\n"
    "GPU2\tNV12\tNV12\t X \tNV12\tSYS\tSYS\tSYS\tSYS\tPXB\tPXB\tNODE\tNODE\tNODE\tNODE"
    "\t64-127\t1\t\tN/A\n"
    "GPU3\tNV12\tNV12\tNV12\t X \tSYS\tSYS\tSYS\tSYS\tPXB\tPXB\tNODE\tNODE\tNODE\tNODE"
    "\t64-127\t1\t\tN/A\n"
    "NIC0\tPXB\tPXB\tSYS\tSYS\t X \tPXB\tNODE\tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\n"
    "NIC5\tSYS\tSYS\tPXB\tPXB\tSYS\tSYS\tSYS\tSYS\tPXB\t X \tNODE\tNODE\tNODE\tNODE\n"
)

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


def test_parses_the_real_pod_matrix_ansi_escape_and_all() -> None:
    """The regression. Every pair NVLink at world 4, not just the ones touching GPU0."""
    topo = parse_topo(REAL_POD_MATRIX)

    assert topo.non_nvlink_pairs(4) == []
    assert topo.nvlink_width(4) == 12
    check_declared(4, topo)


def test_gpu0_data_row_is_never_mistaken_for_the_header() -> None:
    """The precise failure: a header of [0] records GPU0's pairs and nothing else."""
    topo = parse_topo(REAL_POD_MATRIX)

    assert topo.link(1, 2) == "NV12"
    assert topo.link(2, 3) == "NV12"
    assert topo.link(1, 3) == "NV12"


def test_nic_rows_and_columns_are_ignored() -> None:
    """Ten NIC columns sit between the GPUs and the affinity columns on a real node."""
    topo = parse_topo(REAL_POD_MATRIX)

    assert set(topo.matrix.values()) == {"NV12"}


# ---------------------------------------------------------------------------------------------
# The lab note, checked against the data it claims to describe
# ---------------------------------------------------------------------------------------------
#
# T5 learned this the hard way: its note and its CSV disagreed until a test compared them. A number
# in prose is a copy of a number in a file, and copies drift -- especially when a topic is re-run on
# new hardware and only some of the paragraphs get updated. Every figure quoted in README.md is
# listed here with its provenance, so a re-run that changes the data fails the build rather than
# quietly leaving the note describing a machine nobody has now.


def _results() -> list[dict[str, str]]:
    from topics.t09_interconnects.measure import CSV_PATH

    if not CSV_PATH.exists():
        pytest.skip("T9 has not been run in this session")
    return read_rows(CSV_PATH)


def _fit(rows: list[dict[str, str]], world: int, metric: str) -> float:
    return scalar(rows, "fit", f"world{world}", metric)


def _at(rows: list[dict[str, str]], experiment: str, world: int, metric: str, batch: int) -> float:
    hits = [v for x, v in select(rows, experiment, f"world{world}", metric) if int(x) == batch]
    assert len(hits) == 1, f"expected one {experiment}/world{world}/{metric} at batch {batch}"
    return hits[0]


# (world, metric, value quoted in README.md, relative tolerance)
QUOTED_FIT = [
    (2, "alpha_us", 35.45, 0.01),
    (4, "alpha_us", 35.80, 0.01),
    (2, "beta_gbps", 201.5, 0.01),
    (4, "beta_gbps", 221.7, 0.01),
    (2, "fit_r_squared", 0.9997, 0.001),
    (4, "fit_r_squared", 0.9999, 0.001),
]

# (world, batch, comms us/token, modelled TP speedup) — the tensor-parallelism table.
QUOTED_TP = [
    (2, 1, 1987, 1.23),
    (2, 8, 250, 1.53),
    (2, 32, 64, 1.57),
    (2, 128, 17.5, 1.58),
    (4, 1, 2008, 1.59),
    (4, 8, 253, 2.13),
    (4, 32, 65, 2.21),
    (4, 128, 18.4, 2.23),
]

# (world, batch, measured speedup, comms share) — stage 3's real layer.
QUOTED_MEASURED = [
    (2, 1, 1.36, 0.278),
    (2, 128, 1.22, 0.286),
    (4, 1, 1.49, 0.582),
    (4, 128, 1.54, 0.476),
]


@pytest.mark.parametrize(("world", "metric", "quoted", "rel"), QUOTED_FIT)
def test_lab_note_matches_results(world: int, metric: str, quoted: float, rel: float) -> None:
    assert _fit(_results(), world, metric) == pytest.approx(quoted, rel=rel)


@pytest.mark.parametrize(("world", "batch", "comms_us", "speedup"), QUOTED_TP)
def test_lab_note_tp_table_matches_results(
    world: int, batch: int, comms_us: float, speedup: float
) -> None:
    rows = _results()
    assert _at(rows, "decode", world, "comms_us_per_token", batch) == pytest.approx(
        comms_us, rel=0.01
    )
    assert _at(rows, "decode", world, "tp_speedup", batch) == pytest.approx(speedup, rel=0.01)


@pytest.mark.parametrize(("world", "batch", "speedup", "share"), QUOTED_MEASURED)
def test_lab_note_measured_layer_matches_results(
    world: int, batch: int, speedup: float, share: float
) -> None:
    rows = _results()
    assert _at(rows, "tp_matmul", world, "tp_speedup", batch) == pytest.approx(speedup, rel=0.01)
    assert _at(rows, "tp_matmul", world, "comms_share", batch) == pytest.approx(share, rel=0.01)


def test_lab_note_headline_alpha_ratio() -> None:
    """Band 1's verdict: alpha did not scale with 2(N-1). The note quotes 1.01."""
    rows = _results()
    ratio = _fit(rows, 4, "alpha_us") / _fit(rows, 2, "alpha_us")
    assert ratio == pytest.approx(1.01, abs=0.01)


def test_lab_note_alpha_floor_and_hop_decomposition() -> None:
    """The note solves alpha = L + 2(N-1)h and quotes L = 35.28 us, h = 0.087 us, hops = 1.5%."""
    rows = _results()
    a2, a4 = _fit(rows, 2, "alpha_us"), _fit(rows, 4, "alpha_us")
    hop = (a4 - a2) / (ring_hops(4) - ring_hops(2))
    floor = a2 - ring_hops(2) * hop

    assert floor == pytest.approx(35.28, abs=0.05)
    assert hop == pytest.approx(0.087, abs=0.005)
    assert ring_hops(4) * hop / a4 == pytest.approx(0.015, abs=0.002)


def test_lab_note_decode_is_a_rounding_error_of_the_link() -> None:
    """The note quotes 0.300 GB/s at batch 1 on 4 GPUs — 0.135% of the fitted beta."""
    rows = _results()
    call_us = _at(rows, "decode", 4, "allreduce_us", 1)
    achieved = bus_gbps(allreduce_bytes(1, DEFAULT_HIDDEN), 4, call_us)

    assert achieved == pytest.approx(0.300, abs=0.005)
    assert achieved / _fit(rows, 4, "beta_gbps") == pytest.approx(0.00135, abs=0.0001)


def test_lab_note_decode_tax_against_t6() -> None:
    """The note quotes 2.01 ms of collectives per token = 18.2% of T6's measured step."""
    from topics.t09_interconnects.predict import t6_budget

    _, _, step_ms = t6_budget()
    comms_ms = _at(_results(), "decode", 4, "comms_us_per_token", 1) / 1000.0

    assert comms_ms == pytest.approx(2.01, abs=0.02)
    assert comms_ms / step_ms == pytest.approx(0.182, abs=0.002)


def test_lab_note_band_verdicts_are_reported_correctly() -> None:
    """Three of six outside — a note quietly reporting five WITHIN passes every other test."""
    from topics.t09_interconnects.predict import (
        MIN_DECODE_ALPHA_SHARE,
        MIN_SHARE_OF_LINK_SPEC,
        NVLINK_GBPS_PER_LINK,
    )

    rows = _results()
    nv12_spec = 12 * NVLINK_GBPS_PER_LINK

    # Band 2: fails at world 2, passes at world 4 — exactly as the note's table says.
    assert _fit(rows, 2, "beta_gbps") / nv12_spec < MIN_SHARE_OF_LINK_SPEC
    assert _fit(rows, 4, "beta_gbps") / nv12_spec >= MIN_SHARE_OF_LINK_SPEC

    # Band 3: passes at both.
    for world in (2, 4):
        assert _at(rows, "decode", world, "alpha_share", 1) >= MIN_DECODE_ALPHA_SHARE

    # Band 4: passes at world 4, fails at world 2.
    for world, expected_pass in ((4, True), (2, False)):
        ratios = [
            v for _, v in select(rows, "tp_model_check", f"world{world}", "measured_over_predicted")
        ]
        assert ratios, f"no tp_model_check rows for world {world}"
        within = all(1 / 1.5 <= r <= 1.5 for r in ratios)
        assert within is expected_pass


# ---------------------------------------------------------------------------------------------
# separating the fixed floor from the ring's hops
# ---------------------------------------------------------------------------------------------


def test_latency_budget_recovers_a_known_floor_and_hop() -> None:
    """Build alphas from a known (floor, hop) and check both come back."""
    floor, hop = 33.0, 0.5
    alphas = {w: floor + ring_hops(w) * hop for w in (2, 3, 4)}

    budget = fit_latency_budget(alphas)

    assert budget.floor_us == pytest.approx(floor, rel=1e-6)
    assert budget.hop_us == pytest.approx(hop, rel=1e-6)
    assert budget.r_squared == pytest.approx(1.0)
    assert budget.n_worlds == 3


def test_two_world_sizes_cannot_falsify_the_decomposition() -> None:
    """The reason world 3 was added: at two points the fit is exact whatever the data says."""
    budget = fit_latency_budget({2: 35.45, 4: 35.80})

    assert budget.n_worlds == 2
    assert budget.r_squared == pytest.approx(1.0)
    assert budget.alpha_us(2) == pytest.approx(35.45)
    assert budget.alpha_us(4) == pytest.approx(35.80)


def test_three_world_sizes_can_fail_to_fit() -> None:
    """With a residual available, a curve that is not linear in hops shows up as one."""
    budget = fit_latency_budget({2: 35.0, 3: 80.0, 4: 36.0})

    assert budget.n_worlds == 3
    assert budget.r_squared < 0.5


def test_a_flat_alpha_puts_almost_nothing_on_the_hops() -> None:
    """T9's headline, as a property of the estimator rather than of one dataset."""
    budget = fit_latency_budget({2: 35.45, 3: 35.60, 4: 35.80})

    assert budget.hop_us < 0.2
    assert budget.hop_share(4) < 0.05
    assert budget.floor_us == pytest.approx(35.2, abs=0.3)


def test_latency_budget_needs_at_least_two_world_sizes() -> None:
    with pytest.raises(ValueError, match=">= 2 world sizes"):
        fit_latency_budget({4: 35.8})
