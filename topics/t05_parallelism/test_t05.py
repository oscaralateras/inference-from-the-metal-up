"""Unit tests for T5 — the invariants that make the measurements mean anything.

Deliberately network-free: every test uses `random_block()` rather than real weights, so CI never
downloads a checkpoint. The parallelism properties under test are functions of the *shapes*, not
of the weight values, so random weights test them exactly as well.

The tests fall into three groups:

1. **Sharding correctness** — every decomposition must reconstruct the unsharded forward. This is
   the group that already caught a real bug: with batch and sequence collapsed into one axis, data
   parallelism was silently splitting the sequence and producing wrong output at full speed.
2. **The estimator** — the Amdahl fit must recover a serial fraction it is given, and must report
   values outside [0, 1] rather than clamping when a curve is outside the model's domain.
3. **The predictions** — the pipeline-bubble formula and the MoE load factor must agree with their
   closed forms, because the measured numbers are only meaningful against a correct prediction.
"""

from __future__ import annotations

import math

import pytest
import torch
from amdahl import ScalingPoint, fit_parallel_fraction
from model import random_block, rms_norm
from moe import build_moe, load_factor, route
from pipeline import predicted_efficiency
from results_io import FIELDS

BATCH, SEQ = 2, 32


@pytest.fixture(scope="module")
def block():
    return random_block(seed=0)


@pytest.fixture(scope="module")
def x(block):
    return torch.randn(BATCH, SEQ, block.hidden, generator=torch.Generator().manual_seed(1))


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))


# ---- 1. sharding correctness ---------------------------------------------------------------


def test_attention_rejects_2d_input(block):
    """The bug guard: a collapsed (tokens, hidden) axis makes DP silently become SP."""
    flat = torch.randn(SEQ, block.hidden)
    with pytest.raises(ValueError, match="batch, seq, hidden"):
        block.attention(flat)


@pytest.mark.parametrize("world", [1, 2, 3, 4, 6])
def test_tensor_parallel_reconstructs_unsharded(block, x, world):
    """Summing every rank's partial output must equal the unsharded forward.

    Also pins the residual ordering: the residual is added *after* each all-reduce. Adding it
    before would multiply it by `world` and is the classic TP correctness bug.
    """
    ref = block.forward(x)
    shards = [block.tensor_parallel_shard(r, world) for r in range(world)]

    attn = sum(s.attention(rms_norm(x, s.attn_norm)) for s in shards)  # all-reduce #1
    h = x + attn
    mlp = sum(s.mlp(rms_norm(h, s.mlp_norm)) for s in shards)  # all-reduce #2
    got = h + mlp

    assert rel_err(got, ref) < 1e-4


@pytest.mark.parametrize("world", [1, 2, 4])
def test_sequence_parallel_reconstructs_unsharded(block, x, world):
    """SP must match the unsharded forward, which requires the causal mask to be offset.

    A rank holding sequence slice [lo, hi) computes queries from its own slice but keys/values
    from the gathered prefix. If the mask is not offset by `lo`, the slice attends to its own
    future and the output is wrong in a way that still looks numerically plausible.
    """
    ref = block.forward(x)
    per = SEQ // world
    xn = rms_norm(x, block.attn_norm)

    outs = [
        block.attention(xn[:, r * per : (r + 1) * per], kv_source=xn[:, : (r + 1) * per])
        for r in range(world)
    ]
    h = x + torch.cat(outs, dim=1)
    got = h + block.mlp(rms_norm(h, block.mlp_norm))

    assert rel_err(got, ref) < 1e-4


@pytest.mark.parametrize("world", [1, 2, 4])
def test_data_parallel_splits_batch_not_sequence(block, x, world):
    """DP must be exact: sequences are independent, so splitting the batch changes nothing.

    Any error here means the batch axis is being confused with the sequence axis.
    """
    ref = block.forward(x)
    per = BATCH // world if world <= BATCH else 1
    if BATCH % world:
        pytest.skip(f"batch {BATCH} not divisible by world {world}")
    got = torch.cat([block.forward(x[r * per : (r + 1) * per]) for r in range(world)], dim=0)
    assert rel_err(got, ref) == pytest.approx(0.0, abs=1e-6)


def test_tensor_parallel_rejects_indivisible_degree(block):
    """TP degree is not a free parameter — it must divide the head count."""
    with pytest.raises(ValueError, match="attention heads"):
        block.tensor_parallel_shard(0, 5)  # 12 heads is not divisible by 5


def test_shard_holds_a_fraction_of_the_weights(block):
    """TP genuinely divides the footprint; that is why it, not DP, answers 'it does not fit'."""
    full = block.weight_bytes()
    for world in (2, 4):
        shard = block.tensor_parallel_shard(0, world)
        assert shard.weight_bytes() == pytest.approx(full / world, rel=1e-9)


# ---- 2. the Amdahl estimator ---------------------------------------------------------------


@pytest.mark.parametrize("p_true", [1.0, 0.95, 0.8, 0.5])
def test_fit_recovers_known_parallel_fraction(p_true):
    """On noise-free synthetic data the estimator must recover p essentially exactly."""
    points = [ScalingPoint(n, 0.0, 1.0 / ((1.0 - p_true) + p_true / n)) for n in (1, 2, 4, 8, 16)]
    p_hat, r2 = fit_parallel_fraction(points)
    assert p_hat == pytest.approx(p_true, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_reports_negative_p_rather_than_clamping():
    """A curve that gets *slower* with more workers is outside Amdahl's domain.

    Amdahl assumes coordination is free, so its floor is 1.0x. Contention is worse than that.
    Reporting p < 0 makes the model's failure visible; clamping to 0 would hide it and imply
    'perfectly serial' when the truth is 'actively harmful'.
    """
    points = [ScalingPoint(1, 0.0, 1.0), ScalingPoint(2, 0.0, 0.5), ScalingPoint(4, 0.0, 0.25)]
    p_hat, _ = fit_parallel_fraction(points)
    assert p_hat < 0.0


def test_fit_needs_more_than_one_worker():
    with pytest.raises(ValueError, match="workers > 1"):
        fit_parallel_fraction([ScalingPoint(1, 0.0, 1.0)])


# ---- 3. the predictions ---------------------------------------------------------------------


@pytest.mark.parametrize("stages", [2, 4, 8])
@pytest.mark.parametrize("microbatches", [1, 4, 16, 64])
def test_balanced_bubble_matches_closed_form(stages, microbatches):
    """A balanced pipeline must reduce exactly to the textbook M / (M + P - 1)."""
    weights = tuple([1] * stages)
    expected = microbatches / (microbatches + stages - 1)
    assert predicted_efficiency(weights, microbatches) == pytest.approx(expected, rel=1e-12)


def test_bubble_approaches_one_as_microbatches_grow():
    """The bubble is amortised, never eliminated — efficiency rises but stays below 1."""
    weights = (1, 1, 1, 1)
    effs = [predicted_efficiency(weights, m) for m in (1, 4, 16, 64, 1024)]
    assert effs == sorted(effs)
    assert effs[-1] < 1.0
    assert effs[-1] > 0.99


def test_imbalanced_pipeline_is_capped_by_its_slowest_stage():
    """With one stage 2x slow, efficiency asymptotes to total_work / (stages * slowest)."""
    weights = (2, 1, 1, 1)
    ceiling = sum(weights) / (len(weights) * max(weights))  # 5 / 8
    assert predicted_efficiency(weights, 10**7) == pytest.approx(ceiling, rel=1e-4)
    assert ceiling == pytest.approx(0.625)


def test_uniform_routing_is_perfectly_balanced():
    assignment = route(1024, 8, "uniform")
    assert load_factor(assignment, 8, 4) == pytest.approx(1.0)


def test_skewed_routing_creates_a_hot_rank():
    """EP's failure mode: work per rank is decided by the router, not the decomposition."""
    assignment = route(1024, 8, "skewed", seed=0)
    lf = load_factor(assignment, 8, 4)
    assert lf > 1.5, f"expected a clear imbalance, got load factor {lf}"


def test_routing_is_deterministic():
    """Every strategy must see the identical assignment or the comparison is not like-for-like."""
    a = route(512, 8, "skewed", seed=3)
    b = route(512, 8, "skewed", seed=3)
    assert torch.equal(a, b)


def test_moe_forward_matches_per_expert_composition():
    """The unsharded MoE reference must equal running each expert over its own tokens."""
    moe = build_moe(64, 128, n_experts=4, seed=0)
    tokens = torch.randn(96, 64, generator=torch.Generator().manual_seed(2))
    assignment = route(96, 4, "uniform")

    ref = moe.forward(tokens, assignment)
    got = torch.zeros_like(tokens)
    for e in range(4):
        idx = (assignment == e).nonzero(as_tuple=True)[0]
        got[idx] = moe.forward_expert(e, tokens[idx])
    assert rel_err(got, ref) == pytest.approx(0.0, abs=1e-6)


def test_expert_parallel_divides_the_footprint():
    """EP holds E/W experts per rank — the reason MoE models are served this way at all."""
    moe = build_moe(64, 128, n_experts=8, seed=0)
    per_expert = moe.expert_bytes()
    assert per_expert * 8 == pytest.approx(
        sum(t.numel() for t in (moe.gate, moe.up, moe.down)) * 4, rel=1e-9
    )


def test_route_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown routing mode"):
        route(16, 4, "definitely-not-a-mode")


# ---- misc -----------------------------------------------------------------------------------


def test_csv_contract_is_stable():
    """plot.py reads this contract; changing it silently would break every figure."""
    assert FIELDS == ("experiment", "variant", "workers", "metric", "value")


def test_rms_norm_is_per_token(block):
    """RMSNorm normalises the last dim only, so it is unaffected by batch or sequence splits."""
    x = torch.randn(2, 8, block.hidden, generator=torch.Generator().manual_seed(4))
    full = rms_norm(x, block.attn_norm)
    halves = torch.cat(
        [rms_norm(x[:, :4], block.attn_norm), rms_norm(x[:, 4:], block.attn_norm)], 1
    )
    assert rel_err(halves, full) == pytest.approx(0.0, abs=1e-6)


def test_attention_is_causal(block):
    """A token must not see its future — otherwise the SP prefix trick would be unsound."""
    x = torch.randn(1, 16, block.hidden, generator=torch.Generator().manual_seed(6))
    base = block.attention(x)
    perturbed = x.clone()
    perturbed[:, -1] += 100.0  # change only the LAST token
    after = block.attention(perturbed)
    # every earlier position must be untouched
    assert rel_err(after[:, :-1], base[:, :-1]) == pytest.approx(0.0, abs=1e-5)


def test_head_dim_consistency(block):
    assert block.n_heads * block.head_dim == block.hidden
    assert math.isclose(block.hidden / block.n_heads, block.head_dim)
