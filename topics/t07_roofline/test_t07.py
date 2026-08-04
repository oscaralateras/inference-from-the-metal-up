"""Tests for T7. The roofline arithmetic must hold before any GPU time is spent on it."""

from __future__ import annotations

import pytest

from arch_common.gpu import HardwareProfile
from topics.t07_roofline.shapes import (
    DEFAULT_HIDDEN,
    DEFAULT_INTERMEDIATE,
    GemmShape,
    batch_walk_shapes,
    inference_shapes,
)

BF16_BYTES = 2

# An A100 SXM's realistic measured ceilings, used to check the derived quantities without a GPU.
A100 = HardwareProfile(
    session_id="test",
    device="cuda",
    device_name="NVIDIA A100-SXM4-80GB",
    dtype="bfloat16",
    peak_bandwidth_gbps=1700.0,
    peak_tflops=280.0,
)


def test_gemm_flops_are_two_per_multiply_accumulate() -> None:
    shape = GemmShape("t", 4, 5, 6, "prefill")
    assert shape.flops == 2 * 4 * 5 * 6


def test_gemm_bytes_count_both_operands_and_the_result() -> None:
    shape = GemmShape("t", 4, 5, 6, "prefill")
    assert shape.bytes_moved(BF16_BYTES) == (4 * 6 + 6 * 5 + 4 * 5) * BF16_BYTES


def test_decode_intensity_approaches_two_over_bytes_per_element() -> None:
    """M=1 collapses arithmetic intensity to `2/b` — 1.0 in bfloat16.

    This is the same number T6 derives for the whole model. The two topics reach it from opposite
    directions: T6 from total parameter traffic, T7 from a single matmul's operands.
    """
    decode = GemmShape("decode", 1, DEFAULT_INTERMEDIATE, DEFAULT_HIDDEN, "decode")
    assert decode.arithmetic_intensity(BF16_BYTES) == pytest.approx(1.0, rel=0.01)


def test_prefill_intensity_is_orders_of_magnitude_above_decode() -> None:
    shapes = {s.name: s for s in inference_shapes()}
    prefill = shapes["prefill_mlp_up"].arithmetic_intensity(BF16_BYTES)
    decode = shapes["decode_mlp_up"].arithmetic_intensity(BF16_BYTES)
    assert prefill > 100 * decode


def test_the_two_regimes_land_on_opposite_sides_of_the_ridge() -> None:
    """The central claim of T7: same weights, same model, opposite bottlenecks."""
    shapes = {s.name: s for s in inference_shapes()}
    assert shapes["prefill_mlp_up"].arithmetic_intensity(BF16_BYTES) > A100.ridge_point
    assert shapes["decode_mlp_up"].arithmetic_intensity(BF16_BYTES) < A100.ridge_point


def test_ridge_point_is_peak_compute_over_peak_bandwidth() -> None:
    assert A100.ridge_point == pytest.approx(280e12 / 1700e9)


def test_batch_walk_intensity_rises_monotonically() -> None:
    """Every doubling of batch must move the point rightward, or the walk has no meaning."""
    intensities = [s.arithmetic_intensity(BF16_BYTES) for s in batch_walk_shapes()]
    assert intensities == sorted(intensities)
    assert intensities[-1] > 10 * intensities[0]


def test_batch_walk_starts_memory_bound_and_moves_toward_the_ridge() -> None:
    shapes = batch_walk_shapes()
    first = shapes[0].arithmetic_intensity(BF16_BYTES)
    last = shapes[-1].arithmetic_intensity(BF16_BYTES)
    assert first < A100.ridge_point
    assert last / first > 50


def test_batching_does_not_multiply_the_weight_traffic() -> None:
    """Batch 256 does far less than 256x the traffic — the mechanism batching exploits."""
    shapes = {s.m: s for s in batch_walk_shapes()}
    assert shapes[256].bytes_moved(BF16_BYTES) < 256 * shapes[1].bytes_moved(BF16_BYTES)


def test_roof_is_the_lower_of_the_two_ceilings() -> None:
    from topics.t07_roofline.plot import _roof

    below, above = _roof(A100, [1.0, 10_000.0])
    assert below == pytest.approx(A100.peak_bandwidth_gbps * 1e9 * 1.0 / 1e12)
    assert above == pytest.approx(A100.peak_tflops)


def test_no_model_weights_are_loaded_anywhere_in_t07() -> None:
    """T7 is defined as kernel-shape work on synthetic tensors.

    Loading weights would make it slow, expensive, and no longer structurally distinct from T6.
    """
    from pathlib import Path

    banned = ("from_pretrained", "AutoModel", "safetensors")
    for path in Path(__file__).parent.glob("*.py"):
        if path.name == Path(__file__).name:
            continue  # this file names the banned tokens in order to check for them
        source = path.read_text()
        for token in banned:
            assert token not in source, f"{path.name} references {token} — T7 must stay synthetic"
