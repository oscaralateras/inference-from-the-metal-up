"""Correctness for the packing and the kernel.

The packing tests run anywhere, including the authoring Mac — they are pure numpy/torch and they
are where the layout bugs live. Nibble packing has a specific failure mode worth naming: get the
halves the wrong way round and the output is still *approximately* right, because it is a sum of
thousands of terms with the correct magnitudes in the wrong order. Cosine similarity would sit
around 0.9 and look like a quantisation artefact rather than a bug. So the tests check the
reconstruction elementwise, not statistically.

The kernel tests skip without CUDA rather than failing, so `make ci` stays green on the Mac while
still gating the GPU pod.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from topics.t01_number_representation.quantise import dequantise, quantise_symmetric
from topics.t08_gpu_architecture.kernel import HAS_TRITON, int4_gemv, int4_gemv_reference
from topics.t08_gpu_architecture.pack import (
    N_BITS,
    PackedWeight,
    quantise_and_pack,
    unpack_to_dense,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON),
    reason="Triton kernel requires CUDA; run on the GPU pod",
)


def _weight(n: int = 64, k: int = 256, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=generator) * 0.02


def test_packed_shapes_and_dtypes() -> None:
    pw = quantise_and_pack(_weight(), group_size=64)
    assert pw.packed.shape == (64, 128)
    assert pw.packed.dtype == torch.uint8
    assert pw.scales.shape == (64, 4)


def test_packed_codes_match_the_t1_quantiser_exactly() -> None:
    """The integer codes must survive packing bit-for-bit — no tolerance, none is warranted.

    This is the test that catches a swapped nibble half. Both paths quantise the same weight with
    the same function, so any difference at all is a packing bug. Checking the *codes* rather than
    the reconstructed floats is what makes a zero tolerance legitimate: the scales are stored bf16
    (see the next test), so the reconstruction genuinely differs and a float comparison here would
    have to be loosened until it stopped catching the bug it exists for.
    """
    w = _weight()
    pw = quantise_and_pack(w, group_size=64)

    expected_codes, _ = quantise_symmetric(w.numpy().astype(np.float32), N_BITS, "per_group", 64)

    lo = (pw.packed & 0x0F).to(torch.int16) - 8
    hi = (pw.packed >> 4).to(torch.int16) - 8
    packed_codes = torch.cat([lo, hi], dim=1).numpy().astype(np.int8)

    np.testing.assert_array_equal(packed_codes, expected_codes)


def test_reconstruction_differs_only_by_the_bf16_scale_rounding() -> None:
    """Scales are stored bf16, so the round-trip is *not* bit-identical to T1 — and shouldn't be.

    bf16 carries 8 mantissa bits, so a rounded scale is within ~2^-8 (0.4%) of the fp32 one. That
    error is the price of the 16-bits-per-group term in the byte budget, and it is deliberate:
    fp32 scales would cost 32/128 bits per weight instead of 16/128 and move the byte ratio from
    3.88x to 3.76x. Pinning the tolerance here documents that trade rather than leaving a mystery
    0.4% for a future reader to rediscover.
    """
    w = _weight()
    pw = quantise_and_pack(w, group_size=64)

    codes, scale = quantise_symmetric(w.numpy().astype(np.float32), N_BITS, "per_group", 64)
    fp32_reference = dequantise(codes, scale)

    np.testing.assert_allclose(unpack_to_dense(pw).numpy(), fp32_reference, rtol=2**-8, atol=0)


def test_low_nibble_is_the_first_half_of_the_row() -> None:
    """Pin the layout contract the kernel's scale indexing depends on."""
    pw = quantise_and_pack(_weight(n=2, k=8), group_size=4)
    dense = unpack_to_dense(pw)

    low = (pw.packed & 0x0F).to(torch.int16) - 8
    high = (pw.packed >> 4).to(torch.int16) - 8
    scales = pw.scales.to(torch.float32).repeat_interleave(4, dim=1)

    torch.testing.assert_close(dense[:, :4], low.to(torch.float32) * scales[:, :4])
    torch.testing.assert_close(dense[:, 4:], high.to(torch.float32) * scales[:, 4:])


def test_bytes_per_param_matches_the_arithmetic() -> None:
    """4 bits per weight plus one bf16 scale per group — the term the prediction rests on."""
    pw = quantise_and_pack(_weight(k=256), group_size=128)
    expected = (4 + 16 / 128) / 8
    assert pw.bytes_per_param == pytest.approx(expected)


def test_group_size_must_not_straddle_the_nibble_halves() -> None:
    """K//2 must be a multiple of the group size or the kernel's scale index is silently wrong."""
    with pytest.raises(ValueError, match="group_size must divide"):
        quantise_and_pack(_weight(k=192), group_size=128)  # 96 is not a multiple of 128


def test_odd_in_features_rejected() -> None:
    with pytest.raises(ValueError, match="even to pack"):
        quantise_and_pack(_weight(k=255), group_size=5)


def test_reference_gemv_matches_dense_matmul() -> None:
    """The CPU oracle is itself checked, so a kernel failure cannot be blamed on it."""
    pw = quantise_and_pack(_weight(), group_size=64)
    x = torch.randn(256, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(int4_gemv_reference(pw, x), unpack_to_dense(pw) @ x)


@requires_cuda
def test_kernel_matches_the_unfused_reference() -> None:
    """The fused kernel and the materialise-then-multiply path must agree.

    Tolerance is for fp32 summation order over K terms, not for quantisation: both sides consume
    the *same* int4 codes, so the quantisation error is common to both and cancels.
    """
    w = _weight(n=512, k=1024).cuda()
    x = torch.randn(1024, device="cuda")
    pw = quantise_and_pack(w, group_size=128)

    torch.testing.assert_close(int4_gemv(pw, x), int4_gemv_reference(pw, x), rtol=1e-3, atol=1e-4)


@requires_cuda
def test_kernel_handles_a_ragged_row_count() -> None:
    """N not divisible by any BLOCK_N — the masked tail is where GEMV kernels usually break."""
    w = _weight(n=513, k=1024).cuda()
    x = torch.randn(1024, device="cuda")
    pw = quantise_and_pack(w, group_size=128)

    torch.testing.assert_close(int4_gemv(pw, x), int4_gemv_reference(pw, x), rtol=1e-3, atol=1e-4)


@requires_cuda
def test_kernel_accuracy_against_the_original_float_weight() -> None:
    """End to end: int4 must hold T1's cosine floor against the *unquantised* weight."""
    w = _weight(n=512, k=1024).cuda()
    x = torch.randn(1024, device="cuda")
    pw = quantise_and_pack(w, group_size=128)

    reference = (w @ x).cpu().numpy()
    measured = int4_gemv(pw, x).cpu().numpy()
    cosine = float(
        np.dot(reference, measured) / (np.linalg.norm(reference) * np.linalg.norm(measured))
    )
    assert cosine >= 0.99, f"int4 per-group should hold cosine 0.99, got {cosine:.4f}"


def test_packed_weight_is_immutable() -> None:
    pw = quantise_and_pack(_weight(), group_size=64)
    with pytest.raises((AttributeError, TypeError)):
        pw.n = 1  # type: ignore[misc]
    assert isinstance(pw, PackedWeight)
