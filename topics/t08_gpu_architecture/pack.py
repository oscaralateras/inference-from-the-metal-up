"""Pack a weight matrix into the int4 layout the kernel reads.

T1 established *that* int4 works and at what granularity: per-group holds cosine 0.99 where
per-tensor collapses to 0.79, because one outlier channel otherwise sets the scale for the whole
matrix. T1 stopped at the numerics. This module takes the same quantiser and produces the thing a
GPU can actually stream — which is a different problem, and the one the byte budget cares about.

**The layout, and why it is not the obvious one.**

Two int4 codes share one byte. The obvious packing puts adjacent columns `2j` and `2j+1` in the
same byte, which forces the kernel to interleave two half-width vectors back into column order on
every load — awkward in Triton and not free. So instead we pair column `j` with column
`j + K/2`::

    packed[n, j] = nibble(W[n, j]) | nibble(W[n, j + K/2]) << 4      j in [0, K/2)

The low nibbles are then exactly the first half of the row and the high nibbles exactly the
second half, both already in order. The kernel unpacks into two independent accumulations against
two contiguous slices of `x` and never interleaves anything. Layout chosen for the consumer, which
is the whole lesson of the topic.

**The byte budget this produces**, for group size G with bf16 scales::

    bits/param = 4 + 16/G       = 4.125 at G=128
    bytes/param = 0.5156        against bfloat16's 2.0  ->  3.88x less traffic

That 3.88 is the number the pre-registered prediction in `measure.py` is built from, so it is
computed here (`bytes_per_param`) rather than written down as a constant anywhere.

Signed int4 is stored offset by 8 — codes `[-7, 7]` become nibbles `[1, 15]` — because a nibble is
unsigned and sign-extending 4 bits in the kernel costs more than subtracting a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from topics.t01_number_representation.quantise import quantise_symmetric

# int4. T1 swept {8, 4, 3}; 4 is the bit-width production serving actually uses, and the one whose
# accuracy T1 already characterised at this granularity.
N_BITS = 4

# Nibbles are unsigned, so the signed code is stored as `code + ZERO_OFFSET`.
ZERO_OFFSET = 8

# Per-group scales, group size along the reduction (input) axis. T1's finding is that granularity
# is worth 2-3 bits; this is the granularity that bought it.
DEFAULT_GROUP_SIZE = 128

SCALE_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class PackedWeight:
    """An int4 weight matrix in the layout `int4_gemv` expects.

    Attributes:
        packed: `(N, K//2)` uint8. Low nibble is column `j`, high nibble is column `j + K//2`.
        scales: `(N, K//G)` bf16, one per group per output row.
        n: output features (rows of the original `(N, K)` weight).
        k: input features (columns).
        group_size: columns per scale.
    """

    packed: torch.Tensor
    scales: torch.Tensor
    n: int
    k: int
    group_size: int

    @property
    def bytes_stored(self) -> int:
        """Actual bytes the kernel must stream to read this weight, scales included.

        Measured off the tensors rather than derived from a formula, so a layout change that
        quietly costs more traffic shows up in the prediction instead of hiding behind it.
        """
        return self.packed.numel() * self.packed.element_size() + (
            self.scales.numel() * self.scales.element_size()
        )

    @property
    def bytes_per_param(self) -> float:
        """Bytes streamed per weight — the term the whole speedup prediction rests on."""
        return self.bytes_stored / (self.n * self.k)


def quantise_and_pack(
    weight: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> PackedWeight:
    """Quantise a `(N, K)` weight to int4 per-group and pack it for the kernel.

    Args:
        weight: `(out_features, in_features)`, any float dtype, any device.
        group_size: columns sharing one scale. Must divide both `K` and `K//2` — the second
            condition is what keeps a group from straddling the two nibble halves, which would
            make the kernel's scale indexing wrong in a way that is easy to miss because the
            output would still look approximately right.

    Returns:
        A `PackedWeight` on the same device as `weight`.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")

    n, k = weight.shape
    if k % 2 != 0:
        raise ValueError(f"in_features must be even to pack two codes per byte, got K={k}")
    half = k // 2
    if group_size < 1 or k % group_size != 0 or half % group_size != 0:
        raise ValueError(
            f"group_size must divide both K={k} and K//2={half}, got group_size={group_size}"
        )

    # T1's quantiser is numpy and float32 by contract. Round-tripping through the CPU here is
    # deliberate: packing happens once, offline, and reusing the already-tested implementation is
    # worth more than saving milliseconds on a one-off.
    w_np = weight.detach().to(torch.float32).cpu().numpy()
    codes, scale_full = quantise_symmetric(w_np, N_BITS, "per_group", group_size)

    # `quantise_symmetric` returns the scale broadcast back to full width for easy dequantisation.
    # The kernel wants one scale per group, so take the first column of each group. They are equal
    # by construction; assert it rather than trust it, because a silent change to the broadcast
    # shape upstream would otherwise corrupt every weight in the repo without failing anything.
    scales_np = scale_full.reshape(n, k // group_size, group_size)
    if not np.allclose(scales_np, scales_np[:, :, :1]):
        raise ValueError("scale is not constant within a group — quantiser contract changed")
    scales_np = scales_np[:, :, 0]

    nibbles = (codes.astype(np.int16) + ZERO_OFFSET).astype(np.uint8)
    if nibbles.max() > 15:
        raise ValueError(f"code out of nibble range after offset: max {nibbles.max()}")

    lo = nibbles[:, :half]
    hi = nibbles[:, half:]
    packed_np = (lo | (hi << 4)).astype(np.uint8)

    device = weight.device
    return PackedWeight(
        packed=torch.from_numpy(packed_np).to(device),
        scales=torch.from_numpy(scales_np).to(device=device, dtype=SCALE_DTYPE),
        n=n,
        k=k,
        group_size=group_size,
    )


def unpack_to_dense(pw: PackedWeight) -> torch.Tensor:
    """Reconstruct the dequantised `(N, K)` float32 matrix from the packed form.

    Reference implementation only — it materialises exactly the tensor the fused kernel exists to
    avoid ever writing to memory. Used by the tests to prove the kernel computes the same thing,
    and by nothing else.
    """
    lo = (pw.packed & 0x0F).to(torch.int16) - ZERO_OFFSET
    hi = (pw.packed >> 4).to(torch.int16) - ZERO_OFFSET
    codes = torch.cat([lo, hi], dim=1).to(torch.float32)

    scales = pw.scales.to(torch.float32).repeat_interleave(pw.group_size, dim=1)
    return codes * scales
