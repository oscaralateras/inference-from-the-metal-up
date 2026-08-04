# pyright: reportPossiblyUnboundVariable=false, reportIndexIssue=false
#
# Triton is CUDA-only and genuinely absent on the authoring Mac, so `triton`/`tl` are bound inside
# an ImportError guard and every reference to them is unreachable without it. Pyright cannot see
# that invariant and flags each use. Scoped to this one file rather than relaxed repo-wide; the
# `HAS_TRITON` check in `int4_gemv` is what actually enforces it, and it raises with instructions.
"""A fused int4-dequantise GEMV in Triton — the topic's whole intervention.

T7 measured decode at 1.0 FLOPs/byte and 82% of the memory roof: near-optimal for the bytes it
moves, and hopeless in absolute terms because it moves so many. Only two levers exist that far
left of the ridge, and adding compute is neither of them. This kernel pulls the first one: **move
fewer bytes.**

**Why the dequantisation must live inside the kernel.** Storing weights int4 and calling
`torch.matmul(x, dequantise(W))` saves nothing — it writes a full-width bf16 matrix to HBM and
then streams it back, so the traffic is bf16 traffic plus the int4 read. The compression only pays
if the expansion happens *after* the load, in registers, and the expanded value never leaves the
SM. That is what "fused" means here, and it is the difference between a 3.9x win and a loss.

**Why a GEMV and not a GEMM.** At M=1 there is no data reuse to exploit: each weight is read once,
multiplied once, discarded. Tiling, shared-memory staging and tensor cores all exist to amortise
loads across a tile of outputs, and with one output column there is nothing to amortise. So the
kernel deliberately uses none of them — `tl.dot` never appears. A decode kernel is a
*bandwidth-shaped* problem wearing matmul clothing, and writing one is the clearest way to
internalise that the GPU's headline TFLOP/s number is irrelevant here.

**Shape of the work.** One program handles `BLOCK_N` output rows and walks the full reduction
axis. Rows are independent, so there is no cross-program communication and no atomics — the
parallelism is simply `N / BLOCK_N` programs, and with N in the tens of thousands that is ample to
fill every SM.

**A wrong call, corrected by measurement.** The first version loaded a scale alongside every
weight. The reasoning was that those hits are served by L1/L2 rather than HBM, so they cost
latency and instructions but almost no *bandwidth* — and bandwidth is the budget. That is true and
it is beside the point: a kernel this far left of the ridge has nothing to do but wait, so latency
and instruction count are exactly what decide whether it reaches the roof. It measured 37% of the
memory roof — comfortably overhead-bound, and therefore not measuring the thing the topic exists
to measure. Pinning the tile to one quantisation group let the scale multiply leave the inner sum
entirely. See the kernel docstring for the algebra.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - import-time branch, exercised by which machine runs it
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    # Triton is CUDA-only. On the authoring Mac the module must still import so that ruff, pyright
    # and the CPU-side tests run locally; only the kernel itself is unavailable.
    HAS_TRITON = False

from topics.t08_gpu_architecture.pack import ZERO_OFFSET, PackedWeight, unpack_to_dense

# Autotune space. BLOCK_N trades occupancy against per-program work; `num_stages` controls how many
# loop iterations Triton keeps in flight, which is the actual latency-hiding lever in a loop this
# memory-bound — a stage count of 1 leaves every warp waiting on its own load.
#
# The reduction tile is *not* tunable: it is pinned to GROUP_SIZE so that one scale covers the whole
# tile (see the loop body). Trading that knob away is what let the scale multiply leave the inner
# sum, and it was worth far more than the tuning freedom it cost.
_BLOCK_N = (32, 64, 128)
_NUM_WARPS = (4, 8)
_NUM_STAGES = (2, 3, 4)


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=s)
            for bn in _BLOCK_N
            for w in _NUM_WARPS
            for s in _NUM_STAGES
        ],
        key=["n", "k"],
    )
    @triton.jit
    def _int4_gemv_kernel(  # noqa: PLR0913
        packed_ptr,
        scales_ptr,
        x_ptr,
        y_ptr,
        n,
        k,
        half,
        stride_pn,
        stride_pj,
        stride_sn,
        stride_sg,
        GROUP_SIZE: tl.constexpr,
        ZERO_POINT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """y[n] = sum_k dequant(packed[n, k]) * x[k], for a block of BLOCK_N rows.

        The reduction tile is exactly one quantisation group wide. That is the design decision the
        whole kernel turns on: within a group every weight in a row shares one scale, so the scale
        is a constant of the inner sum and factors straight out of it::

            sum_j (code * scale * x)  ==  scale * sum_j (code * x)

        The first form multiplies by the scale once per *element* and must fetch that scale
        alongside every element. The second multiplies once per *row per group* and fetches one
        scale per row per group — 128x fewer loads and one fewer multiply in the hot loop, for an
        identical result. The first version measured 37% of the memory roof; being memory-bound is
        the goal, and it was nowhere near it.
        """
        pid = tl.program_id(axis=0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n

        # fp32 accumulation regardless of the input dtype. T1's whole subject is that the error
        # which matters is the one that compounds; a 3584-term reduction in bf16 would add a
        # summation error on top of the quantisation error being measured, and confound the two.
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        offs_in_group = tl.arange(0, GROUP_SIZE)

        for j0 in range(0, half, GROUP_SIZE):
            offs_j = j0 + offs_in_group
            mask_j = offs_j < half
            tile_mask = mask_n[:, None] & mask_j[None, :]

            # One coalesced load carries two weights per byte. Consecutive lanes read consecutive
            # bytes of a row, which is the access pattern HBM is built to serve.
            byte = tl.load(
                packed_ptr + offs_n[:, None] * stride_pn + offs_j[None, :] * stride_pj,
                mask=tile_mask,
                other=0,
            )
            # `ZERO_POINT` arrives as a constexpr parameter rather than being read from the module.
            # A @triton.jit function cannot close over ordinary Python globals — it compiles an AST
            # in isolation and only constexpr arguments cross that boundary. Passing it in keeps
            # `pack.ZERO_OFFSET` the single definition while satisfying the compiler.
            code_lo = (byte & 0x0F).to(tl.float32) - ZERO_POINT
            code_hi = (byte >> 4).to(tl.float32) - ZERO_POINT

            # x is tiny (K floats) and every program reads all of it, so it lands in L2 after the
            # first program touches it and costs essentially no HBM traffic.
            x_lo = tl.load(x_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            x_hi = tl.load(x_ptr + offs_j + half, mask=mask_j, other=0.0).to(tl.float32)

            # Because the tile is one group wide, both group indices are loop-invariant *scalars*
            # rather than vectors, so each is a single load of BLOCK_N values. Column j sits in
            # group j // G and its partner column j + half exactly `half // G` groups further on;
            # pack-time validation guarantees no group straddles the two halves.
            scale_lo = tl.load(
                scales_ptr + offs_n * stride_sn + (j0 // GROUP_SIZE) * stride_sg,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            scale_hi = tl.load(
                scales_ptr + offs_n * stride_sn + ((j0 + half) // GROUP_SIZE) * stride_sg,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(code_lo * x_lo[None, :], axis=1) * scale_lo
            acc += tl.sum(code_hi * x_hi[None, :], axis=1) * scale_hi

        tl.store(y_ptr + offs_n, acc, mask=mask_n)


def int4_gemv(pw: PackedWeight, x: torch.Tensor) -> torch.Tensor:
    """Compute `W @ x` from the packed int4 weight, dequantising inside the kernel.

    Args:
        pw: the packed weight, on a CUDA device.
        x: `(K,)` activation vector. Any float dtype; read as fp32 inside the kernel.

    Returns:
        `(N,)` float32 result.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is unavailable — it is CUDA-only, so this kernel cannot run on the authoring "
            "machine. Use `int4_gemv_reference` for a CPU check, and see the topic README for the "
            "GPU setup."
        )
    if x.ndim != 1 or x.shape[0] != pw.k:
        raise ValueError(f"expected x of shape ({pw.k},), got {tuple(x.shape)}")
    if not pw.packed.is_cuda:
        raise ValueError("packed weight must live on a CUDA device")

    x = x.contiguous()
    y = torch.empty(pw.n, device=x.device, dtype=torch.float32)
    half = pw.k // 2

    def grid(meta: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(pw.n, meta["BLOCK_N"]),)

    _int4_gemv_kernel[grid](
        pw.packed,
        pw.scales,
        x,
        y,
        pw.n,
        pw.k,
        half,
        pw.packed.stride(0),
        pw.packed.stride(1),
        pw.scales.stride(0),
        pw.scales.stride(1),
        GROUP_SIZE=pw.group_size,
        ZERO_POINT=ZERO_OFFSET,
    )
    return y


def int4_gemv_reference(pw: PackedWeight, x: torch.Tensor) -> torch.Tensor:
    """The same computation, unfused, in plain torch — the correctness oracle.

    Materialises the dequantised matrix, which is precisely what the fused kernel avoids. Runs
    anywhere, including the Mac, so kernel correctness can be specified before there is a GPU to
    test it on.
    """
    dense = unpack_to_dense(pw)
    return dense @ x.to(torch.float32)
