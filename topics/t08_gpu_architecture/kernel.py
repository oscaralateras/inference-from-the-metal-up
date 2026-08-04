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
weight, on the reasoning that those hits are served by L1/L2 rather than HBM and so cost latency
and instructions but almost no *bandwidth* — and bandwidth is the budget. Both halves are true and
the conclusion does not follow: a kernel this far left of the ridge has nothing to do but wait, so
instruction count is exactly what decides whether it reaches the roof. Held against the finished
harness, per-element scales reach 52% of the memory roof and factored scales 80% — 1.95x against
3.00x on the same silicon, same shapes, bit-identical output. The algebra is in the kernel
docstring.
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
# GROUPS sets the reduction tile to a whole number of quantisation groups. It must stay a whole
# number so one scale still covers each group and the scale multiply stays out of the inner sum, but
# within that constraint the tile is free — and it matters: `probe_ceiling` finds this access
# pattern peaks at BLOCK_J=256 on an A100, where a tile pinned to a single 128-wide group leaves
# throughput on the table. The wider the memory system, the more bytes a program must have in flight
# to saturate it, so the best tile is a property of the GPU rather than of the algorithm.
_BLOCK_N = (8, 16, 32, 64)
_GROUPS = (1, 2, 4)
_NUM_WARPS = (2, 4, 8)
_NUM_STAGES = (2, 3, 4)


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": bn, "GROUPS": g}, num_warps=w, num_stages=s)
            for bn in _BLOCK_N
            for g in _GROUPS
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
        GROUPS: tl.constexpr,
    ):
        """y[n] = sum_k dequant(packed[n, k]) * x[k], for a block of BLOCK_N rows.

        The reduction tile is a whole number of quantisation groups wide, and that alignment is the
        design decision the kernel turns on. Within a group every weight in a row shares one scale,
        so the scale is a constant of the inner sum and factors straight out of it::

            sum_j (code * scale * x)  ==  scale * sum_j (code * x)

        The first form multiplies by the scale once per *element* and must fetch that scale
        alongside every element. The second reduces each group first and applies one scale to the
        group total — one multiply per row per group instead of one per element, and one scale load
        instead of one beside every weight, for a bit-identical result.

        `GROUPS` then widens the tile without giving any of that back: the product reshapes to
        (rows, groups, group_size), reduces the innermost axis, and meets a (rows, groups) tile of
        scales. More bytes in flight per program, unchanged scale traffic.
        """
        pid = tl.program_id(axis=0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n

        # fp32 accumulation regardless of the input dtype. T1's whole subject is that the error
        # which matters is the one that compounds; a 3584-term reduction in bf16 would add a
        # summation error on top of the quantisation error being measured, and confound the two.
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        block_j: tl.constexpr = GROUPS * GROUP_SIZE
        offs_in_tile = tl.arange(0, block_j)
        offs_in_groups = tl.arange(0, GROUPS)

        for j0 in range(0, half, block_j):
            offs_j = j0 + offs_in_tile
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

            # Per-group partial sums: (rows, tile) -> (rows, groups, group_size) -> (rows, groups).
            part_lo = tl.sum(
                tl.reshape(code_lo * x_lo[None, :], (BLOCK_N, GROUPS, GROUP_SIZE)), axis=2
            )
            part_hi = tl.sum(
                tl.reshape(code_hi * x_hi[None, :], (BLOCK_N, GROUPS, GROUP_SIZE)), axis=2
            )

            # Column j sits in group j // G; its partner column j + half exactly half // G groups
            # further on. Pack-time validation guarantees no group straddles the two halves, so
            # these indices are exact rather than approximately right.
            groups_lo = j0 // GROUP_SIZE + offs_in_groups
            groups_hi = (j0 + half) // GROUP_SIZE + offs_in_groups
            scale_lo = tl.load(
                scales_ptr + offs_n[:, None] * stride_sn + groups_lo[None, :] * stride_sg,
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)
            scale_hi = tl.load(
                scales_ptr + offs_n[:, None] * stride_sn + groups_hi[None, :] * stride_sg,
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(part_lo * scale_lo, axis=1)
            acc += tl.sum(part_hi * scale_hi, axis=1)

        tl.store(y_ptr + offs_n, acc, mask=mask_n)


def int4_gemv(pw: PackedWeight, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute `W @ x` from the packed int4 weight, dequantising inside the kernel.

    Args:
        pw: the packed weight, on a CUDA device.
        x: `(K,)` activation vector. Any float dtype; read as fp32 inside the kernel.
        out: optional pre-allocated `(N,)` fp32 output. **Pass this when benchmarking.**
            Allocating a fresh output on every call puts the PyTorch caching allocator inside
            the timed region, and for a kernel this short that is not a rounding error: it cost
            ~40% of measured throughput here, which read as a slow kernel rather than as a slow
            harness. A real serving loop reuses its output buffers, so passing `out` is also the
            more faithful setup, not a convenience.

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
    y = torch.empty(pw.n, device=x.device, dtype=torch.float32) if out is None else out
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
