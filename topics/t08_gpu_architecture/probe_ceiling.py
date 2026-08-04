# pyright: reportPossiblyUnboundVariable=false, reportIndexIssue=false
#
# Same Triton import guard as kernel.py: `triton`/`tl` are bound inside a CUDA-only branch and every
# use is unreachable without it. `main()` raises with instructions when Triton is absent.
"""What can this access pattern reach with no arithmetic at all?

This probe exists because it is the measurement that broke a wrong debugging loop, and a claim that
load-bearing does not belong in a throwaway script.

The fused kernel measured 49% of the memory roof, and three rounds of optimisation aimed at its
arithmetic made it worse or did nothing. The question nobody had asked was whether the *access
pattern* could go faster at all. So: strip the kernel to loads and a trivial sum — no unpacking, no
scale, no `x` — and measure the same tiles, from the same rotating pool, at the same shapes.

It reaches ~93% of the roof. That single number relocated the entire problem: the memory path was
already near-saturated, so whatever was costing 44 points was not the loads and not the maths, and
attention moved to everything *around* the kernel. It was the harness — a fresh output allocation
inside the timed region, and single cold launches for a 40 microsecond kernel.

**Rotation is not optional here.** An earlier version of this probe used one tensor and reported
1,031 GB/s — 110% of roof, which is impossible for HBM and is the signature of an L2-resident
working set. The int4 weight is 35 MB against a 4090's 75.5 MB of L2 and an A100's 40 MB, so it
fits entirely in cache on both. The probe that produced the useful answer is the one that streams.

    python -m topics.t08_gpu_architecture.probe_ceiling
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from arch_common.timing import time_op
from topics.t07_roofline.shapes import DEFAULT_HIDDEN, DEFAULT_INTERMEDIATE
from topics.t08_gpu_architecture.kernel import HAS_TRITON
from topics.t08_gpu_architecture.measure import CSV_PATH, LAUNCHES_PER_TIMING, _pool_size

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _load_only_kernel(  # noqa: PLR0913
        packed_ptr,
        y_ptr,
        n,
        half,
        stride_pn,
        stride_pj,
        BLOCK_N: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """Stream the packed tile and sum the raw bytes. Deliberately meaningless arithmetic.

        The sum exists only so the loads cannot be eliminated as dead code — the same discipline
        the C benchmarks in T2-T4 use their sinks and checksums for. What is being measured is the
        journey the bytes take, not anything done to them on arrival.
        """
        pid = tl.program_id(axis=0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for j0 in range(0, half, BLOCK_J):
            offs_j = j0 + tl.arange(0, BLOCK_J)
            byte = tl.load(
                packed_ptr + offs_n[:, None] * stride_pn + offs_j[None, :] * stride_pj,
                mask=mask_n[:, None] & (offs_j < half)[None, :],
                other=0,
            )
            acc += tl.sum(byte.to(tl.float32), axis=1)

        tl.store(y_ptr + offs_n, acc, mask=mask_n)


# Swept rather than autotuned: the point is the achievable ceiling across configurations, and
# reporting the best of an explicit sweep is more legible than trusting a tuner in a diagnostic.
_BLOCK_N = (8, 16, 32, 64)
_BLOCK_J = (128, 256)


def measure_load_ceiling(n: int, k: int, device: torch.device, pool_size: int) -> dict[str, float]:
    """Best achievable GB/s for this access pattern, streaming, with no arithmetic."""
    half = k // 2
    pool = [
        torch.randint(0, 255, (n, half), dtype=torch.uint8, device=device) for _ in range(pool_size)
    ]
    out = torch.empty(n, device=device, dtype=torch.float32)
    moved = pool[0].numel()

    best = 0.0
    best_config = (0, 0)
    for block_n in _BLOCK_N:
        for block_j in _BLOCK_J:
            nxt = itertools.count()

            def launch(bn: int = block_n, bj: int = block_j, c: itertools.count = nxt) -> None:
                _load_only_kernel[(triton.cdiv(n, bn),)](
                    pool[next(c) % len(pool)],
                    out,
                    n,
                    half,
                    pool[0].stride(0),
                    pool[0].stride(1),
                    BLOCK_N=bn,
                    BLOCK_J=bj,
                    num_warps=8,
                    num_stages=3,
                )

            ms = time_op(launch, device, inner=LAUNCHES_PER_TIMING)
            gbps = moved / (ms * 1e-3) / 1e9
            if gbps > best:
                best, best_config = gbps, (block_n, block_j)

    return {"gbps": best, "block_n": float(best_config[0]), "block_j": float(best_config[1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    args = parser.parse_args()

    if not (HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("probe_ceiling needs CUDA and Triton — run it on the GPU pod")

    device = torch.device("cuda")
    profile = load_profile()
    n, k = args.intermediate, args.hidden

    packed_bytes = n * (k // 2)
    pool_size = _pool_size(device, packed_bytes)
    l2_mb = torch.cuda.get_device_properties(device).L2_cache_size / 1e6

    print(f"{profile.device_name}  |  measured roof {profile.peak_bandwidth_gbps:,.1f} GB/s")
    print(
        f"rotating {pool_size} weights "
        f"({pool_size * packed_bytes / 1e6:,.0f} MB against {l2_mb:,.0f} MB of L2)\n"
    )

    result = measure_load_ceiling(n, k, device, pool_size)
    share = result["gbps"] / profile.peak_bandwidth_gbps
    print(f"load-only ceiling   {result['gbps']:>8,.1f} GB/s   ({share:.1%} of roof)")
    print(f"  best config       BLOCK_N={result['block_n']:.0f} BLOCK_J={result['block_j']:.0f}")
    print(
        "\nThe access pattern is not the limit. Anything the full kernel loses against this figure "
        "is arithmetic or harness, not memory."
    )

    append_rows(
        Path(CSV_PATH),
        [
            {
                "session_id": profile.session_id,
                "experiment": "ceiling",
                "variant": "load_only",
                "x": 0,
                "metric": metric,
                "value": value,
            }
            for metric, value in ({"gbps": result["gbps"], "share_of_roof": share}).items()
        ],
    )


if __name__ == "__main__":
    main()
