"""Stages 3 and 4: host memory to HBM, and the copy in between that nobody counts.

    uv run python -m topics.t10_os_virtual_memory.h2d

A GPU's DMA engine can only read from **pinned** host memory — pages the OS has promised never to
relocate. Ordinary pageable memory therefore cannot be a transfer source at all, so CUDA quietly
copies it into a hidden pinned staging buffer first and DMAs from there.

That is the whole mechanism, and it predicts the result before any measurement: the pageable path
moves every byte **twice**, once through the CPU, so it should reach roughly half the bandwidth.
Measuring the staging copy separately (`memcpy_gbps`) turns that from a story into an accounting
identity, since the two halves have to add up:

    1 / pageable_gbps  ~=  1 / pinned_gbps  +  1 / memcpy_gbps

This module is also where the topic's most inference-relevant asymmetry appears. T7 measured
1,736.7 GB/s **inside** the GPU. PCIe Gen4 x16 delivers a small fraction of that. Every weight in
the model crosses the narrow link exactly once, at start-up, and then never again — which is why
cold start is a different engineering problem from steady-state decode, and why the fixes for one
do nothing for the other.
"""

from __future__ import annotations

import argparse

import torch

from arch_common.timing import synchronize, time_op

# Transfer sizes to sweep, in bytes. The small end is where per-transfer fixed cost dominates and
# the ratio between the two paths collapses; the large end is where the bandwidths separate. A
# single size would report whichever half of that story it happened to land in.
SWEEP_BYTES = (
    1 * 1024**2,
    4 * 1024**2,
    16 * 1024**2,
    64 * 1024**2,
    256 * 1024**2,
    1024 * 1024**2,
)

# The transfer size the headline ratio is quoted at. Large enough to be firmly bandwidth-bound,
# which is the regime a model load runs in.
HEADLINE_BYTES = 256 * 1024**2

DTYPE = torch.uint8


def _host_buffer(nbytes: int, *, pinned: bool) -> torch.Tensor:
    """A host tensor of `nbytes`, pinned or pageable.

    `torch.empty(..., pin_memory=True)` calls `cudaHostAlloc` underneath. Pinning is not free — it
    is a kernel-level page-table operation and it takes real time on a multi-gigabyte buffer, which
    is why serving stacks allocate one pinned staging buffer at start-up and reuse it rather than
    pinning each tensor as it arrives. That allocation cost is deliberately outside the timed
    region here: this measures the transfer, not the setup, and the note says so.
    """
    return torch.empty(nbytes, dtype=DTYPE, pin_memory=pinned)


def h2d_gbps(nbytes: int, device: torch.device, *, pinned: bool) -> float:
    """Achieved host-to-device bandwidth for one transfer size.

    `non_blocking=True` is used only on the pinned path, because it is only meaningful there: an
    async copy from pageable memory silently degrades to a synchronous one, since CUDA has to do
    the staging copy on the calling thread before it can start any DMA. Timing is via CUDA events,
    so a copy that did not actually finish cannot be reported as fast.
    """
    host = _host_buffer(nbytes, pinned=pinned)
    dev = torch.empty(nbytes, dtype=DTYPE, device=device)

    ms = time_op(lambda: dev.copy_(host, non_blocking=pinned), device)
    synchronize(device)
    return nbytes / (ms * 1e-3) / 1e9


def memcpy_gbps(nbytes: int) -> float:
    """Host-to-host copy bandwidth — the hidden staging copy, measured on its own.

    This is stage 3, and it is the stage no load-time benchmark reports because it happens inside
    the CUDA runtime. Measuring it separately is what lets the note claim the pageable path is
    "two copies" rather than merely asserting it: the serial combination of this and the pinned
    transfer has to reproduce the measured pageable number.
    """
    src = torch.empty(nbytes, dtype=DTYPE)
    dst = torch.empty(nbytes, dtype=DTYPE)
    cpu = torch.device("cpu")

    ms = time_op(lambda: dst.copy_(src), cpu)
    return nbytes / (ms * 1e-3) / 1e9


def series_gbps(*stages: float) -> float:
    """Combine stage bandwidths in series: reciprocals add, because the times do.

    Used to check the two-copy account of the pageable path against the pageable path as measured.
    If they disagree, the mechanism is not what this module says it is.
    """
    total = sum(1.0 / g for g in stages if g > 0)
    return 1.0 / total if total > 0 else 0.0


def sweep(device: torch.device) -> dict[int, dict[str, float]]:
    """Both paths across every transfer size, plus the staging copy at each."""
    out: dict[int, dict[str, float]] = {}
    for nbytes in SWEEP_BYTES:
        pinned = h2d_gbps(nbytes, device, pinned=True)
        pageable = h2d_gbps(nbytes, device, pinned=False)
        staging = memcpy_gbps(nbytes)
        out[nbytes] = {
            "pinned_gbps": pinned,
            "pageable_gbps": pageable,
            "memcpy_gbps": staging,
            "pinned_over_pageable": pinned / pageable if pageable else 0.0,
            # What the two-copy account predicts the pageable path should reach. Compared against
            # the measured pageable number in the note; agreement is the mechanism check.
            "predicted_pageable_gbps": series_gbps(pinned, staging),
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        raise SystemExit(
            "H2D transfer needs a GPU — there is no host-to-device hop on a CPU-only box. "
            "Run stages 1-2 with `loaders.py` here and this on the pod."
        )

    device = torch.device(args.device)
    print(f"T10 stages 3-4 — host to {torch.cuda.get_device_name(device)}\n")
    print(
        f"{'MiB':>7}  {'pinned':>9}  {'pageable':>9}  {'ratio':>6}  {'memcpy':>9}  {'predicted':>9}"
    )

    for nbytes, stats in sweep(device).items():
        print(
            f"{nbytes // 1024**2:>7}  {stats['pinned_gbps']:>7.2f}    "
            f"{stats['pageable_gbps']:>7.2f}  {stats['pinned_over_pageable']:>5.2f}x  "
            f"{stats['memcpy_gbps']:>7.2f}    {stats['predicted_pageable_gbps']:>7.2f}"
        )


if __name__ == "__main__":
    _main()
