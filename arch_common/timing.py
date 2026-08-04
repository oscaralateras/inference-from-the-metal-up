"""Timing that is actually correct on a GPU.

CUDA kernel launches are **asynchronous**: `some_gpu_op(x)` returns to Python almost immediately,
long before the GPU has finished the work. Wrapping `time.perf_counter()` around it therefore
measures *launch* time, not *execution* time — typically by a factor of 100x or more. This is the
single most common way a GPU benchmark silently produces meaningless numbers, so every GPU timing
in this repo goes through this module.

Two mechanisms, picked by device:

* **CUDA** — `torch.cuda.Event` pairs recorded on the stream. The GPU itself timestamps the work,
  so there is no host-side launch overhead in the measurement at all.
* **CPU** — `time.perf_counter_ns()`, which is already synchronous.

Both paths warm up first (CUDA context creation, cuBLAS autotuning and allocator growth all land
on the first call and are not what we are measuring) and report the **median** of N timed
repetitions, which is robust to the occasional scheduler or clock-throttle outlier in a way the
mean is not.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def synchronize(device: torch.device) -> None:
    """Block until all queued work on `device` has actually finished. No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_op(
    fn: Callable[[], object],
    device: torch.device,
    *,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> float:
    """Return the **median wall-clock milliseconds** of one `fn()` call on `device`.

    `fn` should perform the work under test and nothing else — allocation, `.item()` calls and
    host-device copies inside `fn` are all measured, because from the GPU's point of view they are
    real work that serialises against the kernel you care about.
    """
    if warmup < 0 or iters < 1:
        raise ValueError(f"need warmup >= 0 and iters >= 1, got warmup={warmup} iters={iters}")

    for _ in range(warmup):
        fn()
    synchronize(device)

    if device.type == "cuda":
        return _time_cuda(fn, iters)
    return _time_cpu(fn, iters)


def _time_cuda(fn: Callable[[], object], iters: int) -> float:
    """Time with CUDA events, which are timestamped by the GPU rather than by the host."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()

    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))


def _time_cpu(fn: Callable[[], object], iters: int) -> float:
    """Time with a monotonic host clock. CPU work is synchronous, so this needs no events."""
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)
