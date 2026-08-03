"""Shared workload + timing primitives for the T5 parallelism experiments.

Everything here exists to make two guarantees.

**One worker means one core.** torch (like numpy) silently multi-threads its kernels, so an
un-pinned "1 worker" baseline would already be using every core on the machine and every speedup
measured against it would be a lie that *understates* the true scaling. `pin_single_thread()` is
called by every experiment before it times anything; `assert_pinned()` fails loudly if it did not
take.

**The work unit actually parallelises.** This is not free, and the first choice here was wrong.
The obvious unit of work — a float32 matmul — does *not* scale across threads on Apple Silicon:
numpy and torch both route BLAS matmul to the AMX co-processor, of which there is roughly one per
core cluster, so N threads doing matmuls contend for one shared unit and serialise. Measured on a
10-core M-series: 4 threads doing 4x the matmul work took 3.8x the wall time (1.06x of the ideal
4.0x) — i.e. no parallelism at all, for *both* numpy and torch.

The unit of work is therefore an **elementwise transcendental chain** (`sin -> add -> log -> sum`)
over a plain float32 buffer. It is real arithmetic, it runs in ATen's own vectorised loops rather
than BLAS, it releases the GIL, and it touches NEON/AVX rather than AMX — so it scales across
threads on Apple Silicon (6.2x on 8 threads) *and* on x86. Choosing a workload that can actually
demonstrate the effect being measured is a precondition for the measurement meaning anything.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import torch

# Elements per work unit. Calibrated so one unit lands in the ~1-2 ms range: long enough that
# timer noise and thread hand-off are negligible, short enough that a full sweep stays quick.
UNIT_ELEMS = 500_000


def pin_single_thread() -> None:
    """Force torch to use exactly one thread per process, so 1 worker == 1 core."""
    torch.set_num_threads(1)
    # Only settable before any parallel work starts; already-set is fine and not an error.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def assert_pinned() -> None:
    """Fail loudly rather than silently reporting speedups against a multi-core baseline."""
    n = torch.get_num_threads()
    if n != 1:
        raise RuntimeError(
            f"torch intra-op threads = {n}, expected 1. The 1-worker baseline would already be "
            "parallel and every speedup here would be understated. Call pin_single_thread() first."
        )


def make_buffer(seed: int = 0, elems: int = UNIT_ELEMS) -> torch.Tensor:
    """A fixed random float32 buffer. Each worker gets its own so this is not a cache experiment."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(elems, generator=g, dtype=torch.float32)


def work_unit(x: torch.Tensor) -> float:
    """One indivisible unit of work: an elementwise transcendental chain over `x`.

    Returns a scalar reduction so the computation cannot be optimised away, and does not mutate
    `x`, so a unit is exactly repeatable and every timed run does identical arithmetic.
    """
    return float(torch.sin(x).add_(1.0).log_().sum().item())


@dataclass(frozen=True)
class Timing:
    """A measured wall-clock duration and the checksum of the work that produced it."""

    seconds: float
    checksum: float


def time_units(x: torch.Tensor, n_units: int) -> Timing:
    """Run `n_units` work units back-to-back on the calling thread and time the lot."""
    t0 = time.perf_counter()
    checksum = 0.0
    for _ in range(n_units):
        checksum += work_unit(x)
    return Timing(time.perf_counter() - t0, checksum)


def calibrate_unit_seconds(x: torch.Tensor, reps: int = 30) -> float:
    """Median seconds for one work unit, after a warm-up.

    Median (not mean) because the first iterations pay allocator and cache warm-up costs, and a
    single slow outlier would otherwise inflate every efficiency figure downstream.
    """
    for _ in range(5):  # warm-up: allocator, cache, any lazy init
        work_unit(x)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        work_unit(x)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]
