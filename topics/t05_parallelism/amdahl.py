"""Experiment A — measure Amdahl's constant instead of plotting Amdahl's curve.

Plotting `S = 1/((1-p) + p/n)` proves nothing: it is a closed-form equation and the curve is
whatever you assert `p` to be. This experiment runs it backwards.

1. **Calibrate.** Build a workload with a *known* serial fraction injected by construction, run it
   across 1..N workers, measure the real speedup curve, then fit Amdahl to the measurement and ask
   whether the fit recovers the fraction that was put in. That is a falsifiable check on the
   method, not a restatement of the law.
2. **Apply.** Point the now-validated fit at T4's contention curves, where the serial fraction is
   *not* known, and report it as a measured quantity.

The fit is exact linear least squares, not a search. Amdahl rearranges to

    1/S = (1 - p) + p/n  =>  1/S - 1 = p * (1/n - 1)

so `1/S - 1` is linear in `1/n - 1` through the origin with slope exactly `p`. One division.

    uv run python topics/t05_parallelism/amdahl.py
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from results_io import append_rows
from workload import (
    Timing,
    assert_pinned,
    calibrate_unit_seconds,
    make_buffer,
    pin_single_thread,
    time_units,
)

# Total work units per run. Large enough that thread start-up (~50 us) is noise against the
# ~1.5 ms-per-unit workload, small enough that the whole sweep runs in a couple of minutes.
TOTAL_UNITS = 400

# Serial fractions injected by construction. 0.0 is the control: a perfectly parallel workload
# should recover p = 1.0, and any shortfall is the measurement floor (thread overhead, turbo
# behaviour, memory contention) rather than a property of the program.
INJECTED_SERIAL_FRACTIONS = (0.0, 0.05, 0.15, 0.30)

T4_CSV = Path(__file__).parent.parent / "t04_concurrency" / "results" / "concurrency.csv"


@dataclass(frozen=True)
class ScalingPoint:
    """One (workers, speedup) observation."""

    workers: int
    seconds: float
    speedup: float


def run_split_workload(n_workers: int, serial_units: int, parallel_units: int) -> Timing:
    """Run `serial_units` on one thread, then `parallel_units` split across `n_workers` threads.

    This is Amdahl's model made physical: an irreducibly serial section that no amount of
    hardware can shorten, followed by a section that divides cleanly.
    """
    serial = time_units(make_buffer(seed=0), serial_units)

    # Split the parallel section as evenly as possible; give the remainder to the first workers.
    base, extra = divmod(parallel_units, n_workers)
    shares = [base + (1 if i < extra else 0) for i in range(n_workers)]

    results: list[Timing | None] = [None] * n_workers

    def run(i: int) -> None:
        # Each thread builds its own buffer: sharing one would make this a cache-contention
        # experiment (that is T4's job), not a scaling experiment.
        results[i] = time_units(make_buffer(seed=2 + i), shares[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parallel_seconds = time.perf_counter() - t0

    checksum = serial.checksum + sum(r.checksum for r in results if r is not None)
    return Timing(serial.seconds + parallel_seconds, checksum)


def measure_scaling(serial_fraction: float, max_workers: int) -> list[ScalingPoint]:
    """Sweep worker count for one injected serial fraction and return the speedup curve."""
    serial_units = round(TOTAL_UNITS * serial_fraction)
    parallel_units = TOTAL_UNITS - serial_units

    baseline = run_split_workload(1, serial_units, parallel_units)
    points = [ScalingPoint(1, baseline.seconds, 1.0)]

    for n in range(2, max_workers + 1):
        t = run_split_workload(n, serial_units, parallel_units)
        points.append(ScalingPoint(n, t.seconds, baseline.seconds / t.seconds))
    return points


def fit_parallel_fraction(points: list[ScalingPoint]) -> tuple[float, float]:
    """Least-squares fit of Amdahl's `p` to a measured speedup curve. Returns (p, r_squared).

    Uses the exact linearisation `1/S - 1 = p * (1/n - 1)`, fitted through the origin. The n=1
    point carries no information (both sides are 0) and is skipped.

    `p` is deliberately **not** clamped to [0, 1]. A curve where adding workers makes things
    *slower* yields p < 0, which is Amdahl's way of saying the measurement lies outside what the
    model can express — the model assumes coordination is free, and that is exactly what fails.
    """
    xs = [1.0 / pt.workers - 1.0 for pt in points if pt.workers > 1]
    ys = [1.0 / pt.speedup - 1.0 for pt in points if pt.workers > 1]
    if not xs:
        raise ValueError("need at least one point with workers > 1 to fit")

    denom = sum(x * x for x in xs)
    p = sum(x * y for x, y in zip(xs, ys, strict=True)) / denom

    ss_res = sum((y - p * x) ** 2 for x, y in zip(xs, ys, strict=True))
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return p, r_squared


def load_t4_contention() -> dict[str, list[ScalingPoint]]:
    """Read T4's contention curves and convert throughput to speedup-relative-to-one-thread.

    Reusing T4's committed measurement rather than re-running it: the question here is what
    Amdahl's law has to say about a curve that was produced for a different purpose.
    """
    if not T4_CSV.exists():
        return {}

    by_variant: dict[str, dict[int, float]] = {}
    with T4_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["experiment"] != "contention" or row["metric"] != "mops_per_s":
                continue
            by_variant.setdefault(row["variant"], {})[int(row["threads"])] = float(row["value"])

    curves: dict[str, list[ScalingPoint]] = {}
    for variant, per_thread in by_variant.items():
        if 1 not in per_thread:
            continue
        base = per_thread[1]
        curves[variant] = [
            ScalingPoint(n, 0.0, per_thread[n] / base) for n in sorted(per_thread) if per_thread[n]
        ]
    return curves


def _main() -> None:
    pin_single_thread()
    assert_pinned()

    unit_s = calibrate_unit_seconds(make_buffer(seed=0))
    max_workers = 8
    print(f"unit of work: {unit_s * 1e3:.2f} ms   ({TOTAL_UNITS} units per run)")
    print(f"pinned to 1 torch thread; sweeping 1..{max_workers} workers\n")

    rows: list[dict[str, object]] = []

    print(f"{'injected p':>11} {'recovered p':>12} {'error':>8} {'R^2':>7}  speedups")
    print("-" * 78)
    for s in INJECTED_SERIAL_FRACTIONS:
        points = measure_scaling(s, max_workers)
        p_hat, r2 = fit_parallel_fraction(points)
        p_true = 1.0 - s
        curve = " ".join(f"{pt.speedup:.2f}" for pt in points)
        print(f"{p_true:>11.3f} {p_hat:>12.3f} {p_hat - p_true:>+8.3f} {r2:>7.4f}  {curve}")

        for pt in points:
            rows.append(
                {
                    "experiment": "amdahl_calibration",
                    "variant": f"injected_serial_{s:.2f}",
                    "workers": pt.workers,
                    "metric": "speedup",
                    "value": f"{pt.speedup:.6f}",
                }
            )
        rows.append(
            {
                "experiment": "amdahl_calibration",
                "variant": f"injected_serial_{s:.2f}",
                "workers": 0,
                "metric": "recovered_p",
                "value": f"{p_hat:.6f}",
            }
        )

    t4 = load_t4_contention()
    if t4:
        print(f"\nT4 contention curves fitted with the same estimator ({T4_CSV.name}):")
        print(f"{'variant':>12} {'recovered p':>12} {'R^2':>8}   note")
        print("-" * 78)
        for variant in ("mutex", "atomic", "sharded"):
            if variant not in t4:
                continue
            p_hat, r2 = fit_parallel_fraction(t4[variant])
            note = "outside Amdahl's domain (p<0)" if p_hat < 0 else ""
            print(f"{variant:>12} {p_hat:>12.3f} {r2:>8.4f}   {note}")
            rows.append(
                {
                    "experiment": "amdahl_t4",
                    "variant": variant,
                    "workers": 0,
                    "metric": "recovered_p",
                    "value": f"{p_hat:.6f}",
                }
            )
            for pt in t4[variant]:
                rows.append(
                    {
                        "experiment": "amdahl_t4",
                        "variant": variant,
                        "workers": pt.workers,
                        "metric": "speedup",
                        "value": f"{pt.speedup:.6f}",
                    }
                )

    append_rows(rows)


if __name__ == "__main__":
    _main()
