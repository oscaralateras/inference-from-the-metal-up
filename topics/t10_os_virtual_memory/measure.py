"""The T10 session: all four stages, the bands, and cold start in tokens.

    uv run python -m topics.t10_os_virtual_memory.measure --gib 8            # on the pod, as root
    uv run python -m topics.t10_os_virtual_memory.measure --no-cold --no-gpu # laptop rehearsal

Runs stages 1-2 (`loaders.py`) and stages 3-4 (`h2d.py`), scores the five bands
and then does the thing that makes this an inference topic rather than an I/O benchmark: it
extrapolates
the measured per-stage bandwidths to a real 15.2 GB model and reports the cold start in **tokens not
generated**, against T6's measured decode step.

The extrapolation is deliberate and deliberately labelled. Measuring a 15.2 GB load directly would
be more honest still, but it needs the model on the pod's disk and the whole point of the stage
decomposition is that the per-byte rates are what generalise. The note quotes both: what was
measured at the file size actually used, and what that implies at model scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from topics.t10_os_virtual_memory import h2d, loaders
from topics.t10_os_virtual_memory.pipeline import (
    DEFAULT_BYTES_PER_PARAM,
    DEFAULT_PARAMS,
    LoadPath,
    Stage,
    achieved_gbps,
    cold_start_seconds,
    model_bytes,
    stage_seconds,
    tokens_foregone,
)
from topics.t10_os_virtual_memory.predict import (
    MAX_MMAP_TRUE_SPEEDUP,
    MIN_COLD_SLOWDOWN,
    MIN_COLD_WARM_RATIO,
    MIN_MMAP_APPARENT_SPEEDUP,
    MIN_PINNED_SPEEDUP,
    MIN_SHARE_OF_STAGE_CEILING,
    t6_step,
)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "coldstart.csv"

DEFAULT_GIB = 8.0


def verdict(ok: bool) -> str:
    return "WITHIN" if ok else "OUTSIDE"


def run_loaders(path: Path, *, cold: bool) -> tuple[dict[tuple[str, str], LoadPath], str]:
    """Every (loader, cache-state) combination, each measured from a known cache state.

    Also returns which eviction mechanism was used, because the note has to say: a cold run whose
    method is unstated is indistinguishable from a warm run with a confident label.
    """
    out: dict[tuple[str, str], LoadPath] = {}
    states = ("cold", "warm") if cold else ("warm",)
    method = "none"

    for state in states:
        for name, loader in loaders.LOADERS.items():
            if state == "cold":
                method = loaders.go_cold(path)
            else:
                # Warm means warm, which has to be established rather than assumed: a "warm" run
                # straight after a cache drop is a cold run with a misleading label.
                loader(path)
            out[(name, state)] = loader(path)
    return out, method


def _print_loaders(results: dict[tuple[str, str], LoadPath]) -> None:
    print(
        f"  {'loader':<6} {'cache':<5} {'load s':>8} {'touch s':>8} {'total s':>8} "
        f"{'deferred':>9} {'faults/pg':>10} {'load GB/s':>10}"
    )
    for (name, state), r in sorted(results.items()):
        print(
            f"  {name:<6} {state:<5} {r.load_seconds:>8.3f} {r.first_touch_seconds:>8.3f} "
            f"{r.total_seconds:>8.3f} {r.deferred_share:>8.1%} {r.faults_per_page:>10.2f} "
            f"{achieved_gbps(r.nbytes, r.load_seconds):>10.2f}"
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=DEFAULT_GIB)
    parser.add_argument("--path", default="/tmp/t10_weights.bin")
    parser.add_argument("--no-cold", action="store_true", help="skip cache drops (no root / macOS)")
    parser.add_argument("--no-gpu", action="store_true", help="skip stages 3-4")
    args = parser.parse_args()

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    session = load_profile().session_id if use_gpu else "rehearsal"

    path = loaders.make_weight_file(Path(args.path), int(args.gib * 1024**3))
    nbytes = path.stat().st_size
    rows: list[dict[str, object]] = []

    print(f"T10 — cold start, {nbytes / 1024**3:.2f} GiB payload, session {session}\n")
    print("stages 1-2: disk -> page cache -> process memory")
    paths, method = run_loaders(path, cold=not args.no_cold)
    _print_loaders(paths)
    if not args.no_cold:
        print(f"  cache evicted via {method}")

        # The eviction is verified, not assumed — but NOT via major faults, which was the first
        # attempt and was wrong on this hardware. `read()` never takes page faults at all (the
        # kernel copies out of the page cache, so there is no mapping to fault on), and for `mmap`
        # sequential readahead fetches pages before they are touched, turning what would have been
        # blocking major faults into minor ones. The first pod recorded ONE major fault for 8 GiB
        # that had genuinely just been evicted.
        #
        # The check that does work is the same loader against itself: a cold run must be markedly
        # slower than a warm one. If it is not, the eviction no-oped and the cold column is a
        # second warm column with a misleading heading.
        cold_mmap, warm_mmap_ = paths[("mmap", "cold")], paths[("mmap", "warm")]
        slowdown = (
            cold_mmap.total_seconds / warm_mmap_.total_seconds if warm_mmap_.total_seconds else 0.0
        )
        if slowdown < MIN_COLD_SLOWDOWN:
            print(
                f"  WARNING: the cold mmap run was only {slowdown:.2f}x slower than the warm one. "
                "The eviction did not work and the cold column is not cold — do not publish it."
            )
        else:
            print(
                f"  verified: cold mmap was {slowdown:.1f}x slower than warm, so the pages really "
                "were evicted"
            )

    for (name, state), r in paths.items():
        for metric, value in (
            ("load_seconds", r.load_seconds),
            ("first_touch_seconds", r.first_touch_seconds),
            ("total_seconds", r.total_seconds),
            ("deferred_share", r.deferred_share),
            ("faults_per_page", r.faults_per_page),
            ("load_gbps", achieved_gbps(r.nbytes, r.load_seconds)),
        ):
            rows.append(
                {
                    "session_id": session,
                    "experiment": "load",
                    "variant": f"{name}_{state}",
                    "x": nbytes,
                    "metric": metric,
                    "value": value,
                }
            )

    transfers: dict[int, dict[str, float]] = {}
    if use_gpu:
        device = torch.device("cuda")
        print("\nstages 3-4: process memory -> pinned -> HBM")
        transfers = h2d.sweep(device)
        print(
            f"  {'MiB':>7} {'pinned':>9} {'pageable':>9} {'ratio':>7} {'memcpy':>9} {'2-copy':>9}"
        )
        for size, s in transfers.items():
            print(
                f"  {size // 1024**2:>7} {s['pinned_gbps']:>9.2f} {s['pageable_gbps']:>9.2f} "
                f"{s['pinned_over_pageable']:>6.2f}x {s['memcpy_gbps']:>9.2f} "
                f"{s['serial_bound_gbps']:>9.2f}"
            )
            for metric, value in s.items():
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "h2d",
                        "variant": "transfer",
                        "x": size,
                        "metric": metric,
                        "value": value,
                    }
                )
    else:
        print(
            "\nstages 3-4: SKIPPED (no GPU) — the note must not quote an H2D number from this run"
        )

    print("\n" + "=" * 78)
    print("pre-registered bands")
    print("=" * 78)

    warm_read, warm_mmap = paths[("read", "warm")], paths[("mmap", "warm")]

    if use_gpu:
        head = transfers[h2d.HEADLINE_BYTES]
        pinned_ratio = head["pinned_over_pageable"]
        print(
            f"(1) pinned / pageable at {h2d.HEADLINE_BYTES // 1024**2} MiB: "
            f"{pinned_ratio:.2f}x vs >= {MIN_PINNED_SPEEDUP}  "
            f"{verdict(pinned_ratio >= MIN_PINNED_SPEEDUP)}"
        )
        # The mechanism check, not a band. The serial two-copy bound is what the pageable path
        # would cost if the staging copy and the DMA took turns; measured sits above it by however
        # much the runtime manages to overlap them, and that overlap is why "two copies" does not
        # mean "half the bandwidth".
        overlap = (
            head["pageable_gbps"] / head["serial_bound_gbps"] if head["serial_bound_gbps"] else 0.0
        )
        print(
            f"    serial two-copy bound {head['serial_bound_gbps']:.2f} GB/s, "
            f"measured {head['pageable_gbps']:.2f} — {overlap:.2f}x above it, "
            "which is the staging copy being pipelined against the DMA rather than serialised"
        )

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    apparent = ratio(warm_read.load_seconds, warm_mmap.load_seconds)
    print(
        f"(2) mmap / read, load call only:      {apparent:>8.1f}x vs >= "
        f"{MIN_MMAP_APPARENT_SPEEDUP}  {verdict(apparent >= MIN_MMAP_APPARENT_SPEEDUP)}"
    )

    # Band 3 is scored COLD, and the rehearsal is why. On a warm cache `mmap` wins even after its
    # deferred faults are charged back — not because deferral is free, but because a mapping never
    # copies at all: the pages are already in the page cache and get mapped into the address space
    # rather than duplicated into a second buffer. That is a genuine zero-copy advantage and it has
    # nothing to do with the deferral this band is about.
    #
    # Cold is where the question actually lives. There, the bytes have to come off the device
    # whichever loader asks for them, so the only difference left is *when* — which is what the
    # band was written to test. Both are printed, because the contrast between them is a better
    # result than either alone.
    if not args.no_cold:
        cold_read, cold_mmap = paths[("read", "cold")], paths[("mmap", "cold")]
        true_ratio = ratio(cold_read.total_seconds, cold_mmap.total_seconds)
        print(
            f"(3) mmap / read COLD, faults charged: {true_ratio:>8.2f}x vs <= "
            f"{MAX_MMAP_TRUE_SPEEDUP}  {verdict(true_ratio <= MAX_MMAP_TRUE_SPEEDUP)}"
        )
        warm_ratio = ratio(warm_read.total_seconds, warm_mmap.total_seconds)
        print(f"    cold mmap deferred {cold_mmap.deferred_share:.1%} of its true cost")
        print(
            f"    warm, for contrast: {warm_ratio:.2f}x"
            " — mmap keeps a real lead there because it never copies at all"
        )

        cw = ratio(cold_read.load_seconds, warm_read.load_seconds)
        print(
            f"(4) cold / warm read:                 {cw:>8.2f}x vs >= "
            f"{MIN_COLD_WARM_RATIO}  {verdict(cw >= MIN_COLD_WARM_RATIO)}"
        )
    else:
        print("(3) SKIPPED — scored on the cold path, and cold runs were disabled")
        print(
            f"    warm only: {ratio(warm_read.total_seconds, warm_mmap.total_seconds):>8.2f}x, "
            f"mmap deferred {warm_mmap.deferred_share:.1%}"
        )
        print("(4) cold / warm read:                 SKIPPED — cold runs were disabled")

    # Band 5 and the inference payoff. Every rate below is measured; the model-scale numbers are
    # this run's rates applied to a payload this run did not move, and are labelled as such.
    if use_gpu and not args.no_cold:
        head = transfers[h2d.HEADLINE_BYTES]
        cold_read = paths[("read", "cold")]
        big = model_bytes(DEFAULT_PARAMS, DEFAULT_BYTES_PER_PARAM)
        measured_gbps = {
            "storage_read": achieved_gbps(cold_read.nbytes, cold_read.load_seconds),
            "host_memcpy": head["memcpy_gbps"],
            "h2d_pinned": head["pinned_gbps"],
        }

        # Ceilings, each of which has to be a rate this box actually reached doing a comparable
        # thing — otherwise "share of ceiling" is share of a number someone typed.
        #
        #   warm read vs memcpy    both are DRAM-to-DRAM copies, so the gap is the read path's own
        #                          overhead (syscalls, chunking) rather than anything about a disk
        #   pinned H2D vs its own  the asymptote across the size sweep, so the headline size can be
        #     asymptote            shown to be in the bandwidth-bound regime rather than assumed to
        #
        # Cold storage read is deliberately NOT scored. Nothing in this harness measures the
        # device's sequential ceiling independently, so any denominator would be the measurement
        # itself and the band would be guaranteed to pass. Band 4 is what covers that stage.
        warm_read_gbps = achieved_gbps(warm_read.nbytes, warm_read.load_seconds)
        pinned_asymptote = max(s["pinned_gbps"] for s in transfers.values())

        print("\n(5) each stage against a ceiling this box actually reached:")
        for label, got, ceiling in (
            ("read path", warm_read_gbps, head["memcpy_gbps"]),
            ("h2d pinned", head["pinned_gbps"], pinned_asymptote),
        ):
            share = got / ceiling if ceiling else 0.0
            print(
                f"    {label:<12} {got:>7.2f} GB/s of {ceiling:>7.2f} = {share:>6.1%}  "
                f"{verdict(share >= MIN_SHARE_OF_STAGE_CEILING)}"
            )
        print("    storage read  not scored — no independent device ceiling here; see band (4)")

        stages = [
            Stage(name, big, stage_seconds(big, gbps), gbps) for name, gbps in measured_gbps.items()
        ]

        step_ms, _ = t6_step()
        cold_s = cold_start_seconds(stages)
        print(f"\ncold start for a 15.2 GB model at this run's measured rates: {cold_s:.2f} s")
        print(
            f"against T6's {step_ms:.2f} ms decode step, that is "
            f"{tokens_foregone(cold_s, step_ms):,.0f} tokens not generated."
        )

        for st in stages:
            for metric, value in (
                ("stage_gbps", st.gbps),
                ("stage_seconds", st.seconds),
            ):
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "coldstart",
                        "variant": st.name,
                        "x": big,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, value in (
            ("cold_start_seconds", cold_s),
            ("tokens_foregone", tokens_foregone(cold_s, step_ms)),
        ):
            rows.append(
                {
                    "session_id": session,
                    "experiment": "coldstart",
                    "variant": "total",
                    "x": big,
                    "metric": metric,
                    "value": value,
                }
            )

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    _main()
