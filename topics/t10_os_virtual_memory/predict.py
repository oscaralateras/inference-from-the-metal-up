"""Phase 0 — commit to the bands before renting anything.

    uv run python -m topics.t10_os_virtual_memory.predict --write     # no GPU, no disk, no root

Four of the five bands below are **mechanistic**: they follow from how the OS and the CUDA runtime
work, not from anything about this particular box, so they can be committed to in advance and can
embarrass the run rather than merely describing it.

    1. pinned beats pageable          because the pageable path copies every byte twice
    2. mmap looks much faster         because it returns before reading anything
    3. mmap's lead does not survive   because the pages it did not read still have to arrive
    4. cold is much slower than warm  because a page cache hit is DRAM and a miss is NVMe
    5. each stage reaches most of its own measured ceiling, or the code is the problem

Band 3 is the one worth running the experiment for. It is the only one that could plausibly go the
other way — if lazy mapping genuinely wins after the deferred faults are charged back, then demand
paging is doing something smarter than deferring, and the note has to explain what.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arch_common.results_io import read_rows, scalar
from topics.t10_os_virtual_memory.pipeline import (
    DEFAULT_BYTES_PER_PARAM,
    DEFAULT_PARAMS,
    HUGE_PAGE_BYTES,
    PAGE_BYTES,
    model_bytes,
    page_count,
    tokens_foregone,
)

RESULTS_DIR = Path(__file__).parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# T6's measured decode step, read at runtime rather than restated. Cold start is quoted in tokens
# not generated, and that conversion is only meaningful against a step time from the same repo.
T6_CSV = Path(__file__).resolve().parent.parent / "t06_perf_reasoning" / "results" / "perf.csv"

# --------------------------------------------------------------------------------------------
# Pre-registered bands. Reported WITHIN/OUTSIDE either way — T8 failed two of three and T9 three
# of seven, and both were better notes for it.
# --------------------------------------------------------------------------------------------

# (1) The pageable path stages through a hidden pinned buffer, so it moves each byte twice. Two
# copies against one predicts ~2x; the band is set below that because the staging copy runs at
# DRAM speed while the DMA runs at PCIe speed, so the second copy is cheaper than the first.
MIN_PINNED_SPEEDUP = 1.8

# (2) `mmap` returns after installing a mapping, having read nothing at all. Against a loader that
# copies gigabytes, the apparent gap should be enormous — this band is set at 3x only because a
# band that cannot fail is not a band.
MIN_MMAP_APPARENT_SPEEDUP = 3.0

# (3) The headline, and scored on the COLD path specifically. Once the deferred faults are charged
# to the first pass that touches the pages, the two loaders should converge — because with the cache
# cold they do the same work in a different order.
#
# Warm is a different question with a different answer, and conflating them would have made this
# band meaningless. A rehearsal on a warm 512 MB file had `mmap` ahead by 2.18x even after the
# faults were charged, because a mapping of already-cached pages never copies them at all. That is
# a real zero-copy win and it is not what this band is about. The note reports both.
MAX_MMAP_TRUE_SPEEDUP = 1.2

# (4) A page-cache hit is a DRAM memcpy; a miss is an NVMe read. Cloud NVMe runs a few GB/s and
# single-threaded memcpy runs tens, so 3x is a conservative floor rather than a guess.
MIN_COLD_WARM_RATIO = 3.0

# (5) A stage far below its own measured ceiling is a code problem, not a hardware fact, and the
# note should not attribute it to the hardware. 60% is the line.
MIN_SHARE_OF_STAGE_CEILING = 0.60


@dataclass(frozen=True)
class Prediction:
    """Everything committed to before the run, with each term's provenance."""

    params: int
    bytes_per_param: int
    model_bytes: int
    pages_4k: int
    pages_2m: int
    t6_step_ms: float
    t6_tokens_per_sec: float
    # Cold start under a deliberately optimistic assumption, purely to size the experiment: if
    # every stage ran at a plausible ceiling and none of them overlapped. Labelled assumed
    # wherever it appears and recomputed from measurements in `measure.py`.
    assumed_stage_gbps: dict[str, float] = field(default_factory=dict)
    assumed_cold_start_s: float = 0.0
    assumed_tokens_foregone: float = 0.0
    bands: dict[str, float] = field(default_factory=dict)
    session_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


# Plausible ceilings for sizing only — a cloud NVMe, a single-threaded memcpy, and PCIe Gen4 x16.
# Every one of these is replaced by a measurement before anything is published.
ASSUMED_STAGE_GBPS = {
    "nvme_read": 3.0,
    "page_cache_read": 12.0,
    "host_memcpy": 20.0,
    "h2d_pinned": 25.0,
}


def t6_step() -> tuple[float, float]:
    """Read `(step_ms, tokens_per_sec)` out of T6's measured results.

    Raises rather than defaulting, for the same reason T9's does: cold start is quoted in tokens,
    and a plausible-looking 11 ms typed in here would be a number this repo did not measure.
    """
    if not T6_CSV.exists():
        raise FileNotFoundError(
            f"no T6 results at {T6_CSV} — T10 quotes cold start in tokens not generated, which "
            "needs T6's measured decode step. Run T6 first."
        )
    rows = read_rows(T6_CSV)
    step_ms = scalar(rows, "decomposition", "measured", "step_time_ms")
    return step_ms, 1000.0 / step_ms


def build_prediction(session_id: str = "") -> Prediction:
    """Assemble the pre-registered prediction. Pure arithmetic — no hardware touched."""
    step_ms, tokens_per_sec = t6_step()
    nbytes = model_bytes(DEFAULT_PARAMS, DEFAULT_BYTES_PER_PARAM)

    # Series, not parallel: every byte crosses every hop, and nothing here assumes overlap. A real
    # loader can pipeline stages against each other, which is one of the things the measurement
    # will show this arithmetic does not capture.
    cold_s = sum(
        nbytes / (ASSUMED_STAGE_GBPS[stage] * 1e9)
        for stage in ("nvme_read", "host_memcpy", "h2d_pinned")
    )

    return Prediction(
        params=DEFAULT_PARAMS,
        bytes_per_param=DEFAULT_BYTES_PER_PARAM,
        model_bytes=nbytes,
        pages_4k=page_count(nbytes, PAGE_BYTES),
        pages_2m=page_count(nbytes, HUGE_PAGE_BYTES),
        t6_step_ms=step_ms,
        t6_tokens_per_sec=tokens_per_sec,
        assumed_stage_gbps=dict(ASSUMED_STAGE_GBPS),
        assumed_cold_start_s=cold_s,
        assumed_tokens_foregone=tokens_foregone(cold_s, step_ms),
        bands={
            "min_pinned_speedup": MIN_PINNED_SPEEDUP,
            "min_mmap_apparent_speedup": MIN_MMAP_APPARENT_SPEEDUP,
            "max_mmap_true_speedup": MAX_MMAP_TRUE_SPEEDUP,
            "min_cold_warm_ratio": MIN_COLD_WARM_RATIO,
            "min_share_of_stage_ceiling": MIN_SHARE_OF_STAGE_CEILING,
        },
        session_id=session_id,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="register to results/predictions.json")
    args = parser.parse_args()

    p = build_prediction()

    print("T10 — pre-registered bands, filed before any hardware exists\n")
    # GB decimal, matching every bandwidth in this repo. Dividing a GiB payload by a GB/s rate is
    # a silent 7% error in a seconds figure — small enough to survive review, large enough to
    # matter in a cold-start budget.
    print(f"payload            {p.model_bytes / 1e9:>10.2f} GB    (Qwen2.5-7B, bf16)")
    print(f"pages at 4 KB      {p.pages_4k:>10,}")
    print(f"pages at 2 MB      {p.pages_2m:>10,}   ({p.pages_4k / p.pages_2m:.0f}x fewer)")
    print(f"T6 decode step     {p.t6_step_ms:>10.2f} ms   ({p.t6_tokens_per_sec:,.1f} tok/s)\n")

    print("assumed stage ceilings — for sizing only, all replaced by measurement:")
    for stage, gbps in p.assumed_stage_gbps.items():
        print(f"  {stage:<18} {gbps:>6.1f} GB/s")
    print(f"\n  => cold start   {p.assumed_cold_start_s:>10.2f} s")
    print(f"  => that is      {p.assumed_tokens_foregone:>10,.0f} tokens not generated\n")

    print("bands:")
    print(f"  (1) pinned / pageable H2D            >= {MIN_PINNED_SPEEDUP:.1f}x")
    print(f"  (2) mmap / read, load call only      >= {MIN_MMAP_APPARENT_SPEEDUP:.1f}x")
    print(f"  (3) mmap/read COLD, faults charged   <= {MAX_MMAP_TRUE_SPEEDUP:.1f}x   <- the one")
    print(f"  (4) cold / warm read                 >= {MIN_COLD_WARM_RATIO:.1f}x")
    print(f"  (5) each stage vs its own ceiling    >= {MIN_SHARE_OF_STAGE_CEILING:.0%}")

    if args.write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_PATH.write_text(p.to_json())
        print(f"\nregistered -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    _main()
