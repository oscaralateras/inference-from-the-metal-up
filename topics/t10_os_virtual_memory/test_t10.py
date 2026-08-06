"""Unit tests for T10. All run on any machine — no GPU, no root, no cache drops.

The parts that need a Linux pod (cold-cache reads, H2D transfers) cannot be unit-tested here, so
what is tested is everything they *decide with*: the arithmetic of the pipeline model, the
accounting identity that says the pageable path is two copies, and the loaders' behaviour on a
small file where both paths can actually run.

The loader tests are the important ones. They assert the *mechanism* — that a copying loader has
nothing left to pay and a lazily mapped one does — using fault counts rather than timings, so they
are meaningful on a laptop where the timings would be noise.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path

import pytest

from arch_common.results_io import read_rows, select
from topics.t10_os_virtual_memory.h2d import series_gbps
from topics.t10_os_virtual_memory.loaders import (
    LOADERS,
    counted_faults,
    drop_page_cache,
    evict_file,
    go_cold,
    load_via_mmap,
    load_via_read,
    make_weight_file,
)
from topics.t10_os_virtual_memory.pipeline import (
    DEFAULT_PARAMS,
    HUGE_PAGE_BYTES,
    PAGE_BYTES,
    LoadPath,
    Stage,
    achieved_gbps,
    cold_start_seconds,
    h2d_speedup,
    model_bytes,
    page_count,
    stage_seconds,
    tokens_foregone,
)

# Small enough that the tests stay fast, large enough to span thousands of pages so the fault
# counts are unambiguous rather than a handful either way.
TEST_FILE_BYTES = 8 * 1024 * 1024


@pytest.fixture(scope="module")
def weight_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_weight_file(tmp_path_factory.mktemp("t10") / "w.bin", TEST_FILE_BYTES)


# ---------------------------------------------------------------------------------------------
# the pipeline arithmetic
# ---------------------------------------------------------------------------------------------


def test_model_bytes_matches_the_shape_every_other_topic_uses() -> None:
    """15.2 GB of bf16 weights — the payload T6, T7, T8 and T9 all serve.

    Quoted in GB (decimal) throughout, because every bandwidth in this repo is GB/s decimal and
    dividing a GiB payload by a GB/s rate is a silent 7% error in a seconds figure. That is small
    enough to survive review and large enough to matter in a cold-start budget.
    """
    assert model_bytes(DEFAULT_PARAMS, 2) == DEFAULT_PARAMS * 2
    assert 15.0 < model_bytes() / 1e9 < 15.5


def test_huge_pages_reduce_the_page_count_by_the_size_ratio() -> None:
    """The entire mechanism by which huge pages help, stated as arithmetic."""
    nbytes = model_bytes()
    assert page_count(nbytes, PAGE_BYTES) / page_count(nbytes, HUGE_PAGE_BYTES) == pytest.approx(
        HUGE_PAGE_BYTES / PAGE_BYTES, rel=0.001
    )


def test_page_count_rounds_up() -> None:
    """A partial page is still a page, and still a fault."""
    assert page_count(1) == 1
    assert page_count(PAGE_BYTES) == 1
    assert page_count(PAGE_BYTES + 1) == 2


def test_page_count_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        page_count(-1)


def test_stage_seconds_and_achieved_gbps_are_inverses() -> None:
    nbytes, gbps = 4 * 1024**3, 12.5
    assert achieved_gbps(nbytes, stage_seconds(nbytes, gbps)) == pytest.approx(gbps)


def test_stage_seconds_rejects_a_zero_bandwidth() -> None:
    with pytest.raises(ValueError, match="positive"):
        stage_seconds(1024, 0.0)


def test_cold_start_is_the_sum_of_the_stages_not_the_max() -> None:
    """Series, not parallel — every byte crosses every hop, so the times add."""
    stages = [
        Stage("storage", 1_000_000_000, 1.0, 1.0),
        Stage("memcpy", 1_000_000_000, 0.5, 2.0),
        Stage("h2d", 1_000_000_000, 0.25, 4.0),
    ]
    assert cold_start_seconds(stages) == pytest.approx(1.75)


def test_share_of_ceiling_is_the_number_that_says_whether_to_optimise() -> None:
    stage = Stage("h2d", 25_000_000_000, 2.0, 25.0)
    assert stage.gbps == pytest.approx(12.5)
    assert stage.share_of_ceiling == pytest.approx(0.5)


def test_tokens_foregone_converts_a_cold_start_into_this_repos_currency() -> None:
    """40 s of cold start against an 11.05 ms step is a bit over 3,600 tokens."""
    assert tokens_foregone(40.0, 11.05) == pytest.approx(3620, rel=0.01)


def test_tokens_foregone_rejects_a_zero_step_time() -> None:
    with pytest.raises(ValueError, match="positive"):
        tokens_foregone(1.0, 0.0)


def test_h2d_speedup_is_zero_rather_than_infinite_on_a_missing_measurement() -> None:
    assert h2d_speedup(25.0, 0.0) == 0.0


# ---------------------------------------------------------------------------------------------
# the two-copy account of the pageable path
# ---------------------------------------------------------------------------------------------


def test_series_bandwidths_combine_by_reciprocal() -> None:
    """Two 20 GB/s copies in series take twice as long as one, so they run at 10."""
    assert series_gbps(20.0, 20.0) == pytest.approx(10.0)


def test_a_fast_staging_copy_costs_little() -> None:
    """The band is set below 2x because the staging copy runs at DRAM speed, not PCIe speed.

    This is that reasoning as a test: pairing a 25 GB/s DMA with a 200 GB/s memcpy loses far less
    than half, so predicting exactly 2.0 would have been wrong in a knowable direction.
    """
    assert series_gbps(25.0, 200.0) == pytest.approx(22.2, rel=0.01)


def test_series_ignores_a_stage_that_was_not_measured() -> None:
    assert series_gbps(10.0, 0.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------
# deferred cost, which is the topic's headline
# ---------------------------------------------------------------------------------------------


def test_deferred_share_is_zero_for_a_loader_that_did_the_work() -> None:
    path = LoadPath(
        "read", 1024, load_seconds=1.0, first_touch_seconds=0.0, minor_faults=0, major_faults=0
    )
    assert path.deferred_share == 0.0
    assert path.total_seconds == 1.0


def test_deferred_share_approaches_one_for_a_loader_that_only_promised() -> None:
    path = LoadPath(
        "mmap", 1024, load_seconds=0.001, first_touch_seconds=0.999, minor_faults=0, major_faults=0
    )
    assert path.deferred_share == pytest.approx(0.999, rel=0.01)


def test_faults_per_page_is_the_mechanism_evidence() -> None:
    """One fault per page is the signature of demand paging, and needs no timer to read."""
    nbytes = 100 * PAGE_BYTES
    path = LoadPath("mmap", nbytes, 0.0, 0.1, minor_faults=100, major_faults=0)
    assert path.faults_per_page == pytest.approx(1.0)


def test_counted_faults_sees_faults_from_a_fresh_anonymous_mapping() -> None:
    """The counter must actually count, or every fault-based claim is unfalsifiable.

    Uses an anonymous `mmap` rather than a `bytearray`, because a `bytearray` is not guaranteed to
    be fresh: an earlier version allocated 32 MB and passed in isolation but failed inside the full
    suite, where the allocator handed back pages another test had already faulted in. The test was
    measuring the allocator's freelist, not the fault counter.

    An anonymous mapping is unmapped memory by construction, so touching it must fault.
    Deterministic regardless of what ran before it.

    The bound is loose on purpose, and the reason is a real property worth knowing before reading
    the pod's numbers: kernels **fault around**, mapping several neighbouring pages per fault
    rather than one. This machine takes 512 faults for 2,048 pages — 4 pages per fault, a 16 KB
    granule. So "one fault per page" is the model, not the measurement, and a fault count well
    below the page count is the kernel being efficient rather than the counter being broken.
    """
    size = 8 * 1024 * 1024
    pages = size // PAGE_BYTES
    mapping = mmap.mmap(-1, size)
    try:
        with counted_faults() as faults:
            for offset in range(0, size, PAGE_BYTES):
                mapping[offset] = 1
    finally:
        mapping.close()

    assert 0.05 < faults["minor"] / pages < 2.0, (
        f"{faults['minor']} faults for {pages} fresh pages — the counter is not working"
    )


# ---------------------------------------------------------------------------------------------
# going cold, which is the one thing the measurement cannot fake
# ---------------------------------------------------------------------------------------------


def test_eviction_reports_whether_it_was_possible_rather_than_raising(weight_file: Path) -> None:
    """Both mechanisms return a bool, so `go_cold` can choose between them without exceptions.

    `posix_fadvise` is Linux-only and the global drop needs a privileged container, so on a Mac
    both return False and on a rented pod at least one returns True. Neither is an error on its
    own — only having no mechanism at all is.
    """
    assert isinstance(evict_file(weight_file), bool)
    assert isinstance(drop_page_cache(), bool)


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux-only")
def test_fadvise_is_the_preferred_mechanism_because_it_needs_no_privileges(
    weight_file: Path,
) -> None:
    """On Linux, `go_cold` must not reach for the root-only path when the unprivileged one works.

    This matters in practice rather than in principle: rented containers mount `/proc/sys`
    read-only, so a harness that insisted on the global drop simply could not measure a cold start
    on the hardware it was written to run on.
    """
    assert go_cold(weight_file) == "posix_fadvise"


@pytest.mark.skipif(hasattr(os, "posix_fadvise"), reason="tests the no-mechanism-available path")
def test_going_cold_refuses_rather_than_pretending_when_no_mechanism_exists() -> None:
    """The most important line in the module: a cold run that silently ran warm looks like data.

    On macOS neither mechanism exists, so the only correct behaviour is to refuse — and to say so
    loudly enough that nobody publishes the warm column under a cold heading.
    """
    with pytest.raises(RuntimeError, match="cannot evict the page cache"):
        go_cold(Path(__file__))


# ---------------------------------------------------------------------------------------------
# the loaders themselves, on a real file
# ---------------------------------------------------------------------------------------------


def test_both_loaders_report_the_whole_file(weight_file: Path) -> None:
    for loader in LOADERS.values():
        assert loader(weight_file).nbytes == TEST_FILE_BYTES


def test_the_weight_file_is_incompressible(weight_file: Path) -> None:
    """A file of zeros can be stored as a hole and 'read' at memory speed.

    That would make stage 1 look like DRAM and the entire cold-start decomposition wrong, so the
    generator's use of random bytes is a correctness property, not a stylistic one.
    """
    head = weight_file.read_bytes()[:4096]
    assert len(set(head)) > 200, "generated weights are not high-entropy — check the generator"


def test_the_copying_loader_has_nothing_left_to_defer(weight_file: Path) -> None:
    """The control that makes mmap's deferred cost interpretable.

    Without this, mmap's first-touch cost could be an artefact of the touching method rather than a
    property of lazy mapping.
    """
    path = load_via_read(weight_file)
    assert path.first_touch_seconds < path.load_seconds
    assert path.deferred_share < 0.5


def test_the_mapped_loader_takes_about_one_fault_per_page(weight_file: Path) -> None:
    """Demand paging, observed directly rather than inferred from a stopwatch.

    Bounded loosely on both sides: the kernel's fault-around may bring in several pages per fault
    (pushing this below 1), and Python's own allocations add a few of their own. What the test
    rejects is the two cases that would falsify the mechanism — no faults at all, or wildly more
    than one per page.
    """
    path = load_via_mmap(weight_file)
    assert path.minor_faults > 0, "a lazily mapped file took no faults — it was not lazy"
    assert 0.05 < path.faults_per_page < 2.0


def test_mmap_defers_far_more_than_read_does(weight_file: Path) -> None:
    """The comparison the topic rests on, made with fault counts rather than a stopwatch.

    An earlier version of this test asserted that `mmap`'s load call is faster than its own first
    touch, and it failed on an 8 MB warm file: the mapping's fixed setup (open, `mmap`, teardown)
    took 4.2 ms while touching 2,048 already-cached pages took 0.36 ms. That is not a
    counter-example to demand paging, it is a reminder that at small sizes on a warm cache the
    fixed costs dominate everything — which is precisely why the pod runs this at gigabyte scale
    with the cache dropped.

    Faults are the mechanism and they are scale-free, so this is what the laptop can assert.
    """
    mapped, copied = load_via_mmap(weight_file), load_via_read(weight_file)
    assert mapped.faults_per_page > 10 * copied.faults_per_page


# ---------------------------------------------------------------------------------------------
# The lab note, checked against the data it claims to describe
# ---------------------------------------------------------------------------------------------
#
# T5 learned this the hard way: its note and its CSV disagreed until a test compared them. A number
# in prose is a copy of a number in a file, and copies drift.


def _results() -> list[dict[str, str]]:
    from topics.t10_os_virtual_memory.measure import CSV_PATH

    if not CSV_PATH.exists():
        pytest.skip("T10 has not been run in this session")
    rows = read_rows(CSV_PATH)
    if {r["session_id"] for r in rows} == {"rehearsal"}:
        pytest.skip("T10 holds rehearsal numbers, not measured ones")
    return rows


def _load(rows: list[dict[str, str]], variant: str, metric: str) -> float:
    hits = [v for _, v in select(rows, "load", variant, metric)]
    assert len(hits) == 1, f"expected one load/{variant}/{metric} row, found {len(hits)}"
    return hits[0]


def _h2d(rows: list[dict[str, str]], metric: str, nbytes: int) -> float:
    hits = [v for x, v in select(rows, "h2d", "transfer", metric) if int(x) == nbytes]
    assert len(hits) == 1, f"expected one h2d/{metric} row at {nbytes}"
    return hits[0]


# (variant, load seconds, first touch seconds, total seconds) — the note's headline table.
QUOTED_LOADERS = [
    ("mmap_cold", 0.0001, 0.694, 0.694),
    ("mmap_warm", 0.0001, 0.120, 0.120),
    ("read_cold", 2.415, 0.208, 2.623),
    ("read_warm", 1.374, 0.199, 1.573),
]


@pytest.mark.parametrize(("variant", "load_s", "touch_s", "total_s"), QUOTED_LOADERS)
def test_lab_note_loader_table(variant: str, load_s: float, touch_s: float, total_s: float) -> None:
    rows = _results()
    measured_load = _load(rows, variant, "load_seconds")
    if variant.startswith("mmap"):
        # mmap's load call is sub-millisecond because it moves nothing, so a relative tolerance on
        # it would be comparing noise to noise. The claim the note makes is that it rounds to zero.
        assert measured_load == pytest.approx(load_s, abs=0.0002)
    else:
        assert measured_load == pytest.approx(load_s, rel=0.01)
    assert _load(rows, variant, "first_touch_seconds") == pytest.approx(touch_s, rel=0.01)
    assert _load(rows, variant, "total_seconds") == pytest.approx(total_s, rel=0.01)


def test_lab_note_band_verdicts_are_reported_correctly() -> None:
    """Four of six outside. A note quietly reporting more passes than it earned fails here."""
    rows = _results()
    from topics.t10_os_virtual_memory.predict import (
        MAX_MMAP_TRUE_SPEEDUP,
        MIN_COLD_WARM_RATIO,
        MIN_MMAP_APPARENT_SPEEDUP,
        MIN_PINNED_SPEEDUP,
        MIN_SHARE_OF_STAGE_CEILING,
    )

    head = 256 * 1024**2

    # (1) OUTSIDE — the runtime overlaps the staging copy with the DMA.
    assert _h2d(rows, "pinned_over_pageable", head) < MIN_PINNED_SPEEDUP
    # (2) WITHIN, enormously.
    apparent = _load(rows, "read_warm", "load_seconds") / _load(rows, "mmap_warm", "load_seconds")
    assert apparent >= MIN_MMAP_APPARENT_SPEEDUP
    # (3) OUTSIDE — mmap wins cold too, because it is zero-copy rather than merely lazy.
    cold = _load(rows, "read_cold", "total_seconds") / _load(rows, "mmap_cold", "total_seconds")
    assert cold > MAX_MMAP_TRUE_SPEEDUP
    # (4) OUTSIDE — the storage is fast, so warm is only modestly ahead.
    warm_ratio = _load(rows, "read_cold", "load_seconds") / _load(rows, "read_warm", "load_seconds")
    assert warm_ratio < MIN_COLD_WARM_RATIO
    # (5) read path OUTSIDE, pinned H2D WITHIN.
    read_share = _load(rows, "read_warm", "load_gbps") / _h2d(rows, "memcpy_gbps", head)
    assert read_share < MIN_SHARE_OF_STAGE_CEILING
    asymptote = max(v for _, v in select(rows, "h2d", "transfer", "pinned_gbps"))
    assert _h2d(rows, "pinned_gbps", head) / asymptote >= MIN_SHARE_OF_STAGE_CEILING


def test_lab_note_mmap_beats_read_cold_by_the_quoted_factor() -> None:
    """The note quotes 3.78x cold and 13.1x warm."""
    rows = _results()
    cold = _load(rows, "read_cold", "total_seconds") / _load(rows, "mmap_cold", "total_seconds")
    warm = _load(rows, "read_warm", "total_seconds") / _load(rows, "mmap_warm", "total_seconds")
    assert cold == pytest.approx(3.78, rel=0.01)
    assert warm == pytest.approx(13.1, rel=0.02)


def test_lab_note_the_two_stage_account_of_the_copying_loader() -> None:
    """The note's central mechanism claim, checked as arithmetic.

    mmap cold measures the storage on its own; warm read measures the copy on its own. Composed in
    series they must reproduce cold read — that is what says the copying loader is disk-then-copy
    and the mapped one is disk alone.
    """
    from topics.t10_os_virtual_memory.h2d import series_gbps

    rows = _results()
    nbytes = 8 * 1024**3
    storage = nbytes / _load(rows, "mmap_cold", "first_touch_seconds") / 1e9
    copy = _load(rows, "read_warm", "load_gbps")

    assert storage == pytest.approx(12.4, rel=0.02)
    assert copy == pytest.approx(6.25, rel=0.01)
    assert series_gbps(storage, copy) == pytest.approx(
        _load(rows, "read_cold", "load_gbps"), rel=0.20
    )


def test_lab_note_pageable_beats_the_serial_two_copy_bound() -> None:
    """The note quotes 2.15x above the serial bound, i.e. the staging copy is pipelined."""
    rows = _results()
    head = 256 * 1024**2
    assert _h2d(rows, "pageable_gbps", head) / _h2d(rows, "serial_bound_gbps", head) == (
        pytest.approx(2.15, rel=0.02)
    )


def test_lab_note_cold_start_in_tokens() -> None:
    """The note quotes 6.02 s and 545 tokens, and the stages must sum to the total."""
    rows = _results()
    total = [v for _, v in select(rows, "coldstart", "total", "cold_start_seconds")][0]
    tokens = [v for _, v in select(rows, "coldstart", "total", "tokens_foregone")][0]
    stages = sum(
        [v for _, v in select(rows, "coldstart", name, "stage_seconds")][0]
        for name in ("storage_read", "host_memcpy", "h2d_pinned")
    )

    assert total == pytest.approx(6.02, rel=0.01)
    assert tokens == pytest.approx(545, rel=0.01)
    assert stages == pytest.approx(total, rel=0.001), "the stage times must add up to the total"


def test_lab_note_the_pipelined_floor_is_the_slowest_stage() -> None:
    """The note reports 4.28-6.02 s rather than 6.02 alone.

    Band 1 measured the runtime overlapping a staging copy with the previous chunk's DMA, so a
    serial sum of the stages is an upper bound. The floor is whichever stage is slowest. A note
    that proves pipelining and then quotes only the serial sum contradicts itself.
    """
    rows = _results()
    stages = [
        [v for _, v in select(rows, "coldstart", name, "stage_seconds")][0]
        for name in ("storage_read", "host_memcpy", "h2d_pinned")
    ]

    assert max(stages) == pytest.approx(4.28, rel=0.01)
    assert max(stages) < sum(stages), "a floor equal to the sum would mean nothing overlaps"


def test_lab_note_the_real_checkpoint_was_measured_not_extrapolated() -> None:
    """The note quotes 5.63 s mapped and 14.96 s copied on Qwen2.5-7B's actual shards."""
    rows = _results()
    mapped = [v for _, v in select(rows, "checkpoint", "mapped", "load_seconds")][0]
    copied = [v for _, v in select(rows, "checkpoint", "copied", "load_seconds")][0]

    assert mapped == pytest.approx(5.63, rel=0.02)
    assert copied == pytest.approx(14.96, rel=0.02)
    assert copied / mapped == pytest.approx(2.66, rel=0.03), "the note quotes 2.66x"


def test_lab_note_the_real_load_is_slower_than_the_per_byte_model() -> None:
    """339 tensors moved one at a time cost more than the stage rates predict.

    This is the note's claim that the per-byte model is a floor. If a real mapped load ever came
    in at or below the staged mapped estimate, that paragraph would be wrong.
    """
    rows = _results()
    mapped = [v for _, v in select(rows, "checkpoint", "mapped", "load_seconds")][0]
    memcpy = [v for _, v in select(rows, "coldstart", "host_memcpy", "stage_seconds")][0]
    h2d = [v for _, v in select(rows, "coldstart", "h2d_pinned", "stage_seconds")][0]

    # The mapped stage-1 rate is the measured cold mmap total over the synthetic payload's size,
    # applied to the model's bytes — the same extrapolation the note's staged table performs.
    payload_bytes = [x for x, _ in select(rows, "load", "mmap_cold", "total_seconds")][0]
    model_total_bytes = [x for x, _ in select(rows, "coldstart", "total", "cold_start_seconds")][0]
    mapped_gbps = payload_bytes / _load(rows, "mmap_cold", "total_seconds") / 1e9

    storage_seconds = model_total_bytes / (mapped_gbps * 1e9)
    staged_serial = storage_seconds + memcpy + h2d

    assert mapped > staged_serial, "the real load must exceed the staged estimate, not undercut it"
    assert mapped / max(storage_seconds, memcpy, h2d) == pytest.approx(4.6, rel=0.1)


def test_lab_note_the_mapped_staged_range() -> None:
    """The note quotes 1.23-2.96 s = 111-268 tokens for a mapped stage 1.

    Same three stages as the copying estimate with storage running at the measured cold mmap rate
    instead of the copying one, floor and ceiling both.
    """
    rows = _results()
    payload = [x for x, _ in select(rows, "load", "mmap_cold", "total_seconds")][0]
    model = [x for x, _ in select(rows, "coldstart", "total", "cold_start_seconds")][0]
    step_s = [v for _, v in select(rows, "coldstart", "total", "cold_start_seconds")][0] / [
        v for _, v in select(rows, "coldstart", "total", "tokens_foregone")
    ][0]

    mapped_gbps = payload / _load(rows, "mmap_cold", "total_seconds") / 1e9
    storage = model / (mapped_gbps * 1e9)
    memcpy = [v for _, v in select(rows, "coldstart", "host_memcpy", "stage_seconds")][0]
    h2d = [v for _, v in select(rows, "coldstart", "h2d_pinned", "stage_seconds")][0]

    assert max(storage, memcpy, h2d) == pytest.approx(1.23, rel=0.02)
    assert storage + memcpy + h2d == pytest.approx(2.96, rel=0.02)
    assert max(storage, memcpy, h2d) / step_s == pytest.approx(111, rel=0.02)
    assert (storage + memcpy + h2d) / step_s == pytest.approx(268, rel=0.02)


def test_lab_note_the_storage_speeds_at_which_the_failed_bands_would_pass() -> None:
    """The note's portability table: band 3 needs storage <= 1.25 GB/s, band 4 <= 3.13.

    Both fall out of the two-stage model. `read` is capped by the copy however fast storage gets,
    so mmap/read = 1 + storage/copy and cold/warm = 1 + copy/storage. Solving each against its
    registered threshold is what turns "on a slower disk it would pass" into a number.
    """
    from topics.t10_os_virtual_memory.predict import MAX_MMAP_TRUE_SPEEDUP, MIN_COLD_WARM_RATIO

    rows = _results()
    copy_gbps = _load(rows, "read_warm", "load_gbps")
    payload = [x for x, _ in select(rows, "load", "mmap_cold", "total_seconds")][0]
    storage_gbps = payload / _load(rows, "mmap_cold", "total_seconds") / 1e9

    band3_max = (MAX_MMAP_TRUE_SPEEDUP - 1) * copy_gbps
    band4_max = copy_gbps / (MIN_COLD_WARM_RATIO - 1)

    assert band3_max == pytest.approx(1.25, rel=0.01)
    assert band4_max == pytest.approx(3.13, rel=0.01)
    assert storage_gbps == pytest.approx(12.38, rel=0.01)
    assert storage_gbps > band4_max > band3_max, "this box is past both thresholds"
