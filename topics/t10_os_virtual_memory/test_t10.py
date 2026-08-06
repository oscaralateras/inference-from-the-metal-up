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
from pathlib import Path

import pytest

from topics.t10_os_virtual_memory.h2d import series_gbps
from topics.t10_os_virtual_memory.loaders import (
    LOADERS,
    counted_faults,
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
