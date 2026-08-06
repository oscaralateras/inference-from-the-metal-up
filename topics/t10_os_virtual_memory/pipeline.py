"""The cost model for getting weights from a disk into HBM. Pure arithmetic, no hardware.

Every topic before this one begins with the weights already in HBM. T10 is the only one about how
they got there, and the framing that makes it an inference topic rather than an I/O benchmark is
**cold start**: the time from "this pod has never seen the model" to "first token out".

That path is four stages, each with its own ceiling and its own fix:

    NVMe  --(1)-->  page cache  --(2)-->  process memory  --(3)-->  pinned  --(4)-->  HBM
          storage read          copy or map           the staging copy      DMA over PCIe
                                                      nobody counts

The stages are wired in series, so the slowest one sets the floor and optimising any other is
wasted effort. That is the whole reason to decompose rather than to time the lump: `from_pretrained`
took 40 seconds is not an actionable fact, and "36 of those 40 were stage 1 on a cold page cache"
is.

Two properties of this path are worth stating before any of it is measured, because they are what
make the result surprising:

* **Stage 4 is ~70x narrower than HBM.** T7 measured 1,736.7 GB/s inside the GPU; PCIe Gen4 x16
  tops out near 25 GB/s in practice. The link every topic ignores is the slowest one in the box.
* **Stage 2 can be made to look free by not doing it.** `mmap` returns before it has read anything;
  the pages arrive later, as faults, charged to whoever touches them first. That is not a saving,
  it is a transfer of the bill to the first forward pass — and it is why apparent load time is the
  wrong metric. `LoadPath` below exists to charge it back.
"""

from __future__ import annotations

from dataclasses import dataclass

# Qwen2.5-7B in bf16 — the model T6, T7, T8 and T9 all use, so the cold-start numbers this topic
# produces compose with their steady-state ones rather than describing a different system.
DEFAULT_PARAMS = 7_615_616_512
DEFAULT_BYTES_PER_PARAM = 2

# x86-64's default page size. Every fault, every TLB entry and every `mmap` residency decision is
# quantised to this, so it is the unit the fault counts below are in.
PAGE_BYTES = 4096

# The huge-page size the same hardware supports. 512x fewer pages for the same bytes, which is the
# entire mechanism by which huge pages help: fewer faults to take and fewer TLB entries to miss.
HUGE_PAGE_BYTES = 2 * 1024 * 1024


def model_bytes(
    params: int = DEFAULT_PARAMS, bytes_per_param: int = DEFAULT_BYTES_PER_PARAM
) -> int:
    """Bytes of weights to move. The payload every stage below is quoted against."""
    return params * bytes_per_param


def page_count(nbytes: int, page_bytes: int = PAGE_BYTES) -> int:
    """How many pages `nbytes` spans — the faults a fully lazy mapping must eventually take.

    7,615,616,512 parameters in bf16 is 15.2 GB, which is 3.7 million 4 KB pages. At a few
    microseconds of kernel time each, the fault handling alone is seconds — before a single byte
    has been read from anywhere. This is the arithmetic that makes huge pages interesting.
    """
    if nbytes < 0:
        raise ValueError(f"bytes must be non-negative, got {nbytes}")
    return -(-nbytes // page_bytes)


def stage_seconds(nbytes: int, gbps: float) -> float:
    """Time for one stage to move `nbytes` at `gbps`. The series model's only primitive."""
    if gbps <= 0:
        raise ValueError(f"stage bandwidth must be positive, got {gbps}")
    return nbytes / (gbps * 1e9)


def achieved_gbps(nbytes: int, seconds: float) -> float:
    """Bandwidth actually reached. Returns 0 for a zero-length measurement rather than dividing."""
    return nbytes / seconds / 1e9 if seconds > 0 else 0.0


@dataclass(frozen=True)
class Stage:
    """One hop of the load path, scored against its own measured ceiling.

    The ceiling is measured on the same box, never quoted: a cloud NVMe is not its datasheet and a
    PCIe slot is not its specification. `share_of_ceiling` is the only number here that says
    whether a stage is worth optimising — a stage at 95% of its ceiling needs different hardware,
    a stage at 30% needs different code.
    """

    name: str
    nbytes: int
    seconds: float
    ceiling_gbps: float

    @property
    def gbps(self) -> float:
        return achieved_gbps(self.nbytes, self.seconds)

    @property
    def share_of_ceiling(self) -> float:
        return self.gbps / self.ceiling_gbps if self.ceiling_gbps > 0 else 0.0


@dataclass(frozen=True)
class LoadPath:
    """One way of getting a weight file into process memory, with its deferred cost charged back.

    `load_seconds` is what a stopwatch around the load call reports, and it is the number every
    loader benchmark publishes. `first_touch_seconds` is what it costs to then touch every page
    once — nothing for a loader that already copied the bytes, and everything for one that only
    promised to.

    `total_seconds` is the honest comparison, and the gap between the two is the topic's headline.
    """

    name: str
    nbytes: int
    load_seconds: float
    first_touch_seconds: float
    minor_faults: int
    major_faults: int

    @property
    def total_seconds(self) -> float:
        return self.load_seconds + self.first_touch_seconds

    @property
    def deferred_share(self) -> float:
        """Fraction of the true cost that the load call did not report.

        ~0 means the loader did the work it claimed to. Approaching 1 means the stopwatch was
        measuring a promise.
        """
        return self.first_touch_seconds / self.total_seconds if self.total_seconds > 0 else 0.0

    @property
    def faults_per_page(self) -> float:
        """Faults taken per page of the file — the direct evidence of who deferred what.

        A copying loader reads through the page cache and takes essentially none of these against
        the mapping. A lazily mapped one takes one per page, which is why this number separates
        the two mechanisms without needing a timer at all.
        """
        pages = page_count(self.nbytes)
        return (self.minor_faults + self.major_faults) / pages if pages else 0.0


def cold_start_seconds(stages: list[Stage]) -> float:
    """Total time through a series of stages. Series, not max — every byte crosses every hop."""
    return sum(stage.seconds for stage in stages)


def tokens_foregone(cold_seconds: float, step_ms: float) -> float:
    """Cold start expressed in the only currency this repo trades in: tokens not generated.

    This is the line that makes T10 an inference topic. A cold start is not a number of seconds,
    it is a number of tokens the GPU did not emit while it was waiting for a disk — and against
    T6's measured 11.05 ms step, seconds are thousands of tokens.

    It is also the argument for and against scale-to-zero in one number: if a cold start costs
    2,000 tokens and your traffic gap is 200 tokens' worth of idle, keeping the pod warm is cheaper.
    """
    if step_ms <= 0:
        raise ValueError(f"step time must be positive, got {step_ms}")
    return cold_seconds / (step_ms * 1e-3)


def h2d_speedup(pinned_gbps: float, pageable_gbps: float) -> float:
    """Pinned over pageable H2D bandwidth.

    Pageable memory cannot be the source of a DMA — the OS is free to relocate it mid-transfer —
    so CUDA copies it into a hidden pinned staging buffer first and DMAs from there. The pageable
    path therefore moves the bytes **twice**, once through the CPU. That predicts a ratio near 2x
    from the copy count alone, before any measurement, which is what makes it a band rather than
    a hope.
    """
    return pinned_gbps / pageable_gbps if pageable_gbps > 0 else 0.0
