"""Stages 1 and 2: getting a weight file off a disk and into a process, four different ways.

    uv run python -m topics.t10_os_virtual_memory.loaders --gib 4

The four ways are `read` and `mmap`, each cold and warm, and the interesting axis is not which is
fastest but **which one is telling the truth about what it did**.

`read` copies: the call returns when every byte is in the process's buffer, so a stopwatch around
it measures the whole cost. `mmap` maps: the call returns after installing a mapping, having read
nothing, and the pages arrive one fault at a time as they are touched. Both eventually move the
same bytes over the same wire. Only one of them charges you at the till.

So every path here is measured twice — once for the load call, once for a pass that touches every
page afterwards — and reported as the sum. `resource.getrusage` supplies the corroboration: a
copying loader takes almost no faults against its buffer, a lazily mapped one takes one per page,
and that count is mechanism-level evidence that does not depend on a timer being fair.

**Going cold without root.** The textbook way to evict the page cache is to write to
`/proc/sys/vm/drop_caches`, which needs root *and* a container privileged enough to have `/proc/sys`
mounted writable — which rented pods generally are not. `posix_fadvise(POSIX_FADV_DONTNEED)` drops
one file's cached pages instead, needs no privileges at all, and is the better instrument anyway:
it evicts the weight file rather than the entire system's cache, so the measurement is not
perturbed by having just discarded the pages backing Python and torch. `go_cold` prefers it and
falls back to the global drop.
"""

from __future__ import annotations

import argparse
import mmap
import os
import resource
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from topics.t10_os_virtual_memory.pipeline import PAGE_BYTES, LoadPath, page_count

# Chunk size for the copying loader. Large enough that per-call syscall overhead is negligible
# against a multi-gigabyte file, small enough not to be a second allocation strategy in disguise.
READ_CHUNK_BYTES = 64 * 1024 * 1024

# What the synthetic weight file is filled with. Incompressible on purpose: a file of zeros can be
# stored as a hole by the filesystem and then "read" at memory speed, which would make stage 1 look
# like DRAM and the whole topic wrong. Random bytes cannot be elided by any layer of the stack.
GENERATOR_SEED = 10


def evict_file(path: Path) -> bool:
    """Evict just this file's pages from the page cache. **No root required.**

    `posix_fadvise(POSIX_FADV_DONTNEED)` asks the kernel to drop the cached pages backing one file
    descriptor. It is the better instrument for this experiment than the global cache drop, not
    merely the more available one:

    * **Targeted.** It evicts the weight file and nothing else, so the measurement is not perturbed
      by having just thrown away the page cache entries for Python, torch, and every shared library
      the process is about to touch.
    * **Unprivileged.** Rented containers usually mount `/proc/sys` read-only, so the global drop
      is unavailable however root you are. This works in an ordinary container.

    Clean pages only — the kernel will not discard dirty ones — hence the `sync` first. Returns
    whether the call was even possible; `posix_fadvise` is Linux-only, so macOS gets False and the
    caller falls back or refuses.
    """
    if not hasattr(os, "posix_fadvise"):
        return False

    os.sync()
    with path.open("rb") as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return True


def drop_page_cache() -> bool:
    """Evict the **entire** page cache. Linux and root, and usually unavailable in a container.

    Kept as the fallback rather than the primary. It is the blunter of the two instruments: it
    discards every cached page on the system, including ones this process is about to need for
    reasons unrelated to the measurement.
    """
    sysfs = Path("/proc/sys/vm/drop_caches")
    if not sysfs.exists():
        return False
    try:
        subprocess.run(["sync"], check=True)
        sysfs.write_text("3\n")
    except (PermissionError, OSError, subprocess.SubprocessError):
        return False
    return True


def go_cold(path: Path) -> str:
    """Make `path` genuinely uncached, by whichever mechanism this box allows.

    Returns the name of the method used, so the lab note can state it rather than implying a cold
    run happened by unspecified means.

    Raises if neither works. That refusal is deliberate and it is the most important line in this
    module: a "cold" measurement that silently ran warm is worse than no measurement, because it
    looks exactly like data. Every published model-load number that does not say how it dropped the
    cache is suspect for precisely this reason.
    """
    if evict_file(path):
        return "posix_fadvise"
    if drop_page_cache():
        return "drop_caches"
    raise RuntimeError(
        "cannot evict the page cache by either mechanism — posix_fadvise is unavailable (not "
        "Linux) and /proc/sys/vm/drop_caches is missing or not writable. Cold numbers cannot be "
        "measured here. Pass --no-cold to run the warm path only, and the note must then say the "
        "cold column is absent rather than quoting a warm number as a load time"
    )


def make_weight_file(path: Path, nbytes: int) -> Path:
    """Write an incompressible file of `nbytes`, unless one of the right size is already there.

    Synthetic rather than a real checkpoint on purpose, and for the same reason T7 and T8 use
    synthetic tensors: the cost of moving a file depends on its size and its layout, not on what
    the numbers mean. It also keeps the topic runnable without a 15 GB download on a metered box.
    """
    if path.exists() and path.stat().st_size == nbytes:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GENERATOR_SEED)
    written = 0
    with path.open("wb") as f:
        while written < nbytes:
            chunk = min(READ_CHUNK_BYTES, nbytes - written)
            f.write(rng.integers(0, 256, size=chunk, dtype=np.uint8).tobytes())
            written += chunk
        f.flush()
        os.fsync(f.fileno())
    return path


@contextmanager
def counted_faults() -> Iterator[dict[str, int]]:
    """Record the minor and major page faults taken inside the block.

    Minor: the page was already in the page cache and only needed mapping. Major: it had to be
    fetched from the device. The ratio is the direct read on whether a "cold" run was actually
    cold, independently of how long it took — which is exactly the corroboration a timing-only
    experiment lacks.
    """
    before = resource.getrusage(resource.RUSAGE_SELF)
    counts = {"minor": 0, "major": 0}
    yield counts
    after = resource.getrusage(resource.RUSAGE_SELF)
    counts["minor"] = after.ru_minflt - before.ru_minflt
    counts["major"] = after.ru_majflt - before.ru_majflt


def _touch_every_page(buf: memoryview | mmap.mmap, nbytes: int) -> int:
    """Read one byte from every page and return a checksum, forcing residency page by page.

    One byte per page rather than a full traversal, deliberately. A full read would conflate two
    costs — the fault handling and the memory traffic — and it is the fault handling that `mmap`
    deferred. Touching a byte per page isolates it: the traffic is identical either way, so the
    difference between paths is the faults alone.

    The checksum is returned and printed so the loop cannot be optimised away, the same defensive
    habit T2 needed in C.
    """
    total = 0
    for offset in range(0, nbytes, PAGE_BYTES):
        total += buf[offset]
    return total


def load_via_read(path: Path) -> LoadPath:
    """Copy the whole file into process memory, then confirm there is nothing left to pay.

    The first-touch pass is still run, and is expected to cost almost nothing — that near-zero is
    the control. Without it, `mmap`'s first-touch cost would look like an artefact of the touching
    method rather than a property of lazy mapping.
    """
    nbytes = path.stat().st_size
    buf = bytearray(nbytes)
    view = memoryview(buf)

    with counted_faults() as faults:
        start = time.perf_counter()
        with path.open("rb", buffering=0) as f:
            done = 0
            while done < nbytes:
                got = f.readinto(view[done : done + READ_CHUNK_BYTES])
                if not got:
                    break
                done += got
        load_s = time.perf_counter() - start

    touch_start = time.perf_counter()
    _touch_every_page(view, nbytes)
    touch_s = time.perf_counter() - touch_start

    return LoadPath(
        name="read",
        nbytes=nbytes,
        load_seconds=load_s,
        first_touch_seconds=touch_s,
        minor_faults=faults["minor"],
        major_faults=faults["major"],
    )


def load_via_mmap(path: Path) -> LoadPath:
    """Map the file and return immediately, then measure what that deferral actually cost.

    The fault counter is scoped to the touching pass rather than to the mapping call, because that
    is where the faults happen — and a reader who expects the load call to be where the work is
    should see the count land somewhere else.
    """
    nbytes = path.stat().st_size

    start = time.perf_counter()
    with path.open("rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    load_s = time.perf_counter() - start

    try:
        with counted_faults() as faults:
            touch_start = time.perf_counter()
            _touch_every_page(mapping, nbytes)
            touch_s = time.perf_counter() - touch_start
    finally:
        mapping.close()

    return LoadPath(
        name="mmap",
        nbytes=nbytes,
        load_seconds=load_s,
        first_touch_seconds=touch_s,
        minor_faults=faults["minor"],
        major_faults=faults["major"],
    )


LOADERS = {"read": load_via_read, "mmap": load_via_mmap}


def _report(path: LoadPath, cache: str) -> str:
    return (
        f"  {path.name:<5} {cache:<5}  load {path.load_seconds:>7.3f}s  "
        f"first touch {path.first_touch_seconds:>7.3f}s  total {path.total_seconds:>7.3f}s  "
        f"deferred {path.deferred_share:>5.1%}  faults/page {path.faults_per_page:>5.2f}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=4.0, help="size of the synthetic weight file")
    parser.add_argument("--path", default="/tmp/t10_weights.bin")
    parser.add_argument(
        "--no-cold",
        action="store_true",
        help="skip the cold-cache runs. Use on a Mac or without root; the note must then say the "
        "cold column is missing rather than quoting the warm one as a load time",
    )
    args = parser.parse_args()

    path = make_weight_file(Path(args.path), int(args.gib * 1024**3))
    nbytes = path.stat().st_size
    print(f"T10 stages 1-2 — {nbytes / 1024**3:.2f} GiB, {page_count(nbytes):,} pages\n")

    for cache in ("cold", "warm"):
        if cache == "cold" and args.no_cold:
            continue
        for loader in LOADERS.values():
            if cache == "cold":
                go_cold(path)
            else:
                loader(path)  # warm the cache with a discarded run
            print(_report(loader(path), cache))


if __name__ == "__main__":
    _main()
