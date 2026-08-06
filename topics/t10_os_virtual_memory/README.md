# T10 — OS & virtual memory: what a cold start actually costs

> **Status: pre-registered, not yet measured.** The bands below are filed and the harness is
> written, rehearsed and tested; the GPU session has not run. Everything here is a prediction, and
> the numbers that replace it may embarrass it. That is the point of writing it down first.

**Question:** T1–T9 all begin with the weights already in HBM. **How did they get there, and what
does that cost the first time?**

Framed as cold start, because that is where the question has consequences: a pod that scales to
zero, a spot instance that gets preempted, a multi-model server swapping checkpoints, an autoscaler
adding a replica under load. All four pay this path, and none of them pay it again.

---

## The path, and why it is four stages rather than one

```
NVMe  ──①──▶  page cache  ──②──▶  process memory  ──③──▶  pinned  ──④──▶  HBM
   storage read        copy or map          the staging copy      DMA over PCIe
                                            nobody counts
```

They are in **series** — every byte crosses every hop — so the slowest stage sets the floor and
optimising any other is wasted effort. "`from_pretrained` took 40 seconds" is not an actionable
fact. "36 of those 40 were stage 1 on a cold page cache" is, and it points at a different fix than
"31 of those 40 were stage 4" would.

Two things are worth stating before any of it is measured, because they are what make the result
worth having:

**Stage 4 is roughly 70× narrower than HBM.** T7 measured **1,736.7 GB/s** inside the GPU. PCIe
Gen4 ×16 tops out near 25 GB/s in practice. The link every other topic ignores is the slowest one in
the box, and the whole model crosses it exactly once.

**Stage 2 can be made to look free by not doing it.** `mmap` returns after installing a mapping,
having read nothing; the pages arrive later as faults, charged to whoever touches them first. That
is not a saving, it is a transfer of the bill to the first forward pass — and it is why apparent
load time is the wrong metric and why every path here is measured twice.

## Pre-registered bands

Four of the five follow from how the OS and the CUDA runtime work rather than from anything about
the box, so they can be committed to in advance.

| # | Band | Prediction | Why, mechanically |
|---|---|---|---|
| 1 | pinned / pageable H2D | **≥ 1.8×** | pageable memory cannot be a DMA source, so CUDA stages it through a hidden pinned buffer — every byte is copied **twice** |
| 2 | mmap / read, load call only | **≥ 3×** | `mmap` returns having read nothing at all |
| 3 | **mmap / read cold, faults charged back** | **≤ 1.2×** | the pages it did not read still have to arrive |
| 4 | cold / warm read | **≥ 3×** | a page-cache hit is a DRAM memcpy; a miss is an NVMe read |
| 5 | each stage vs a ceiling this box reached | **≥ 60%** | below that, the code is the story, not the hardware |

**Band 3 is why the experiment is worth running.** It is the only one that could plausibly go the
other way, and if lazy mapping genuinely wins after its deferred faults are charged back, then
demand paging is doing something smarter than deferring and the note has to explain what.

### Band 3 is scored cold, and the rehearsal is why

The first version scored it warm and it failed immediately — `mmap` was ahead by **2.18×** on a
warm 512 MB file even after every deferred fault was charged. That is not a counter-example to
deferral, it is a **different mechanism**: a mapping of already-cached pages never copies them at
all, while `read` copies them into a second buffer. Zero-copy, not laziness.

Cold is where the band's question actually lives, because there the bytes must come off the device
whichever loader asks for them and the only difference left is *when*. Both are now measured and
both go in the note, because the contrast is a better result than either alone:

- **warm** — `mmap` should win, and the win should be real (one copy instead of two)
- **cold** — `mmap` should converge, because the deferral bought nothing

Which, if it holds, has a directly useful serving conclusion: **memory-mapped loading pays off on
the second pod on a node, not the first.**

## What the arithmetic already says

Filed by `predict.py` against T6's measured 11.05 ms decode step:

| | |
|---|---|
| payload | **15.23 GB** (Qwen2.5-7B, bf16) |
| pages at 4 KB | **3,718,563** |
| pages at 2 MB | 7,263 — 512× fewer, which is the entire mechanism of huge pages |
| assumed cold start | 6.45 s, at plausible-but-unmeasured stage rates |
| **in this repo's currency** | **584 tokens not generated** |

That last line is the one that makes this an inference topic. A cold start is not a number of
seconds, it is a number of tokens the GPU did not emit while it waited for a disk. It is also the
scale-to-zero argument in one number: if a cold start costs 584 tokens and the traffic gap you are
scaling into is worth 200, keeping the pod warm is cheaper.

## Method notes

**The cold measurement needs root, and the gate refuses without it.** `/proc/sys/vm/drop_caches`
is the only way to make a cold run cold. Without it every load benchmark after the first measures a
memcpy out of DRAM and reports it as disk bandwidth — the single most common reason a published
model-load number does not reproduce on a fresh pod, because a fresh pod is by definition cold and
the benchmark never was.

**The weight file is incompressible on purpose.** A file of zeros can be stored as a hole by the
filesystem and then "read" at memory speed, which would make stage 1 look like DRAM and the whole
decomposition wrong. A test asserts the generated file's entropy.

**First touch is one byte per page, not a full traversal.** A full read would conflate fault
handling with memory traffic, and it is the fault handling that `mmap` deferred. Touching one byte
per page isolates it: the traffic is identical either way, so the difference between paths is the
faults alone.

**Faults are the corroboration a timer cannot give.** `getrusage` counts minor and major faults
around each phase, so the mechanism is observable independently of whether the timing is fair. It
also reveals **fault-around**: this laptop takes 512 faults for 2,048 fresh pages — the kernel maps
four neighbours per fault. "One fault per page" is the model, not the measurement, and the tests
band accordingly.

**Rehearsal is warm-only and never published.** `make t10-rehearse` runs the whole harness on a
small file with no GPU and no cache drops, stamping `session_id = "rehearsal"` so its numbers can
never be mistaken for results.

## Reproduce

Off the clock, on any machine:

```bash
uv sync
make t10-predict     # file the bands — laptop, no GPU, no root
make t10-rehearse    # the same harness, warm only, small file; numbers not published
uv run pytest topics/t10_os_virtual_memory tests
```

On a single-GPU Linux pod, **as root** — see [`scripts/t10_t11_session.sh`](../../scripts/t10_t11_session.sh),
which shares one pod and one hardware probe with T11:

```bash
bash s.sh gate      # 1 GPU, root, writable drop_caches, enough disk. ~10s.
bash s.sh setup     # clone + uv sync + make probe
bash s.sh t10       # cold/warm x read/mmap, then H2D, then the bands
```

**Put the weight file on the container's NVMe, not `/workspace`.** RunPod mounts `/workspace` as
network storage (MooseFS). For most topics that is merely slow; for this one it is fatal, because
stage 1 would be measuring a network filesystem and the note would describe MooseFS rather than a
disk.
