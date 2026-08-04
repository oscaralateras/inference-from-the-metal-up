# T7 — Roofline and arithmetic intensity

**Question:** Where do real inference matmuls sit on my GPU's roofline, and what moves them?

**Status:** built, tested and rehearsed end-to-end on CPU. Awaiting the GPU session — the numbers
below are the method, not the results.

---

## The model

Two ceilings, both **measured on this machine, never quoted from a spec sheet**:

- **peak compute** — the largest square bf16 GEMM the hardware sustains
- **peak bandwidth** — a large device-to-device copy, sized far beyond last-level cache

The roof at any arithmetic intensity is `min(bandwidth × AI, peak_compute)` — a diagonal on the
left, a ceiling on the right. Where they meet is the **ridge point**, `peak_compute / bandwidth`.
Left of it a kernel is memory-bound; right of it, compute-bound. It is a property of the silicon;
no amount of tuning moves it.

For a `(M,K) @ (K,N)` matmul:

```
FLOPs = 2 · M · N · K            one multiply and one add per multiply-accumulate
bytes = (M·K + K·N + M·N) · b    both operands read, the result written
AI    = FLOPs / bytes
```

## Why prefill and decode are different machines

| regime | M | arithmetic intensity | side of the ridge |
|---|---|---|---|
| prefill | prompt length | hundreds | compute-bound |
| decode | 1 | `2/b` → **1.0** in bf16 | memory-bound |

Same weights, same model, opposite bottlenecks — and it falls straight out of the arithmetic. That
single fact drives most of how inference systems are built.

## The batch walk

Batching B decode requests reads the weights **once** and serves all B. FLOPs scale with B; the
dominant byte term does not. So arithmetic intensity rises with batch size and the roofline point
walks rightward toward the ridge.

`batch_walk.png` is that walk, M from 1 to 256. It is the whole mechanical argument for continuous
batching in one figure.

## Pre-registered

| band | value | what falsification means |
|---|---|---|
| best prefill shape | ≥ **70%** of measured peak | the benchmark methodology is suspect, not the GPU |

Reported either way, and investigated in this note rather than buried if it fails.

## Reproduce

```bash
python -m arch_common.probe                    # once per session, shared with T6
python -m topics.t07_roofline.measure
python -m topics.t07_roofline.plot
```

Rehearsal on any laptop, seconds, no GPU:

```bash
python -m arch_common.probe --device cpu
python -m topics.t07_roofline.measure --device cpu --hidden 512 --intermediate 1024 \
    --prefill-tokens 128
python -m topics.t07_roofline.plot
```

On a CPU every shape lands *right* of the ridge — a CPU has little compute relative to its
bandwidth, so its ridge point is ~0.05 rather than ~170. The walk comes out flat. That is the
correct physics for that machine, and it is a useful check that the plot is reading real ceilings
rather than hardcoded GPU numbers.

## Figures

| figure | shows |
|---|---|
| `roofline.png` | prefill and decode projections placed under the measured roof |
| `batch_walk.png` | the decode point climbing toward the ridge, B = 1 → 256 |

## What is not measured

- **Attention's own QK^T and AV matmuls are excluded.** Their cost scales with sequence length
  rather than with the weights, so they belong to a separate analysis.
- **Analytic intensity is a lower bound on traffic.** Real kernels re-read tiles that miss in L2, so
  the true intensity is slightly worse and every point sits a little below where it is drawn.
- **One GPU, one precision, dense matmuls only.** No quantised or sparse kernels.
- **No weights are ever loaded.** Every measurement is a synthetic tensor of the right shape. This
  is deliberate: it keeps the topic fast, cheap and portable, and it is enforced by a test.

## Relationship to T6

T7 owns the **shape** domain: isolated kernels, FLOPs per byte, position against the ridge — no
clock on either axis and no model on disk. T6 owns the **time** domain: whole model, real KV cache,
tokens per second.

Both read their ceilings from the same `results/hardware.json`, so the two lab notes cannot end up
quoting different bandwidths for the same GPU. `tests/test_distinctness.py` fails CI if the metric
vocabularies overlap, if either topic strays into the other's domain, or if the two were measured
in different sessions.

The two batch sweeps are the one genuine near-collision, and they measure different things on
purpose: **T7 sweeps the matmul shape in isolation** — no model, no KV cache, pure kernel. **T6
sweeps the whole model with a real cache** and measures latency percentiles. T7 shows the kernel
reaching toward the ridge; T6 shows what the rest of the system does to that gain.
