# T7 — Roofline and arithmetic intensity

**Question:** Where do real inference matmuls sit on my GPU's roofline, and what moves them?

**Setup:** NVIDIA A100-SXM4-80GB, bfloat16, SM clock 1230 MHz / memory 1593 MHz, torch 2.8.0+cu128.
Qwen2.5-7B's dimensions (hidden 3584, intermediate 18944); **no weights are ever loaded** — every
measurement is a synthetic tensor of the right shape. Session `6c79f20d6c13`, shared with T6.

---

## Result

![roofline](results/roofline.png)

**Decode runs at 0.5% of the GPU's compute — and that is near-optimal, not wasteful.**

The decode MLP projection reaches 1.42 TFLOP/s against a 263.6 TFLOP/s ceiling. But the ceiling it
is actually under is the *memory* roof, which at 1.0 FLOP/byte is `1734.8 GB/s × 1` = **1.73
TFLOP/s**. It hits **82% of that.** The kernel is running nearly as fast as the hardware permits;
the hardware simply does not permit much when each byte is used once.

That distinction — "underutilised" versus "memory-bound" — is the entire reason to draw a roofline.

### The two ceilings, measured

| | measured | A100 spec | achieved |
|---|---|---|---|
| memory bandwidth | **1,734.8 GB/s** | 2,039 | 85% |
| compute (bf16) | **263.6 TFLOP/s** | 312 | 85% |
| **ridge point** | **152.0 FLOPs/byte** | | |

Both ceilings land at 85% of the marketing figure. Quoting the spec sheet would have shifted every
conclusion by the same 15%.

Peak is also strongly **shape**-dependent, which is why the whole sweep is reported:

| square GEMM | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| TFLOP/s | 50.2 | 126.5 | 219.3 | 251.9 | **263.6** |

A 1024² matmul reaches **19%** of what a 16384² one does on identical silicon. "Peak FLOP/s"
measured at one arbitrary size is not a hardware property.

### The two regimes

| shape | AI (FLOPs/byte) | TFLOP/s | % of peak | bound by |
|---|---|---|---|---|
| prefill MLP up-projection | 1,219 | 231.3 | **87.8%** | compute |
| prefill QKV projection | 956 | 159.2 | 60.4% | compute |
| decode MLP up-projection | 1.0 | 1.42 | 0.5% | memory |
| decode QKV projection | 1.0 | 0.74 | 0.3% | memory |

Same weights, same model, **three orders of magnitude apart in arithmetic intensity** and on
opposite sides of the ridge. Prefill is a different machine from decode.

**Pre-registered band:** best prefill shape ≥ 70% of measured peak → **WITHIN (87.8%)** ✓

## Batching walks decode to the ridge, and then stops paying

![batch walk](results/batch_walk.png)

| batch | AI | TFLOP/s | implied GB/s |
|---|---|---|---|
| 1 | 1.0 | 1.41 | 1,414 |
| 8 | 8.0 | 11.2 | 1,408 |
| 64 | 62.7 | 88.1 | 1,405 |
| 128 | 122.8 | 149.1 | 1,214 |
| 256 | 236.0 | 155.0 | 657 |

**Batch 1 → 128: throughput ×105. Batch 128 → 256: ×1.04.**

The `implied GB/s` column is the mechanism, and it is the most useful number here. From batch 1 to
64 it is **pinned at ~1,410 GB/s** — the kernel is bandwidth-saturated the entire way up, and every
doubling of batch buys a near-doubling of throughput for free, because the weights are read once
regardless. Batch 256 is the first point whose intensity (236) exceeds the ridge (152); it has
crossed into compute-bound territory, bandwidth utilisation collapses to 657 GB/s, and the gains
stop dead.

Continuous batching is not a heuristic. It is the exploitation of a flat line on this plot, and it
runs out at a point the hardware fixes in advance.

## Inference payoff

1. **Decode is memory-bound by two orders of magnitude** (AI 1.0 vs ridge 152). Buying FLOP/s to
   speed up decode is buying the wrong thing.
2. **Quantisation is a bandwidth optimisation, not a compute one.** AI = `2/bytes_per_param`, so
   int8 doubles arithmetic intensity and int4 quadruples it — it moves the point rightward, which
   is the only direction that helps.
3. **Batch until the ridge, then stop.** The ceiling is `peak_FLOPs / bandwidth` and it is knowable
   before you run anything.
4. **Prefill and decode should be scheduled as different workloads** — one saturates compute, the
   other saturates memory. Disaggregated prefill/decode serving is this table, productised.
5. **Compute-bound is necessary but not sufficient.** Both prefill shapes cleared the ridge; one
   reached 87.8% of peak and the other 60.4%.

## What surprised me

- **The QKV projection got 60.4% where the MLP got 87.8%** — same regime, same hardware, both
  firmly compute-bound. The difference is shape: 3584 output columns tile onto the SMs worse than
  18944. I had assumed "compute-bound" was the end of the analysis; it is the beginning of one.
- **The same effect appears in decode**, larger: 1,420 GB/s for the MLP projection versus 744 GB/s
  for QKV. A 2× spread between two kernels in the same layer of the same model.
- **The implied-bandwidth column was an afterthought** and turned out to be the strongest evidence
  in the topic — a flat ~1,410 GB/s across six batch sizes says "bandwidth-saturated" far more
  convincingly than any single point could.
- **Both ceilings landing at exactly 85% of spec** is a coincidence, but a useful one for
  remembering roughly what a datasheet is worth.
- **The GEMM sweep spans 5.2×.** I expected shape sensitivity; I did not expect the small end to be
  under a fifth of the large end.

## What is not measured

- **Attention's own QK^T and AV matmuls are excluded.** Their cost scales with sequence length
  rather than with weights, so they belong to a separate analysis.
- **Analytic intensity is a lower bound on traffic** — real kernels re-read tiles that miss in L2,
  so true intensity is slightly worse and every point sits marginally left of where it is drawn.
- **Dense bf16 only.** No quantised or sparse kernels, so claim 2 above is arithmetic, not
  measurement.
- **One GPU, one session.** No claim about other hardware.
- **Isolated kernels, not a running model.** What the rest of a system does to these numbers is
  T6's question, not this one.

## Reproduce

```bash
python -m arch_common.probe            # once per session — shared with T6
python -m topics.t07_roofline.measure
python -m topics.t07_roofline.plot
```

Rehearsal on any laptop, seconds, no GPU:

```bash
python -m arch_common.probe --device cpu
python -m topics.t07_roofline.measure --device cpu --hidden 512 --intermediate 1024 \
    --prefill-tokens 128
```

On a CPU every shape lands *right* of the ridge — little compute relative to bandwidth puts the
ridge near 0.05 rather than 152, and the batch walk comes out flat. That is the correct physics for
that machine, and a useful check that the plot reads real measured ceilings rather than hardcoded
GPU constants.

## Relationship to T6

T7 owns the **shape** domain: isolated kernels, FLOPs per byte, position against the ridge — no
clock on either axis, no weights on disk. T6 owns the **time** domain: whole model, real KV cache,
tokens per second.

Both read their ceilings from the same `results/hardware.json`, so the two notes cannot end up
quoting different roofs for the same GPU. `tests/test_distinctness.py` fails CI if the metric
vocabularies overlap, if either topic strays into the other's domain, or if the two were measured
in different sessions.

The two batch sweeps are the one genuine near-collision, and they measure different things on
purpose: **T7 sweeps the matmul shape in isolation** — no model, no cache, pure kernel. **T6 sweeps
the whole model with a real cache.** T7 shows the kernel reaching toward the ridge; T6 shows what
the rest of the system does to that gain.
