# T8 — GPU architecture: what quantisation actually costs

**Question:** Decode is bandwidth-bound (T7). If I make it read **4× fewer bytes**, do I get 4× the
tokens — and if not, where does the difference go?

**Setup:** NVIDIA A100-SXM4-80GB, SM 1215 MHz idle / 1410 MHz boost, memory 1593 MHz, torch
2.8.0+cu128, Triton 3.4.0. Every "% of roof" below is against the **measured** roof of 1,737 GB/s
(`make probe`, shared with T6 and T7) rather than the 2,039 GB/s datasheet figure. I never mix the
two. The shape is Qwen2.5-7B's decode MLP up-projection (N=18944, K=3584), with synthetic weights as
in T7, since the byte budget depends on the shape and not the values. Session `ea734b39914c`, shared
with T6 and T7.

---

## Result

![the decode point moves](results/int4_roofline.png)

**Storing weights in int4 cuts the bytes 3.88× and buys 1.54×.**

The bytes did drop as predicted. The speedup did not follow, because the work that quantisation
takes off the memory system lands on the vector units instead, and at batch 1 on this GPU that
second cost eats most of the first.

| kernel | bytes / launch | ms | GB/s | % of memory roof |
|---|---|---|---|---|
| bf16, cuBLAS (`torch.matmul`) | 135.8 MB | 0.094 | 1,444 | 83.2% |
| **bf16, Triton** (control) | 135.8 MB | 0.082 | **1,659** | **95.5%** |
| int8 fused, Triton (control) | 69.0 MB | 0.055 | 1,256 | 72.3% |
| load-only, int4 pattern, no arithmetic | 35.0 MB | 0.030 | 1,185 | 68.2% |
| **int4 fused, Triton** | 35.0 MB | **0.053** | 660 | 38.0% |

Median of 5 runs; the int4 spread is 659–662 GB/s.

| pre-registered band | predicted | measured | verdict |
|---|---|---|---|
| kernel ≥ 75% of the byte ratio | 3.88× ceiling | **1.54×** (40%) | **OUTSIDE** ✗ |
| cosine ≥ 0.99 vs fp32 | — | **0.9932** | WITHIN ✓ |
| end-to-end within ±25% | 2.21× | **1.35×** | **OUTSIDE** ✗ |

Two of the three bands failed. I've left them as failures and spent the rest of the note working out
why, since that turned out to be the interesting part.

### Checking it against int8

A kernel that misses its band has an obvious competing explanation: maybe the kernel is just slow, or
the harness is wrong. Arguing about that from one data point is hopeless, so I ran the same kernel
structure at 8 bits and looked at whether the band moved with the arithmetic.

int8 is a useful comparison because it halves both things at once. It saves half as many bytes as
int4, and it also does half the work per byte, since one byte carries one weight — no nibble to
unpack, one multiply-accumulate per byte instead of two.

| | byte ratio | speedup | **% of its own ceiling** | band |
|---|---|---|---|---|
| int8 fused | 1.97× | 1.49× | **75.7%** | **WITHIN ✓** |
| int4 fused | 3.88× | 1.54× | **39.8%** | OUTSIDE ✗ |

Same code, same harness, same pre-registered 75% line — and it passes at 8 bits while failing at 4.
A slow kernel or a broken harness would have failed both. So the band was reasonable; it's int4's
arithmetic that misses it.

The comparison gets sharper if you stop counting bytes and count weights instead:

| | GB/s | bytes/weight | **Gweights/s** |
|---|---|---|---|
| bf16 Triton | 1,659 | 2.000 | 830 |
| int8 fused | 1,256 | 1.016 | **1,237** |
| int4 fused | 660 | 0.516 | **1,281** |

The two integer kernels land within 3.6% of each other on weights per second, even though their
bandwidths differ by a factor of 1.9. They're stuck at the same rate, and that rate is measured in
weights rather than bytes. Below bf16 this kernel stops being limited by how much it reads and starts
being limited by how much it has to do to each weight it reads. Halving the bits again buys almost
nothing, because by then the constraint doesn't have bits in it.

### Where the prediction came from

| term | value | source |
|---|---|---|
| bytes/param, bf16 | 2.000 | — |
| bytes/param, int4 g128 | **0.516** | `pack.py`, measured off the allocated tensors |
| byte ratio | **3.88×** | the kernel's ceiling — it cannot beat this |
| weight share of a decode step | **73.7%** | T6's error budget, read live from `t06/results/perf.csv` |
| **predicted end-to-end** | **2.21×** | Amdahl (T5): `1 / (0.737/3.88 + 0.263)` |

Every term is either measured off the tensors or read from an earlier topic's CSV at runtime. None
of it is typed into this file.

---

## Why the band is scored against Triton rather than cuBLAS

Benchmarking my Triton kernel against cuBLAS would measure two things at once: the byte reduction,
and the gap between my code and NVIDIA's hand-tuned assembly. Only the first one is what this topic
is about.

So the control is the same GEMV in Triton over bf16 weights — same author, same framework, same
tiling, same reduction, same reused output buffer, with only the data format changed.

That control also answers a question I had about my own code. It hits **95.5% of the memory roof,
ahead of cuBLAS's 83.2%**, so neither Triton nor the way I've structured the kernel is what's holding
the int4 version back.

I've kept the cuBLAS row anyway, because "should I ship this?" is a real question and it has a
different answer: against what a decode step actually runs today, the int4 kernel is **1.77×**.

## Two separate costs, and only one of them is quantisation

The int4 kernel reaches 40% of the bandwidth the same framework gets on bf16. The tempting move is to
blame all of that on the dequantisation, but that's wrong, and the load-only row is what shows it.
Strip the kernel down to loads and a trivial sum — same access pattern, same 35 MB, no arithmetic at
all — and it still only reaches 68.2%.

| step | % of roof | cause |
|---|---|---|
| bf16 Triton, 135.8 MB/launch | 95.5% | — |
| load-only, 35.0 MB/launch | 68.2% | **the smaller transfer**, nothing to do with quantisation |
| int4 fused, 35.0 MB/launch | 38.0% | **the arithmetic** |

So roughly half the shortfall is just that reading a quarter as much per launch gives the memory
system a quarter as long to get up to speed. That is a genuine cost of quantisation — a compressed
weight really is a shorter read — but it's a different mechanism from the arithmetic, and a kernel
that only loaded int4 bytes and threw them away would still fall well short of 95%.

One thing to flag before someone spots it in the table: int8 reaches 1,256 GB/s, which is above the
1,185 GB/s load-only ceiling. That isn't a measurement error. The probe streams the int4 volume of
35 MB, while int8 moves 69 MB per launch and spreads the ramp-up over twice as long a read. Different
transfer sizes have different ceilings, so int8 landing above this one is the same shorter-read
effect showing up again from the other direction. The integrity guard only bounds int4 against the
probe, since that's the pair that actually shares an access pattern.

The other half of the shortfall is the dequantisation, and the per-byte accounting covers it:

| | weights per byte loaded | work per byte |
|---|---|---|
| bf16 | 0.5 | one multiply-accumulate per 2 bytes → **~2 ops/byte** |
| int8 g128 | 1 | subtract bias, convert, multiply-accumulate → **~4 ops/byte** |
| int4 g128 | 2 | unpack two nibbles, two multiply-accumulates → **~8 ops/byte** |

Reading 4× fewer bytes means doing 4× more arithmetic per byte read, and an A100 has roughly **11.2
fp32 vector FLOPs available per byte of bandwidth** — 19.5 TFLOP/s (6,912 fp32 lanes × 1.41 GHz boost
× 2) against the 1,737 GB/s measured here. At around 8 ops/byte the dequantisation eats most of that
budget. At int8's 4 it doesn't, which is why int8 clears the band and int4 doesn't, and why the two
integer kernels converge on about 1.25 Tweights/s: they're sharing a vector-issue limit that doesn't
care how many bits each weight was stored in.

The general version of this: whether quantisation pays on a given GPU depends on that GPU's compute
per byte, and a datacentre part built for tensor-core throughput can have much less *general-purpose
vector* throughput per byte than the headline numbers suggest.

### The optimisations that didn't help

I tried five structural variants and every one came back inside the noise: a wider reduction tile,
fp16 instead of fp32 arithmetic, the zero-point folded out of the inner loop, single-stream
accumulation joining the two nibble halves into one reduction, and unpacking by bit-stuffing codes
into an fp16 mantissa to skip the integer→float conversion entirely, which is the trick production
kernels use.

Those nulls fit an arithmetic-volume limit but don't establish one. Each removes roughly one
operation out of eight, so each predicts about a 4% change, and that's inside the 659–662 GB/s
spread — too small to resolve one at a time. The claim rests on the two controls instead, both of
which move the result far more than the noise does: the load-only probe is a 30-point effect on the
transfer term, and the int8 kernel is a 36-point effect on the arithmetic term with a band that
changes verdict.

## How this compares to production kernels

1.54× is well short of what int4 can deliver, and the measurement points at why. Production kernels —
Marlin, AWQ, the GPTQ family — go after exactly the term this topic finds binding, in two ways:

1. **They avoid the integer→float conversion.** A 4-bit code gets OR-ed into the mantissa of a
   constant-exponent fp16 and the bias subtracted, so unpacking is a mask and an add rather than a
   conversion instruction.
2. **They dequantise into tensor-core layouts**, which moves the multiply off the vector units.

Both cut operations per byte, which the accounting above says is the quantity that decides the
outcome. I'm not going to quote a number for what they achieve, because I haven't run them, and
quoting someone else's benchmark is the thing this repo exists to avoid.

Two things worth being clear about, since the obvious misreading runs the other way. This is not
evidence that quantisation doesn't pay. It's a measurement of what a straightforward fused kernel
costs on one GPU at batch 1, plus an identification of which cost the production kernels are built to
remove.

And the second mechanism isn't even available here: at M=1 there's a single output column, so there's
nothing for a tensor core to do. Batch-1 decode is the hardest case for a quantised kernel, not the
typical one. A batched serving path has a lever this one doesn't.

## Why the dequantisation has to happen inside the kernel

Storing weights in int4 and then calling `torch.matmul(x, dequantise(W))` saves nothing. It writes a
full-width bf16 matrix out to HBM and streams it straight back, so the traffic ends up being bf16
traffic *plus* the int4 read. Compression only pays if the expansion happens after the load, in
registers, and the expanded weight never leaves the SM.

`int4_gemv_reference` in `kernel.py` is that unfused version. I've kept it as the correctness oracle
precisely because it does the thing the fused kernel is designed to avoid — if the two agree, the
fusion didn't change the maths.

## Why there's no tiling, shared memory or tensor cores here

At M=1 there's no reuse to exploit. Every weight is loaded once, multiplied once and discarded.
Shared-memory staging, register tiling and `tl.dot` all exist to amortise a load across a tile of
outputs, and with one output column there's nothing to amortise. `tl.dot` never appears in the file.

This inverts the usual GPU lesson, which is why it's worth spelling out: the A100's tensor-core
throughput — 312 TFLOP/s on the datasheet, 260 measured by `make probe` — is irrelevant to decode.
The much smaller fp32 vector throughput is what binds.

## Caveats & reproduce

- **The tokens/sec figure is a projection**, computed from T6's error budget. It is not a re-run of
  vLLM with this kernel spliced in, and it's labelled as modelled wherever it appears.
- **L2 contamination.** Timing one weight repeatedly leaves it sitting in L2 and reports cache
  bandwidth as if it were HBM. The int4 weight is 35 MB against this A100's 42 MB of L2, so it fits
  entirely in cache, and the variant whose whole claim is that it streams fewer bytes would have
  streamed none of them. Both variants rotate through 5 distinct weights (175 MB int4, 679 MB bf16),
  sized from the device. This also happens to be the faithful setup, since a decode step reads every
  layer once and reuses none of it.
- **Steady state rather than cold launches.** Each timing window contains 16 back-to-back launches. A
  decode step fires around 200 of these in sequence and never pays for one in isolation, so
  single-launch timing would mostly measure the dispatch path. Applied identically to every variant.
  One caveat on the caveat: these launches are independent, whereas a real step's are dependent layer
  to layer, so reality has slightly less launch-level parallelism than this setup gives it.
- **cuBLAS allocates its own output** while both Triton kernels get reused buffers. That asymmetry
  flatters cuBLAS, which is the safe direction to be wrong in, and it doesn't touch the band, which
  is scored Triton against Triton.
- **Autotuner variance.** Triton re-searches in every fresh process, so the whole benchmark runs 5
  times and I report the median with its spread.
- **One shape, one GPU, synthetic weights.** The MLP up-projection only — attention's QK^T and AV
  matmuls scale with sequence length rather than with weights and belong in a different analysis.
  Accuracy is measured against the unquantised *synthetic* tensor, so it validates the kernel rather
  than the model. T1 is where int4's accuracy on a real Qwen weight was established.
- **The compute-per-byte argument is untested across GPUs.** It predicts that a card with more vector
  throughput per byte would land closer to the byte ratio, but that's read off a datasheet, not
  measured. Running this same commit on a second GPU would test it. This topic doesn't.
- **No profiler evidence.** The ~8 ops/byte figure is counted by hand from the kernel source, not
  read out of Nsight Compute. The int8 trend makes the arithmetic explanation hard to argue with, but
  an `ncu` capture of issue-slot utilisation and the SASS instruction mix would turn the strongest
  inference here into a direct observation. That's the next thing I'd measure.

```bash
uv sync
make probe                                  # once per pod; T6, T7 and T8 share the profile
make t8-predict                             # prediction only — no GPU needed
make t8                                     # measure + plot (CUDA + Triton)
make t8-ceiling                             # load-only ceiling for this access pattern
uv run pytest topics/t08_gpu_architecture   # packing tests run anywhere; kernel tests need CUDA
```
