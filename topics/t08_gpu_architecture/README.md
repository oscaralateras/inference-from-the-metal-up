# T8 — GPU architecture: what quantisation actually costs

**Question:** Decode is bandwidth-bound (T7). If I make it read **4× fewer bytes**, do I get 4× the
tokens — and if not, what ate the difference?

**Setup:** NVIDIA A100-SXM4-80GB, SM 1215 MHz idle / 1410 MHz boost, memory 1593 MHz, torch
2.8.0+cu128, Triton 3.4.0. **Every "% of roof" below is against the *measured* roof of 1,737 GB/s**
(`make probe`, shared with T6 and T7), not the 2,039 GB/s datasheet figure — the two are never mixed.
Qwen2.5-7B's decode MLP up-projection (N=18944, K=3584); as in T7, synthetic weights, because the
byte budget depends on the shape and not the values. Session `ea734b39914c`, shared with T6 and T7.

---

## Result

![the decode point moves](results/int4_roofline.png)

**Storing weights in int4 cuts the bytes 3.88× and buys 1.54×.**

Quantisation does not remove work; it relocates it. What it takes off the memory system it puts
onto the vector units, and on this GPU at batch 1 the second cost eats most of the first.

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

Two of three bands failed. They are reported as failures, and the mechanism behind them is the
finding.

### The int8 control is what makes that mechanism a measurement rather than an excuse

A kernel that misses its band has an obvious alternative explanation: the kernel is bad, or the
harness is. The cheapest way to distinguish "quantisation costs arithmetic" from "Oscar writes slow
Triton" is to run the *same kernel structure* at a second bit width and see whether the band moves
with the arithmetic. int8 halves the byte saving and halves the work per byte with it — one byte
carries one weight, so there is no nibble to unpack and one multiply-accumulate per byte instead of
two.

| | byte ratio | speedup | **% of its own ceiling** | band |
|---|---|---|---|---|
| int8 fused | 1.97× | 1.49× | **75.7%** | **WITHIN ✓** |
| int4 fused | 3.88× | 1.54× | **39.8%** | OUTSIDE ✗ |

**The same code, the same harness and the same pre-registered 75% line pass at 8 bits and fail at
4.** A bad kernel or a bad harness would fail both. The band was not miscalibrated; it is int4's
arithmetic that misses it.

The sharpest form of the result appears when the two integer kernels are normalised to *weights
processed* rather than bytes moved:

| | GB/s | bytes/weight | **Gweights/s** |
|---|---|---|---|
| bf16 Triton | 1,659 | 2.000 | 830 |
| int8 fused | 1,256 | 1.016 | **1,237** |
| int4 fused | 660 | 0.516 | **1,281** |

The two integer kernels sit within 3.6% of each other on weights per second while their
*bandwidths* differ by 1.9×. They are pinned to the same rate — and it is a rate denominated in
weights, not in bytes. That is the whole topic in one line: **below bf16, this kernel stops being
limited by how much it reads and starts being limited by how much it must do to each thing it
reads.** Halving the bits again buys almost nothing, because the binding constraint no longer has
bits in it.

### Everything that fed the prediction came from an earlier topic

| term | value | source |
|---|---|---|
| bytes/param, bf16 | 2.000 | — |
| bytes/param, int4 g128 | **0.516** | `pack.py`, measured off the allocated tensors |
| byte ratio | **3.88×** | the kernel's ceiling — it cannot beat this |
| weight share of a decode step | **73.7%** | T6's error budget, read live from `t06/results/perf.csv` |
| **predicted end-to-end** | **2.21×** | Amdahl (T5): `1 / (0.737/3.88 + 0.263)` |

---

## Why the band is scored against Triton, not cuBLAS

Comparing a hand-written Triton kernel against cuBLAS measures two things at once: the byte
reduction, and the distance between one person's kernel and NVIDIA's hand-tuned assembly. Only the
first is what this topic is about.

So the control is **the same GEMV in Triton over bf16 weights** — same author, same framework, same
tiling, same reduction, same reused output buffer. Only the data format differs.

That control also settles a question worth settling: it reaches **95.5% of the memory roof, ahead of
cuBLAS's 83.2%.** The framework is not the limitation and neither is the kernel structure. Whatever
the int4 kernel loses, it loses to its own arithmetic.

The cuBLAS row stays because "should I ship this?" is a real question with a different answer:
against what a decode step runs today, the int4 kernel is **1.77×**.

## The gap has two causes, and only one of them is quantisation

The int4 kernel reaches 40% of the bandwidth the same framework achieves on bf16. It is tempting to
charge all of that to the dequantisation, and wrong. The load-only row above is the control that
separates them: **same access pattern, same 35 MB, no arithmetic at all — and it reaches 68.2%, not
95%.**

| step | % of roof | cause |
|---|---|---|
| bf16 Triton, 135.8 MB/launch | 95.5% | — |
| load-only, 35.0 MB/launch | 68.2% | **the smaller transfer**, nothing to do with quantisation |
| int4 fused, 35.0 MB/launch | 38.0% | **the arithmetic** |

Roughly half the shortfall is simply that reading a quarter as much per launch gives the memory
system a quarter as long to reach steady state. That is a real cost of quantisation — a compressed
weight *is* a shorter read — but it is a different mechanism from the arithmetic, and a kernel that
merely loaded int4 bytes and threw them away would still not reach 95%.

**The ceiling is a property of the transfer size, not of "loading" in general** — which is worth
stating because the int8 row appears to violate it. int8 reaches 1,256 GB/s against a load-only
ceiling of 1,185, and that is not a measurement error: the probe streams the *int4* volume of
35 MB, while int8 moves 69 MB per launch and therefore amortises ramp-up over twice as long a read.
Two transfer sizes, two ceilings. The int8 row is a second, independent sighting of the same
shorter-read effect the probe was built to isolate, and the integrity guard bounds int4 against the
probe only, because that is the pair the probe actually shares an access pattern with.

The other half is the dequantisation, and the per-byte accounting explains it:

| | weights per byte loaded | work per byte |
|---|---|---|
| bf16 | 0.5 | one multiply-accumulate per 2 bytes → **~2 ops/byte** |
| int8 g128 | 1 | subtract bias, convert, multiply-accumulate → **~4 ops/byte** |
| int4 g128 | 2 | unpack two nibbles, two multiply-accumulates → **~8 ops/byte** |

**Five structural variants were tried and every one measured within noise**: a wider reduction tile,
fp16 rather than fp32 arithmetic, the zero-point folded out of the inner loop, single-stream
accumulation joining the two nibble halves into one reduction, and unpacking by bit-stuffing codes
into an fp16 mantissa to remove the integer→float conversion entirely — the technique production
kernels use.

Those nulls are **consistent with** an arithmetic-volume limit without proving one. Each removes
roughly one operation of the eight, which predicts a ~4% change — inside the 659–662 GB/s spread,
so individually unresolvable. What carries the claim is the two controls, both of which move the
result by far more than the noise: the load-only probe (a 30-point effect on the transfer term) and
the int8 kernel (a 36-point effect on the arithmetic term, and a band that flips verdict).

**Reading 4× fewer bytes means doing 4× more arithmetic per byte read**, and an A100 has roughly
**11.2 fp32 vector FLOPs available per byte of bandwidth** — 19.5 TFLOP/s (6,912 fp32 lanes × 1.41
GHz boost × 2) against the 1,737 GB/s measured here. At ~8 ops/byte the dequantisation consumes
most of that budget; at int8's ~4 it does not, which is why int8 clears the band and int4 does not.

That budget is why the two integer kernels converge on ~1.25 Tweights/s: the limit they share is
the vector-issue rate, and it is indifferent to how many bits each weight was stored in.

The general form of the result: whether quantisation pays on a given GPU depends on that GPU's
compute-per-byte, and a datacentre part optimised for tensor-core throughput can have far less
*general-purpose vector* throughput per byte than the headline figures suggest.

## Where this sits against production kernels

**1.54× is short of what the technique can deliver, and the measurement points at why.** Production
int4 kernels — Marlin, AWQ, the GPTQ family — attack precisely the term this topic identifies as
binding, by two mechanisms:

1. **They avoid the integer→float conversion.** A 4-bit code is OR-ed into the mantissa of a
   constant-exponent fp16 and the bias subtracted, so unpacking is a mask and an add rather than a
   conversion instruction.
2. **They dequantise into tensor-core layouts**, moving the multiply off the vector units entirely.

Both reduce operations per byte, which is the quantity the per-byte accounting above says decides
the outcome. No claim is made here about what multiple they achieve — that would be quoting someone
else's benchmark rather than reporting one, which is what the rest of this repo exists not to do.

Worth stating plainly because the alternative reading is wrong: **this is not evidence that
quantisation fails to pay.** It is a measurement of what a straightforward fused kernel costs on one
GPU at batch 1, and an identification of which cost the production kernels are built to remove.

Note also that mechanism 2 is unavailable at M=1: with a single output column there is nothing for
a tensor core to do. So the batch-1 decode case measured here is the *hardest* case for quantised
kernels, not the representative one — a batched serving path has a lever this one does not.

## Why the dequantisation has to live inside the kernel

Storing weights int4 and calling `torch.matmul(x, dequantise(W))` saves nothing — it writes a
full-width bf16 matrix to HBM and streams it straight back, so the traffic is bf16 traffic *plus*
the int4 read. Compression only pays if the expansion happens after the load, in registers, and the
expanded weight never leaves the SM.

`int4_gemv_reference` in `kernel.py` is deliberately that unfused version, kept as the correctness
oracle precisely because it is what the fused kernel exists not to be.

## Why this kernel has no tiling, no shared memory and no tensor cores

At M=1 there is no reuse to exploit: every weight is loaded once, multiplied once, discarded.
Shared-memory staging, register tiling and `tl.dot` all exist to amortise a load across a tile of
outputs, and with one output column there is nothing to amortise. `tl.dot` never appears in this
file.

Worth stating because it inverts the usual GPU lesson: the A100's 312 TFLOP/s of tensor-core
throughput is **irrelevant** to decode. Its far smaller fp32 vector throughput is what binds.

## Caveats & reproduce

- **The tokens/sec figure is a projection**, computed from T6's error budget — not a re-run of vLLM
  with this kernel spliced in. Labelled as modelled wherever it appears.
- **L2 contamination, and how it is avoided.** Timing one weight repeatedly leaves it resident in
  L2 and reports cache bandwidth as if it were HBM. The int4 weight is 35 MB against this A100's
  42 MB of L2, so it fits entirely in cache — the variant whose whole claim is that it streams
  fewer bytes would have streamed none of them. Both variants rotate through 5 distinct weights
  (175 MB int4, 679 MB bf16), sized from the device. This is also the faithful setup: a decode step
  reads every layer once and reuses none of it.
- **Steady state, not cold launches.** Each timing window contains 16 back-to-back launches. A
  decode step fires ~200 of these in sequence and never pays for one in isolation, so single-launch
  timing would report the dispatch path rather than the kernel. Applied identically to every
  variant. The launches here are mutually independent, whereas a real step's are dependent layer to
  layer — so reality has slightly less launch-level parallelism than this measures.
- **cuBLAS allocates its own output**; both Triton kernels are given reused buffers. That asymmetry
  flatters cuBLAS, which is the safe direction, and it does not touch the band, which is scored
  Triton-against-Triton.
- **Autotuner variance.** Triton re-searches in every fresh process, so the whole benchmark is
  repeated 5 times and the median reported with its spread.
- **One shape, one GPU, synthetic weights.** The MLP up-projection only; attention's QK^T and AV
  matmuls scale with sequence length rather than weights and belong to a different analysis.
  Accuracy is measured against the unquantised *synthetic* tensor, so it validates the kernel, not
  the model — T1 is where int4's accuracy on a real Qwen weight was established.
- **One GPU.** The compute-per-byte argument above predicts that a card with more vector throughput
  per byte would land closer to the byte ratio, but that is an inference from the datasheet, not a
  measurement. Running the same commit on a second GPU would test it; this topic does not.

```bash
uv sync
make probe                                  # once per pod; T6, T7 and T8 share the profile
make t8-predict                             # prediction only — no GPU needed
make t8                                     # measure + plot (CUDA + Triton)
make t8-ceiling                             # load-only ceiling for this access pattern
uv run pytest topics/t08_gpu_architecture   # packing tests run anywhere; kernel tests need CUDA
```
