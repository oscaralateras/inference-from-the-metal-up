# T8 — GPU architecture: what quantisation actually costs

**Question:** Decode is bandwidth-bound (T7). If I make it read **4× fewer bytes**, do I get 4× the
tokens — and if not, what ate the difference?

**Setup:** NVIDIA A100-SXM4-80GB, SM 1215 MHz / memory 1593 MHz, torch 2.8.0+cu128, Triton 3.4.0.
Qwen2.5-7B's decode MLP up-projection (N=18944, K=3584); as in T7, synthetic weights, because the
byte budget depends on the shape and not the values. Session `ea734b39914c`, shared with T6 and T7.

---

## Result

![the decode point moves](results/int4_roofline.png)

**Storing weights in int4 cuts the bytes 3.88× and buys 1.56×.**

Quantisation does not remove work; it trades one cost for two others. Half the shortfall is that a
compressed weight is a *shorter read*, and a shorter read gives the memory system less time to reach
steady state. The other half is the dequantisation arithmetic, which grows exactly as fast as the
byte count falls. A load-only control — same shape, same bytes, no maths — separates them.

| kernel | bytes / launch | ms | GB/s | % of memory roof |
|---|---|---|---|---|
| bf16, cuBLAS (`torch.matmul`) | 135.8 MB | 0.094 | 1,442 | 83.1% |
| **bf16, Triton** (control) | 135.8 MB | 0.082 | **1,658** | **95.4%** |
| load-only, same pattern, no arithmetic | 35.0 MB | 0.030 | 1,185 | 68.2% |
| **int4 fused, Triton** | 35.0 MB | **0.053** | 665 | 38.3% |

Median of 5 runs; the int4 spread is 664–666 GB/s.

| pre-registered band | predicted | measured | verdict |
|---|---|---|---|
| kernel ≥ 75% of the byte ratio | 3.88× ceiling | **1.56×** (40%) | **OUTSIDE** ✗ |
| cosine ≥ 0.99 vs fp32 | — | **0.9932** | WITHIN ✓ |
| end-to-end within ±25% | 2.21× | **1.36×** | **OUTSIDE** ✗ |

Two of three bands failed. They are reported as failures, and the mechanism behind them is the
finding.

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

That control also settles a question worth settling: it reaches **95.4% of the memory roof, ahead of
cuBLAS's 83.1%.** The framework is not the limitation and neither is the kernel structure. Whatever
the int4 kernel loses, it loses to its own arithmetic.

The cuBLAS row stays because "should I ship this?" is a real question with a different answer:
against what a decode step runs today, the int4 kernel is **1.79×**.

## The gap has two causes, and only one of them is quantisation

The int4 kernel reaches 40% of the bandwidth the same framework achieves on bf16. It is tempting to
charge all of that to the dequantisation, and wrong. The load-only row above is the control that
separates them: **same access pattern, same 35 MB, no arithmetic at all — and it reaches 68.2%, not
95%.**

| step | % of roof | cause |
|---|---|---|
| bf16 Triton, 135.8 MB/launch | 95.4% | — |
| load-only, 35.0 MB/launch | 68.2% | **the smaller transfer**, nothing to do with quantisation |
| int4 fused, 35.0 MB/launch | 38.3% | **the arithmetic** |

Roughly half the shortfall is simply that reading a quarter as much per launch gives the memory
system a quarter as long to reach steady state. That is a real cost of quantisation — a compressed
weight *is* a shorter read — but it is a different mechanism from the arithmetic, and a kernel that
merely loaded int4 bytes and threw them away would still not reach 95%.

The other half is the dequantisation, and the per-byte accounting explains it:

| | weights per byte loaded | work per byte |
|---|---|---|
| bf16 | 0.5 | one multiply-accumulate per 2 bytes → **~2 ops/byte** |
| int4 g128 | 2 | unpack two nibbles, two multiply-accumulates → **~8 ops/byte** |

**Four structural variants of the kernel — a wider reduction tile, fp16 instead of fp32 arithmetic,
the zero-point folded out of the inner loop, and single-stream accumulation joining the two nibble
halves into one reduction — all measured within noise of each other.** That is the evidence that
this is a limit on arithmetic *volume* rather than on kernel structure: when four independent ways
of rearranging the work change nothing, the work itself is the constraint.

**Reading 4× fewer bytes means doing 4× more arithmetic per byte read**, and an A100 has roughly
**9.6 fp32 vector FLOPs available per byte of bandwidth** (19.5 TFLOP/s against 2,039 GB/s). At ~8
ops/byte the dequantisation consumes most of that budget, which is why the kernel sits where it
does against its own load-only ceiling.

The general form of the result: whether quantisation pays on a given GPU depends on that GPU's
compute-per-byte, and a datacentre part optimised for tensor-core throughput can have far less
*general-purpose vector* throughput per byte than the headline figures suggest.

## Where this sits against production kernels

**1.56× is well short of what the technique can deliver, and the shortfall is implementation, not
physics.** Marlin and the AWQ/GPTQ kernels reach close to the full byte ratio at batch 1 on this
class of hardware. So the ceiling is reachable; this kernel does not reach it.

The difference is specific and nameable. Those kernels avoid the integer→float conversion entirely,
bit-stuffing nibbles into fp16 mantissas so the unpack is a mask and an add rather than a
conversion instruction, and they dequantise directly into tensor-core layouts so the multiply
leaves the vector units altogether. That takes the per-byte arithmetic from ~8 operations to
roughly 1–2 — which is exactly the term this measurement identifies as binding.

Stated plainly because the alternative reading is wrong: **this is not evidence that quantisation
fails to pay on an A100.** It is a measurement of what a straightforward fused kernel costs, and an
identification of which cost the production kernels engineer away. A naive fused kernel is the
right thing to write first and the wrong thing to ship, and the gap between the two is the ~6
operations per byte above.

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
