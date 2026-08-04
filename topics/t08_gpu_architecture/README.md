# T8 — GPU architecture: what quantisation actually costs

**Question:** Decode is bandwidth-bound (T7). If I make it read **4× fewer bytes**, do I get 4× the
tokens — and if not, what ate the difference?

**Setup:** NVIDIA A100-SXM4-80GB, SM 1215 MHz / memory 1593 MHz, torch 2.8.0+cu128, Triton 3.4.0.
Qwen2.5-7B's decode MLP up-projection (N=18944, K=3584); as in T7, synthetic weights, because the
byte budget depends on the shape and not the values. Session `ea734b39914c`, shared with T6 and T7.

---

## Result

![the decode point moves](results/int4_roofline.png)

**Storing weights in int4 cuts the bytes 3.88× and buys 1.56×. The difference is arithmetic:
quantisation does not remove work, it trades memory traffic for compute — and on an A100 that trade
is close to a wash.**

| kernel | ms | GB/s | % of memory roof |
|---|---|---|---|
| bf16, cuBLAS (`torch.matmul`) | 0.094 | 1,442 | 83.1% |
| **bf16, Triton** (control) | 0.082 | **1,658** | **95.4%** |
| **int4 fused, Triton** | **0.053** | 665 | 38.3% |

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

## What surprised me: quantisation is a trade, not a saving

The int4 kernel reaches 40% of the bandwidth the *same framework* achieves on bf16. That is not a
tuning failure — it is what the arithmetic costs, and the per-byte accounting makes it obvious:

| | weights per byte loaded | work per byte |
|---|---|---|
| bf16 | 0.5 | one multiply-accumulate per 2 bytes → **~2 ops/byte** |
| int4 g128 | 2 | unpack two nibbles, two multiply-accumulates → **~8 ops/byte** |

**Four structural variants of the kernel — a wider reduction tile, fp16 instead of fp32 arithmetic,
the zero-point folded out of the inner loop, and single-stream accumulation joining the two nibble
halves into one reduction — all measured within noise of each other.** That is the evidence that
this is a limit on arithmetic *volume* rather than on kernel structure: when four independent ways
of rearranging the work change nothing, the work itself is the constraint.

**Reading 4× fewer bytes means doing 4× more arithmetic per byte read.** Whether that trade pays
depends entirely on how much compute a GPU has per byte of bandwidth:

| GPU | fp32 vector | bandwidth | FLOPs available per byte |
|---|---|---|---|
| RTX 4090 | ~82 TFLOP/s | 1,008 GB/s | **~82** |
| A100-SXM4-80GB | ~19.5 TFLOP/s | 2,039 GB/s | **~9.6** |

At ~8 ops/byte the int4 kernel uses a tenth of a 4090's budget and most of an A100's. The same
kernel, unchanged, is memory-bound on the consumer card and arithmetic-bound on the datacentre one —
because the datacentre card has nearly twice the bandwidth and under a quarter the fp32 vector
throughput.

**This is why production int4 kernels are exotic.** Marlin, AWQ and the GPTQ kernels go to
considerable lengths — bit-stuffing nibbles into fp16 mantissas to dodge the integer→float
conversion instruction, dequantising directly into tensor-core layouts — and from the outside that
reads as over-engineering. It isn't. Without it the win evaporates on exactly the hardware people
deploy on. A naive fused kernel is the right thing to write first, and this is the measurement that
explains why it is not the thing to ship.

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
- **The cross-GPU comparison is not like-for-like.** The 4090 figure quoted above came from an
  earlier development session on different silicon, driver and clocks. It is used to make a
  qualitative point about compute-per-byte, not as a paired measurement.

```bash
uv sync
make probe                                  # once per pod; T6, T7 and T8 share the profile
make t8-predict                             # prediction only — no GPU needed
make t8                                     # measure + plot (CUDA + Triton)
make t8-ceiling                             # load-only ceiling for this access pattern
uv run pytest topics/t08_gpu_architecture   # packing tests run anywhere; kernel tests need CUDA
```
