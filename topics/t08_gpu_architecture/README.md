# T8 — GPU architecture: buying back decode throughput

**Question:** Decode is bandwidth-bound (T7). If I make it read **4× fewer bytes**, do I get 4× the
tokens — and if not, what ate the difference?

**Setup:** *(development results below are from an NVIDIA RTX 4090, driver 570.195.03, SM 2580 MHz /
memory 10501 MHz, torch 2.8.0+cu128, Triton 3.4.0, session `8dd030949035`. The reportable run is on
A100-SXM4-80GB — see Caveats.)* Qwen2.5-7B's decode MLP up-projection (N=18944, K=3584), synthetic
weights: as in T7, the byte budget depends on the shape, not the values.

---

## Result

![the decode point moves](results/int4_roofline.png)

**Cutting the bytes 3.88× made the kernel 3.43× faster and the decode step 2.20× faster. The
missing 1.68× is the 23% of the step that was never weight traffic** — Amdahl's law, first plotted
in T5 as a theoretical curve, arriving as a hard ceiling on real silicon.

The end-to-end prediction was registered before the kernel ran: **2.33× predicted, 2.20× measured,
a 5.6% miss.**

| | measured | roof | % of roof |
|---|---|---|---|
| bf16 torch (baseline) | 0.139 ms | 976.4 GB/s | 103.9% |
| **int4 fused (this kernel)** | **0.040 ms** | **864.8 GB/s** | **92.0%** |

| pre-registered band | predicted | measured (median of 5) | verdict |
|---|---|---|---|
| kernel ≥ 75% of the byte ratio | 3.88× ceiling | **3.43×** (88%) | **WITHIN** ✓ |
| cosine ≥ 0.99 vs fp32 | — | **0.9933** | **WITHIN** ✓ |
| end-to-end within ±25% | 2.33× | **2.20×** | **WITHIN** ✓ |

Run-to-run spread across five runs: **3.10×–3.53×**, median 3.43×. The variation is Triton's
autotuner re-searching in each fresh process, not thermal — the GPU idles at 39 °C between runs.
Reported as a spread rather than as a single flattering number.

### Everything that fed the prediction came from an earlier topic

| term | value | source |
|---|---|---|
| bytes/param, bf16 | 2.000 | — |
| bytes/param, int4 g128 | **0.516** | `pack.py`, measured off the allocated tensors |
| byte ratio | **3.88×** | the kernel's ceiling — it cannot beat this |
| weight share of a decode step | **76.9%** | T6's error budget, read live from `t06/results/perf.csv` |
| **predicted end-to-end** | **2.33×** | Amdahl (T5): `1 / (0.769/3.88 + 0.231)` |

---

## What surprised me: the kernel was never the slow part

The first working version measured **49% of the memory roof** and I spent four rounds optimising it.
Three of those four made it worse or did nothing:

| change | result |
|---|---|
| factor the scale out of the inner sum | 52% → 80% ✓ **real** |
| widen the reduction tile to several groups | 52% → 48% ✗ reverted |
| widen the autotune space downward | no change ✗ kept anyway |
| lift the zero-point out of the loop | 49% → 36% ✗ reverted |

The actual cause was in the **measurement harness**, twice over:

1. **`torch.empty` on every call.** The wrapper allocated a fresh output tensor inside the timed
   region. For a 40 µs kernel the PyTorch caching allocator is not a rounding error.
2. **Timing single cold launches.** A decode step launches ~200 of these GEMVs back-to-back and
   never pays for one in isolation, so single-launch timing reports the dispatch path rather than
   the kernel.

Fixing both took the kernel from 49% to 92% of roof with **no change to the kernel at all**. The
optimisation that did matter — factoring the scale out — was worth 1.95× → 3.00×, but I could only
prove that after the harness was honest. Before then I was A/B-testing against noise.

**The lesson is the one worth publishing:** the first three rounds of "optimisation" were confident,
reasoned, and aimed at a bottleneck that did not exist. What broke the loop was building a stripped
load-only probe of the same access pattern and finding it reached 93% of roof — which localised the
problem to everything *around* the kernel rather than inside it.

## Why the dequantisation has to live inside the kernel

Storing weights int4 and calling `torch.matmul(x, dequantise(W))` saves nothing at all — it writes a
full-width bf16 matrix to HBM and streams it straight back, so the traffic is bf16 traffic *plus* the
int4 read. Compression only pays if the expansion happens after the load, in registers, and the
expanded weight never leaves the SM.

That is the whole content of the word "fused". `int4_gemv_reference` in `kernel.py` is deliberately
the unfused version, kept as the correctness oracle precisely because it is what the fused kernel
exists not to be.

## Why this kernel has no tiling, no shared memory and no tensor cores

At M=1 there is no reuse to exploit: every weight is loaded once, multiplied once, discarded.
Shared-memory staging, register tiling and `tl.dot` all exist to amortise a load across a tile of
outputs, and with one output column there is nothing to amortise. `tl.dot` never appears in this
file.

Worth saying plainly because it inverts the usual GPU lesson: the A100's 312 TFLOP/s and its tensor
cores are **irrelevant** to decode.

## Caveats & reproduce

- **Development hardware.** These numbers are from an RTX 4090. T6 and T7 measured an
  A100-SXM4-80GB in session `6c79f20d6c13`, which no longer exists. What crosses the gap is the
  *ratio* 76.9% — a property of the model's parameter count against its non-weight work — while
  T6's absolute 94.3 tok/s does not. **The tokens/sec figure is therefore a projection, not a
  measurement**, and is labelled as such everywhere it appears.
- **The baseline exceeds 100% of roof, legitimately.** `arch_common.gpu` measures the roof with a
  device-to-device *copy* — N bytes read **and** N written. A GEMV is read-only (135 MB in, 76 KB
  out), and GPUs sustain higher bandwidth on pure reads than on mixed traffic, so a copy-derived
  roof understates the pure-read ceiling by a few percent. This is a different phenomenon from the
  **112%** an earlier version of this benchmark reported, which was genuine cache contamination —
  see below.
- **L2 contamination, and how it was removed.** Timing one weight repeatedly leaves it resident in
  L2 and reports cache bandwidth as if it were HBM. The int4 weight is 35 MB; a 4090's L2 is
  75.5 MB and an A100's is 40 MB, so the compressed side fits **entirely in cache on both** — the
  very variant whose claim is that it streams fewer bytes was streaming none of them. Both variants
  now rotate through 9 distinct weights (315 MB int4, 1.2 GB bf16), sized from the device's L2. This
  is also the faithful setup: a decode step reads every layer once and reuses none of it.
- **The baseline still allocates its own output**, because `torch.matmul` does; the int4 path is
  given a reused buffer. That asymmetry flatters the *baseline*, which is the safe direction for
  the claim being made.
- **Independent launches.** The 16 back-to-back launches per timing window use different weights but
  are mutually independent, whereas a real decode step's ~200 kernels are dependent layer to layer.
  Reality has slightly less launch-level parallelism than this measures.
- **One shape, one GPU, synthetic weights.** The MLP up-projection only; attention's QK^T and AV
  matmuls scale with sequence length rather than with weights and belong to a different analysis.
  Accuracy here is measured against the unquantised *synthetic* tensor, so it validates the kernel,
  not the model — T1 is where int4's accuracy on a real Qwen weight was established.

```bash
uv sync
uv run python -m arch_common.probe          # once per session; T6/T7/T8 share the profile
make t8-predict                             # prediction only — no GPU needed
make t8                                     # measure + plot (CUDA + Triton)
uv run pytest topics/t08_gpu_architecture   # packing tests run anywhere; kernel tests need CUDA
```

`make t8-predict` reproduces the prediction table on any machine, with no GPU and no numbers copied
by hand — `bytes/param` comes off the allocated tensors and `weight_share` is read live from T6's
CSV:

```
  bytes/param  2.000 bf16 -> 0.516
  byte ratio   3.88x           <- the kernel's ceiling
  weight share 76.9%           <- T6 error budget
  end-to-end   2.33x           <- Amdahl (T5)
  decode       94.3 -> 219.7 tok/s
```
