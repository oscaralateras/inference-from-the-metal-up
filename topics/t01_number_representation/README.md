# T1 — Number representation / floating point / quantization foundation

**Artefact (a01):** a symmetric integer quantizer built from scratch, used to measure **quantization
error vs. bit-width and granularity** on a real LLM weight tensor — reproducing, from first
principles, the **outlier problem** that forces per-channel/per-group quantization.

**Status:** complete.

---

## Lab note

**Question.** How does quantization error scale with bit-width (8 → 2) and granularity (per-tensor
vs per-channel vs per-group), and where do weight outliers break it?

**Setup.**
- **Tensor:** `model.layers.0.mlp.gate_proj.weight` from `HuggingFaceTB/SmolLM2-135M`, shape
  `(1536, 576)`, read straight from safetensors as float32. `max|w| = 3.25` vs `mean|w| = 0.15` — a
  ~22× outlier that turns out to be the whole story.
- **Method:** symmetric integer quantization (round-to-nearest) with absmax calibration — the
  simplest textbook baseline, chosen deliberately to expose *where the naive method fails*.
- **Sweep:** bit-widths `{8,7,6,5,4,3,2}` × granularities `{per-tensor, per-channel, per-group
  (group_size = 64)}`.
- **Metrics:** SQNR (dB) and cosine similarity — the two standard, independent layer-level fidelity
  axes (magnitude and direction) — plus relative error (%) as the human-readable form of SQNR. All
  applied to the weights *and* to the layer output `W @ x` vs `Ŵ @ x` on a fixed random Gaussian
  input `x` (512 columns, seed 0). Output fidelity is what inference actually experiences; weight
  error is only a proxy for it.
- **Hardware:** CPU only — this measures a numerical *property* of the weights, not inference speed,
  so no GPU is needed.
- **Reproduce:** `uv run python topics/t01_number_representation/probe.py` (table) and
  `uv run python topics/t01_number_representation/plot.py` (figures).

**Result.**

| n_bits | granularity | out SQNR (dB) | out error % | out cosine |
|---|---|---|---|---|
| 8 | per-tensor | 27.9 | 4.0 | 0.9992 |
| 8 | per-group | 44.3 | 0.6 | 1.0000 |
| 4 | per-tensor | **3.0** | **70.9** | **0.790** |
| 4 | per-channel | 16.7 | 14.7 | 0.9894 |
| 4 | per-group | 19.1 | 11.1 | 0.9939 |
| 3 | per-tensor | 0.1 | 98.9 | 0.212 |
| 2 | per-tensor | 0.0 | 99.9 | 0.053 |

![Output SQNR and cosine vs bit-width, per granularity](results/quality_vs_bits.png)

*Output fidelity vs bit-width, magnitude (SQNR, left) and direction (cosine, right). Per-tensor
(orange) sits well below per-channel/per-group across the whole range; its cosine holds near 1.0
until ~4 bits and then falls off a cliff, while the finer schemes stay usable down to ~3 bits.*

![Weight-magnitude histogram showing the outlier that sets the scale](results/weight_outliers.png)

*Distribution of `|weight|` (log-count). The black dotted line is the single largest weight (3.25),
which alone sets the per-tensor scale; the red dashed line is the resulting int4 step (3.25 / 7 ≈
0.46). Almost the entire mass of the 885k weights sits **left of the red line** — below one
quantization step — so per-tensor int4 rounds them to 0 or ±1. One outlier, stranded far out on its
own, wastes the whole 16-level grid.*

**Headline finding.** *Granularity barely matters at int8 but is the difference between broken and
usable at int4.* Naive per-tensor int4 produces **71% output error (cosine 0.79 — direction
destroyed)**; simply switching to per-group recovers it to **11% error (cosine 0.99)**. On the
quality-vs-bits curve, per-group at 3 bits matches per-tensor at ~5.5 bits — **granularity is worth
~2–3 bits.** Two secondary findings: (1) weight SQNR tracks output SQNR almost exactly, so weight
error is a faithful (and cheap) proxy here; (2) worst-case (max-abs) error is nearly constant across
granularities — it would have *hidden* the collapse, which is why the bulk metrics (SQNR, cosine)
are the ones that matter.

**Inference payoff.** This is precisely why production int4 quantization uses per-group scales, and
the motivation behind **GPTQ / AWQ / SmoothQuant**. The histogram shows the mechanism directly: a
single outlier at 3.25 sets the per-tensor scale, making the int4 step ≈ 0.46, and the entire bulk
of weights (mean 0.15) sits below one step — so they round to 0/±1 and vanish. Per-channel/per-group
quarantine the outlier to its own channel/group, so every other channel keeps a fine scale. A
~40-line round-to-nearest baseline reproduces the motivation for the entire modern weight-
quantization literature.

**What surprised me.** Three things caught me off guard:
- **The collapse is a cliff, not a slope.** I assumed int4 would be "a bit worse" than int8. Instead
  per-tensor int4 lost 71% of the output and cosine fell to 0.79 — the layer basically stopped
  pointing where it should. Quality doesn't degrade gently as you drop bits; below ~5 bits per-tensor
  falls off an edge.
- **One number does all the damage.** I understood the outlier problem in the abstract, but the
  histogram made it visceral: a single weight at 3.25, while 99% of weights sit under 0.5, sets the
  scale for all 885k of them and wastes almost the entire int4 grid. It's the clearest "skyscraper
  and an ant on one ruler" example I've built.
- **The metric you pick changes the story.** Max-abs error barely moved between per-tensor and
  per-channel — if that had been my only metric, I'd have wrongly concluded per-tensor was fine. It
  took the bulk metrics (SQNR, cosine) to expose the collapse. A good reminder not to trust a single
  number.

The lasting takeaway: I now understand *why* GPTQ/AWQ exist, not just *that* they do.

**On the metrics.** SQNR (dB) and cosine are the standard layer-level fidelity metrics; relative
error (%) is the same magnitude axis as SQNR in readable units. The metric that *ultimately* matters
for LLM quantization is downstream **perplexity / task accuracy**, which requires running the full
model — the natural next step beyond this single-tensor study. Weight/output fidelity here is the
honest, cheap proxy for it.

**Caveats.**
- **Baseline method, not SOTA:** round-to-nearest + absmax. GPTQ/AWQ do meaningfully better at low
  bits; the point here was to show the baseline's cliff, not to compete with them.
- **Random-input proxy:** output fidelity uses a random Gaussian `x`. Real activations have their own
  distribution and outliers, so real-world output error could differ.
- **One tensor, one small model:** this is a layer-level fidelity study, not a full-model
  perplexity/accuracy result.
- **Quality only:** memory and latency (the *payoff* of quantization) aren't measured here — that's
  deliberately deferred to the roofline (T7) and serving artefacts.
