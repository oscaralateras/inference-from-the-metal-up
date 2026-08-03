# T5 — Parallelism taxonomy: five ways to split a transformer

**What this is:** a from-scratch implementation of **all five ways a large language model is split
across devices** — data, tensor, pipeline, sequence and expert parallelism — each running on real
`torch.distributed` collectives over a real transformer block, measured on the three axes that
decide which one you pick: **throughput**, **bytes communicated**, and **model bytes held per
device**. Plus two supporting experiments: Amdahl's law used as a *measuring instrument* rather
than a plotted formula, and the pipeline bubble measured against its closed form.

**Status:** in progress. Correctness is complete and verified; the CPU rehearsal is green;
canonical throughput numbers await a multi-GPU session (see *What is not measured yet*).

## Reproduce

```bash
uv sync                                                            # once, from the repo root
uv run python topics/t05_parallelism/amdahl.py                     # experiment A
uv run python topics/t05_parallelism/pipeline.py                   # experiment B
uv run python topics/t05_parallelism/strategies.py --backend gloo --world-sizes 1,2,4   # C
uv run python topics/t05_parallelism/strategies.py --backend gloo --strategies ep --routing skewed
uv run python topics/t05_parallelism/plot.py                       # figures
uv run pytest topics/t05_parallelism                               # 45 unit tests
```

The harness is **backend-agnostic**: `--backend gloo` runs on CPU, `--backend nccl` runs the
*identical* strategy code on GPUs. CPU is for correctness and rehearsal; GPU is for the numbers.

---

## Lab note

**Question.** A model no longer fits, or no longer runs fast enough, on one device. There are
exactly five axes you can cut it along. What does each one actually cost — in communication, in
memory per device, and in throughput — and how does each one fail?

### Setup

*In plain terms:* take one real transformer block, split it five different ways across several
processes, and make each split talk over the same machinery real serving engines use. Then measure
not just which is fastest, but what each one is *spending* to get there — because the fastest
option on one axis is often the one that cannot help you at all on another.

- **Model.** One Llama-architecture transformer block — attention **and** SwiGLU MLP, RMSNorm, both
  residuals — with real weights from `JackFram/llama-160m` (hidden 768, 12 heads, intermediate
  3072). Attention is load-bearing here and not decoration: see *Why attention had to be included*.
- **The five strategies**, each on real `torch.distributed` collectives:

  | | what gets split | collectives per block | fails when |
  |---|---|---|---|
  | **DP** data | the batch | none | the model does not fit on one device |
  | **TP** tensor | every weight matrix (heads, intermediate) | 2 × all-reduce | the interconnect is slow |
  | **PP** pipeline | the layers (depth) | 1 × send/recv per seam | microbatches are few (the bubble) |
  | **SP** sequence | the sequence | 1 × all-gather | sequences are short (gather ≈ pure overhead) |
  | **EP** expert | the experts of an MoE | all-gather + all-reduce | routing is uneven |

- **Metrics.** Throughput (tokens/s); **bytes communicated per step**; **model bytes held per
  rank**. The last two are computed from the shapes, so they are *device-independent* — they stay
  true on hardware this study did not rent, which is what lets a CPU rehearsal say something
  honest about GPUs.
- **Correctness.** Every strategy is checked against the unsharded forward every run. A fast wrong
  answer is the easiest thing to produce here, and this check caught two real bugs (below).
- **Hardware.** Correctness and communication volume: any machine. Canonical throughput: pending a
  4× A100 SXM (NVLink) session, with a PCIe node as a controlled comparison.

### Result A — Amdahl's law, measured backwards

Plotting `S = 1/((1-p) + p/n)` proves nothing: it is a closed form, and the curve is whatever you
assert `p` to be. So this runs it the other way. Inject a **known** serial fraction, measure the
real speedup curve, fit Amdahl to the measurement, and ask whether the fit recovers what was put
in. The fit is exact linear least squares via `1/S - 1 = p(1/n - 1)`, not a search.

| injected p | recovered p | error | R² |
|---|---|---|---|
| 1.00 | 0.976 | −0.024 | 0.9993 |
| 0.95 | 0.919 | −0.031 | 0.9947 |
| 0.85 | 0.808 | −0.042 | 0.9904 |
| 0.70 | 0.673 | −0.027 | 0.9943 |

![Measured speedup curves with fitted Amdahl curves](results/amdahl_calibration.png)

The estimator recovers the injected fraction to within 0.04 with R² > 0.99, and is **consistently
biased low** — which is the honest answer, not an error: real parallel overhead (thread start-up,
memory contention) is genuinely there, so the measured curve really is slightly worse than the
injected fraction alone predicts.

Now point the validated estimator at T4's contention curves, where the serial fraction was never
known:

![The same estimator applied to T4's mutex, atomic and sharded curves](results/amdahl_t4.png)

| T4 variant | recovered p | R² |
|---|---|---|
| sharded | **0.954** | 0.9561 |
| atomic | **−3.400** | 0.3651 |
| mutex | **−7.665** | 0.8189 |

**Sharded dispatch fits Amdahl cleanly at p = 0.954.** The mutex and the atomic fit **p < 0** —
they are *outside the model's domain entirely*. Amdahl's floor is 1.0×, because the law assumes
coordination is free: the worst a program can do is refuse to parallelise. T4's contended cases are
worse than that — adding workers made them actively slower. The estimator is deliberately not
clamped to [0, 1], so this shows up as a negative number rather than being quietly rounded to
"perfectly serial."

### Result B — the pipeline bubble

*(Balanced and imbalanced sweeps implemented and running; canonical numbers pending the x86/GPU
session — the development Mac cannot produce honest parallel matmul timings, see Caveats.)*

The prediction under test is `efficiency = M/(M+P-1)` for M microbatches over P stages, and its
generalisation to uneven stages, `M·ΣW / (P·(ΣW + (M-1)·max W))`. Both are unit-tested against
their closed forms.

### Result C — what the five strategies actually cost

Communication volume and per-rank memory are functions of the shapes, so these hold regardless of
device. At world size 4, one step over 8 layers:

| strategy | MB communicated / step | MB held per rank | correctness |
|---|---|---|---|
| **DP** data | **0.00** | 302.0 *(full copy)* | exact |
| **TP** tensor | **100.66** | 75.5 | 1.1e-06 |
| **SP** sequence | 37.75 | 302.0 *(full copy)* | exact |
| **PP** pipeline | 6.29 | 75.5 | exact |
| **EP** expert | 6.29 | 28.3 | exact |

![Communication volume and memory per rank for the five strategies](results/strategies_cost.png)

Two structural facts fall straight out:

**Tensor parallelism moves 16× more data than pipeline parallelism.** TP all-reduces the full
hidden state twice per block; PP hands off an activation once per seam. That single ratio is the
mechanical reason production stacks put **TP inside a node on NVLink and PP across nodes** — it was
never a convention, it is a bandwidth budget.

**DP and SP replicate the entire model.** Both hold 302 MB per rank at every world size, while TP,
PP and EP divide it. So neither can help a model that does not fit on one device, no matter how
many devices you add. Throughput plots alone hide this completely.

**EP's failure mode is different in kind.** With uniform routing every rank does 1/N of the work.
With Zipf-skewed routing the imbalance grows as ranks are added:

| world size | load factor (uniform) | load factor (skewed) |
|---|---|---|
| 1 | 1.00 | 1.00 |
| 2 | 1.00 | **1.78** |
| 4 | 1.00 | **2.93** |

The busiest rank does ~2.9× the average work, and every other rank waits for it at the next
collective. DP/TP/PP/SP all split a *fixed* amount of work into equal pieces; EP's split is decided
by the **router**, at runtime, from the data. Adding hardware does not fix it — which is why MoE
serving stacks fight this with auxiliary load-balancing losses and expert capacity limits.

### Result D — TP degree is not a free parameter

Tensor-parallel degree must **divide the attention head count**, and with grouped-query attention
it is additionally capped at `num_key_value_heads`. Surveying small open models:

| model | heads | KV heads | usable TP degrees |
|---|---|---|---|
| SmolLM2-135M *(T1's model)* | 9 | 3 | 1, 3 |
| Qwen2.5-0.5B | 14 | 2 | 1, 2 |
| TinyLlama-1.1B | 32 | 4 | 1, 2, 4 |
| llama-160m | 12 | 12 | 1, 2, 3, 4, 6, 12 |

This is why production models choose power-of-two head counts, and it is a real deployment
constraint rather than a detail: you cannot simply decide to serve a model at TP=8. It also forced
this study off T1's model — SmolLM2's 9 heads would have pinned the sweep to a single usable point.

### Why attention had to be included

The block here is attention **and** MLP, and that was not for realism. In an MLP every token is
processed independently, so "split the sequence across ranks" and "split the batch across ranks"
are *the same operation* — sequence parallelism would be data parallelism wearing a different name,
and every SP number would be meaningless. **Attention is the only thing in a transformer that mixes
tokens together**, and therefore the only reason SP exists as a distinct strategy with a distinct
communication cost. An MLP-only study cannot measure four of the five honestly.

### Inference payoff

- **The comms ratio explains the deployment topology.** TP's 16× communication premium over PP is
  why TP is confined to a single node on NVLink while PP spans nodes over Ethernet. The rule
  everyone repeats falls out of a number you can compute from the shapes.
- **The memory column is the one that decides "it does not fit."** DP and SP replicate. If a 70B
  model will not fit on one GPU, adding DP replicas cannot help — only TP, PP or EP divide the
  footprint. Throughput comparisons hide this, and it is the first question a serving deployment
  actually has to answer.
- **The bubble explains why PP is wrong for decode.** Efficiency is `M/(M+P-1)`, so PP needs many
  microbatches in flight to amortise its bubble. Decode batches are small by construction (T3) —
  the thing that fills a pipeline is exactly the thing decode does not have. PP buys capacity
  across nodes, not latency.
- **Amdahl is the lens, not the finding.** Every strategy has a non-parallelisable part: PP's
  bubble, TP's blocking all-reduce, EP's imbalance. Fitting `p` to a measured curve turns "it
  scaled badly" into a number — and the T4 result shows where the model itself gives out.
- **Through-line.** T2: decode is a serial dependent chain. T3: so you batch, because decode is
  bandwidth-bound. T4: coordinate the workers by sharing less. **T5: and when one device is not
  enough, here are the five ways to cut the model — each with a different bill.** T9 then measures
  the interconnect those bills are paid on.

### What surprised me

- **Amdahl's law cannot describe contention at all.** I expected the mutex curve to fit with a
  small `p`. Instead it fit p = −7.7, outside the model entirely. Amdahl assumes coordination is
  free, so its floor is 1.0× — the model has no way to express "adding workers made this worse."
  The law is a ceiling on *parallelisable* work, not a general theory of scaling.
- **Splitting the batch and splitting the sequence are not the same thing, and the code did not
  know.** With batch and sequence collapsed into one axis, data parallelism was silently splitting
  the sequence — running fine, producing plausible throughput, and returning wrong output (rel err
  1.1e-2). It only surfaced because every strategy is checked against the unsharded forward.
- **Sequence parallelism replicates the whole model.** I had filed SP mentally next to TP as a
  "model parallel" strategy. It is not: it splits *activations*, not weights, so its per-rank
  memory is identical to DP's. It solves an activation-memory problem, not a model-size one.
- **The 64-byte cache line has an analogue in head counts.** TP degree must divide the head count —
  a hardware-flavoured divisibility constraint sitting in the middle of what looks like a pure
  software choice.

### What is not measured yet

Stated plainly rather than papered over:

- **Canonical throughput.** The throughput column needs a real multi-GPU run. On CPU, "communication"
  is a memcpy through shared memory, so the compute-to-communication ratio is nothing like a GPU's —
  TP in particular looks far better than it would over a real interconnect. **Communication volume
  and per-rank memory are already device-independent and stand as measured; throughput does not.**
- **NVLink vs PCIe.** Planned as a controlled comparison — identical code, two node types.
- **Experiment B's canonical numbers**, for the same reason.

### Caveats

- **The CPU is a testbed for structure, not a stand-in for a GPU.** Everything above that depends on
  compute-to-communication *ratios* is explicitly deferred, not estimated.
- **BLAS matmul does not parallelise across threads on Apple Silicon.** numpy and torch both route
  to the shared AMX co-processor: 4 threads on 4× the matmul work took 3.8× the wall time (1.06× of
  the ideal 4×). On the x86 EPYC box the same test gives 3.65×. Development happened on the Mac;
  no timing number here comes from it.
- **One block, not a full model.** These are the decompositions a serving engine is built from,
  measured on one transformer block. The end-to-end result is a serving artefact, not this one.
- **The MoE is synthetic** — correctly shaped, randomly weighted, top-1 routing. No small open MoE
  has a convenient single-file checkpoint, and EP's behaviour depends on the routing distribution
  and the shapes rather than the weight values.
- **EP over-communicates by construction.** This implementation all-gathers and reduces rather than
  using all-to-all, which is what a production stack does. That isolates the load-imbalance effect
  cleanly but overstates EP's communication volume.
- **Sequence lengths are short.** SP's all-gather cost grows with sequence length, so its true case
  (very long context) is understated here.

---

### CSV contract

All three experiments write `results/parallelism.csv` in one long format; `plot.py` reads it.

```
experiment,variant,workers,metric,value
amdahl_calibration,injected_serial_0.05,4,speedup,3.22
amdahl_calibration,injected_serial_0.05,0,recovered_p,0.919
amdahl_t4,mutex,0,recovered_p,-7.665
pipeline_bubble,balanced,16,efficiency,0.447
strategies_gloo,tp,4,comms_bytes_per_step,100663296
strategies_gloo,ep_skewed,4,load_factor,2.93
```

- `experiment` — `amdahl_calibration`, `amdahl_t4`, `pipeline_bubble`, or `strategies_{backend}`
- `variant` — the injected fraction, the T4 lock variant, the stage layout, or the strategy
- `workers` — world size / thread count / microbatch count. **`0` marks a per-curve summary row**
  (a fitted constant) rather than a per-worker observation
- `metric` — `speedup`, `recovered_p`, `efficiency`, `predicted_efficiency`, `wall_seconds`,
  `tokens_per_s`, `comms_bytes_per_step`, `weight_bytes_per_rank`, `max_rel_err`, `load_factor`
- `value` — the measured number

Writes are idempotent per **(experiment, variant)** pair, so re-running one strategy refreshes its
own rows and leaves its siblings intact.
