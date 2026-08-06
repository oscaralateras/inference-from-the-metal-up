# T11 — Compiler & runtime: fusion and graph capture are two wins, not one

**Question:** Compilation is sold as one number. It is two mechanisms with different costs and
**opposite payoffs**. Where does the dominant one flip?

    fusion          merges ops into one kernel, so intermediates stay in registers
                    instead of round-tripping through HBM            -> removes BYTES
    graph capture   records the launch sequence once and replays it
                    with a single call                               -> removes LAUNCHES

**Setup:** 1× NVIDIA A100-SXM4-80GB, driver 580.159.04, torch 2.8.0+cu128. Measured memory
bandwidth **1,737.1 GB/s** — the roof every figure below is scored against, and within **0.02%** of
T6/T7/T8's 1,736.7. Session `7e6c8b9fba8e`, shared with T10. Chain shape is Qwen2.5-7B's hidden
size (3584), bf16, as in T5–T9.

---

## Result

![the crossover](results/mechanism_crossover.png)

**Both mechanisms work, neither substitutes for the other, and which one dominates flips at batch
648.**

| batch | eager | compile | graph | compile+graph | fusion | graphs | both |
|---|---|---|---|---|---|---|---|
| 1 | 141.5 µs | 95.4 | 25.8 | **8.6** | 1.48× | **5.50×** | **16.4×** |
| 8 | — | — | — | — | 1.88× | 5.01× | 20.2× |
| 32 | — | — | — | — | 1.85× | 3.91× | 19.4× |
| 128 | — | — | — | — | 1.85× | 3.50× | 19.0× |
| 512 | — | — | — | — | 1.84× | 2.18× | 11.9× |
| 2048 | 249.7 µs | 91.7 | 233.7 | **53.0** | **2.72×** | 1.07× | 4.7× |

At batch 1 graph capture is worth **5.50×** and fusion **1.48×**. At batch 2048 they have swapped:
fusion **2.72×**, capture **1.07×** — capture has become worth almost nothing. Applying both is
worth **up to 20×**, which is more than either alone at every batch size measured.

| # | pre-registered band | predicted | measured | verdict |
|---|---|---|---|---|
| 1 | fusion speedup at batch 1 | ≤ 1.1× | **1.48×** | **OUTSIDE** ✗ |
| 2 | graph speedup at batch 1 | ≥ 2× | **5.50×** | WITHIN ✓ |
| 3 | crossover batch | in [64, 512] | **648** | **OUTSIDE** ✗ |
| 4 | fused chain vs memory roof | ≥ 70% | **27.6%** | **OUTSIDE** ✗ |
| 5 | measured fusion vs byte model | within 1.5× | **0.82×** | WITHIN ✓ |

**Three of five failed.** Two of the three failures have clean mechanical explanations that the run
itself produced, and the third — band 3's chain-length control — genuinely refuted part of the
model. All three are more useful than the passes.

---

## The crossover is real, and the model's *input* was what was wrong

Band 3 registered a crossover in [64, 512], from a model fed an **assumed** 5 µs launch cost —
because nothing in this repo had ever measured one for a plain kernel. T9 measured a *collective's*
fixed cost at ~34 µs and explicitly left this untested.

This run can measure it. The gap between eager and graph replay is exactly the launch cost capture
removed:

    L = (eager − graph) / ops = (141.5 − 25.8) / 5 = 23.15 µs per chain op

Feeding that into the **same unchanged formula** predicts batch **801** against a measured **648** —
**24% out**, against the 3.7× the assumed input was out by. The model's structure survives; the
number it was given did not.

(Per *kernel* rather than per chain op it is 10.5 µs, because eager runs 11 kernels for a 5-op
chain — see below, that discrepancy matters.)

## The control refuted the chain-length prediction, and that is the most useful result here

The model says fusion removes `2k−3` round trips while capture removes `k` launches, so a shorter
chain should cross over **later** by exactly `(2·5−3)/(2·2−3)` = **7×**. `make t11-control` re-runs
the whole sweep at 2 and 3 ops to test that.

| chain | crossover | model predicted | measured shift |
|---|---|---|---|
| 5 ops | 648 | — | — |
| 3 ops | **631** | 2.33× later | **0.97×** |
| 2 ops | **748** | 7.00× later | **1.16×** |

**The crossover barely moves.** The prediction is refuted — not marginally, but by a factor of six.

The data says why, and both halves of the model were wrong in offsetting directions:

|  | model said | measured (2 → 3 → 5 ops) |
|---|---|---|
| `compile` time | constant in `ops` (one launch + boundary traffic) | **62.3 → 63.3 → 91.7 µs** — grows |
| `graph` time @2048 | linear in `ops` (`2k·A/BW`) | **135.3 → 156.2 → 233.7 µs** — sublinear |

`compile` is not constant because Dynamo's guard and dispatch cost grows with the traced graph.
`graph` is sublinear because each kernel carries a fixed device-side cost on top of its traffic. The
crossover is where those two curves meet, and since both rise with `ops`, the intersection barely
moves. **Two modelling errors that happen to cancel.**

Worth being precise about what this does and does not vindicate. An earlier draft of the derivation
cancelled `ops` and predicted no chain-length dependence — which is what the data shows. That draft
was not right; it reached the correct conclusion by asserting `compile = 1·L + 3A/BW`, which the
measurements above show is false. **Getting the answer from wrong premises is not a validated
model**, and the note is not going to claim it was.

What survives is empirical and still useful: **the crossover sits at 630–750 and is insensitive to
chain length over 2–5 ops.**

## Band 4: the chain does not fully fuse, and the kernel counts prove it

Registered at ≥ 70% of the memory roof; measured **27.6%**. Two separate things are wrong with that
number, and the run diagnosed both.

**First, it was scored against the wrong mode.** `compile` runs the fused kernel *and* pays Dynamo's
guard evaluation on every call. That cost is roughly constant, which is why `compile` times barely
move across four decades of batch (95.4 µs at batch 1, 91.7 µs at batch 2048 — for 2,048× the work).
The same kernel graph-replayed reaches **831.7 GB/s = 47.9% of roof**, not 27.6%. **The band stays
failed as registered** — rescoring after seeing the data is the thing this repo exists not to do —
but 20 percentage points of it were framework overhead, not the fuser.

**Second, and this is the real finding: the 5-op chain does not fuse into one kernel.**

| chain | eager kernels | after `torch.compile` | fused bandwidth @2048 | vs roof |
|---|---|---|---|---|
| 2 ops | 7 | **1** | 1,592 GB/s | **91.7%** ✓ |
| 3 ops | 8 | **1** | 1,391 GB/s | **80.1%** ✓ |
| 5 ops | 11 | **2** | 832 GB/s | **47.9%** ✗ |

The fifth op is a rotary-style `chunk` then `cat`, and Inductor emits a **second kernel** for it —
the concatenation materialises rather than fusing away. One extra HBM round trip, and the achieved
bandwidth halves.

So band 4 fails on the 5-op chain and would have **passed comfortably on either shorter one**. That
is a statement about which ops fuse, not about how well Inductor fuses the ones that do: where the
chain collapses to a single kernel, it reaches **80–92% of the measured memory roof**.

## Band 1: fusion helps more than bytes alone predict

Registered at ≤ 1.1× — the reasoning being that at batch 1 the whole chain is 7,168 B, about
**41 nanoseconds** of traffic at 1,737 GB/s, so there is essentially no traffic for fusion to remove.
Measured **1.48×**.

The arithmetic was right and the model was incomplete. Fusion does not only remove HBM round trips;
it removes **kernels**, and each kernel eager-dispatched costs Python and dispatch time on the host.
Eager runs **11** kernels for this chain and `compile` runs **2**. At batch 1, that is the whole
effect — the byte model predicted only the part of fusion that is about bytes.

This also explains part of why band 3's chain-length prediction failed: the model's `ops` (5) and
the real kernel count (11) are not the same number, and they do not scale together.

## What this is for

The practical shape is regime-dependent, which was the hypothesis and is what the data shows:

- **Low-batch, latency-sensitive serving** → **capture the graph**. Worth 5.5× at batch 1; fusion is
  worth 1.5×. This is where a chat endpoint at batch 1–32 lives.
- **High-batch, throughput serving** → **fuse**. Past batch ~650 capture is worth almost nothing
  (1.07× at 2048) while fusion is worth 2.7×.
- **Do both** — 16–20× across the low and middle batches, more than either alone everywhere.
- **Check that your chain actually fused.** A single non-fusing op in the middle costs half the
  achievable bandwidth, and nothing warns you. The kernel count does.

It also closes a thread T9 opened deliberately: T9 measured a *collective's* fixed per-call cost at
~34 µs and noted plain kernel launches were the same phenomenon, untested. Measured here at
**10.5 µs per kernel** on the same class of hardware.

## Why this is not T6 again

T6 measured CUDA graphs on a whole vLLM step and found 15–36%, with the loss growing with batch.
Same direction as band 2 here, and it is a *bundle*: both mechanisms at once, on a stack you do not
control, reported as one number.

| | owns | payload | units |
|---|---|---|---|
| T6 | the whole engine, bundled | a 7B model | ms/step, tok/s |
| **T11** | **one weightless subgraph, unbundled** | **7 KB per activation** | µs, kernel launches |
| T10 | the cold path, before any of this | 15.2 GB, once | seconds, page faults |

`tests/test_t10_t11_distinctness.py` asserts T11 never emits a step time or a tokens/sec, so it
cannot quietly become T6 with a smaller model.

## Method notes

**The 2×2 only works if `compile` does not secretly capture graphs.** `mode="reduce-overhead"` turns
cudagraphs on inside Inductor, which would put both mechanisms in the `compile` cell and invalidate
every attribution here while all the numbers still looked plausible.
`assert_fusion_is_not_secretly_graphs()` checks Inductor's own config flag at runtime rather than
trusting the mode argument, and a unit test calls it.

**The chain is weightless deliberately.** Put a matmul in it and the weight read dwarfs everything:
at batch 1 the MLP's projections read 271 MB while their elementwise epilogue moves 37 KB, so
fusion's saving would be 0.01% and unmeasurable.

**Kernel counts come from the profiler in a separate pass** from the timing, because the profiler's
overhead inflates wall-clock badly. They are the mechanism evidence that does not depend on a timer
being fair — and they are what caught the 5-op chain emitting two kernels.

**16 back-to-back calls per timed window**, as in T8 and T9: a decode step runs this chain 28 times
in sequence and never pays for one in isolation.

## Caveats

- **`compile` is host-bound here, and that bounds what this says about fusion.** ~90 µs per call of
  guard and dispatch overhead sits on top of every `compile` measurement, and on a machine with a
  faster host CPU the fusion column would look better and the crossover would move. The container
  reported 255 CPUs while being limited to 16, which will not have helped.
- **The crossover is a property of this hardware pairing**, not a universal batch size. It is where
  a ~90 µs host-side cost meets a rising traffic cost; both terms are machine-specific.
- **Chain lengths 2, 3 and 5 only.** The refutation of the length-scaling prediction rests on three
  points, and the two shorter chains both fuse to one kernel while the longest does not — so chain
  length and fusion completeness are confounded in this data. Separating them needs a 5-op chain
  whose ops all fuse, which is the next thing I would measure.
- **One chain shape, one dtype, one hidden size.**
- **Inductor's default mode only.** `max-autotune` was not tried; it would change the fusion column
  and not the graph one.

## Reproduce

Off the clock, on any machine:

```bash
uv sync
uv run python -m topics.t11_compiler_runtime.predict --bandwidth 1736.7 --write
make t11-rehearse    # CPU, graph modes skipped, numbers not published
uv run pytest topics/t11_compiler_runtime tests
```

The rehearsal needs `setuptools` — `torch.compile` on CPU builds a C++ wrapper through Inductor,
which imports it at codegen time and which torch does not declare. It is in the dev group for that
reason: a rehearsal that cannot run means the first execution of this code is the one being paid for
by the hour.

On a single GPU — see [`scripts/t10_t11_session.sh`](../../scripts/t10_t11_session.sh), which shares
one pod and one hardware probe with T10:

```bash
bash s.sh setup     # clone + uv sync + make probe
bash s.sh t11       # the 2x2 across the batch sweep, then the crossover
bash s.sh control   # band 3's falsification run at shorter chain lengths
```

Every number quoted here is asserted against `results/compiler.csv` by the `test_lab_note_*` tests
in `test_t11.py`, so the prose cannot drift from the data.
