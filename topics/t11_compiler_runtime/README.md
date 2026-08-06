# T11 — Compiler & runtime: fusion and graph capture are two wins, not one

> **Status: pre-registered, not yet measured.** The bands below are filed and the harness is
> written, rehearsed and tested; the GPU session has not run. Everything here is a prediction.

**Question:** Compilation is sold as one number. It is two mechanisms with different costs and
**opposite payoffs**. Where does the dominant one flip?

    fusion          merges ops into one kernel, so intermediates stay in registers
                    instead of round-tripping through HBM            → removes BYTES
    graph capture   records the launch sequence once and replays it
                    with a single call                               → removes LAUNCHES

---

## Why this is not T6 again

T6 measured CUDA graphs on a whole vLLM step and found **15–36%**, with the loss growing with
batch. That is a real result and it is a *bundle*: both mechanisms at once, on a stack you do not
control, reported as one number.

T11 unbundles it on a subgraph you do control, and scores each half against a ceiling another topic
already measured. `tests/test_t10_t11_distinctness.py` asserts T11 never emits a step time or a
tokens/sec, so it cannot quietly become T6 with a smaller model.

| | owns | payload | units |
|---|---|---|---|
| T6 | the whole engine, bundled | a 7B model | ms/step, tok/s |
| **T11** | **one weightless subgraph, unbundled** | **7 KB per activation** | µs, kernel launches |
| T10 | the cold path, before any of this | 15.2 GB, once | seconds, page faults |

## The design: a 2×2, and the guard that keeps it one

| | many launches | one launch |
|---|---|---|
| **many round trips** | eager *(baseline)* | `graph` — **launch cost isolated** |
| **few round trips** | `compile` — **fusion isolated** | `compile_graph` |

The whole thing rests on `torch.compile` being able to fuse **without** also capturing graphs. It
can — but not by accident: `mode="reduce-overhead"` turns cudagraphs on inside Inductor, which would
silently bundle the two mechanisms back together while every number still looked plausible. So
`assert_fusion_is_not_secretly_graphs()` checks Inductor's own config flag rather than trusting the
argument, and a unit test calls it. That is the failure this repo cares most about: valid-looking
numbers, invalid conclusion, nothing raising.

## The chain, and why it has no weights in it

    RMSNorm → residual add → SiLU → gate multiply → rotary half-rotation

Five real ops from a transformer's residual path, and **not** the MLP — deliberately. Put a matmul
in the chain and the weight read dwarfs everything: at batch 1 the MLP's two projections read
271 MB while their elementwise epilogue moves 37 KB, so fusion's saving is 0.01% and unmeasurable.
A weightless chain is the only place in a transformer where fusion's bytes *are* the story.

Written as plain PyTorch with no fused primitives, because the compiler has to find the fusion
itself. Calling a pre-fused op would measure NVIDIA's kernel, not the compiler.

## The prediction, derived rather than guessed

Bytes are countable before anything runs. An unfused chain of `k` ops over an activation of `A`
bytes reads and writes it once per op (`2kA`); perfectly fused, it touches only the boundary
tensors (`3A`). Model each mode as launches plus traffic:

    eager    = k·L  +  2k·A/BW
    compile  = 1·L  +   3·A/BW
    graph    =   0  +  2k·A/BW

Compile beats graph when `L + 3A/BW < 2k·A/BW`, i.e. **`A > L·BW / (2k − 3)`**.

| # | Band | Prediction |
|---|---|---|
| 1 | fusion speedup at batch 1 | **≤ 1.1× — invisible** |
| 2 | graph speedup at batch 1 | **≥ 2×** |
| 3 | **crossover batch** | **in [64, 512]**; the model says **173** |
| 4 | fused chain vs T7's memory roof | ≥ 70% |
| 5 | measured fusion vs the byte model | within 1.5× (T9's tolerance) |

Band 1's justification is arithmetic, not intuition: at batch 1 the whole chain is 7,168 B, which at
T7's measured 1,736.7 GB/s is **41 nanoseconds** of traffic against a launch costing microseconds.
Three orders of magnitude of headroom for launch overhead to hide in. **Fusion cannot help what is
not bandwidth-bound.** Graph capture can.

### Band 3's control, and the derivation error it exists to catch

The op count does **not** cancel here, and an earlier version of this model said it did — cancelling
`k` from an equation where fusion removes `2k−3` round trips while capture removes `k` launches.
That predicted a crossover 3.5× too high.

The corrected model makes a sharp, falsifiable claim: **a longer chain crosses over earlier**,
because fusion has more to remove. Going from 5 ops to 2 should move the crossover by a factor of
`(2·5−3)/(2·2−3) = 7`. `make t11-control` re-runs the whole sweep at shorter chain lengths to test
exactly that, and a unit test pins the direction so the claim cannot regress to the wrong one.

A model that survives a 7× prediction is doing real work. One that only fits at a single chain
length is a curve.

## What this is for

Serving stacks apply both mechanisms and rarely say which is doing the work. If the crossover lands
where the model says, the guidance is concrete and regime-dependent:

- **low-batch / latency-sensitive serving** → capture the graph; fusion is noise
- **high-batch / throughput serving** → fuse; the launches are already amortised
- **and the two do not substitute for each other**, which is why engines ship both

It also closes a thread T9 opened deliberately. T9 measured the fixed per-call cost of a
*collective* at ~34 µs and noted that plain kernel launches are the same phenomenon untested. This
is that test.

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
