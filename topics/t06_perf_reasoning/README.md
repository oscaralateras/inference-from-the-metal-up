# T6 — Performance reasoning

**Question:** Can I predict an LLM's decode throughput from first principles, and account for
every token I don't get?

**Status:** built, tested and rehearsed end-to-end on CPU. Awaiting the GPU session — the numbers
below are the method, not the results.

---

## Why the prediction is not the point

`tokens/sec = bandwidth / bytes_per_token` is a five-line calculation. Anyone can do it, and on its
own it proves nothing except that you have read a blog post.

The finding is the **error budget**: taking the measured per-token time and attributing it to named
causes until what remains is small enough to admit is unexplained. That is the skill that separates
reasoning about inference performance from reading `nvidia-smi`.

## The arithmetic

For a model with `P` parameters at `b` bytes each, decoding one token at batch 1:

| quantity | formula | why |
|---|---|---|
| FLOPs / token | `2P` | one multiply and one add per weight |
| bytes / token | `P × b` | each weight **read** once; the 2 above is compute, not traffic |
| arithmetic intensity | `2 / b` | **1.0 FLOP/byte** in bfloat16 |

An A100's ridge point is ~170 FLOPs/byte. Decode sits at 1. It is memory-bound by two orders of
magnitude, and no amount of extra compute would help.

Two refinements that the textbook version omits and that `model_math.py` gets right:

- **The input embedding is a lookup, not a matmul.** One row of the table is read, not the whole
  matrix. The LM head *is* a matmul and is fully read. Counting both inflates bytes/token by ~7% on
  a 7B model.
- **The KV cache grows with context.** `2 × layers × kv_heads × head_dim × seq_len × b`. Unlike the
  weight term this is not constant, which is why decode measurably slows as a sequence lengthens.

## The decomposition

Times add; throughputs do not. So the gap is closed in the time domain:

```
measured_ms  =  weights + kv_cache + activations + launch_overhead + unexplained
```

The first three are bytes ÷ the session's **measured** bandwidth. The fourth is measured
per-launch cost × module calls per token. The residual is reported, not absorbed.

## Pre-registered, before the GPU run

`predict.py` writes `results/predictions.json` and it is **committed in its own commit, before**
the measurement commit. The git history is the evidence that predict-then-measure was real rather
than reconstructed. Two bands are fixed at the same time:

| band | value | what falsification means |
|---|---|---|
| naive prediction vs measured | within **2×** | the byte model is wrong, not the GPU |
| unexplained residual | ≤ **25%** | the decomposition is missing a real term |

`tests/test_distinctness.py` asserts the prediction is pinned to the same session as the results.
The CPU rehearsal reports both bands as **OUTSIDE** — correct, since a 1M-parameter toy model is
pure launch overhead. A harness that cannot fail is not measuring anything.

## Reproduce

```bash
# once per session — both T6 and T7 read these ceilings
python -m arch_common.probe

python -m topics.t06_perf_reasoning.predict --model Qwen/Qwen2.5-7B   # then COMMIT
python -m topics.t06_perf_reasoning.measure --model Qwen/Qwen2.5-7B
python -m topics.t06_perf_reasoning.decompose
python -m topics.t06_perf_reasoning.plot
```

Rehearsal on any laptop, no GPU and no weights of consequence:

```bash
python -m arch_common.probe --device cpu
TINY=hf-internal-testing/tiny-random-LlamaForCausalLM
python -m topics.t06_perf_reasoning.predict --model $TINY --seq-len 64
python -m topics.t06_perf_reasoning.measure --model $TINY --device cpu --seq-len 64 \
    --tokens 40 --batches 1 2 4 --contexts 64 128 256
python -m topics.t06_perf_reasoning.decompose && python -m topics.t06_perf_reasoning.plot
```

## Figures

| figure | shows |
|---|---|
| `error_budget.png` | where every millisecond of a decode step goes, next to the measured total |
| `serving_tradeoff.png` | throughput against p99 latency across batch sizes — the real serving decision |
| `context_decay.png` | tokens/sec falling as the KV cache grows |

## Measurement discipline

- **CUDA events, not `perf_counter`.** Kernel launches are asynchronous; a host timer measures the
  launch and reports numbers ~100× too fast. Every GPU timing goes through `arch_common.timing`.
- **Each decode step is timed once and its output reused.** Running the step twice would append a
  duplicate KV entry and silently corrupt the context — wrong, not merely wasteful.
- **Warmup before timing.** Decode shapes differ from prefill shapes, so cuBLAS has not autotuned
  for them and the first tokens are unrepresentative.
- **Median, not mean**, for the central value; **nearest-rank p99** for the tail, with the sample
  count stated — p99 over 32 samples is just the maximum wearing a hat.

## What is not measured

- **Module calls are a lower bound on kernel launches.** Forward hooks cannot see elementwise ops,
  fused internals or anything in functional form. The overhead term is therefore a floor.
- **Activation traffic is an estimate**, not a profile. Precise accounting needs kernel-level
  tooling; at batch 1 the term is small enough that the estimate does not carry the argument.
- **One model, one GPU, one precision.** No claim beyond that.
- **No continuous batching or paged attention** — this is a static batch, so the throughput figures
  are a floor relative to a real serving stack.

## Relationship to T7

T6 owns the **time** domain: whole model, real KV cache, wall-clock. T7 owns the **shape** domain:
isolated matmuls, FLOPs per byte, position against the ridge. Both read their hardware ceilings
from the same `results/hardware.json`, and `tests/test_distinctness.py` fails CI if the two topics'
metric vocabularies overlap or if they were measured in different sessions.

T6 predicts a number and finds a gap. T7 draws the map that explains the gap's shape.
