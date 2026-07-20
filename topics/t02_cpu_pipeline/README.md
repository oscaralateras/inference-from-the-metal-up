# T2 — CPU execution and the pipeline

**Artefact (a02):** a C microbenchmark measuring the real cost of **branch misprediction** (sorted
vs unsorted data through a branchy loop) and **instruction-level parallelism** (a serial dependent
chain vs independent accumulators), tying both back to why CPU token/sampling loops are slow relative
to GPU throughput.

**Status:** in progress — scaffold only. Numbers below are placeholders until measured on Linux x86.

---

## Lab note

**Question.** What do a branch misprediction and an instruction-level dependency actually *cost* in
cycles, and why does that make branchy CPU loops slow?

**Setup.**
- **Experiment (a) — branch prediction:** an array of random ints; a loop summing only the elements
  above a threshold. Time the loop on a **sorted** vs **unsorted** copy of the same data — identical
  work, but the branch is perfectly predictable when sorted and ~50/50 when shuffled.
- **Experiment (b) — ILP:** the same number of multiply-adds arranged as a **serial dependent chain**
  (`acc = acc*a + b`, each step waits on the last) vs **N independent accumulators** (the CPU can
  overlap them). This exposes the latency-vs-throughput gap of a single execution unit.
- **Method:** compiled `-O2`; a `volatile` sink / returned checksum prevents the optimizer from
  eliding the work; warm up, repeat, take the median; emit CSV.
- **Hardware:** **Linux x86** — authored on the Mac (Apple Silicon), but ARM mutes the
  sorted-vs-unsorted effect and lacks `rdtsc`, so the canonical numbers come from an x86 box. The CPU
  model and clock are recorded with the results.
- **Reproduce:** `make run` (writes `results/pipeline.csv`), then `uv run python plot.py`.

**Result.** _(TBD — fill in once measured on Linux x86.)_

**Headline finding.** _(TBD.)_

**Inference payoff.** _(TBD — frames CPU-vs-GPU throughput and why branchy per-token sampling/decode
loops don't map well onto latency-bound CPU execution.)_

**What surprised me.** _(TBD.)_

**Caveats.** _(TBD.)_

---

### CSV contract

`bench.c` writes `results/pipeline.csv`; `plot.py` reads it. Agreed columns:

```
experiment,variant,n,ns_per_elem,checksum
branch,unsorted,...
branch,sorted,...
ilp,dependent,...
ilp,independent,...
simd,scalar,...
simd,vector,...
```

- `experiment` — `branch`, `ilp`, or `simd`
- `variant` — `sorted`/`unsorted` (branch), `dependent`/`independent` (ilp), or `scalar`/`vector` (simd)
- `n` — number of elements processed
- `ns_per_elem` — median nanoseconds per element (the headline number)
- `checksum` — the sink value, printed so the compiler can't elide the loop and so runs are verifiable
