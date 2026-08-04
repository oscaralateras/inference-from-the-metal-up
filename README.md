# inference-from-the-metal-up

**Why does an LLM generate tokens as slowly as it does?** Not the hand-waving answer — the measured
one. This repo works up from the hardware: one small, reproducible microbenchmark per
computer-architecture topic, each ending in a concrete, mechanical explanation of a real inference
behaviour.

Every result here was produced by running something. No topic is included unless a measurement in
this repo — not a citation — carries the argument.

![Decode is memory-bound on a GPU: GEMV vs GEMM across batch size](topics/t03_memory_hierarchy/results/crossover_gpu.png)

*The thesis in one plot (T3): on a GPU, single-token decode sits at ~77% of HBM bandwidth and is
unambiguously memory-bound — so its speed ceiling is bandwidth ÷ weight-bytes, and no amount of extra
compute moves it. Batching is what escapes the wall.*

## Shipped topics

| # | Topic | Headline finding | Inference payoff |
|---|-------|------------------|------------------|
| [T1](topics/t01_number_representation) | Number representation & quantization | Granularity is worth ~2–3 bits. Per-tensor int4 loses **71%** of the layer output (cos 0.79); per-group holds **11%** (cos 0.99). | Why production int4 needs per-group scales — and why GPTQ/AWQ exist. One outlier sets the scale for all 885k weights. |
| [T2](topics/t02_cpu_pipeline) | CPU execution & the pipeline | Fitting the pipeline is worth **1.8×–3.2×** on identical arithmetic: branch 1.76×, ILP 3.20×, SIMD 2.35×. | Decode lands on the wrong side of all three. The serial token dependency is why batching exists. |
| [T3](topics/t03_memory_hierarchy) | Memory hierarchy & the memory wall | Traversal order is worth **~20×** — and the memory-bound crossover is set by the device's **ridge point**, not by the kernel. | Decode is memory-bound *specifically on GPUs* (ridge ≈ 200 vs CPU ≈ 0.18). Batching raises arithmetic intensity; that is continuous batching's whole lever. |
| [T4](topics/t04_concurrency) | Concurrency & synchronization | Coordination taxes, all on identical work: memory layout alone **17×**, an unsynchronised counter is **78% wrong**, a locked queue costs **456×** vs sharding. | A single-lock scheduler caps tokens/sec no matter how much GPU you attach. Real engines shard dispatch — this is why. |
| [T5](topics/t05_parallelism) | Parallelism: five ways to split a transformer | Scaling order is **not** communication order. On 4× A100 NVLink: DP **3.77×** (0 MB/step), TP **2.96×** (940 MB), PP **2.48×** (59 MB). PP is capped by its bubble at 2.91× before a byte moves; TP's bandwidth is only ~4% of its step. Routing skew alone costs EP **55%**. | Communication *volume* doesn't predict communication *cost*, let alone scaling. And DP/SP scale best while replicating the whole model — so neither can serve one that doesn't fit. |
| [T7](topics/t07_roofline) | Roofline model & arithmetic intensity | Decode runs at **0.5%** of an A100's compute — and that is near-optimal, not wasteful: it hits **82%** of the *memory* roof it is actually under. Batching walks it from 1 to 236 FLOPs/byte, **×105 throughput to batch 128**, then **×1.04** to 256 once it crosses the ridge at 152. | Decode is memory-bound by two orders of magnitude, so quantisation (which raises intensity) beats buying FLOP/s. Batch until the ridge; the ceiling is knowable before you run anything. |

## Roadmap

Eleven topics, built in order. Five shipped, two built and awaiting their GPU session;
the rest are scoped and planned.

| | Topic | Language | Status |
|---|---|---|---|
| T1 | Number representation / quantization | Python | ✅ shipped |
| T2 | CPU execution & the pipeline | C | ✅ shipped |
| T3 | Memory hierarchy | C + PyTorch | ✅ shipped |
| T4 | Concurrency & synchronization | Rust | ✅ shipped |
| T5 | Parallelism: five ways to split a transformer | Python · 4× GPU | ✅ shipped |
| T6 | Performance reasoning | Python · GPU | 🔧 built, awaiting GPU run |
| T7 | Roofline model & arithmetic intensity | Python · GPU | ✅ shipped |
| T8 | GPU architecture (tiled matmul) | Triton · GPU | planned |
| T9 | Interconnects & multi-device | Python · multi-GPU | planned |
| T10 | OS & virtual memory | C + Python | planned |
| T11 | Compiler / runtime layer | Python · GPU | planned |

## How these are measured

The numbers are only worth reading if the method is honest, so:

- **Author on the Mac; measure on real hardware.** Canonical CPU numbers come from an **AMD
  EPYC-Milan** box (Hetzner CCX33, Ubuntu 24.04, 64-byte cache line); GPU results from a T4 (T3)
  and **4× A100 SXM on NV12 NVLink** (T5). `rdtsc`, cache sizes,
  cache-line width and branch-prediction behaviour all differ on Apple Silicon, which would make the
  numbers non-canonical. Every lab note states the hardware it ran on.
- **Use the language that tells the truth.** C and Rust where cycle-level effects matter (T2–T4) —
  Python overhead would swamp a 0.36 ns/op result. Python everywhere else. Every benchmark emits CSV
  that a Python `plot.py` consumes, so the analysis layer stays uniform across languages.
- **Defeat the optimiser, then prove it.** Benchmarks carry checksums and sinks so the compiler
  cannot delete the work being timed, and fail loudly on checksum mismatch rather than reporting a
  fast lie.
- **Every lab note states what surprised me and what the result does *not* show.** The caveats are
  load-bearing: baseline-not-SOTA, single-tensor, microbenchmark-not-server. Several findings here
  contradicted what I expected going in, and are written up that way.
- **CI on every push:** ruff, pyright, pytest, plus `cargo fmt`, `clippy -D warnings`, `cargo test`.

## Layout

Each topic is self-contained in `topics/tNN_.../`: a `README.md` lab note (question → setup → result
→ headline finding → inference payoff → surprises → caveats), the benchmark source, a `plot.py`, and
a committed `results/` folder with the CSV and figures.

```bash
uv sync        # environment (Python 3.12, pinned)
make ci        # everything CI runs: lint, format, types, tests
```

Per-topic reproduce steps are in each lab note.

## License

MIT — see [`LICENSE`](LICENSE).
