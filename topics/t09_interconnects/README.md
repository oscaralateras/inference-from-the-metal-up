# T9 — Interconnects: what a collective costs when the message is small

**Question:** T5 measured tensor parallelism end to end and found that its communication *volume*
does not explain its cost — 940 MB per step is only ~4% of the step at NVLink bandwidth, yet TP
scaled worst of the three dense strategies. T5 called the loss "frequency and shape" rather than
volume. **So what does one all-reduce actually cost, and how much of that cost is there before it
moves a single byte?**

**Setup:** 4× NVIDIA A100-SXM4-80GB, **NV12 NVLink between every pair** (12 bonded links), NCCL
2.27.3, torch 2.8.0+cu128, bfloat16. Measured memory bandwidth **1,739.2 GB/s** — T6, T7 and T8 ran
at 1,737 GB/s, so this is the same silicon to within 0.1% and the numbers below compose with theirs
directly. Session `f3f296a657bc`. Model shape throughout is Qwen2.5-7B (hidden 3584, 28 layers), as
in T5–T8.

---

## Result

![the cost curve](results/allreduce_cost.png)

**A decode all-reduce is 99.9% fixed cost, and that fixed cost has almost nothing to do with the
interconnect.**

A collective is priced as `t(n) = α + n/β`. Fitted on this node:

| | α (fixed) | β (bandwidth) | R² | crossover | β vs NV12 spec |
|---|---|---|---|---|---|
| 2 GPUs | **35.45 µs** | 201.5 GB/s | 0.9997 | 6,978 KB | 67.2% |
| 4 GPUs | **35.80 µs** | 221.7 GB/s | 0.9999 | 5,168 KB | 73.9% |

Decode at batch 1 sends **7,168 B**. That is three orders of magnitude below the crossover, so the
call costs 35.85 µs of which **99.9% is α**. It achieves **0.300 GB/s of bus bandwidth — 0.135% of
the 221.7 GB/s the same link delivers to a large message.**

Fifty-six of those per token (2 per layer × 28 layers) is **2.01 ms**, against T6's measured 11.05 ms
step: an **18.2% tax** on a step that tensor parallelism is supposed to be shrinking.

| pre-registered band | predicted | measured | verdict |
|---|---|---|---|
| 1. α(4)/α(2), from ring `2(N-1)` | 3.0 ± 30% | **1.01** | **OUTSIDE** ✗ |
| 2. β vs NV12 spec, world 2 | ≥ 70% | 67.2% | **OUTSIDE** ✗ |
| 2. β vs NV12 spec, world 4 | ≥ 70% | **73.9%** | WITHIN ✓ |
| 3. batch-1 call is fixed cost | ≥ 90% | **99.9%** | WITHIN ✓ |
| 4. model vs real TP layer, world 4 | within 1.5× | 0.83–1.06× | WITHIN ✓ |
| 4. model vs real TP layer, world 2 | within 1.5× | **0.47–0.66×** | **OUTSIDE** ✗ |

Three of six failed. Band 1 failed by the widest margin and turned out to be the most useful thing
here, so the rest of this note is mostly about it.

---

## Band 1: α did not scale, and the reason is not the ring

The ring model says an all-reduce costs `2(N-1)` dependent hops, so going 2 → 4 GPUs should triple
the fixed cost. It didn't move at all: **35.45 → 35.80 µs, a ratio of 1.01.**

Solving `α = L + 2(N-1)·h` across the two world sizes separates a size-independent floor from a
per-hop term:

    L = 35.28 µs        the floor — paid once, regardless of world size
    h = 0.087 µs        per dependent hop

At 4 GPUs the six hops contribute **1.5% of α**. The ring is there; it is simply not what you are
paying for.

The obvious suspicion is that NCCL wasn't running a ring at all — it switches to tree algorithms for
small payloads on some topologies. So I asked it, with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING`
(full capture in [`results/nccl_algo.txt`](results/nccl_algo.txt)):

```
AllReduce:           4 Bytes -> Algo RING proto LL     channel{Lo..Hi}={0..0}
AllReduce:       5,756 Bytes -> Algo RING proto LL     channel{Lo..Hi}={0..0}
AllReduce:     182,092 Bytes -> Algo RING proto LL     channel{Lo..Hi}={0..9}
AllReduce:   5,758,372 Bytes -> Algo RING proto LL128  channel{Lo..Hi}={0..23}
AllReduce: 182,095,808 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..23}
```

It is a ring, at every size. What changes is the protocol and — decisively — **the channel count**.
The node has 24 collective channels. A decode-sized message uses **one of them**.

So the mechanism is: at 7 KB, NCCL runs the low-latency protocol on a single channel, because
spreading such a small payload across 24 channels would cost more in synchronisation than it saves.
The cost that remains is launch and synchronisation, which is why it does not care how many GPUs are
in the ring.

**This is the same lesson as T8, on different hardware.** T8 found a batch-1 GEMV bound by work per
weight rather than by bytes. Here a collective is bound by per-call overhead rather than by the wire.
Decode makes every operation small, and small operations are priced by their fixed costs — which is
also the hypothesis T11 will test directly on kernel launches.

## Band 2: the link is not the constraint either

β reaches 67.2% of NV12's 300 GB/s spec at 2 GPUs and 73.9% at 4. The 2-GPU number misses the band.

Worth noting the direction: **4 GPUs achieve *more* bus bandwidth than 2**, which is the opposite of
the naive expectation that a longer ring is worse. Bus bandwidth already contains the `2(N-1)/N`
factor, so a wider ring genuinely uses more of the fabric — with two GPUs there are simply fewer
paths in play. The band's denominator is a datasheet figure, and 67% vs 74% of it is a much smaller
story than the four orders of magnitude separating decode from the regime where β matters at all.

## What this predicts for tensor parallelism

Amdahl with a communication penalty, using T6's measured error budget (weight share **73.7%**, step
**11.05 ms**, **90.5 tok/s**), read from `t06/results/perf.csv` at runtime rather than retyped:

    speedup = 1 / ( (1-w) + w/N + comms_per_token/step )

| batch | comms/token, TP2 | TP2 | comms/token, TP4 | TP4 |
|---|---|---|---|---|
| 1 | 1,987 µs | **1.23×** | 2,008 µs | **1.59×** |
| 8 | 250 µs | 1.53× | 253 µs | 2.13× |
| 32 | 64 µs | 1.57× | 65 µs | 2.21× |
| 128 | 17.5 µs | 1.58× | 18.4 µs | 2.23× |

![sharding buys least where latency matters most](results/decode_tax.png)

Two things fall out, and they point in opposite directions from the pre-registered story:

**Sharding wider is nearly free in latency terms.** I predicted TP4 would be punished because α would
triple. It doesn't, so TP4 beats TP2 at every batch size. The prediction registered before the run
said 1.49× and 1.76×, from an assumed α of 8 µs; the measured α of ~35 µs makes batch 1 *worse* than
predicted at TP2 and *better* at TP4, because the assumed model had the world-size scaling backwards.

**Batching is what pays for the collectives.** The fixed cost is per call, not per token, so it
amortises: 1,987 µs/token at batch 1 becomes 17.5 µs at batch 128. TP2 goes 1.23× → 1.58× on that
alone. Same conclusion T6 and T7 reached by different routes — batch first.

## Band 4: measured on a real layer, and half of it disagrees

The fit above comes from an all-reduce running alone, with nothing else on the GPU and the same
buffer every time — the cleanest possible setting and therefore the most flattering. Stage 3 runs the
actual thing: a row-parallel down-projection (K=18944, N=3584, T7 and T8's shape), taking the
collective's cost as *step with it* minus *step without it*.

| | batch 1 | batch 128 |
|---|---|---|
| TP2 measured speedup | **1.36×** (comms 27.8% of step) | 1.22× (28.6%) |
| TP4 measured speedup | **1.49×** (comms 58.2% of step) | 1.54× (47.6%) |

Against the fitted model:

| | b1 | b8 | b32 | b128 |
|---|---|---|---|---|
| world 4, measured/predicted | 1.06× | 1.03× | 0.99× | 0.83× |
| world 2, measured/predicted | **0.56×** | **0.47×** | **0.55×** | **0.66×** |

**At 4 GPUs the microbenchmark predicts a real layer's communication to within a few percent.** At 2
GPUs it over-predicts by roughly a factor of two — the collective inside real work is *faster* than
the same collective measured alone.

I don't know why, and I am not going to pick whichever explanation sounds best. The candidates:

- **Overlap.** The difference method attributes to comms anything the all-reduce cannot hide behind
  the matmul. At TP2 each rank's matmul is twice the size it is at TP4 (51.9 µs vs 27.5 µs at
  batch 1), so there is more work for the collective's launch to overlap with. This predicts the
  effect shrinks as the matmul shrinks, and TP4's agreement is consistent with that.
- **A different code path at N=2.** With a single peer, a "ring" is a bidirectional exchange, and
  NCCL may take a cheaper path than the general one the isolated sweep exercised.

Separating them needs a run I did not do: the same measurement with `NCCL_ALGO` pinned, plus an
Nsight timeline showing whether the collective actually overlaps. **That is the next thing I would
measure.** Until then band 4 is recorded as failed at world 2.

## Why this is not T5 again

Both topics run NCCL collectives on 4× A100 NVLink and both concern tensor parallelism. The split is
by regime, and it is the reason this topic exists:

| | payload per all-reduce | regime |
|---|---|---|
| T5's TP, batch 16 × seq 512 | **58,720,256 B** | far out on `n/β` — bandwidth-bound |
| Decode, batch 1 | **7,168 B** | entirely inside α — latency-bound |

**8,192× apart.** T5 measured the fast end; inference runs at the slow one.
`tests/test_distinctness.py` asserts that separation and the disjointness of the two metric
vocabularies, so the claim is enforced rather than promised.

## Method notes

**The topology gate aborts rather than warns, and checks twice.** A rented node whose GPUs lack
NVLink produces no error — just a bandwidth roughly 10× low that would land here as a fact about
NVLink. The declared check parses `nvidia-smi topo -m`; the empirical one moves 256 MB and requires
≥ 100 GB/s (measured: 172.6 GB/s at world 2, 192.9 GB/s at world 4). The matrix is a claim; only the
measurement is evidence. Both are recorded in [`results/topology.txt`](results/topology.txt).

The gate earned this on its first contact with real hardware, by failing for the wrong reason:
`nvidia-smi` underlines its header row with an ANSI escape even when its output is piped, so the
header was skipped and the **GPU0 data row was mistaken for it** — leaving only GPU0's pairs in the
matrix. That passed at world 2 and failed at world 4, which reads exactly like a partially-connected
node. The pod was fine. The regression test is now that pod's matrix, captured verbatim with the
escape and all ten NIC columns intact; the fixtures that missed it were hand-written from what the
output *looks like* rather than from what it is.

**α and β are fitted from separate regions.** A single least-squares fit across the whole sweep
recovered 13.2 µs from a synthetic curve built with α = 7.5 µs — a 76% error in this topic's headline
number, because across six decades the top decade takes almost all the leverage. α is now the median
of the flat floor (≤ 16 KB); β is the slope of the ramp (≥ 4 MB), whose intercept is discarded; R² is
the ramp's, so an algorithm switch shows up rather than averaging into a confident wrong bandwidth.
`test_t09.py` builds curves from known parameters and checks the estimator recovers them.

**Steady state, and the slowest rank.** Each timing window holds 16 back-to-back collectives — a
35 µs call timed alone would report the dispatch path, and a TP decode step fires 56 in sequence
anyway. Every measurement is reduced across ranks with `MAX`, because a collective is a barrier and
its cost is the cost to its slowest participant.

## Caveats

- **4-way measured; nothing beyond it is claimed.** 8-way brings NVSwitch behaviour this
  configuration cannot observe.
- **NVLS was unavailable on this node** (`NVLS_NCHANNELS 0`). The multicast path that could collapse
  the ring entirely was never an option here, and on hardware where it is available the small-message
  numbers could differ. Untested.
- **The tokens/sec and TP-speedup figures are modelled**, not a re-run of vLLM under tensor
  parallelism. They compose T6's measured error budget with a measured α, and the Amdahl model is
  deliberately optimistic: it holds the non-weight 26% fixed under sharding, which T5 measured to be
  false. Read them as upper bounds. Stage 3's numbers, by contrast, are measured — but on a single
  layer, not an end-to-end serving stack.
- **α is attributed to launch and synchronisation by inference, not by direct measurement.** The
  evidence is that it is flat across world size (hops contribute 1.5%) and that NCCL uses one channel
  at this size. An Nsight capture separating launch from in-kernel time would turn that inference
  into an observation. This topic does not.
- **One fabric.** NVLink on one node. The interesting contrast is PCIe, where α is several times
  larger; the model predicts what that does to decode, but this topic has not measured it.
- **Synthetic tensors**, as in T7 and T8 — the collective's cost depends on shape and fabric, not on
  values.

## Reproduce

Off the clock, on any machine:

```bash
uv sync
make t9-predict     # register the bands — laptop, no GPU
make t9-rehearse    # identical harness over gloo on CPU; numbers not published
uv run pytest topics/t09_interconnects tests    # unit tests + distinctness, no GPU
```

On a 4× A100 SXM NVLink pod, on-demand. [`scripts/t9_session.sh`](../../scripts/t9_session.sh) is the
whole session; run `gate` on its own first, since it needs nothing but `nvidia-smi` and so decides
whether the pod is worth keeping before anything is downloaded onto it:

```bash
curl -fsSL https://raw.githubusercontent.com/oscaralateras/inference-from-the-metal-up/main/scripts/t9_session.sh -o t9.sh
bash t9.sh gate     # ~30s. If this fails, destroy the pod and re-rent.
bash t9.sh setup    # clone + uv sync + make probe
bash t9.sh run      # stages 1-2: sweep, fit, decode operating points, plots
bash t9.sh tp       # stage 3: the real layer, band (4)
```

**Put the build on local disk, not `/workspace`.** RunPod mounts `/workspace` as network storage
(MooseFS), which manages roughly 250 small-file creates per second; unpacking torch and vLLM writes
well over 100,000 files, so the install stalls there while the network itself is idle at 197 MB/s.
The script defaults to `/workspace` for persistence — override it, and point uv's cache there too:

```bash
export UV_CACHE_DIR=/root/.cache/uv
WORKDIR=/root/ifmu bash t9.sh setup
```

The NCCL algorithm capture behind band 1:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  uv run python -m topics.t09_interconnects.measure --backend nccl --world-sizes 4 \
  2>&1 | grep -E "AllReduce:.*Algo"
```

Every number quoted in this note is asserted against `results/interconnect.csv` by
`test_lab_note_matches_results` in `test_t09.py`, so the prose cannot drift from the data.
