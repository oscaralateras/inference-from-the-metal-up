"""The cost of a collective, as a two-term model — and where decode lands on it.

T5 measured tensor parallelism end to end and found something it could not fully explain: TP's
940 MB per step is only ~4% of the step time at NVLink bandwidth, yet TP scales 21% worse than
data parallelism. Its conclusion was that the loss is "frequency and shape" rather than volume.
This module is the arithmetic that turns that sentence into something measurable.

A collective is **not** priced in bytes. It is priced as

    t(n) = alpha + n / beta

`alpha` is what the call costs before it has moved a single useful byte — kernel launch, ring
setup, and N-1 dependent hops each of which must complete before the next begins. `beta` is the
rate once it is moving. Every all-reduce pays both terms, and which one dominates is decided
entirely by `n`.

That is the whole topic, because **T5 and decode sit on opposite ends of this curve**:

* T5 ran TP at batch 16 x seq 512, so each all-reduce carried 16*512*3584*2 = **58.7 MB**. Far out
  on the `n/beta` term, where a fast link is exactly the thing that helps.
* Decode runs one token at a time, so seq = 1 and each all-reduce carries batch*3584*2 =
  **7.2 KB at batch 1** — four orders of magnitude smaller, and entirely inside `alpha`, where the
  link's bandwidth is close to irrelevant.

So the interesting claim is not "communication is expensive". It is that the *same* collective on
the *same* wire has two completely different cost structures, and inference runs in the regime
that the headline bandwidth number does not describe.

### Ring all-reduce, and why alpha grows with world size

NCCL's bandwidth-optimal algorithm is a ring: reduce-scatter over N-1 steps, then all-gather over
another N-1. Each step ships `n/N` bytes to one neighbour, so the totals are

    hops        = 2 * (N - 1)                    dependent, hence 2(N-1) latencies
    bytes/rank  = 2 * (N - 1) / N * n            the "bus" volume NCCL reports

giving

    t(n, N) = 2*(N-1)*alpha_step  +  2*(N-1)/N * n / beta_link

Two consequences worth pre-registering as predictions, because they are falsifiable and cheap:

1. **The fixed cost scales with `2(N-1)`, not with N.** Going 2 -> 4 GPUs should multiply alpha by
   3.0, not by 2. If decode is alpha-bound, sharding wider costs *more per call* even though each
   call carries the same bytes.
2. **The moving cost saturates.** `2(N-1)/N` runs 1.0 -> 1.5 -> 1.75 -> 2.0, so bytes on the wire
   approach twice the payload however wide the ring gets.

`bus_gbps` below is the standard NCCL busbw figure, and it is the one to compare against a link's
spec sheet: it already contains the `2(N-1)/N` factor, so a healthy ring holds it roughly flat
across world sizes while the naive `n/t` figure would appear to fall.
"""

from __future__ import annotations

from dataclasses import dataclass

# Qwen2.5-7B, the same model T5, T6, T7 and T8 all describe. Hidden size fixes the all-reduce
# payload; layer count fixes how many of them a token pays for.
DEFAULT_HIDDEN = 3584
DEFAULT_LAYERS = 28
BYTES_PER_ELEMENT_BF16 = 2

# Tensor parallelism all-reduces twice per block: once after attention's output projection, once
# after the MLP's down projection. Both are row-parallel, so each rank holds a partial sum over
# the hidden dimension and the full vector has to exist before the next RMSNorm can run. This is
# the count T5's harness used and the reason its step contained 16 all-reduces over 8 layers.
ALLREDUCES_PER_LAYER = 2

# The decode batch walk, matching T7's so the two topics' x-axes line up.
BATCH_WALK = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Message sizes for the sweep: 1 KB to 1 GB, four points per decade. Deliberately spans both
# regimes — the flat alpha floor at the bottom, the linear beta ramp at the top — because the
# whole point is to show the corner between them and where decode sits relative to it.
SWEEP_MIN_BYTES = 1024
SWEEP_MAX_BYTES = 1024**3
SWEEP_POINTS_PER_DECADE = 4


def allreduce_bytes(batch: int, hidden: int = DEFAULT_HIDDEN, dtype_bytes: int = 2) -> int:
    """Bytes in one TP all-reduce during **decode**, where the sequence length is 1 by definition.

    The payload is the hidden activation for every sequence in the batch. Sequence length does not
    appear because decode produces exactly one token per sequence per step — which is precisely
    what makes this number so much smaller than T5's, and it is a property of the regime rather
    than of the configuration.
    """
    if batch < 1 or hidden < 1 or dtype_bytes < 1:
        raise ValueError(
            f"need batch, hidden, dtype_bytes >= 1, got {batch}, {hidden}, {dtype_bytes}"
        )
    return batch * hidden * dtype_bytes


def prefill_allreduce_bytes(
    batch: int, seq: int, hidden: int = DEFAULT_HIDDEN, dtype_bytes: int = 2
) -> int:
    """The same collective during prefill, where sequence length multiplies the payload.

    Present so the note can put T5's operating point (batch 16, seq 512) on the same axis as
    decode's rather than asserting the contrast in prose.
    """
    if seq < 1:
        raise ValueError(f"need seq >= 1, got {seq}")
    return allreduce_bytes(batch, hidden, dtype_bytes) * seq


def ring_hops(world: int) -> int:
    """Dependent latencies in a ring all-reduce: `2(N-1)`, from N-1 reduce-scatter + N-1 gather."""
    if world < 1:
        raise ValueError(f"need world >= 1, got {world}")
    return 2 * (world - 1)


def bus_factor(world: int) -> float:
    """The `2(N-1)/N` volume factor relating payload bytes to bytes actually crossing the wire."""
    if world < 1:
        raise ValueError(f"need world >= 1, got {world}")
    return 2.0 * (world - 1) / world


def bus_gbps(n_bytes: int, world: int, time_us: float) -> float:
    """NCCL's `busbw`: wire bytes over elapsed time, comparable to a link's spec sheet.

    The naive `n/t` figure ("algorithm bandwidth") is not comparable across world sizes, because a
    wider ring moves more wire bytes for the same payload. Dividing by the bus factor is what makes
    a flat line across N mean "the link is saturated" rather than "the benchmark is confused".
    """
    if time_us <= 0:
        raise ValueError(f"need time_us > 0, got {time_us}")
    if world == 1:
        return 0.0
    return bus_factor(world) * n_bytes / (time_us * 1e-6) / 1e9


@dataclass(frozen=True)
class RingCost:
    """A fitted `t(n) = alpha + n/beta` for one world size.

    `alpha_us` is the whole call's fixed cost, not the per-hop cost — `alpha_step_us` divides it
    back out by `2(N-1)` so the per-hop latency can be compared across world sizes, which is what
    the ring model actually predicts should hold constant.
    """

    world: int
    alpha_us: float
    beta_gbps: float
    r_squared: float
    n_points: int

    @property
    def alpha_step_us(self) -> float:
        """Fixed cost per dependent hop. Constant across N if the ring model holds."""
        hops = ring_hops(self.world)
        return self.alpha_us / hops if hops else 0.0

    def time_us(self, n_bytes: int) -> float:
        """Predicted microseconds for one all-reduce of `n_bytes`."""
        return self.alpha_us + bus_factor(self.world) * n_bytes / (self.beta_gbps * 1e9) * 1e6

    def alpha_share(self, n_bytes: int) -> float:
        """Fraction of the call spent on fixed cost. ~1.0 for decode, ~0 for T5's prefill."""
        return self.alpha_us / self.time_us(n_bytes)

    def crossover_bytes(self) -> float:
        """Payload at which the moving term first equals the fixed term.

        The corner of the curve, and the single most useful number here: below it you are buying
        latency, above it you are buying bandwidth, and the two are different purchases.
        """
        return self.alpha_us * 1e-6 * self.beta_gbps * 1e9 / bus_factor(self.world)


# The two parameters are estimated from the two regions that actually constrain them, rather than
# from one regression across the whole sweep. This is not a refinement — the single-regression
# version is simply wrong for this data, and it was caught by feeding it a curve with a known
# alpha: across six decades of message size, ordinary least squares on raw bytes gives the top
# decade almost all the leverage, because the x-spread there dwarfs everything below it. Fitting
# a synthetic curve built with alpha = 7.5 us recovered **13.2 us**, a 76% error in the one number
# this topic exists to report. The floor and the ramp are separate measurements and are treated as
# such.
#
# Below this, the moving term is negligible: 16 KB across a 300 GB/s link is ~0.05 us against a
# fixed cost of several microseconds, so the region is flat and its height *is* alpha.
FLOOR_MAX_BYTES = 16 * 1024

# Above this, the fixed term is negligible and the slope is bandwidth. 4 MB at 300 GB/s is ~13 us,
# already comfortably past a single-digit-microsecond alpha.
RAMP_MIN_BYTES = 4 * 1024 * 1024


def fit_ring_cost(
    world: int,
    points: list[tuple[int, float]],
    *,
    floor_max_bytes: int = FLOOR_MAX_BYTES,
    ramp_min_bytes: int = RAMP_MIN_BYTES,
) -> RingCost:
    """Estimate `t = alpha + n/beta` from `(bytes, microseconds)` observations.

    Two regions, two estimators, because the two parameters are identified by different parts of
    the curve:

    * **alpha** is the median of the flat floor (`n <= floor_max_bytes`). The median rather than
      the mean because a single scheduler hiccup in a microsecond-scale measurement is a large
      outlier, and the floor is exactly where such an outlier does the most damage.
    * **beta** is the slope of an ordinary least squares fit over the ramp
      (`n >= ramp_min_bytes`) only. Its intercept is discarded: extrapolating a line fitted at
      hundreds of megabytes back to zero is not a measurement of anything.

    `r_squared` is the ramp's, so it answers the question it is actually being asked — is the
    large-message region a straight line, or did the collective change algorithm partway up? NCCL
    switches between tree and ring by message size, so this is a real thing that happens, and
    averaging across the switch would produce a confident bandwidth describing neither regime.
    """
    if world < 2:
        raise ValueError(f"a collective needs world >= 2, got {world}")
    if len(points) < 2:
        raise ValueError(f"need >= 2 points to fit two parameters, got {len(points)}")

    floor = sorted(t for n, t in points if n <= floor_max_bytes)
    ramp = [(float(n), t) for n, t in points if n >= ramp_min_bytes]

    if not floor:
        raise ValueError(
            f"no points at or below {floor_max_bytes:,} B, so alpha is unconstrained. The sweep "
            "must reach down into the latency floor — that region is the whole subject of this "
            "topic, not a lead-in to it."
        )
    if len(ramp) < 2:
        raise ValueError(
            f"only {len(ramp)} point(s) at or above {ramp_min_bytes:,} B, so beta is "
            "unconstrained. Extend the sweep upward: a bandwidth read off the latency floor is "
            "not a bandwidth."
        )

    mid = len(floor) // 2
    alpha_us = floor[mid] if len(floor) % 2 else (floor[mid - 1] + floor[mid]) / 2

    n_ramp = len(ramp)
    mean_x = sum(x for x, _ in ramp) / n_ramp
    mean_y = sum(y for _, y in ramp) / n_ramp
    sxx = sum((x - mean_x) ** 2 for x, _ in ramp)
    if sxx == 0:
        raise ValueError("every ramp point has the same message size — slope is undefined")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in ramp) / sxx

    if slope <= 0:
        raise ValueError(
            f"fitted a non-positive slope ({slope:.3e} us/byte) at world={world} — time is not "
            "increasing with message size across the ramp, so either the sweep never left the "
            "latency floor or the timing is wrong. Neither produces a usable bandwidth."
        )

    ss_tot = sum((y - mean_y) ** 2 for _, y in ramp)
    ss_res = sum((y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in ramp)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # slope is us per byte of payload; the wire carries `bus_factor` times that.
    beta_gbps = bus_factor(world) / (slope * 1e-6) / 1e9

    return RingCost(
        world=world,
        alpha_us=max(alpha_us, 0.0),
        beta_gbps=beta_gbps,
        r_squared=r_squared,
        n_points=len(points),
    )


@dataclass(frozen=True)
class LatencyBudget:
    """`alpha(N) = floor + 2(N-1)*hop`, fitted across world sizes.

    The point of fitting this rather than solving it: with only two world sizes there are as many
    equations as unknowns, so the decomposition reproduces the data exactly and cannot be wrong. A
    third world size gives the model a residual and therefore the ability to fail — which matters,
    because the claim being made is that the ring's hop count explains almost none of the fixed
    cost, and a decomposition that cannot fail is not evidence for anything.
    """

    floor_us: float
    hop_us: float
    r_squared: float
    n_worlds: int

    def alpha_us(self, world: int) -> float:
        return self.floor_us + ring_hops(world) * self.hop_us

    def hop_share(self, world: int) -> float:
        """Fraction of `alpha(N)` attributable to ring traversal."""
        alpha = self.alpha_us(world)
        return ring_hops(world) * self.hop_us / alpha if alpha else 0.0


def fit_latency_budget(alphas: dict[int, float]) -> LatencyBudget:
    """Least squares of `alpha` on hop count across world sizes.

    Exactly determined at two world sizes (R^2 is then 1.0 by construction and means nothing);
    genuinely tested at three or more.
    """
    if len(alphas) < 2:
        raise ValueError(f"need >= 2 world sizes to separate floor from hop, got {len(alphas)}")

    xs = [float(ring_hops(w)) for w in sorted(alphas)]
    ys = [alphas[w] for w in sorted(alphas)]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("every world size has the same hop count — the split is undefined")
    hop = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sxx
    floor = mean_y - hop * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (floor + hop * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return LatencyBudget(floor_us=floor, hop_us=hop, r_squared=r_squared, n_worlds=n)


def sweep_sizes(
    min_bytes: int = SWEEP_MIN_BYTES,
    max_bytes: int = SWEEP_MAX_BYTES,
    per_decade: int = SWEEP_POINTS_PER_DECADE,
) -> list[int]:
    """Log-spaced message sizes, rounded to a multiple of 4 bytes so element counts stay whole."""
    if min_bytes < 4 or max_bytes < min_bytes or per_decade < 1:
        raise ValueError(f"bad sweep bounds: {min_bytes}, {max_bytes}, {per_decade}")

    sizes: list[int] = []
    value = float(min_bytes)
    step = 10.0 ** (1.0 / per_decade)
    while value <= max_bytes * 1.0001:
        rounded = max(4, int(value) // 4 * 4)
        if not sizes or rounded > sizes[-1]:
            sizes.append(rounded)
        value *= step
    return sizes


def comms_per_token_us(
    cost: RingCost, batch: int, *, layers: int = DEFAULT_LAYERS, hidden: int = DEFAULT_HIDDEN
) -> float:
    """Total all-reduce microseconds a decode step spends, for one token of every sequence.

    `layers * ALLREDUCES_PER_LAYER` calls, each carrying the batch's hidden activation. Divided by
    batch at the end because the step produces `batch` tokens, so the per-token cost is what
    composes with T6's per-token budget.
    """
    per_call = cost.time_us(allreduce_bytes(batch, hidden))
    return layers * ALLREDUCES_PER_LAYER * per_call / batch


def predicted_tp_speedup(
    weight_share: float, world: int, step_ms: float, comms_us_per_token: float
) -> float:
    """Amdahl with a communication penalty — the same law T5 calibrated and T8 reused.

    Tensor parallelism divides the weight traffic by `world`; everything else in the step is held
    fixed, and the collectives are added on top:

        speedup = 1 / ( (1 - w) + w/N + comms_per_token / step )

    Deliberately **optimistic**, and it should be read as an upper bound rather than a forecast.
    It assumes the non-weight 26% is untouched by sharding, when T5 measured that it is not: each
    rank runs a matmul 1/N the size and gets proportionally less out of the GPU, and every
    all-reduce is a barrier that costs the slowest rank's jitter. Both push the real number down.
    An optimistic prediction that still lands near the measurement is a stronger result than a
    hedged one that cannot be wrong.
    """
    if not 0.0 <= weight_share <= 1.0:
        raise ValueError(f"weight_share must be a fraction, got {weight_share}")
    if world < 1 or step_ms <= 0:
        raise ValueError(f"need world >= 1 and step_ms > 0, got {world}, {step_ms}")

    compute = (1.0 - weight_share) + weight_share / world
    comms = comms_us_per_token / 1e3 / step_ms
    return 1.0 / (compute + comms)
