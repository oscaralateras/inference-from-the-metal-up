"""The subgraph under test, and the byte arithmetic that predicts what fusing it is worth.

Compilation is sold as one win. It is two, with different mechanisms and opposite payoffs:

    fusion          merges ops into one kernel, so intermediates stay in registers instead of
                    round-tripping through HBM        -> removes BYTES
    graph capture   records the launch sequence once and replays it with a single call
                                                      -> removes LAUNCHES

T6 measured both at once on a vLLM step and reported a bundle (15-36%). This topic separates them,
and the separation only works on a chain with **no weights in it**. Put a matmul in the chain and
the weight read dwarfs everything: at batch 1 the MLP's two projections read 271 MB while their
elementwise epilogue moves 37 KB, so fusion's saving is 0.01% and unmeasurable. A weightless
residual chain — the norm, the residual add, the activation, the rotary — is the only place in a
transformer where fusion's bytes are the whole story.

**The prediction.** Bytes are countable in advance. An unfused chain of `k` elementwise ops over a
tensor of `n` bytes reads and writes it once per op: `2kn`. Perfectly fused, the tensor is read
once and written once: `2n`. So fusion's ceiling is a factor of `k`, and against T7's *measured*
1,736.7 GB/s that becomes a predicted time, not a hoped-for speedup. Nothing here is fitted.

**And the reason it will not deliver that at batch 1.** The chain at batch 1 is 7 KB. At 1,700 GB/s
that is 4 nanoseconds of traffic against a kernel launch that costs microseconds — three orders of
magnitude of headroom for launch overhead to hide in. Fusion cannot help what is not
bandwidth-bound. Graph capture can. That flip is what this topic goes looking for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Qwen2.5-7B's hidden size, as in T5-T9, so the chain is the shape a real decode step normalises.
DEFAULT_HIDDEN = 3584

BYTES_PER_ELEMENT = 2  # bfloat16, as everywhere else in this repo

# The chain, in order. Each entry is one elementwise or reduction op that an unfused execution runs
# as its own kernel, with its own launch and its own HBM round trip.
#
# It is deliberately the *residual path* of a transformer block rather than the MLP: RMSNorm,
# residual add, SiLU, the gate multiply, and a rotary-style rotation. Real, weightless, and exactly
# the region `torch.compile` is best at and hand-written kernels usually skip.
CHAIN_OPS = ("rmsnorm", "residual_add", "silu", "gate_mul", "rotary_half")

# How many distinct tensors the chain reads at the boundary (the hidden state and the residual)
# and writes at the boundary (the output). Fusion cannot remove these — they are the chain's
# contract with the rest of the model, not its internals.
BOUNDARY_READS = 2
BOUNDARY_WRITES = 1


def activation_bytes(batch: int, hidden: int = DEFAULT_HIDDEN) -> int:
    """Bytes in one activation tensor of the chain."""
    if batch < 1 or hidden < 1:
        raise ValueError(f"batch and hidden must be positive, got {batch}, {hidden}")
    return batch * hidden * BYTES_PER_ELEMENT


def unfused_bytes(batch: int, hidden: int = DEFAULT_HIDDEN, ops: int = len(CHAIN_OPS)) -> int:
    """HBM traffic when every op is its own kernel: each reads its input and writes its output.

    This is the honest upper bound rather than the worst case — a few of these ops read two
    tensors, and the allocator may keep a small one in L2 between adjacent kernels. Both effects
    push the real number *below* this, which is the right direction for a bound the measurement is
    scored against.
    """
    return 2 * ops * activation_bytes(batch, hidden)


def fused_bytes(batch: int, hidden: int = DEFAULT_HIDDEN) -> int:
    """HBM traffic when the whole chain is one kernel: the boundary tensors and nothing else.

    Every intermediate stays in registers or shared memory. This is what a perfect fuser achieves
    and what T8's hand-written kernel achieved for its own chain, so it is the same ceiling
    concept, applied to the compiler instead of to a person.
    """
    return (BOUNDARY_READS + BOUNDARY_WRITES) * activation_bytes(batch, hidden)


def fusion_ceiling(batch: int, hidden: int = DEFAULT_HIDDEN) -> float:
    """The most fusion can be worth here, from byte counts alone. Independent of the hardware."""
    return unfused_bytes(batch, hidden) / fused_bytes(batch, hidden)


def traffic_seconds(nbytes: int, gbps: float) -> float:
    """Time to move `nbytes` at a measured bandwidth — byte counts into microseconds."""
    if gbps <= 0:
        raise ValueError(f"bandwidth must be positive, got {gbps}")
    return nbytes / (gbps * 1e9)


def bandwidth_bound_batch(gbps: float, launch_us: float) -> float:
    """The batch at which the unfused chain's traffic finally costs as much as its own launches.

    Below it the chain is launch-bound; above it, bandwidth-bound.

        2 * ops * batch * hidden * 2 / (gbps * 1e9)  =  ops * launch_us * 1e-6

    `ops` appears on both sides and cancels, so this threshold is a property of the hardware alone.
    It is *not* the crossover the topic pre-registers — see `fusion_crossover_batch`, which asks a
    different and harder question and does depend on chain length.
    """
    if gbps <= 0 or launch_us <= 0:
        raise ValueError(f"need positive bandwidth and launch cost, got {gbps}, {launch_us}")
    return launch_us * 1e-6 * gbps * 1e9 / (4 * DEFAULT_HIDDEN)


def fusion_crossover_batch(
    gbps: float, launch_us: float, ops: int = len(CHAIN_OPS), hidden: int = DEFAULT_HIDDEN
) -> float:
    """The batch at which fusion overtakes graph capture. The topic's headline, derived.

    Model each mode as launches plus traffic, serial — crude, but every term in it is a quantity
    this repo has measured, and it is falsifiable:

        eager    = ops * L  +  2 * ops * A / BW          many launches, many round trips
        compile  =   1 * L  +      3 * A / BW            one launch, boundary tensors only
        graph    =       0  +  2 * ops * A / BW          no launches, still every round trip

    where `A` is one activation tensor's bytes and `L` is a kernel launch. Compile beats graph when

        L + 3A/BW  <  2*ops*A/BW    =>    A  >  L * BW / (2*ops - 3)

    which in batch terms is the expression below. **The op count does not cancel here**, because
    fusion removes `2*ops - 3` round trips while graph capture removes `ops` launches, and those
    scale differently. So a longer chain crosses over *earlier* — fusion has more to remove.

    That is the control `measure.py --chain-lengths` runs, and it is a sharp prediction rather than
    a vague one: going from 5 ops to 2 should move the crossover by a factor of
    `(2*5-3) / (2*2-3)` = 7. A model that survives that is doing real work.
    """
    if gbps <= 0 or launch_us <= 0:
        raise ValueError(f"need positive bandwidth and launch cost, got {gbps}, {launch_us}")
    if 2 * ops - 3 <= 0:
        raise ValueError(
            f"with {ops} ops, fusion removes no net traffic (2*ops <= 3) and never overtakes "
            "graph capture — the crossover is undefined, not merely large"
        )
    return gbps * 1e9 * launch_us * 1e-6 / ((2 * ops - 3) * hidden * BYTES_PER_ELEMENT)


@dataclass(frozen=True)
class ChainInputs:
    """The tensors the chain runs on. Held together so every mode gets identical inputs."""

    hidden_state: torch.Tensor
    residual: torch.Tensor
    gate: torch.Tensor
    weight: torch.Tensor

    @property
    def batch(self) -> int:
        return self.hidden_state.shape[0]


def make_inputs(batch: int, hidden: int, device: torch.device, dtype: torch.dtype) -> ChainInputs:
    """Allocate the chain's inputs once, outside any timed region.

    Allocation inside the timed region cost T8 about 40% of its measured throughput, which is the
    kind of mistake that looks like a result. Everything here is allocated up front and reused.
    """
    gen = torch.Generator(device="cpu").manual_seed(11)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)

    return ChainInputs(
        hidden_state=rand(batch, hidden),
        residual=rand(batch, hidden),
        gate=rand(batch, hidden),
        weight=rand(hidden),
    )


def decode_chain(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    w: torch.Tensor,
    ops: int = len(CHAIN_OPS),
) -> torch.Tensor:
    """The chain itself: up to five weightless ops, written the way a model file writes them.

    Deliberately plain PyTorch with no fused primitives. `torch.compile` has to find the fusion
    itself, which is exactly the capability under test — calling a pre-fused op would measure
    NVIDIA's kernel, not the compiler.

    `ops` truncates the chain, which exists for band 3's control: `fusion_crossover_batch` predicts
    that shortening the chain moves the crossover by a specific factor, and truncating is how that
    gets tested rather than assumed. Python-level branching on `ops` is fine here — `torch.compile`
    specialises on it, so each length compiles once into a graph with no branches left in it.
    """
    if not 1 <= ops <= len(CHAIN_OPS):
        raise ValueError(f"ops must be in 1..{len(CHAIN_OPS)}, got {ops}")

    # RMSNorm, spelled out rather than via a fused module so the reduction is visible to the fuser.
    variance = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(variance + 1e-6) * w
    if ops == 1:
        return out

    # Residual add — the op that makes this a chain rather than a pointwise map.
    out = out + residual
    if ops == 2:
        return out

    # SwiGLU's activation and gate, the two ops that dominate an unfused elementwise epilogue.
    out = torch.nn.functional.silu(out)
    if ops == 3:
        return out

    out = out * gate
    if ops == 4:
        return out

    # A rotary-style half-rotation: cheap arithmetic, another full round trip when unfused.
    first, second = out.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)
