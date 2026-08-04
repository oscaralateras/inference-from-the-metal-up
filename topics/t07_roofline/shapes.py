"""The matmul shapes inference actually runs, and their arithmetic intensity.

T7 measures **isolated kernels on synthetic tensors** — no model weights are ever loaded. The
model's `config.json` supplies the dimensions and nothing else. That keeps this topic fast, cheap
and portable, and it is what structurally separates it from T6: T6 measures a whole model in the
time domain, T7 measures individual kernels in the shape domain.

Arithmetic intensity for a `(M,K) @ (K,N)` matmul:

    FLOPs = 2 * M * N * K              one multiply and one add per multiply-accumulate
    bytes = (M*K + K*N + M*N) * b      read both operands, write the result
    AI    = FLOPs / bytes

Take the two regimes an LLM runs in:

* **Prefill** processes the whole prompt at once, so M is the prompt length. Both `M*K` and `M*N`
  grow with it, but the FLOPs grow as `M*N*K`, so AI rises roughly linearly in M. Prompt-sized M
  puts it in the hundreds — right of the ridge, compute-bound.
* **Decode** produces one token, so M = 1. The weight matrix `K*N` dominates the byte count while
  the FLOPs collapse to `2*N*K`, giving AI -> `2/b`: exactly **1 FLOP per byte** in bfloat16, two
  orders of magnitude left of the ridge, hard memory-bound.

Same weights, same model, opposite regimes. That is the central fact of inference systems, and it
falls straight out of this arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

# Qwen2.5-7B, so T6 and T7 describe the same model from two different angles. Overridable on the
# command line; these are the defaults used in the committed results.
DEFAULT_HIDDEN = 3584
DEFAULT_INTERMEDIATE = 18944
DEFAULT_PREFILL_TOKENS = 2048

# The batch walk for decode: M rising from a single token toward server-sized batches. This is the
# sweep that shows the roofline point climbing toward the ridge.
BATCH_WALK = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class GemmShape:
    """One matmul, named for the thing in a transformer that actually runs it."""

    name: str
    m: int
    n: int
    k: int
    regime: str  # "prefill" | "decode"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def bytes_moved(self, bytes_per_element: int) -> int:
        """Both operands read, the result written — the minimum traffic any implementation needs.

        A lower bound, and deliberately so. Real kernels re-read tiles that miss in L2, so the
        achieved intensity is never better than this and the roofline point sits below the roof.
        """
        return (self.m * self.k + self.k * self.n + self.m * self.n) * bytes_per_element

    def arithmetic_intensity(self, bytes_per_element: int) -> float:
        return self.flops / self.bytes_moved(bytes_per_element)


def inference_shapes(
    hidden: int = DEFAULT_HIDDEN,
    intermediate: int = DEFAULT_INTERMEDIATE,
    prefill_tokens: int = DEFAULT_PREFILL_TOKENS,
) -> list[GemmShape]:
    """The four matmuls that dominate a transformer, in both regimes.

    Attention's own QK^T and AV matmuls are excluded: their cost scales with sequence length
    rather than with the weights, so they belong to a different analysis. Stated as a limit in the
    lab note rather than quietly folded in.
    """
    return [
        GemmShape("prefill_qkv_proj", prefill_tokens, hidden, hidden, "prefill"),
        GemmShape("prefill_mlp_up", prefill_tokens, intermediate, hidden, "prefill"),
        GemmShape("decode_qkv_proj", 1, hidden, hidden, "decode"),
        GemmShape("decode_mlp_up", 1, intermediate, hidden, "decode"),
    ]


def batch_walk_shapes(
    hidden: int = DEFAULT_HIDDEN,
    intermediate: int = DEFAULT_INTERMEDIATE,
) -> list[GemmShape]:
    """The decode MLP projection at rising batch size — a GEMV growing into a GEMM.

    Batching B decode requests reads the weights **once** and serves all B, so the FLOPs scale
    with B while the dominant byte term does not. Arithmetic intensity therefore rises with batch
    size, and the roofline point walks rightward toward the ridge. This is the whole mechanical
    argument for continuous batching, in one sweep.
    """
    return [GemmShape(f"decode_batch_{m}", m, intermediate, hidden, "decode") for m in BATCH_WALK]
