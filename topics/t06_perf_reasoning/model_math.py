"""Decode cost, derived from a model's config file alone — no weights, no GPU, no measurement.

This is the "first principles" half of T6. Everything here is arithmetic over a `config.json`, so
it can be checked on a laptop and committed *before* the GPU run. `test_t06.py` verifies the
analytic parameter count against the real loaded model to within 0.5%, which is what stops this
from being plausible-looking algebra that happens to be wrong.

The three quantities that matter for batch-1 decode:

* **FLOPs per token = 2P.** Every weight is used exactly once, in a multiply-accumulate: one
  multiply plus one add. Two floating-point operations per parameter.
* **Bytes per token = P x bytes_per_param.** Every weight is *read* once. The factor of two above
  is compute, not traffic — you use each loaded byte twice.
* **Arithmetic intensity = 2 / bytes_per_param.** For bfloat16 that is **1 FLOP per byte**, which
  is roughly two orders of magnitude below any modern GPU's ridge point. That single number is why
  decode is memory-bound and why bandwidth, not FLOP/s, sets the speed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Activation traffic per layer per token, counted in units of the tensor width involved.
#
# Decode reads and writes a handful of small intermediate tensors per layer on top of the weights:
# the qkv projections' outputs, the attention output, the two norms, the two residual adds (each a
# read of two operands and a write), and the MLP's gate/up/down intermediates. Counting each
# tensor touch once gives roughly these multipliers. They are an **estimate**, stated as such —
# precise accounting would need a kernel-level profile, and at batch 1 this whole term is small.
HIDDEN_TENSOR_TOUCHES = 14  # tensors of width `hidden_size` touched per layer per token
INTERMEDIATE_TENSOR_TOUCHES = 4  # tensors of width `intermediate_size` (gate, up, their product)


@dataclass(frozen=True)
class ModelShape:
    """The architecture numbers T6 needs, lifted straight out of a HuggingFace `config.json`."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    tie_word_embeddings: bool
    bytes_per_param: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_config(cls, config: dict[str, object], bytes_per_param: int = 2) -> ModelShape:
        """Build from a parsed `config.json`. Raises on anything missing rather than guessing.

        Guessing was a real T5 bug: a hardcoded head count silently produced a wrong `head_dim`
        for any model that did not happen to match, and the results looked entirely reasonable.
        """

        def _as_int(value: object, default: int) -> int:
            return value if isinstance(value, int) else default

        def need(key: str) -> int:
            value = config.get(key)
            if not isinstance(value, int):
                raise KeyError(f"config is missing required integer field {key!r}")
            return value

        heads = need("num_attention_heads")
        hidden = need("hidden_size")
        if hidden % heads:
            raise ValueError(f"hidden_size {hidden} is not divisible by {heads} heads")

        return cls(
            name=str(config.get("_name_or_path", "unknown")),
            hidden_size=hidden,
            intermediate_size=need("intermediate_size"),
            num_hidden_layers=need("num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=_as_int(config.get("num_key_value_heads"), default=heads),
            vocab_size=need("vocab_size"),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            bytes_per_param=bytes_per_param,
        )

    # -- parameter counts -------------------------------------------------------------------

    @property
    def params_per_layer(self) -> int:
        """Weights in one transformer block: attention projections + MLP + two RMSNorms."""
        h, kv = self.hidden_size, self.num_key_value_heads * self.head_dim
        attention = h * h + 2 * (h * kv) + h * h  # q, k, v, o
        mlp = 3 * h * self.intermediate_size  # gate, up, down
        norms = 2 * h  # input_layernorm, post_attention_layernorm
        return attention + mlp + norms

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def total_params(self) -> int:
        """Every weight in the model, including embeddings and the LM head."""
        body = self.num_hidden_layers * self.params_per_layer + self.hidden_size  # + final norm
        heads = self.embedding_params if self.tie_word_embeddings else 2 * self.embedding_params
        return body + heads

    @property
    def params_read_per_token(self) -> int:
        """Weights actually *read* to decode one token — the number that drives bytes/token.

        Not the same as `total_params`, and the difference is the point. The input embedding is a
        **lookup**: one row of the table, a few kilobytes, not the whole matrix. The LM head is a
        **matmul** against the full vocabulary, so every one of its weights is read. Counting the
        input embedding as traffic overstates bytes/token by ~7% on a 7B model and makes the
        prediction pessimistic for a reason that has nothing to do with the hardware.
        """
        return self.total_params - self.embedding_params

    # -- per-token cost ---------------------------------------------------------------------

    @property
    def flops_per_token(self) -> int:
        """One multiply and one add per weight read: `2P`."""
        return 2 * self.params_read_per_token

    @property
    def weight_bytes_per_token(self) -> int:
        """Each weight read exactly once."""
        return self.params_read_per_token * self.bytes_per_param

    def kv_cache_bytes(self, seq_len: int, batch: int = 1) -> int:
        """Bytes of KV cache read to decode one token, across the whole model.

        Two tensors (K and V) per layer, one entry per key/value head per past position:

            2 x layers x kv_heads x head_dim x seq_len x batch x bytes_per_param

        Unlike the weight term this **grows with context**, which is why decode throughput decays
        as a sequence gets longer — the same model gets measurably slower the more it has said.
        """
        if seq_len < 0 or batch < 1:
            raise ValueError(f"need seq_len >= 0 and batch >= 1, got {seq_len=} {batch=}")
        per_layer = 2 * self.num_key_value_heads * self.head_dim
        return per_layer * self.num_hidden_layers * seq_len * batch * self.bytes_per_param

    def activation_bytes_per_token(self, batch: int = 1) -> int:
        """Estimated traffic in intermediate tensors — norms, residual adds, MLP intermediates.

        These carry almost no FLOPs but do move bytes, and on a memory-bound workload bytes are
        what cost time. Small next to the weight term at batch 1; it grows linearly with batch
        while the weight term stays flat, so it matters more as you batch up.
        """
        per_layer = (
            HIDDEN_TENSOR_TOUCHES * self.hidden_size
            + INTERMEDIATE_TENSOR_TOUCHES * self.intermediate_size
        )
        return per_layer * self.num_hidden_layers * batch * self.bytes_per_param

    def bytes_per_token(self, seq_len: int, batch: int = 1) -> int:
        """Total bytes moved to decode one token per sequence, at a given context length.

        Weights are read once no matter how large the batch — that is exactly why batching raises
        throughput — so the weight term is *not* multiplied by `batch`.
        """
        return (
            self.weight_bytes_per_token
            + self.kv_cache_bytes(seq_len, batch)
            + self.activation_bytes_per_token(batch)
        )

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte for batch-1 decode, weights only: `2 / bytes_per_param`.

        bfloat16 gives 1.0. Compare against the hardware's ridge point (~170 FLOPs/byte on an
        A100) to see how far into memory-bound territory decode sits.
        """
        return self.flops_per_token / self.weight_bytes_per_token

    # -- the prediction ---------------------------------------------------------------------

    def predicted_tokens_per_sec(
        self, bandwidth_gbps: float, seq_len: int, batch: int = 1
    ) -> float:
        """Tokens/sec if the run were perfectly bandwidth-bound: `bandwidth / bytes_per_token`.

        The ceiling, not a forecast. Every real effect — imperfect bandwidth utilisation, kernel
        launch overhead, non-overlapped work — can only push the measurement below this line.
        Measuring *above* it would mean the model is wrong, not that the GPU is fast.
        """
        if bandwidth_gbps <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth_gbps}")
        return bandwidth_gbps * 1e9 / self.bytes_per_token(seq_len, batch) * batch
