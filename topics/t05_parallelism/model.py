"""A real transformer block — the thing T5's five parallelism strategies actually split.

Attention **and** MLP, not MLP alone, and that is a load-bearing choice. In an MLP every token is
processed independently, so "split the sequence across ranks" and "split the batch across ranks"
are the *same operation* — sequence parallelism would just be data parallelism wearing a different
name, and any SP number measured on an MLP would be meaningless. Attention is the only thing in a
transformer that mixes tokens together, so it is the only thing that makes SP a distinct strategy
with a distinct communication cost.

The block is Llama-architecture (RMSNorm, GQA-capable attention, SwiGLU MLP, two residuals) with
real weights from `JackFram/llama-160m`:

    h = x + attn(rms_norm(x))
    y = h + mlp(rms_norm(h))

**Why this model.** TP degree is not a free parameter: it must divide the attention head count,
and with grouped-query attention it is further capped at `num_key_value_heads`. Surveying small
open models:

    SmolLM2-135M      9 heads,  3 kv  -> TP degree must divide 9, capped at 3
    Qwen2.5-0.5B     14 heads,  2 kv  -> capped at 2
    TinyLlama-1.1B   32 heads,  4 kv  -> capped at 4
    llama-160m       12 heads, 12 kv  -> 1, 2, 3, 4, 6, 12 all valid

llama-160m's 12 MHA heads give a rich sweep on an 8-core box; SmolLM2 (T1's model) would have
pinned TP to degree 1. This is exactly why production models choose power-of-two head counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

MODEL_ID = "JackFram/llama-160m"
HIDDEN = 768
N_HEADS = 12
N_KV_HEADS = 12
HEAD_DIM = HIDDEN // N_HEADS
INTERMEDIATE = 3072


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Llama's RMSNorm. Per-token, so it is unaffected by every split except TP-on-hidden."""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


@dataclass
class TransformerBlock:
    """One transformer block, held as plain tensors so every shard is explicit and visible.

    Shapes follow HF's Llama convention: each projection is (out_features, in_features), so a
    forward is `x @ W.T`.
    """

    q: torch.Tensor  # (n_heads * head_dim, hidden)
    k: torch.Tensor  # (n_kv_heads * head_dim, hidden)
    v: torch.Tensor  # (n_kv_heads * head_dim, hidden)
    o: torch.Tensor  # (hidden, n_heads * head_dim)
    gate: torch.Tensor  # (intermediate, hidden)
    up: torch.Tensor  # (intermediate, hidden)
    down: torch.Tensor  # (hidden, intermediate)
    attn_norm: torch.Tensor  # (hidden,)
    mlp_norm: torch.Tensor  # (hidden,)
    n_heads: int = N_HEADS
    n_kv_heads: int = N_KV_HEADS
    head_dim: int = HEAD_DIM

    @property
    def hidden(self) -> int:
        return self.attn_norm.shape[0]

    @property
    def intermediate(self) -> int:
        return self.gate.shape[0]

    def n_params(self) -> int:
        return sum(
            t.numel() for t in (self.q, self.k, self.v, self.o, self.gate, self.up, self.down)
        )

    def weight_bytes(self) -> int:
        return self.n_params() * self.q.element_size()

    def to_device(self, device: torch.device) -> TransformerBlock:
        """Move every tensor to `device`. Returns self so it can be chained onto a loader."""
        for name in ("q", "k", "v", "o", "gate", "up", "down", "attn_norm", "mlp_norm"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ---- forward -------------------------------------------------------------------------

    def attention(self, x: torch.Tensor, kv_source: torch.Tensor | None = None) -> torch.Tensor:
        """Causal multi-head attention over `x`, shape **(batch, seq, hidden)**.

        The batch and sequence axes are kept **separate and explicit**, which matters more here
        than it would in ordinary model code: the whole point of T5 is that different strategies
        split different axes. Data parallelism splits `batch`; sequence parallelism splits `seq`.
        Collapse them into one "tokens" axis and DP silently becomes SP-without-the-gather — it
        will run, it will be fast, and it will be wrong, because attention mixes tokens along
        `seq` but never along `batch`.

        `kv_source` exists for sequence parallelism: a rank holding only a slice of the sequence
        computes queries from its own slice but keys and values from the gathered full sequence.
        When None (every other strategy) it is just `x`, and this is ordinary self-attention.
        """
        if x.dim() != 3:
            raise ValueError(
                f"expected (batch, seq, hidden), got shape {tuple(x.shape)}. The batch and "
                "sequence axes must stay distinct — see this method's docstring."
            )
        kv_input = x if kv_source is None else kv_source
        b, q_len, _ = x.shape
        kv_len = kv_input.shape[1]

        # (b, len, n*hd) -> (b, n, len, hd)
        q = (x @ self.q.T).view(b, q_len, -1, self.head_dim).transpose(1, 2)
        k = (kv_input @ self.k.T).view(b, kv_len, -1, self.head_dim).transpose(1, 2)
        v = (kv_input @ self.v.T).view(b, kv_len, -1, self.head_dim).transpose(1, 2)

        if k.shape[1] != q.shape[1]:  # GQA: repeat each kv head to cover its query group
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask. Under SP the queries are a slice starting at `kv_len - q_len`, so the mask
        # must be offset by that amount or the slice would attend to its own future.
        offset = kv_len - q_len
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=x.device).tril(diagonal=offset)
        scores = scores.masked_fill(~causal, float("-inf"))

        out = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, q_len, -1)
        return out @ self.o.T

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP: the part TP shards via a column/row pair needing one all-reduce."""
        return (F.silu(x @ self.gate.T) * (x @ self.up.T)) @ self.down.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full unsharded block — the reference every parallel variant must reproduce."""
        h = x + self.attention(rms_norm(x, self.attn_norm))
        return h + self.mlp(rms_norm(h, self.mlp_norm))

    # ---- sharding ------------------------------------------------------------------------

    def tensor_parallel_shard(self, rank: int, world_size: int) -> TransformerBlock:
        """This rank's TP shard: attention split by *head*, MLP split by *intermediate*.

        Attention is column-parallel on q/k/v (each rank owns whole heads) and row-parallel on o.
        The MLP is column-parallel on gate/up and row-parallel on down. Both halves are arranged
        so each rank produces a *partial sum* over the full hidden dim, and one all-reduce per
        half reconstructs the true output — two collectives per block, which is why TP wants
        NVLink-class bandwidth and stays inside a node.
        """
        if self.n_heads % world_size:
            raise ValueError(
                f"TP degree {world_size} must divide the {self.n_heads} attention heads. "
                "TP degree is not a free parameter — see this module's docstring."
            )
        if self.n_kv_heads % world_size:
            raise ValueError(
                f"TP degree {world_size} exceeds/misaligns the {self.n_kv_heads} KV heads. "
                "Real engines replicate KV heads past this point; here it is an explicit error."
            )
        if self.intermediate % world_size:
            raise ValueError(f"TP degree {world_size} must divide intermediate {self.intermediate}")

        hq = self.n_heads // world_size
        hkv = self.n_kv_heads // world_size
        qs, ks = hq * self.head_dim, hkv * self.head_dim
        i_per = self.intermediate // world_size

        return TransformerBlock(
            q=self.q[rank * qs : (rank + 1) * qs].contiguous(),
            k=self.k[rank * ks : (rank + 1) * ks].contiguous(),
            v=self.v[rank * ks : (rank + 1) * ks].contiguous(),
            o=self.o[:, rank * qs : (rank + 1) * qs].contiguous(),
            gate=self.gate[rank * i_per : (rank + 1) * i_per].contiguous(),
            up=self.up[rank * i_per : (rank + 1) * i_per].contiguous(),
            down=self.down[:, rank * i_per : (rank + 1) * i_per].contiguous(),
            attn_norm=self.attn_norm.clone(),
            mlp_norm=self.mlp_norm.clone(),
            n_heads=hq,
            n_kv_heads=hkv,
            head_dim=self.head_dim,
        )


def random_block(seed: int = 0) -> TransformerBlock:
    """A correctly-shaped block with random weights, for tests that must not hit the network."""
    g = torch.Generator().manual_seed(seed)
    s = HIDDEN**-0.5
    return TransformerBlock(
        q=torch.randn(N_HEADS * HEAD_DIM, HIDDEN, generator=g) * s,
        k=torch.randn(N_KV_HEADS * HEAD_DIM, HIDDEN, generator=g) * s,
        v=torch.randn(N_KV_HEADS * HEAD_DIM, HIDDEN, generator=g) * s,
        o=torch.randn(HIDDEN, N_HEADS * HEAD_DIM, generator=g) * s,
        gate=torch.randn(INTERMEDIATE, HIDDEN, generator=g) * s,
        up=torch.randn(INTERMEDIATE, HIDDEN, generator=g) * s,
        down=torch.randn(HIDDEN, INTERMEDIATE, generator=g) * s,
        attn_norm=torch.ones(HIDDEN),
        mlp_norm=torch.ones(HIDDEN),
    )


def load_block(layer: int = 0, model_id: str = MODEL_ID) -> TransformerBlock:
    """Load one real transformer block straight from safetensors, without building the model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    prefix = f"model.layers.{layer}"
    names = {
        "q": f"{prefix}.self_attn.q_proj.weight",
        "k": f"{prefix}.self_attn.k_proj.weight",
        "v": f"{prefix}.self_attn.v_proj.weight",
        "o": f"{prefix}.self_attn.o_proj.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
        "attn_norm": f"{prefix}.input_layernorm.weight",
        "mlp_norm": f"{prefix}.post_attention_layernorm.weight",
    }
    with safe_open(path, framework="pt") as f:
        available = set(f.keys())
        missing = [n for n in names.values() if n not in available]
        if missing:
            raise KeyError(f"missing tensors in {model_id}: {missing}")
        p = {k: f.get_tensor(v).float() for k, v in names.items()}

    head_dim = p["q"].shape[0] // N_HEADS
    return TransformerBlock(
        q=p["q"],
        k=p["k"],
        v=p["v"],
        o=p["o"],
        gate=p["gate"],
        up=p["up"],
        down=p["down"],
        attn_norm=p["attn_norm"],
        mlp_norm=p["mlp_norm"],
        n_heads=N_HEADS,
        n_kv_heads=p["k"].shape[0] // head_dim,
        head_dim=head_dim,
    )
