"""A synthetic mixture-of-experts layer — the thing expert parallelism splits.

Dense models have nothing to shard by expert, so EP needs an MoE block. This is a deliberately
small, explicit one: `n_experts` independent SwiGLU MLPs plus a router that assigns each token to
exactly one expert (top-1).

EP's failure mode is different in kind from the other four strategies. DP, TP, PP and SP all split
a *fixed* amount of work into equal pieces; the costs they pay are communication and pipeline
bubbles. EP's work is **data-dependent**: how much each rank does depends on where the router
sends the tokens. If routing is uniform, every rank does 1/N of the work. If one expert is popular,
the rank holding it becomes the critical path and every other rank waits at the next collective —
no amount of extra hardware helps, because the imbalance is in the routing, not the machine.

The router here is therefore configurable rather than learned, so the imbalance can be dialled:

  - `uniform`  — tokens spread evenly across experts (the best case, and the one papers assume)
  - `skewed`   — a Zipf-like distribution where one expert attracts a large share (the real case
                 that MoE serving stacks fight with auxiliary load-balancing losses and expert
                 capacity limits)

That contrast is the whole experiment: identical FLOPs, identical hardware, and throughput decided
purely by how evenly the router spread the work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MoELayer:
    """`n_experts` independent SwiGLU MLPs plus a top-1 router."""

    gate: torch.Tensor  # (n_experts, expert_intermediate, hidden)
    up: torch.Tensor  # (n_experts, expert_intermediate, hidden)
    down: torch.Tensor  # (n_experts, hidden, expert_intermediate)

    @property
    def n_experts(self) -> int:
        return self.gate.shape[0]

    @property
    def hidden(self) -> int:
        return self.gate.shape[2]

    def expert_bytes(self) -> int:
        """Bytes for ONE expert — this is what a rank actually holds under EP."""
        per = self.gate[0].numel() + self.up[0].numel() + self.down[0].numel()
        return per * self.gate.element_size()

    def to_device(self, device: torch.device) -> MoELayer:
        """Move every expert's tensors to `device`. Returns self for chaining."""
        for name in ("gate", "up", "down"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def forward_expert(self, expert: int, x: torch.Tensor) -> torch.Tensor:
        """Run one expert over the tokens routed to it."""
        if x.shape[0] == 0:  # a rank can legitimately receive zero tokens under skewed routing
            return x
        g, u, d = self.gate[expert], self.up[expert], self.down[expert]
        return (F.silu(x @ g.T) * (x @ u.T)) @ d.T

    def forward(self, x: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        """Unsharded reference: every expert run locally, tokens reassembled in order."""
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            idx = (assignment == e).nonzero(as_tuple=True)[0]
            if idx.numel():
                out[idx] = self.forward_expert(e, x[idx])
        return out


def build_moe(hidden: int, expert_intermediate: int, n_experts: int, seed: int = 0) -> MoELayer:
    """Random-weight MoE of the given shape.

    Random rather than real weights because no small open MoE has a convenient single-file
    checkpoint, and EP's behaviour depends on the *routing distribution* and the shapes, not on
    the weight values. This is stated in the lab note's caveats.
    """
    g = torch.Generator().manual_seed(seed)
    s = hidden**-0.5
    return MoELayer(
        gate=torch.randn(n_experts, expert_intermediate, hidden, generator=g) * s,
        up=torch.randn(n_experts, expert_intermediate, hidden, generator=g) * s,
        down=torch.randn(n_experts, hidden, expert_intermediate, generator=g) * s,
    )


def route(
    n_tokens: int, n_experts: int, mode: str = "uniform", seed: int = 0, skew: float = 1.6
) -> torch.Tensor:
    """Assign each token to exactly one expert. Returns an int64 tensor of expert ids.

    `uniform` spreads tokens round-robin — the idealised case. `skewed` draws from a Zipf-like
    distribution over experts (probability proportional to 1/(rank+1)^skew), so expert 0 attracts
    a large share. Deterministic given `seed`, so every strategy sees the identical assignment
    and the comparison is like-for-like.
    """
    if mode == "uniform":
        return torch.arange(n_tokens, dtype=torch.int64) % n_experts
    if mode == "skewed":
        g = torch.Generator().manual_seed(seed)
        weights = torch.tensor([1.0 / (i + 1) ** skew for i in range(n_experts)])
        probs = weights / weights.sum()
        return torch.multinomial(probs, n_tokens, replacement=True, generator=g).to(torch.int64)
    raise ValueError(f"unknown routing mode {mode!r}; expected 'uniform' or 'skewed'")


def load_factor(assignment: torch.Tensor, n_experts: int, world_size: int) -> float:
    """Ratio of the busiest rank's token count to the average — the imbalance in one number.

    1.0 means perfectly balanced. 2.0 means the busiest rank does twice the average, so the step
    takes twice as long as it should because everyone waits for it. This is the number that
    predicts EP's throughput, and it is a property of the *routing*, not of the hardware.
    """
    experts_per_rank = n_experts // world_size
    counts = torch.bincount(assignment, minlength=n_experts)
    per_rank = counts.view(world_size, experts_per_rank).sum(dim=1).float()
    return float(per_rank.max() / per_rank.mean())
