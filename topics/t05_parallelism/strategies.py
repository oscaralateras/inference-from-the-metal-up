"""Experiment C — the five ways to split a transformer, measured head-to-head.

One model, one workload, five decompositions, real collectives via `torch.distributed`:

  DP  data parallelism      — replicate the block, split the *requests*.        no collectives
  TP  tensor parallelism    — split every weight matrix, all-reduce partials.   2 all-reduce/block
  PP  pipeline parallelism  — split by *depth*, stream microbatches.            send/recv per seam
  SP  sequence parallelism  — split the *sequence*, all-gather K/V.             1 all-gather/block
  EP  expert parallelism    — split the *experts* of an MoE.                    gather + reduce

Backend-agnostic on purpose: `gloo` on CPU for free development and correctness, `nccl` on GPU for
the numbers that count. The strategy code is identical either way, so the CPU run is a genuine
rehearsal and the GPU session is pure measurement.

Three things are recorded per (strategy, world_size):

  **throughput**       — tokens/sec, the headline comparison.
  **comms bytes**      — payload actually handed to collectives per step. Device-independent: it
                         falls out of the shapes, so it stays true on hardware we did not rent,
                         and it is what lets a CPU run say something honest about GPUs.
  **weight bytes/rank** — what each rank must hold. DP replicates and cannot help a model that
                         does not fit; TP/PP/EP genuinely divide it. This is the axis throughput
                         alone hides.

Every strategy is also checked against the unsharded forward, because a fast wrong answer is the
easiest thing to produce here.

    uv run python topics/t05_parallelism/strategies.py --backend gloo --world-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from model import TransformerBlock, load_block, random_block, rms_norm
from moe import build_moe, load_factor, route

STRATEGIES = ("dp", "tp", "pp", "sp", "ep")

# Workload. TOKENS is the sequence the block processes per step; LAYERS is the model depth that
# pipeline parallelism has to divide. Both are deliberately modest on CPU and overridden for GPU.
# Batch and sequence are kept as SEPARATE axes because the strategies split different ones:
# DP and PP split `batch`, SP splits `seq`. Collapsing them into one "tokens" axis makes DP
# silently become sequence-parallelism-without-the-gather — fast, plausible, and wrong.
DEFAULT_BATCH = 16
DEFAULT_SEQ = 128
DEFAULT_LAYERS = 8
DEFAULT_MICROBATCHES = 8
N_EXPERTS = 8
WARMUP_STEPS = 2
TIMED_STEPS = 5


@dataclass
class Result:
    """One (strategy, world_size) observation, gathered from rank 0."""

    strategy: str
    world_size: int
    backend: str
    tokens_per_s: float
    comms_bytes_per_step: int
    weight_bytes_per_rank: int
    max_rel_err: float
    routing: str = ""
    load_factor: float = 1.0


def _elem_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# --------------------------------------------------------------------------------------------
# the five strategies. each returns (output_or_None, comms_bytes) for ONE step.
# only rank 0 returns a reconstructed output; the rest return None.
# --------------------------------------------------------------------------------------------


def step_dp(
    block: TransformerBlock, x: torch.Tensor, rank: int, world: int, layers: int
) -> tuple[torch.Tensor | None, int]:
    """Data parallel: every rank holds the whole block and takes a slice of the tokens.

    Zero collectives at inference time — requests are independent, so there is nothing to
    coordinate. That is DP's whole appeal, and also its whole limit: it needs the entire model to
    fit on one device, so it buys throughput and never capacity.
    """
    per = x.shape[0] // world  # split the BATCH axis: sequences are independent of each other
    local = x[rank * per : (rank + 1) * per]
    for _ in range(layers):
        local = block.forward(local)

    # all_gather rather than gather: NCCL does not implement gather, and this harness must run
    # unchanged on both backends. The reassembly exists only so the correctness check can compare
    # against the unsharded forward — a real DP server never does it (each rank returns its own
    # requests) — so it is excluded from the comms accounting below.
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.contiguous())
    return (torch.cat(gathered, dim=0) if rank == 0 else None), 0


def step_tp(
    shard: TransformerBlock, x: torch.Tensor, rank: int, world: int, layers: int
) -> tuple[torch.Tensor | None, int]:
    """Tensor parallel: every rank holds a slice of every matrix and sees every token.

    Two all-reduces per layer — one after attention, one after the MLP — because the column/row
    split is arranged so each rank produces a *partial sum* over the full hidden dim. The residual
    add must happen after the all-reduce, not before, or the residual gets counted `world` times.
    """
    comms = 0
    h = x
    for _ in range(layers):
        attn = shard.attention(rms_norm(h, shard.attn_norm))
        dist.all_reduce(attn)
        comms += _elem_bytes(attn)
        h = h + attn  # residual AFTER the reduce

        mlp = shard.mlp(rms_norm(h, shard.mlp_norm))
        dist.all_reduce(mlp)
        comms += _elem_bytes(mlp)
        h = h + mlp
    return (h if rank == 0 else None), comms


def step_pp(
    block: TransformerBlock,
    x: torch.Tensor,
    rank: int,
    world: int,
    layers: int,
    microbatches: int,
) -> tuple[torch.Tensor | None, int]:
    """Pipeline parallel: rank r owns a contiguous slice of the depth; microbatches stream through.

    Communication is tiny — one activation hand-off per seam per microbatch — but the *bubble* is
    structural: rank r cannot start microbatch 0 until ranks 0..r-1 have finished it, and it runs
    dry at the end. Efficiency tops out at M/(M+P-1), which is why PP wants many microbatches in
    flight and why small decode batches suit it so badly.
    """
    layers_here = layers // world
    per = x.shape[0] // microbatches  # microbatches partition the BATCH axis
    comms = 0
    collected: list[torch.Tensor] = []

    for m in range(microbatches):
        if rank == 0:
            h = x[m * per : (m + 1) * per].clone()
        else:
            h = torch.empty(per, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)
            dist.recv(h, src=rank - 1)

        for _ in range(layers_here):
            h = block.forward(h)

        if rank == world - 1:
            collected.append(h)
        else:
            dist.send(h, dst=rank + 1)
            comms += _elem_bytes(h)

    # Last stage owns the answer; ship it to rank 0 for the correctness check only.
    if world > 1:
        if rank == world - 1:
            out = torch.cat(collected, dim=0)
            dist.send(out, dst=0)
            return None, comms
        if rank == 0:
            out = torch.empty_like(x)
            dist.recv(out, src=world - 1)
            return out, comms
        return None, comms
    return torch.cat(collected, dim=0), comms


def step_sp(
    block: TransformerBlock, x: torch.Tensor, rank: int, world: int, layers: int
) -> tuple[torch.Tensor | None, int]:
    """Sequence parallel: every rank holds the whole block but only a slice of the *sequence*.

    In the MLP this is indistinguishable from DP — tokens are independent, so each rank just does
    its own. Attention is what makes SP a distinct strategy: a query can attend to any earlier
    token, so each rank must all-gather the normed hidden state to build keys and values over the
    full sequence. That gather is SP's characteristic cost and it grows with sequence length.
    """
    per = x.shape[1] // world  # split the SEQUENCE axis - this is what makes SP not DP
    lo = rank * per
    local = x[:, lo : lo + per]
    comms = 0

    for _ in range(layers):
        normed_local = rms_norm(local, block.attn_norm)
        parts = [torch.empty_like(normed_local) for _ in range(world)]
        dist.all_gather(parts, normed_local)
        comms += _elem_bytes(normed_local) * (world - 1)
        full = torch.cat(parts, dim=1)

        # Causal: this slice only ever attends to tokens at or before its own end.
        attn = block.attention(normed_local, kv_source=full[:, : lo + per])
        h = local + attn
        local = h + block.mlp(rms_norm(h, block.mlp_norm))  # per-token, no comms

    # all_gather, not gather: NCCL has no gather. Reassembly is for the correctness check only.
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local.contiguous())
    return (torch.cat(gathered, dim=1) if rank == 0 else None), comms


def step_ep(
    moe, x: torch.Tensor, assignment: torch.Tensor, rank: int, world: int
) -> tuple[torch.Tensor | None, int]:
    """Expert parallel: rank r owns experts [r*E/W, (r+1)*E/W) and processes only their tokens.

    Unlike the other four, the work per rank is **data-dependent**: it is decided by the router,
    not by the decomposition. Under skewed routing the rank holding the popular expert becomes the
    critical path and every other rank idles at the all-reduce. Adding hardware does not fix it —
    the imbalance is in the routing.
    """
    experts_per_rank = moe.n_experts // world
    lo = rank * experts_per_rank
    flat = x.reshape(-1, x.shape[-1])  # routing is per token, so batch and seq collapse here
    out = torch.zeros_like(flat)

    for e in range(lo, lo + experts_per_rank):
        idx = (assignment == e).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = moe.forward_expert(e, flat[idx])

    # Combine: every rank filled disjoint rows, so summing reconstructs the full output. A
    # production stack uses all-to-all instead (send tokens to the owning rank rather than
    # broadcasting all of them); this over-communicates but isolates the imbalance cleanly.
    dist.all_reduce(out)
    return (out.view_as(x) if rank == 0 else None), _elem_bytes(out)


# --------------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------------


def _worker(rank: int, world: int, cfg: dict, out_path: str) -> None:
    backend = cfg["backend"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(cfg["port"]))
    dist.init_process_group(backend=backend, rank=rank, world_size=world)

    device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    else:
        torch.set_num_threads(max(1, cfg["threads_per_rank"]))

    block = (random_block() if cfg["random_weights"] else load_block(0)).to_device(device)
    batch, seq, layers = cfg["batch"], cfg["seq"], cfg["layers"]
    tokens = batch * seq
    x = torch.randn(
        batch, seq, block.hidden, generator=torch.Generator().manual_seed(11), dtype=torch.float32
    ).to(device)

    moe = build_moe(block.hidden, block.intermediate // 2, N_EXPERTS).to_device(device)
    assignment = route(tokens, N_EXPERTS, cfg["routing"], seed=5).to(device)

    # Reference output, computed identically on every rank so the check needs no extra comms.
    with torch.no_grad():
        if cfg["strategy"] == "ep":
            reference = moe.forward(x.reshape(-1, x.shape[-1]), assignment).view_as(x)
        else:
            ref = x
            for _ in range(layers):
                ref = block.forward(ref)
            reference = ref

    shard = block.tensor_parallel_shard(rank, world) if cfg["strategy"] == "tp" else block

    def one_step() -> tuple[torch.Tensor | None, int]:
        s = cfg["strategy"]
        if s == "dp":
            return step_dp(block, x, rank, world, layers)
        if s == "tp":
            return step_tp(shard, x, rank, world, layers)
        if s == "pp":
            return step_pp(block, x, rank, world, layers, cfg["microbatches"])
        if s == "sp":
            return step_sp(block, x, rank, world, layers)
        return step_ep(moe, x, assignment, rank, world)

    out: torch.Tensor | None = None
    comms = 0
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            one_step()

        if backend == "nccl":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            out, comms = one_step()
        if backend == "nccl":
            torch.cuda.synchronize()
        dist.barrier()
        wall = time.perf_counter() - t0

    if rank == 0:
        err = 0.0
        if out is not None:
            denom = reference.abs().max().clamp_min(1e-12)
            err = float((out - reference).abs().max() / denom)
        weight_bytes = (
            moe.expert_bytes() * (N_EXPERTS // world)
            if cfg["strategy"] == "ep"
            else _strategy_weight_bytes(cfg["strategy"], block, shard, layers, world)
        )
        result = Result(
            strategy=cfg["strategy"],
            world_size=world,
            backend=backend,
            tokens_per_s=(tokens * layers * TIMED_STEPS) / wall,
            comms_bytes_per_step=comms,
            weight_bytes_per_rank=weight_bytes,
            max_rel_err=err,
            routing=cfg["routing"] if cfg["strategy"] == "ep" else "",
            load_factor=load_factor(assignment.cpu(), N_EXPERTS, world)
            if cfg["strategy"] == "ep"
            else 1.0,
        )
        Path(out_path).write_text(json.dumps(asdict(result)))

    dist.destroy_process_group()


def _strategy_weight_bytes(
    strategy: str, block: TransformerBlock, shard: TransformerBlock, layers: int, world: int
) -> int:
    """What one rank must actually hold — the axis throughput hides.

    DP and SP replicate every layer, so they cannot help a model that does not fit. TP holds a
    slice of every layer; PP holds every parameter of a slice of the layers. Both genuinely divide
    the footprint, which is why they, and not DP, are the answers to "the model is too big".
    """
    if strategy in ("dp", "sp"):
        return block.weight_bytes() * layers
    if strategy == "tp":
        return shard.weight_bytes() * layers
    if strategy == "pp":
        return block.weight_bytes() * (layers // world)
    raise ValueError(strategy)


def run(strategy: str, world: int, cfg: dict) -> Result:
    """Spawn `world` ranks for one (strategy, world_size) point and return rank 0's result."""
    full = {**cfg, "strategy": strategy}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world, full, out_path), nprocs=world, join=True
    )
    return Result(**json.loads(Path(out_path).read_text()))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-sizes", default="1,2,4")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--microbatches", type=int, default=DEFAULT_MICROBATCHES)
    parser.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--threads-per-rank", type=int, default=1)
    parser.add_argument("--port", type=int, default=29511)
    args = parser.parse_args()

    world_sizes = [int(w) for w in args.world_sizes.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    cfg = {
        "backend": args.backend,
        "batch": args.batch,
        "seq": args.seq,
        "layers": args.layers,
        "microbatches": args.microbatches,
        "routing": args.routing,
        "random_weights": args.random_weights,
        "threads_per_rank": args.threads_per_rank,
        "port": args.port,
    }

    print(
        f"backend={args.backend}  batch={args.batch}  seq={args.seq}  layers={args.layers}  "
        f"microbatches={args.microbatches}  routing={args.routing}"
    )
    print(
        f"{'strategy':>9} {'W':>3} {'tok/s':>12} {'comms MB/step':>14} "
        f"{'wt MB/rank':>11} {'rel err':>10} {'load':>6}"
    )
    print("-" * 74)

    rows: list[dict[str, object]] = []
    for strategy in strategies:
        for world in world_sizes:
            if strategy == "pp" and args.layers % world:
                continue  # depth must divide across stages
            if strategy == "ep" and N_EXPERTS % world:
                continue
            if strategy == "dp" and args.batch % world:
                continue
            if strategy == "sp" and args.seq % world:
                continue
            try:
                r = run(strategy, world, cfg)
            except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                print(f"{strategy:>9} {world:>3}   FAILED: {type(exc).__name__}: {exc}")
                continue
            print(
                f"{r.strategy:>9} {r.world_size:>3} {r.tokens_per_s:>12,.0f} "
                f"{r.comms_bytes_per_step / 1e6:>14.2f} {r.weight_bytes_per_rank / 1e6:>11.1f} "
                f"{r.max_rel_err:>10.2e} {r.load_factor:>6.2f}"
            )
            for metric, value in (
                ("tokens_per_s", r.tokens_per_s),
                ("comms_bytes_per_step", r.comms_bytes_per_step),
                ("weight_bytes_per_rank", r.weight_bytes_per_rank),
                ("max_rel_err", r.max_rel_err),
                ("load_factor", r.load_factor),
            ):
                rows.append(
                    {
                        "experiment": f"strategies_{args.backend}",
                        "variant": f"{r.strategy}{'_' + r.routing if r.routing else ''}",
                        "workers": r.world_size,
                        "metric": metric,
                        "value": f"{value:.6f}",
                    }
                )

    from results_io import append_rows

    append_rows(rows)


if __name__ == "__main__":
    _main()
