"""Measure what a real model actually does, so the prediction has something to be wrong about.

Three experiments, all end-to-end over the whole model with a real KV cache:

* **decode**   - single-stream tokens/sec at a fixed context length. The headline number.
* **batching** - tokens/sec and per-token p50/p99 across batch sizes. Little's law, and the
                 throughput-versus-tail-latency tradeoff that decides real serving configs.
* **context**  - tokens/sec across context lengths. The KV cache term made visible: the same
                 model gets slower the more it has already said.

Plus two calibration measurements that feed the gap decomposition: achieved per-launch overhead,
and the number of module calls per decoded token.

    python measure.py --device cuda --model Qwen/Qwen2.5-7B
    python measure.py --device cpu --model hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from typing import cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from arch_common.timing import synchronize, time_op
from topics.t06_perf_reasoning.model_math import ModelShape

CSV_PATH = Path(__file__).parent / "results" / "perf.csv"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32)
DEFAULT_CONTEXTS = (512, 2048, 8192)
DEFAULT_SEQ_LEN = 512
DEFAULT_TOKENS = 128  # enough samples for p99 to mean something
WARMUP_TOKENS = 5

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(name: str, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    """Load a causal LM in eval mode on `device`, with gradients globally off."""
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True)
    module = cast(torch.nn.Module, model)
    module.to(device)
    module.eval()
    module.requires_grad_(False)
    return module


def shape_from_model(name: str, bytes_per_param: int) -> ModelShape:
    """The analytic shape, read from the same config the weights were built from."""
    config = AutoConfig.from_pretrained(name).to_dict()
    config.setdefault("_name_or_path", name)
    return ModelShape.from_config(config, bytes_per_param=bytes_per_param)


@torch.inference_mode()
def decode_latencies(
    model, device: torch.device, batch: int, seq_len: int, n_tokens: int, vocab: int
) -> list[float]:
    """Prefill a batch, then decode `n_tokens` one at a time, returning per-token milliseconds.

    Timed per token rather than once around the whole loop, because the per-token *distribution*
    is half the point — an outer timer reports a mean and hides the tail completely.

    Each step is timed and its output reused. Decoding the same step twice (once to time it, once
    to advance) would be wrong, not merely wasteful: the KV cache is mutated in place, so the
    second call would append a duplicate entry and silently corrupt the context.
    """
    prompt = torch.randint(0, vocab, (batch, seq_len), device=device)
    out = model(input_ids=prompt, use_cache=True)
    past = out.past_key_values
    token = out.logits[:, -1:].argmax(dim=-1)

    # Untimed steps first: the decode shapes differ from the prefill shapes, so cuBLAS has not yet
    # autotuned for them and the first few tokens are unrepresentatively slow.
    for _ in range(WARMUP_TOKENS):
        out = model(input_ids=token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
    synchronize(device)

    latencies: list[float] = []
    for _ in range(n_tokens):
        with _StepTimer(device) as timer:
            out = model(input_ids=token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        latencies.append(timer.elapsed_ms)

    return latencies


class _StepTimer:
    """Time one decode step, using CUDA events on GPU and a monotonic clock on CPU.

    `time_op` cannot be used here because it discards the callable's return value, and each decode
    step's output is needed to produce the next one.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.elapsed_ms = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._start_ns = 0

    def __enter__(self) -> _StepTimer:
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1e6


def summarise(latencies: list[float], batch: int) -> dict[str, float]:
    """Turn a per-token latency sample into the three numbers that describe a serving config."""
    if not latencies:
        raise ValueError("no latency samples — the decode loop produced nothing")
    ordered = sorted(latencies)
    # Nearest-rank percentile: the smallest sample at or above 99% of the distribution. With N
    # samples the resolution is 1/N, so p99 is only meaningful once N is comfortably over 100 —
    # `--tokens` defaults high enough for that and the lab note states the sample count.
    p99_index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
    median_ms = statistics.median(ordered)
    return {
        "tokens_per_sec": batch / (median_ms * 1e-3),
        "latency_p50_ms": median_ms,
        "latency_p99_ms": ordered[p99_index],
    }


@torch.inference_mode()
def measure_launch_overhead(device: torch.device) -> float:
    """Milliseconds per kernel launch, from a tensor small enough that the work is free.

    A 32-element add does essentially no memory traffic and no arithmetic, so what is left is the
    fixed cost of getting a kernel onto the GPU. Multiplied by the launches per token, this is the
    part of decode that no amount of bandwidth would fix.
    """
    tiny = torch.ones(32, device=device)
    return time_op(lambda: tiny.add_(1.0), device, iters=200)


@torch.inference_mode()
def count_module_calls(model, device: torch.device, vocab: int) -> int:
    """Leaf modules invoked to decode one token — a lower bound on kernel launches.

    A lower bound, not a count: elementwise ops, norms fused inside a module and anything in
    functional form are invisible to module hooks. Stated as a floor in the lab note rather than
    quietly presented as exact.
    """
    calls = 0

    def bump(*_args: object) -> None:
        nonlocal calls
        calls += 1

    leaves = [m for m in model.modules() if not list(m.children())]
    handles = [m.register_forward_hook(bump) for m in leaves]
    try:
        prompt = torch.randint(0, vocab, (1, 8), device=device)
        past = model(input_ids=prompt, use_cache=True).past_key_values
        synchronize(device)
        calls = 0
        model(input_ids=prompt[:, -1:], past_key_values=past, use_cache=True)
        synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    profile = load_profile()
    shape = shape_from_model(args.model, torch.finfo(dtype).bits // 8)

    print(f"model {args.model} on {profile.device_name} ({args.dtype})")
    print(
        f"analytic params {shape.total_params / 1e9:.2f}B, read per token "
        f"{shape.params_read_per_token / 1e9:.2f}B\n"
    )

    model = load_model(args.model, dtype, device)
    rows: list[dict[str, object]] = []

    def record(experiment: str, variant: str, x: int, metrics: dict[str, float]) -> None:
        rows.extend(
            {
                "session_id": profile.session_id,
                "experiment": experiment,
                "variant": variant,
                "x": x,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        )

    # -- calibration ------------------------------------------------------------------------
    overhead_ms = measure_launch_overhead(device)
    module_calls = count_module_calls(model, device, shape.vocab_size)
    record("calibration", "launch", 0, {"per_launch_ms": overhead_ms})
    record("calibration", "launch", 0, {"module_calls_per_token": float(module_calls)})
    print(
        f"per-launch {overhead_ms * 1e3:.1f} us x >={module_calls} module calls "
        f"= >={overhead_ms * module_calls:.2f} ms/token of pure overhead\n"
    )

    # -- batching sweep (also supplies the headline single-stream number at batch 1) ----------
    print(f"{'batch':>6} {'tok/s':>12} {'p50 ms':>9} {'p99 ms':>9} {'concurrency':>12}")
    print("-" * 54)
    for batch in args.batches:
        latencies = decode_latencies(
            model, device, batch, args.seq_len, args.tokens, shape.vocab_size
        )
        stats = summarise(latencies, batch)
        # Little's law: concurrency = throughput x latency. Recovering the batch size we set is
        # the check that the law holds and that the measurement is self-consistent.
        stats["littles_law_concurrency"] = stats["tokens_per_sec"] * stats["latency_p50_ms"] * 1e-3
        record("batching", "measured", batch, stats)
        if batch == 1:
            record("decode", "measured", args.seq_len, stats)
        print(
            f"{batch:>6} {stats['tokens_per_sec']:>12,.1f} {stats['latency_p50_ms']:>9.2f} "
            f"{stats['latency_p99_ms']:>9.2f} {stats['littles_law_concurrency']:>12.2f}"
        )

    # -- context sweep: the KV cache term, made visible ---------------------------------------
    print(f"\n{'context':>8} {'tok/s':>12} {'kv MB/token':>13}")
    print("-" * 36)
    for context in args.contexts:
        latencies = decode_latencies(model, device, 1, context, args.tokens, shape.vocab_size)
        stats = summarise(latencies, 1)
        record("context", "measured", context, stats)
        kv_mb = shape.kv_cache_bytes(context) / 1e6
        print(f"{context:>8} {stats['tokens_per_sec']:>12,.1f} {kv_mb:>13,.1f}")

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    main()
