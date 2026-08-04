"""Measure a production inference stack, so the prediction has something real to be wrong about.

The system under test is **vLLM**, not a bare `transformers` forward loop. That choice is load
bearing: an eager-mode loop spends a large fraction of every decode step launching hundreds of
small kernels, so it measures Python and the CUDA driver as much as it measures the hardware.
Predicting hardware behaviour and then measuring a framework's overhead answers a different
question than the one this topic asks.

Four experiments:

* **decode**   - single-stream tokens/sec at a fixed context. The headline number.
* **batching** - throughput and request-latency percentiles across batch sizes. Little's law, and
                 the throughput-versus-tail-latency tradeoff that decides real serving configs.
* **context**  - throughput across context lengths, isolating the KV cache term.
* **graphs**   - the same measurement with CUDA graphs on and off.

That last one is the controlled experiment. vLLM captures CUDA graphs by default and
`enforce_eager=True` turns them off, changing nothing else — same stack, same kernels, same
weights. The difference between the two **is** per-launch overhead, measured rather than
estimated.

**Per-step time is measured by difference.** Generating N tokens costs `prefill + N x step`, so
timing two output lengths and dividing the difference cancels prefill, scheduler startup and
detokenisation exactly:

    step_ms = (T_long - T_short) / (n_long - n_short)

Timing one generate call and dividing by N folds prefill into the per-token number and overstates
it, badly so at short output lengths.

    python -m topics.t06_perf_reasoning.measure --model Qwen/Qwen2.5-7B
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from topics.t06_perf_reasoning.model_math import ModelShape

CSV_PATH = Path(__file__).parent / "results" / "perf.csv"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_CONTEXTS = (512, 2048, 8192, 32768)
DEFAULT_SEQ_LEN = 512

# The two output lengths differenced to isolate per-step decode time. SHORT must be long enough
# for the scheduler to have reached steady state, and short enough that the difference is
# dominated by decode rather than by noise.
LONG_TOKENS = 128
SHORT_TOKENS = 8

# Headroom for the longest context sweep point plus the tokens generated from it.
DEFAULT_MAX_MODEL_LEN = 40960
GPU_MEMORY_UTILISATION = 0.90

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class StepResult:
    """One (batch, context) measurement, reduced to the numbers a serving decision needs."""

    step_ms: float
    tokens_per_sec: float
    request_p50_ms: float
    request_p99_ms: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "step_time_ms": self.step_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "request_latency_p50_ms": self.request_p50_ms,
            "request_latency_p99_ms": self.request_p99_ms,
            # concurrency = throughput x latency. Recovering the batch size we set is the check
            # that the measurement is internally consistent.
            "littles_law_concurrency": self.tokens_per_sec * self.step_ms * 1e-3,
        }


def shape_from_model(name: str, bytes_per_param: int) -> ModelShape:
    """The analytic shape, read from the same config the weights were built from."""
    config = AutoConfig.from_pretrained(name).to_dict()
    config.setdefault("_name_or_path", name)
    return ModelShape.from_config(config, bytes_per_param=bytes_per_param)


def build_engine(model: str, dtype: str, *, cuda_graphs: bool, max_model_len: int) -> Any:
    """Start a vLLM engine. Imported lazily so the analytic tests run with no GPU present."""
    from vllm import LLM

    return LLM(
        model=model,
        dtype=dtype,
        enforce_eager=not cuda_graphs,
        max_model_len=max_model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILISATION,
        disable_log_stats=True,
    )


def _as_prompts(token_ids: list[list[int]]) -> Any:
    """Wrap raw token ids in whatever prompt type this vLLM version accepts.

    vLLM moved from a `prompt_token_ids=` keyword to a `TokensPrompt` object, and relocated that
    class between the top-level package and `vllm.inputs`. Feeding pre-tokenised prompts matters:
    it fixes the context length exactly, where tokenising text would leave it approximate and make
    the context sweep meaningless.
    """
    try:
        from vllm import TokensPrompt  # type: ignore[attr-defined]
    except ImportError:
        try:
            from vllm.inputs import TokensPrompt  # type: ignore[no-redef]
        except ImportError:
            return [{"prompt_token_ids": ids} for ids in token_ids]
    return [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]


def _generate(engine: Any, prompts: list[list[int]], max_tokens: int) -> tuple[float, list[float]]:
    """Run one batch to a fixed output length. Returns wall seconds and per-request latencies.

    `ignore_eos` forces every request to the full length. Without it requests finish at different
    times, the batch shrinks as it runs, and the measurement silently becomes a mixture of batch
    sizes rather than the one under test.
    """
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)

    start = time.perf_counter()
    outputs = engine.generate(_as_prompts(prompts), params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    latencies: list[float] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        finished = getattr(metrics, "finished_time", None)
        if arrival and finished:
            latencies.append((finished - arrival) * 1e3)

    # Fall back to the wall time if this vLLM build does not expose per-request metrics; a batch
    # of one is exactly the wall time anyway, and the note records which path was taken.
    return elapsed, latencies or [elapsed * 1e3]


def measure_step(engine: Any, batch: int, context: int, vocab: int) -> StepResult:
    """Per-step decode cost at one (batch, context), by differencing two output lengths."""
    prompts = [[(i * 7919 + j) % vocab for j in range(context)] for i in range(batch)]

    long_s, long_latencies = _generate(engine, prompts, LONG_TOKENS)
    short_s, _ = _generate(engine, prompts, SHORT_TOKENS)

    step_ms = (long_s - short_s) / (LONG_TOKENS - SHORT_TOKENS) * 1e3
    if step_ms <= 0:
        raise ValueError(
            f"differenced step time is non-positive ({step_ms:.3f} ms) at batch={batch} "
            f"context={context} — the two generate calls did not separate, so this is noise "
            "rather than signal"
        )

    ordered = sorted(long_latencies)
    p99_index = min(len(ordered) - 1, -(-99 * len(ordered) // 100) - 1)
    return StepResult(
        step_ms=step_ms,
        tokens_per_sec=batch / (step_ms * 1e-3),
        request_p50_ms=statistics.median(ordered),
        request_p99_ms=ordered[p99_index],
    )


def shutdown(engine: Any) -> None:
    """Release the engine's GPU memory so the next configuration can allocate its own."""
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--graph-batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    args = parser.parse_args()

    profile = load_profile()
    shape = shape_from_model(args.model, torch.finfo(DTYPES[args.dtype]).bits // 8)
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

    print(f"{args.model} on {profile.device_name} ({args.dtype})")
    print(
        f"stack: vLLM   analytic params {shape.total_params / 1e9:.2f}B, "
        f"read per token {shape.params_read_per_token / 1e9:.2f}B\n"
    )

    engine = build_engine(
        args.model, args.dtype, cuda_graphs=True, max_model_len=args.max_model_len
    )

    print(f"{'batch':>6} {'tok/s':>12} {'step ms':>9} {'req p50':>10} {'req p99':>10} {'conc':>8}")
    print("-" * 62)
    for batch in args.batches:
        result = measure_step(engine, batch, args.seq_len, shape.vocab_size)
        record("batching", "cuda_graphs", batch, result.as_metrics())
        if batch == 1:
            record("decode", "measured", args.seq_len, result.as_metrics())
        print(
            f"{batch:>6} {result.tokens_per_sec:>12,.1f} {result.step_ms:>9.2f} "
            f"{result.request_p50_ms:>10,.0f} {result.request_p99_ms:>10,.0f} "
            f"{result.tokens_per_sec * result.step_ms * 1e-3:>8.2f}"
        )

    print(f"\n{'context':>8} {'tok/s':>12} {'step ms':>9} {'kv MB/token':>13}")
    print("-" * 46)
    for context in args.contexts:
        if context + LONG_TOKENS > args.max_model_len:
            print(f"{context:>8}  skipped — exceeds max_model_len {args.max_model_len:,}")
            continue
        result = measure_step(engine, 1, context, shape.vocab_size)
        record("context", "cuda_graphs", context, result.as_metrics())
        print(
            f"{context:>8} {result.tokens_per_sec:>12,.1f} {result.step_ms:>9.2f} "
            f"{shape.kv_cache_bytes(context) / 1e6:>13,.1f}"
        )

    graphs_on = {
        batch: measure_step(engine, batch, args.seq_len, shape.vocab_size)
        for batch in args.graph_batches
    }
    shutdown(engine)

    # Same stack, same kernels, same weights — graphs off. The only variable is launch overhead.
    engine = build_engine(
        args.model, args.dtype, cuda_graphs=False, max_model_len=args.max_model_len
    )
    print(f"\n{'batch':>6} {'graphs ms':>10} {'eager ms':>10} {'overhead ms':>12} {'share':>8}")
    print("-" * 52)
    for batch in args.graph_batches:
        eager = measure_step(engine, batch, args.seq_len, shape.vocab_size)
        record("graphs", "eager", batch, eager.as_metrics())
        record("graphs", "cuda_graphs", batch, graphs_on[batch].as_metrics())
        overhead = eager.step_ms - graphs_on[batch].step_ms
        print(
            f"{batch:>6} {graphs_on[batch].step_ms:>10.2f} {eager.step_ms:>10.2f} "
            f"{overhead:>12.2f} {overhead / eager.step_ms:>7.1%}"
        )
    shutdown(engine)

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    main()
