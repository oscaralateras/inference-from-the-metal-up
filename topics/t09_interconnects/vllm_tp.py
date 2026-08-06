"""Stage 4 — stop modelling the end-to-end number and measure it.

    uv run python -m topics.t09_interconnects.vllm_tp --tp 1,2,4 --batches 1,8,32

The headline TP speedups in the lab note are **modelled**: Amdahl over T6's measured error budget,
with a measured alpha supplying the communication term. That model is deliberately optimistic — it
holds the non-weight fraction fixed under sharding, which T5 measured to be false, since each rank
runs a matmul 1/N the size and gets proportionally less from the GPU. So the model should
over-predict, and the size of the gap is worth knowing rather than guessing.

This runs the real thing: the same vLLM engine T6 measured, with `tensor_parallel_size` set, and
the same differencing method T6 used to isolate a decode step from prefill. Two engines cannot
share a process — vLLM holds CUDA and NCCL state per engine — so each TP size is a separate spawn,
which is also why the world sizes are looped over subprocesses rather than configured in one run.

The comparison this makes possible:

    modelled speedup    Amdahl(T6 weight share, N, measured alpha)      -- an upper bound
    measured speedup    step(TP=1) / step(TP=N)                         -- what actually happens

A model that lands close is a model worth carrying to hardware nobody has rented; a model that
over-predicts by a lot means the terms it holds fixed are the ones that matter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar
from topics.t09_interconnects.measure import CSV_PATH
from topics.t09_interconnects.model import RingCost, comms_per_token_us, predicted_tp_speedup
from topics.t09_interconnects.predict import t6_budget

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_BATCHES = (1, 8, 32)

# Same differencing constants T6 uses: generate to two output lengths and subtract, so prefill and
# engine start-up cancel and what remains is decode.
SHORT_TOKENS = 8
LONG_TOKENS = 64
CONTEXT_TOKENS = 512

# Left well below 1.0 because four engines' worth of fragmentation across a session adds up, and an
# OOM at TP=1 (which holds the whole model on one card) would cost the comparison its baseline.
GPU_MEMORY_UTILISATION = 0.85


def _child(model: str, dtype: str, tp: int, batches: list[int], out_path: str) -> None:
    """One engine, one TP size, in its own process. Never imported by the parent."""
    import time

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=tp,
        enforce_eager=False,
        max_model_len=CONTEXT_TOKENS + LONG_TOKENS + 8,
        gpu_memory_utilization=GPU_MEMORY_UTILISATION,
        disable_log_stats=True,
    )

    def run(prompts: list[list[int]], max_tokens: int) -> float:
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
        start = time.perf_counter()
        engine.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)
        return time.perf_counter() - start

    results: dict[str, float] = {}
    for batch in batches:
        prompts = [[(i * 7919 + j) % 30000 for j in range(CONTEXT_TOKENS)] for i in range(batch)]
        long_s = run(prompts, LONG_TOKENS)
        short_s = run(prompts, SHORT_TOKENS)
        step_ms = (long_s - short_s) / (LONG_TOKENS - SHORT_TOKENS) * 1e3
        if step_ms <= 0:
            raise ValueError(
                f"differenced step time is non-positive ({step_ms:.3f} ms) at batch={batch}, "
                f"tp={tp} — the two generate calls did not separate, so this is noise not signal"
            )
        results[str(batch)] = step_ms

    Path(out_path).write_text(json.dumps(results))


def run_tp(model: str, dtype: str, tp: int, batches: list[int]) -> dict[str, float]:
    """Spawn a fresh interpreter for one TP size and return its per-batch step times."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "topics.t09_interconnects.vllm_tp",
            "--child",
            "--model",
            model,
            "--dtype",
            dtype,
            "--tp",
            str(tp),
            "--batches",
            ",".join(str(b) for b in batches),
            "--out",
            out_path,
        ],
        check=True,
    )
    return json.loads(Path(out_path).read_text())


def _fitted(world: int) -> RingCost | None:
    """The measured cost model for this world size, if `measure.py` has been run."""
    if not CSV_PATH.exists():
        return None
    rows = read_rows(CSV_PATH)
    try:
        return RingCost(
            world=world,
            alpha_us=scalar(rows, "fit", f"world{world}", "alpha_us"),
            beta_gbps=scalar(rows, "fit", f"world{world}", "beta_gbps"),
            r_squared=1.0,
            n_points=0,
        )
    except KeyError:
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", default="1,2,4")
    parser.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    batches = [int(b) for b in args.batches.split(",")]

    if args.child:
        _child(args.model, args.dtype, int(args.tp), batches, args.out)
        return

    session = load_profile().session_id
    weight_share, _, _ = t6_budget()
    sizes = [int(t) for t in args.tp.split(",")]

    print(f"T9 stage 4 — vLLM under real tensor parallelism, {args.model}\n")

    baseline: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for tp in sizes:
        steps = run_tp(args.model, args.dtype, tp, batches)
        if tp == 1:
            baseline = steps

        print(f"TP{tp}:")
        fit = _fitted(tp)
        for batch in batches:
            step_ms = steps[str(batch)]
            measured = baseline.get(str(batch), 0.0) / step_ms if step_ms else 0.0

            line = (
                f"  batch {batch:>3}: step {step_ms:>7.2f} ms  "
                f"{1000 / step_ms * batch:>8.1f} tok/s  measured {measured:.2f}x"
            )
            modelled = 0.0
            if fit is not None and tp > 1:
                modelled = predicted_tp_speedup(
                    weight_share, tp, baseline[str(batch)], comms_per_token_us(fit, batch)
                )
                line += f"   modelled {modelled:.2f}x  (model/measured {modelled / measured:.2f}x)"
            print(line)

            for metric, value in (
                ("step_ms", step_ms),
                ("tokens_per_sec", 1000 / step_ms * batch),
                ("measured_speedup", measured),
                ("modelled_speedup", modelled),
            ):
                if value:
                    rows.append(
                        {
                            "session_id": session,
                            "experiment": "vllm_tp",
                            "variant": f"tp{tp}",
                            "x": batch,
                            "metric": metric,
                            "value": value,
                        }
                    )

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    _main()
