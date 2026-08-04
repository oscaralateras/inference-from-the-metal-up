"""Account for the gap between predicted and measured decode, term by term.

Predicting `bandwidth / bytes_per_token` is a five-line calculation. The finding is not the
prediction — it is the **error budget**: taking the measured step time and attributing it to named
causes until what remains is small enough to admit is unexplained.

The decomposition works in the **time** domain, because times add and throughputs do not:

    measured_ms  =  weights + kv_cache + activations + unexplained

Each explained term is bytes divided by the session's measured bandwidth. The residual is
reported, not absorbed into a fudge factor.

Launch overhead is deliberately **not** a term here. The measured step comes from vLLM with CUDA
graphs captured, which is exactly the mechanism that removes per-launch cost — charging for it
again would be double-counting. What launch overhead *would* have cost is measured separately, by
turning graphs off and changing nothing else, and reported on its own.

    python -m topics.t06_perf_reasoning.decompose
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar, select
from topics.t06_perf_reasoning.measure import CSV_PATH, DTYPES, shape_from_model
from topics.t06_perf_reasoning.predict import PREDICTIONS_PATH


@dataclass(frozen=True)
class Term:
    """One named contribution to the measured per-token time."""

    name: str
    ms: float
    note: str


def _bytes_to_ms(num_bytes: float, bandwidth_gbps: float) -> float:
    """Time to move `num_bytes` at the session's measured bandwidth, in milliseconds."""
    return num_bytes / (bandwidth_gbps * 1e9) * 1e3


def predicted_bytes(prediction: dict[str, object]) -> float:
    """Total bytes the pre-registered model says one decode step must move."""
    keys = ("weight_bytes_per_token", "kv_bytes_per_token", "activation_bytes_per_token")
    total = 0.0
    for key in keys:
        value = prediction[key]
        if not isinstance(value, (int, float)):
            raise TypeError(f"prediction field {key!r} is not numeric: {value!r}")
        total += float(value)
    return total


def build_terms(seq_len: int) -> tuple[list[Term], float]:
    """Return the explained terms plus the measured per-token time they are explaining."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"no pre-registered prediction at {PREDICTIONS_PATH} — run predict.py before "
            "measuring, or the prediction is not a prediction"
        )
    prediction = json.loads(PREDICTIONS_PATH.read_text())
    profile = load_profile()
    rows = read_rows(CSV_PATH)

    if prediction["session_id"] != profile.session_id:
        raise ValueError(
            f"prediction was made against session {prediction['session_id']} but the current "
            f"hardware profile is {profile.session_id} — re-run predict.py against this session"
        )

    dtype = DTYPES[prediction["dtype"]]
    shape = shape_from_model(prediction["model"], torch.finfo(dtype).bits // 8)
    bandwidth = profile.peak_bandwidth_gbps
    measured_ms = scalar(rows, "decode", "measured", "step_time_ms")

    terms = [
        Term(
            "weights",
            _bytes_to_ms(shape.weight_bytes_per_token, bandwidth),
            "every weight read once, at the session's measured streaming bandwidth",
        ),
        Term(
            "kv_cache",
            _bytes_to_ms(shape.kv_cache_bytes(seq_len), bandwidth),
            f"K and V for {seq_len} past positions; grows with context",
        ),
        Term(
            "activations",
            _bytes_to_ms(shape.activation_bytes_per_token(), bandwidth),
            "norms, residual adds and MLP intermediates — bytes without FLOPs",
        ),
    ]
    return terms, measured_ms


def report_graph_overhead(rows: list[dict[str, str]]) -> None:
    """What per-launch overhead costs, from turning CUDA graphs off and changing nothing else."""
    on = dict(select(rows, "graphs", "cuda_graphs", "step_time_ms"))
    off = dict(select(rows, "graphs", "eager", "step_time_ms"))
    shared = sorted(set(on) & set(off))
    if not shared:
        return

    print(f"\n{'batch':>6} {'graphs ms':>10} {'eager ms':>10} {'overhead ms':>12} {'share':>8}")
    print("-" * 52)
    for batch in shared:
        overhead = off[batch] - on[batch]
        print(
            f"{batch:>6,.0f} {on[batch]:>10.2f} {off[batch]:>10.2f} "
            f"{overhead:>12.2f} {overhead / off[batch]:>7.1%}"
        )


def main() -> None:
    prediction = json.loads(PREDICTIONS_PATH.read_text())
    seq_len = int(prediction["seq_len"])
    terms, measured_ms = build_terms(seq_len)
    rows = read_rows(CSV_PATH)
    profile = load_profile()

    explained_ms = sum(term.ms for term in terms)
    unexplained_ms = measured_ms - explained_ms
    unexplained_fraction = unexplained_ms / measured_ms

    print(f"measured per-token time  {measured_ms:>8.2f} ms   (vLLM, CUDA graphs on)\n")
    print(f"{'term':<14} {'ms':>8} {'share':>8}   note")
    print("-" * 96)
    for term in terms:
        print(f"{term.name:<14} {term.ms:>8.2f} {term.ms / measured_ms:>7.1%}   {term.note}")
    print(
        f"{'unexplained':<14} {unexplained_ms:>8.2f} {unexplained_fraction:>7.1%}   "
        "small-GEMV kernels not reaching streaming bandwidth; attention; non-overlapped work"
    )
    print("-" * 96)

    # The effective bandwidth decode actually achieves, against what a large streaming copy does.
    # The single most useful number to fall out of the decomposition.
    effective_gbps = predicted_bytes(prediction) / (measured_ms * 1e-3) / 1e9
    print(
        f"\neffective decode bandwidth {effective_gbps:>8,.0f} GB/s  = "
        f"{effective_gbps / profile.peak_bandwidth_gbps:.0%} of the "
        f"{profile.peak_bandwidth_gbps:,.0f} GB/s a large streaming copy sustains"
    )

    band = float(prediction["max_unexplained_fraction"])
    verdict = "WITHIN" if abs(unexplained_fraction) <= band else "OUTSIDE"
    print(
        f"\npre-registered band: unexplained <= {band:.0%}  ->  {verdict} "
        f"({unexplained_fraction:.1%})"
    )

    naive = float(prediction["naive_tokens_per_sec"])
    measured_tps = scalar(rows, "decode", "measured", "tokens_per_sec")
    ratio = naive / measured_tps
    factor_band = float(prediction["naive_within_factor"])
    factor_verdict = "WITHIN" if ratio <= factor_band else "OUTSIDE"
    print(
        f"naive prediction {naive:,.1f} tok/s vs measured {measured_tps:,.1f} tok/s "
        f"= {ratio:.2f}x  ->  {factor_verdict} (band {factor_band:.1f}x)"
    )

    report_graph_overhead(rows)

    out: list[dict[str, object]] = [
        {
            "session_id": prediction["session_id"],
            "experiment": "decomposition",
            "variant": term.name,
            "x": 0,
            "metric": "step_time_ms",
            "value": term.ms,
        }
        for term in [
            *terms,
            Term("unexplained", unexplained_ms, ""),
            Term("measured", measured_ms, ""),
        ]
    ]
    out.append(
        {
            "session_id": prediction["session_id"],
            "experiment": "decomposition",
            "variant": "effective_bandwidth",
            "x": 0,
            "metric": "effective_bandwidth_gbps",
            "value": effective_gbps,
        }
    )
    append_rows(CSV_PATH, out)


if __name__ == "__main__":
    main()
