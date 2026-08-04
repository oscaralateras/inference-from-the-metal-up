"""Account for the gap between predicted and measured decode, term by term.

Predicting `bandwidth / bytes_per_token` is a five-line calculation. The finding is not the
prediction — it is the **error budget**: taking the measured step time and attributing it to named
causes until what remains is small enough to admit is unexplained.

The decomposition works in the **time** domain, because times add and throughputs do not:

    measured_ms = weights + kv_cache + activations + launch_overhead + unexplained

The first three are bytes divided by the session's measured bandwidth. The fourth is measured
per-launch cost times the module calls per token. Whatever is left is the residual, reported
honestly rather than absorbed into a fudge factor.

    python decompose.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows, read_rows, scalar
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


def build_terms(seq_len: int) -> tuple[list[Term], float]:
    """Return the explained terms plus the measured per-token time they are explaining."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"no pre-registered prediction at {PREDICTIONS_PATH} — run predict.py and commit it "
            "before measuring, or the prediction is not a prediction"
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

    measured_ms = scalar(rows, "decode", "measured", "latency_p50_ms")
    per_launch_ms = scalar(rows, "calibration", "launch", "per_launch_ms")
    module_calls = scalar(rows, "calibration", "launch", "module_calls_per_token")

    terms = [
        Term(
            "weights",
            _bytes_to_ms(shape.weight_bytes_per_token, bandwidth),
            "every weight read once, at the session's measured bandwidth",
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
        Term(
            "launch_overhead",
            per_launch_ms * module_calls,
            f">={int(module_calls)} module calls x {per_launch_ms * 1e3:.1f} us; "
            "bandwidth cannot fix this",
        ),
    ]
    return terms, measured_ms


def main() -> None:
    prediction = json.loads(PREDICTIONS_PATH.read_text())
    seq_len = int(prediction["seq_len"])
    terms, measured_ms = build_terms(seq_len)

    explained_ms = sum(t.ms for t in terms)
    unexplained_ms = measured_ms - explained_ms
    unexplained_fraction = unexplained_ms / measured_ms

    print(f"measured per-token time  {measured_ms:>8.2f} ms\n")
    print(f"{'term':<18} {'ms':>8} {'share':>8}   note")
    print("-" * 92)
    for term in terms:
        print(f"{term.name:<18} {term.ms:>8.2f} {term.ms / measured_ms:>7.1%}   {term.note}")
    print(
        f"{'unexplained':<18} {unexplained_ms:>8.2f} {unexplained_fraction:>7.1%}   "
        "imperfect bandwidth utilisation, non-overlapped work, attention kernel efficiency"
    )
    print("-" * 92)

    band = float(prediction["max_unexplained_fraction"])
    verdict = "WITHIN" if abs(unexplained_fraction) <= band else "OUTSIDE"
    print(
        f"\npre-registered band: unexplained <= {band:.0%}  ->  {verdict} "
        f"({unexplained_fraction:.1%})"
    )

    naive = float(prediction["naive_tokens_per_sec"])
    measured_tps = scalar(read_rows(CSV_PATH), "decode", "measured", "tokens_per_sec")
    ratio = naive / measured_tps
    factor_band = float(prediction["naive_within_factor"])
    factor_verdict = "WITHIN" if ratio <= factor_band else "OUTSIDE"
    print(
        f"naive prediction {naive:,.1f} tok/s vs measured {measured_tps:,.1f} tok/s "
        f"= {ratio:.2f}x  ->  {factor_verdict} (band {factor_band:.1f}x)"
    )

    rows: list[dict[str, object]] = [
        {
            "session_id": prediction["session_id"],
            "experiment": "decomposition",
            "variant": term.name,
            "x": 0,
            "metric": "step_time_ms",
            "value": term.ms,
        }
        for term in [*terms, Term("unexplained", unexplained_ms, "")]
    ]
    rows.append(
        {
            "session_id": prediction["session_id"],
            "experiment": "decomposition",
            "variant": "measured",
            "x": 0,
            "metric": "step_time_ms",
            "value": measured_ms,
        }
    )
    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    main()
