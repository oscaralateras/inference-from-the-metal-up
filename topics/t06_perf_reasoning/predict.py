"""Write the from-first-principles prediction to disk **before** the model is ever run.

The whole value of T6 is predict-then-measure. Computing the prediction after seeing the
measurement is not prediction, it is storytelling — and it is very easy to do accidentally, by
"checking" a formula against a number you already have and adjusting until it fits.

So the prediction is committed to `results/predictions.json` as its own git commit, before the
measurement commit. The commit history is the evidence that the ordering was real. The acceptance
bands below are fixed at the same time, so no outcome can be narrated as a success after the fact.

    python predict.py --model Qwen/Qwen2.5-7B
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from topics.t06_perf_reasoning.measure import (
    DEFAULT_MODEL,
    DEFAULT_SEQ_LEN,
    DTYPES,
    shape_from_model,
)

PREDICTIONS_PATH = Path(__file__).parent / "results" / "predictions.json"

# Pre-registered acceptance bands. Fixed before the run; the lab note reports the outcome against
# them whichever way it falls.
#
# NAIVE_WITHIN_FACTOR - the weights-only prediction is a ceiling, so the measurement must land
#   below it and within this factor. Outside the band means the *model* is wrong (a mis-derived
#   byte count), not that the GPU underperformed.
# MAX_UNEXPLAINED_FRACTION - after attributing time to weights, KV cache, activations and launch
#   overhead, this much of the measured step time may remain unexplained. This is T6's real
#   pass/fail criterion.
NAIVE_WITHIN_FACTOR = 2.0
MAX_UNEXPLAINED_FRACTION = 0.25


@dataclass(frozen=True)
class Prediction:
    """Everything derivable before the model runs, plus the bands it will be judged against."""

    model: str
    dtype: str
    seq_len: int
    session_id: str
    device_name: str
    peak_bandwidth_gbps: float
    ridge_point_flops_per_byte: float
    total_params: int
    params_read_per_token: int
    flops_per_token: int
    weight_bytes_per_token: int
    kv_bytes_per_token: int
    activation_bytes_per_token: int
    arithmetic_intensity: float
    naive_tokens_per_sec: float
    full_tokens_per_sec: float
    naive_within_factor: float
    max_unexplained_fraction: float


def build(model_name: str, dtype_name: str, seq_len: int) -> Prediction:
    """Derive every predicted quantity from the config file and the measured ceilings."""
    profile = load_profile()
    dtype = DTYPES[dtype_name]
    shape = shape_from_model(model_name, torch.finfo(dtype).bits // 8)
    bandwidth = profile.peak_bandwidth_gbps

    return Prediction(
        model=model_name,
        dtype=dtype_name,
        seq_len=seq_len,
        session_id=profile.session_id,
        device_name=profile.device_name,
        peak_bandwidth_gbps=bandwidth,
        ridge_point_flops_per_byte=profile.ridge_point,
        total_params=shape.total_params,
        params_read_per_token=shape.params_read_per_token,
        flops_per_token=shape.flops_per_token,
        weight_bytes_per_token=shape.weight_bytes_per_token,
        kv_bytes_per_token=shape.kv_cache_bytes(seq_len),
        activation_bytes_per_token=shape.activation_bytes_per_token(),
        arithmetic_intensity=shape.arithmetic_intensity,
        # Weights only — the textbook back-of-the-envelope, and deliberately optimistic.
        naive_tokens_per_sec=bandwidth * 1e9 / shape.weight_bytes_per_token,
        # Every byte we know about: weights + KV cache + activation traffic.
        full_tokens_per_sec=shape.predicted_tokens_per_sec(bandwidth, seq_len),
        naive_within_factor=NAIVE_WITHIN_FACTOR,
        max_unexplained_fraction=MAX_UNEXPLAINED_FRACTION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    args = parser.parse_args()

    prediction = build(args.model, args.dtype, args.seq_len)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(asdict(prediction), indent=2) + "\n")

    print(f"{prediction.model} on {prediction.device_name} ({prediction.dtype})\n")
    print(f"  params (total)        {prediction.total_params / 1e9:>10.2f} B")
    print(f"  params read / token   {prediction.params_read_per_token / 1e9:>10.2f} B")
    print(f"  FLOPs / token         {prediction.flops_per_token / 1e9:>10.2f} GFLOP")
    print(f"  weight bytes / token  {prediction.weight_bytes_per_token / 1e9:>10.2f} GB")
    print(
        f"  KV bytes / token      {prediction.kv_bytes_per_token / 1e6:>10.2f} MB "
        f"(at {prediction.seq_len} ctx)"
    )
    print(f"  activation bytes/tok  {prediction.activation_bytes_per_token / 1e6:>10.2f} MB")
    print(f"\n  arithmetic intensity  {prediction.arithmetic_intensity:>10.2f} FLOPs/byte")
    print(f"  hardware ridge point  {prediction.ridge_point_flops_per_byte:>10.2f} FLOPs/byte")
    ratio = prediction.ridge_point_flops_per_byte / prediction.arithmetic_intensity
    if ratio > 1.0:
        print(f"  -> decode sits {ratio:,.0f}x below the ridge: firmly memory-bound\n")
    else:
        # True on a CPU, where compute is scarce relative to bandwidth. Never true on a GPU.
        print(f"  -> decode sits {1 / ratio:,.1f}x above the ridge: compute-bound on this device\n")
    print(f"  predicted (weights only)  {prediction.naive_tokens_per_sec:>10,.1f} tok/s")
    print(f"  predicted (all bytes)     {prediction.full_tokens_per_sec:>10,.1f} tok/s")
    print(f"\npre-registered -> {PREDICTIONS_PATH}")
    print("commit this file BEFORE running measure.py")


if __name__ == "__main__":
    main()
