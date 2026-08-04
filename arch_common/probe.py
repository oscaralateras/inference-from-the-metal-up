"""Measure this machine's ceilings once, for every topic to share.

    python -m arch_common.probe                 # auto-detect device, bfloat16
    python -m arch_common.probe --device cpu    # rehearsal on a laptop

Run this **first** in a GPU session. T6 and T7 both read `results/hardware.json`; running it again
mid-session mints a new session_id and will fail the cross-topic agreement test.
"""

from __future__ import annotations

import argparse

import torch

from arch_common.gpu import PROFILE_PATH, format_tflops, probe_hardware

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=default_device(), choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"probing {args.device} in {args.dtype} — this takes a minute\n")

    profile = probe_hardware(device, DTYPES[args.dtype])

    print(f"device            {profile.device_name}")
    if profile.clocks:
        print(f"clocks (sm,mem)   {profile.clocks}")
    print(f"peak bandwidth    {profile.peak_bandwidth_gbps:>10,.1f} GB/s   (measured, not spec)")
    print(f"peak compute      {format_tflops(profile.peak_tflops):>10} TFLOP/s")
    print(f"ridge point       {profile.ridge_point:>10,.2f} FLOPs/byte")
    print("\nGEMM sweep (TFLOP/s by square size — peak is shape-sensitive):")
    for size, tflops in sorted(profile.gemm_sweep.items(), key=lambda kv: int(kv[0])):
        print(f"  {int(size):>6,}  {format_tflops(tflops):>8}")
    print(f"\nsession {profile.session_id} -> {PROFILE_PATH}")


if __name__ == "__main__":
    main()
