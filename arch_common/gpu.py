"""The two hardware ceilings, measured once per session and shared by every topic.

T6 needs achieved memory bandwidth (it is the first term in its throughput prediction). T7 needs
achieved bandwidth *and* achieved peak FLOP/s (they are the two halves of the roofline). If each
topic probed independently they would land on slightly different numbers — different thermal
state, different clocks, possibly a different pod — and the two lab notes would quietly disagree
with each other with nothing failing to flag it.

So the probe runs **once**, writes `results/hardware.json` at the repo root, and both topics read
that file. The profile carries a `session_id`; a cross-topic test asserts both topics' results
were produced against the same one. Same-session reproducibility becomes a property of where the
number lives rather than a discipline anyone has to remember.

Every ceiling here is **measured, never quoted from a spec sheet.** The gap between the two is
itself worth reporting: an A100 SXM is specified at 2039 GB/s and delivers closer to 1400-1800.
"""

from __future__ import annotations

import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from arch_common.timing import time_op

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = REPO_ROOT / "results" / "hardware.json"

# Square GEMM sizes swept to find achieved peak FLOP/s. Peak is genuinely shape-sensitive — small
# matrices are launch-bound, and sizes that tile badly onto the SMs lose several TFLOP/s — so the
# whole sweep is reported, not just its maximum.
GEMM_SIZES_CUDA = (1024, 2048, 4096, 8192, 16384)
GEMM_SIZES_CPU = (256, 512, 1024)

# Bandwidth buffer sizes. Must be far larger than last-level cache or the probe measures cache
# bandwidth and reports a number several times too high.
COPY_BYTES_CUDA = 512 * 1024 * 1024
COPY_BYTES_CPU = 64 * 1024 * 1024


@dataclass(frozen=True)
class HardwareProfile:
    """The measured ceilings for one machine, in one session."""

    session_id: str
    device: str
    device_name: str
    dtype: str
    peak_bandwidth_gbps: float
    peak_tflops: float
    gemm_sweep: dict[str, float] = field(default_factory=dict)
    torch_version: str = ""
    platform: str = ""
    clocks: str = ""

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte at which the compute and bandwidth ceilings meet.

        Left of it a kernel is memory-bound, right of it compute-bound. A property of the hardware
        alone — no amount of tuning moves it.
        """
        return (self.peak_tflops * 1e12) / (self.peak_bandwidth_gbps * 1e9)


def format_tflops(value: float) -> str:
    """Format TFLOP/s legibly across four orders of magnitude.

    An A100 reports ~280; a laptop CPU rehearsal reports ~0.005. A single fixed precision renders
    one of those two as "0.0", which reads as a broken benchmark rather than a slow one.
    """
    return f"{value:,.3f}" if value < 1.0 else f"{value:,.1f}"


def device_label(device: torch.device) -> str:
    """Human-readable name for the device under test."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def measure_peak_bandwidth(device: torch.device, dtype: torch.dtype) -> float:
    """Achieved HBM/DRAM bandwidth in GB/s, via a large device-to-device copy.

    A copy touches each element twice — one read from the source, one write to the destination —
    so the bytes moved are `2 * numel * itemsize`. Forgetting the factor of two halves the
    reported bandwidth, which is a comfortable-looking number and therefore an easy mistake to
    leave in.
    """
    total_bytes = COPY_BYTES_CUDA if device.type == "cuda" else COPY_BYTES_CPU
    bytes_per_element = torch.finfo(dtype).bits // 8
    numel = total_bytes // bytes_per_element

    src = torch.randn(numel, dtype=dtype, device=device)
    dst = torch.empty_like(src)

    ms = time_op(lambda: dst.copy_(src), device)
    moved_bytes = 2 * numel * src.element_size()
    return moved_bytes / (ms * 1e-3) / 1e9


def measure_peak_tflops(device: torch.device, dtype: torch.dtype) -> tuple[float, dict[str, float]]:
    """Achieved peak compute in TFLOP/s, plus the full square-GEMM sweep behind it.

    An `n x n @ n x n` matmul is `2 * n^3` FLOPs — `n^3` multiplies and `n^3` adds. Large square
    GEMMs are the most compute-dense thing the hardware does, which is why they are the standard
    way to find the compute ceiling.
    """
    sizes = GEMM_SIZES_CUDA if device.type == "cuda" else GEMM_SIZES_CPU
    sweep: dict[str, float] = {}

    for n in sizes:
        sweep[str(n)] = gemm_tflops(n, n, n, device, dtype)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return max(sweep.values()), sweep


def gemm_tflops(m: int, n: int, k: int, device: torch.device, dtype: torch.dtype) -> float:
    """Achieved TFLOP/s for one `(m,k) @ (k,n)` matmul: `2*m*n*k` FLOPs over the measured time."""
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    ms = time_op(lambda: torch.matmul(a, b), device)
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e12


def _read_clocks(device: torch.device) -> str:
    """Record GPU clocks alongside the results — a throttled run is a different machine.

    Best-effort: absence of `nvidia-smi` is not an error, it just means the field is empty.
    """
    if device.type != "cuda":
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def probe_hardware(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    *,
    write: bool = True,
) -> HardwareProfile:
    """Measure both ceilings and (by default) persist them for every topic to read."""
    bandwidth = measure_peak_bandwidth(device, dtype)
    tflops, sweep = measure_peak_tflops(device, dtype)

    profile = HardwareProfile(
        session_id=uuid.uuid4().hex[:12],
        device=device.type,
        device_name=device_label(device),
        dtype=str(dtype).removeprefix("torch."),
        peak_bandwidth_gbps=bandwidth,
        peak_tflops=tflops,
        gemm_sweep=sweep,
        torch_version=torch.__version__,
        platform=platform.platform(),
        clocks=_read_clocks(device),
    )

    if write:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(asdict(profile), indent=2) + "\n")

    return profile


def load_profile(path: Path = PROFILE_PATH) -> HardwareProfile:
    """Read the session's measured ceilings. Raises if the probe has not been run."""
    if not path.exists():
        raise FileNotFoundError(
            f"no hardware profile at {path} — run `python -m arch_common.probe` first; "
            "T6 and T7 both read their ceilings from it so they agree with each other"
        )
    return HardwareProfile(**json.loads(path.read_text()))
