"""Shared measurement plumbing for the topic artefacts.

Extracted on the *second* real use, not designed up front (see the plan's rule of three): T6 and
T7 both need the same hardware ceilings and the same timing discipline, so those live here rather
than being written twice and drifting apart.

Deliberately small. Anything used by exactly one topic stays in that topic's folder.
"""

from arch_common.gpu import (
    HardwareProfile,
    device_label,
    gemm_tflops,
    load_profile,
    measure_peak_bandwidth,
    measure_peak_tflops,
    probe_hardware,
)
from arch_common.results_io import append_rows, read_rows
from arch_common.timing import synchronize, time_op

__all__ = [
    "HardwareProfile",
    "append_rows",
    "device_label",
    "gemm_tflops",
    "load_profile",
    "measure_peak_bandwidth",
    "measure_peak_tflops",
    "probe_hardware",
    "read_rows",
    "synchronize",
    "time_op",
]
