"""The 2x2: fusion on/off crossed with graph capture on/off, plus a launch counter.

The whole design rests on the two mechanisms being separable, and they are only separable if
`torch.compile` can be made to fuse **without** also capturing CUDA graphs. It can, but not by
accident: `mode="reduce-overhead"` turns cudagraphs on inside Inductor, which would silently
bundle the two mechanisms back together and reproduce T6's undifferentiated number while looking
like a controlled experiment.

So the compile mode here is the default one, and `assert_fusion_is_not_secretly_graphs` checks
Inductor's own config flag rather than trusting the argument. That check is load-bearing: without
it, a torch upgrade that changes the default would quietly invalidate the topic.

    eager           many launches, many HBM round trips     the baseline
    compile         fewer launches, FEWER round trips       fusion isolated
    graph           ONE launch,    many round trips         launch cost isolated
    compile+graph   one launch,    fewer round trips        both
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from topics.t11_compiler_runtime.chain import CHAIN_OPS, ChainInputs, decode_chain

MODES = ("eager", "compile", "graph", "compile_graph")

# Warmup calls before capture or timing. Three is enough for the allocator to settle and for
# Inductor to have compiled and autotuned; capture on a cold callable records the compilation.
WARMUP_CALLS = 3


def assert_fusion_is_not_secretly_graphs() -> None:
    """Refuse to run if Inductor would apply CUDA graphs on its own.

    If it did, the `compile` cell of the 2x2 would contain both mechanisms and every attribution in
    the note would be wrong — while every number still looked plausible. That is the failure mode
    this repo cares most about, so it is checked rather than assumed.
    """
    from torch._inductor import config as inductor_config

    if getattr(inductor_config.triton, "cudagraphs", False):
        raise RuntimeError(
            "Inductor is configured to apply CUDA graphs, so the 'compile' mode would measure "
            "fusion AND graph capture together — which is exactly what this topic exists to "
            "separate. Set torch._inductor.config.triton.cudagraphs = False."
        )


def build_callable(
    inputs: ChainInputs, *, compiled: bool, ops: int, fusing: bool = False
) -> Callable[[], torch.Tensor]:
    """A zero-argument callable running the chain on fixed inputs, eager or compiled."""
    fn = decode_chain
    if compiled:
        assert_fusion_is_not_secretly_graphs()
        fn = torch.compile(decode_chain)

    def run() -> torch.Tensor:
        return fn(inputs.hidden_state, inputs.residual, inputs.gate, inputs.weight, ops, fusing)

    return run


def capture(fn: Callable[[], torch.Tensor], device: torch.device) -> Callable[[], None]:
    """Record `fn` into a CUDA graph and return a replay callable.

    The warmup on a side stream is not hygiene — a first call inside capture would try to allocate
    and to build kernels, and capture rejects both. T9 learned the harder version of this lesson
    when NCCL's watchdog thread deadlocked a capture for eleven minutes; there is no NCCL here, so
    the default error mode is left alone.
    """
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_CALLS):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    return graph.replay


def build_mode(
    mode: str,
    inputs: ChainInputs,
    device: torch.device,
    ops: int = len(CHAIN_OPS),
    fusing: bool = False,
) -> Callable[[], object]:
    """One of the four cells, ready to time. Identical inputs across all four by construction."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    fn = build_callable(inputs, compiled=mode.startswith("compile"), ops=ops, fusing=fusing)

    for _ in range(WARMUP_CALLS):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return capture(fn, device) if mode.endswith("graph") else fn


def count_kernel_launches(fn: Callable[[], object], device: torch.device) -> int:
    """How many device kernels one call of `fn` actually launches.

    This is the mechanism evidence, and it is what stops the note attributing a speedup to fusion
    when the compiler merely reordered something. Fusion must show up as a *smaller kernel count*
    at the same batch size; graph capture must show up as one launch regardless of count.

    Counts device-side kernels only. The profiler's own overhead inflates wall-clock badly, so
    this is deliberately a separate pass from the timing and its timings are discarded.
    """
    if device.type != "cuda":
        return -1

    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize(device)

    total = 0
    for event in prof.key_averages():
        device_time = getattr(event, "self_device_time_total", None)
        if device_time is None:
            device_time = getattr(event, "self_cuda_time_total", 0.0)
        if device_time and device_time > 0:
            total += event.count
    return total
