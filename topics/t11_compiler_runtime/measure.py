"""The T11 session: the 2x2 across a batch sweep, and the crossover it exists to locate.

    uv run python -m topics.t11_compiler_runtime.measure                     # on the pod
    uv run python -m topics.t11_compiler_runtime.measure --device cpu        # laptop rehearsal
    uv run python -m topics.t11_compiler_runtime.measure --chain-lengths 2,3 # band 3's control

Four modes, six batch sizes, and one question: where does the dominant win flip from removing
launches to removing bytes?

The crossover is located from the measurements rather than eyeballed off a plot — `crossover_batch`
finds where fusion's speedup overtakes graph capture's and interpolates between the bracketing
points, in log space because the sweep is geometric.

The chain-length control is what makes band 3 more than a curve fit. Fusion removes `2k-3` round
trips while capture removes `k` launches, so the model says a longer chain crosses over *earlier*,
by exactly `(2*5-3)/(2*2-3)` = 7 going from five ops to two. That is a sharp number to be wrong
against, and no amount of agreement at a single chain length would have tested it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from arch_common.gpu import load_profile
from arch_common.results_io import append_rows
from arch_common.timing import time_op
from topics.t11_compiler_runtime.chain import (
    CHAIN_OPS,
    DEFAULT_HIDDEN,
    fused_bytes,
    fusion_crossover_batch,
    make_inputs,
    unfused_bytes,
)
from topics.t11_compiler_runtime.modes import MODES, build_mode, count_kernel_launches
from topics.t11_compiler_runtime.predict import (
    BATCHES,
    CROSSOVER_BATCH_RANGE,
    FUSION_MODEL_TOLERANCE,
    MAX_FUSION_SPEEDUP_AT_BATCH_1,
    MIN_GRAPH_SPEEDUP_AT_BATCH_1,
    MIN_SHARE_OF_MEMORY_ROOF,
    build_prediction,
)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "compiler.csv"

# Back-to-back calls inside one timed window, as in T8 and T9. A decode step runs this chain 28
# times in sequence and never pays for one in isolation, so steady state is both the faithful
# measurement and the one that does not report the dispatch path instead of the kernel.
CALLS_PER_TIMING = 16


def verdict(ok: bool) -> str:
    return "WITHIN" if ok else "OUTSIDE"


def crossover_batch(batches: list[int], fusion: list[float], graphs: list[float]) -> float:
    """The batch at which fusion's speedup overtakes graph capture's, interpolated in log2.

    Returns 0.0 if fusion never overtakes within the sweep — a real answer the note must be able to
    report, not a failure to crash on. Linear interpolation would bias every result toward the
    upper bracket, because the sweep doubles rather than steps.
    """
    for i in range(1, len(batches)):
        before, after = fusion[i - 1] - graphs[i - 1], fusion[i] - graphs[i]
        if before < 0 <= after:
            if after == before:
                return float(batches[i])
            frac = -before / (after - before)
            lo, hi = math.log2(batches[i - 1]), math.log2(batches[i])
            return 2 ** (lo * (1 - frac) + hi * frac)
    return 0.0


def run_sweep(
    ops: int,
    batches: list[int],
    hidden: int,
    device: torch.device,
    modes: list[str],
    fusing: bool = False,
) -> tuple[dict[int, dict[str, float]], list[float], list[float], dict[int, dict[str, int]]]:
    """Time every mode at every batch for one chain length."""
    per_batch: dict[int, dict[str, float]] = {}
    launches: dict[int, dict[str, int]] = {}
    fusion_speedups: list[float] = []
    graph_speedups: list[float] = []

    header = "  ".join(f"{m:>13}" for m in modes)
    print(f"\nchain length {ops} of {len(CHAIN_OPS)}{' (all-fusing variant)' if fusing else ''}")
    print(f"{'batch':>7}  {header}  {'fusion':>7} {'graphs':>7} {'both':>7}")

    for batch in batches:
        inputs = make_inputs(batch, hidden, device, torch.bfloat16)
        times: dict[str, float] = {}
        counts: dict[str, int] = {}

        for mode in modes:
            fn = build_mode(mode, inputs, device, ops=ops, fusing=fusing)
            times[mode] = time_op(fn, device, inner=CALLS_PER_TIMING) * 1e3
            counts[mode] = count_kernel_launches(fn, device)

        # Each ratio isolates one mechanism by changing exactly one thing.
        fusion = times["eager"] / times["compile"]
        graphs = times["eager"] / times["graph"] if "graph" in times else 0.0
        both = times["eager"] / times["compile_graph"] if "compile_graph" in times else 0.0

        fusion_speedups.append(fusion)
        graph_speedups.append(graphs)
        per_batch[batch] = {"fusion": fusion, "graphs": graphs, "both": both, **times}
        launches[batch] = counts

        cells = "  ".join(f"{times[m]:>13.2f}" for m in modes)
        print(f"{batch:>7}  {cells}  {fusion:>6.2f}x {graphs:>6.2f}x {both:>6.2f}x")

    return per_batch, fusion_speedups, graph_speedups, launches


def _rows(
    session: str,
    ops: int,
    hidden: int,
    per_batch: dict[int, dict[str, float]],
    launches: dict[int, dict[str, int]],
    modes: list[str],
    suffix: str | None = None,
) -> list[dict[str, object]]:
    """One row per observation. Chain length is part of the variant, so the control's sweeps sit
    alongside the main one instead of overwriting it."""
    if suffix is None:
        suffix = "" if ops == len(CHAIN_OPS) else f"_ops{ops}"
    rows: list[dict[str, object]] = []

    for batch, times in per_batch.items():
        for mode in modes:
            for metric, value in (
                ("latency_us", times[mode]),
                ("kernel_launches", float(launches[batch][mode])),
            ):
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "modes",
                        "variant": f"{mode}{suffix}",
                        "x": batch,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, value in (
            ("fusion_speedup", times["fusion"]),
            ("graph_speedup", times["graphs"]),
            ("combined_speedup", times["both"]),
            ("fused_gbps", fused_bytes(batch, hidden) / (times["compile"] * 1e-6) / 1e9),
            (
                "byte_model_speedup",
                unfused_bytes(batch, hidden, ops) / fused_bytes(batch, hidden),
            ),
        ):
            rows.append(
                {
                    "session_id": session,
                    "experiment": "mechanism",
                    "variant": f"chain{suffix}",
                    "x": batch,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--batches", default=",".join(str(b) for b in BATCHES))
    parser.add_argument(
        "--chain-lengths",
        default="",
        help="band 3's control: extra op counts to repeat the sweep at. The model predicts the "
        "crossover moves by (2*5-3)/(2*ops-3), so this is what can falsify it",
    )
    parser.add_argument(
        "--all-fusing",
        action="store_true",
        help="repeat the sweep with the 5-op chain whose fifth op fuses (post-norm instead of the "
        "rotary cat). Holds chain length fixed and changes only fusion completeness, which is the "
        "confound the first control could not separate",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    on_gpu = device.type == "cuda"
    batches = [int(b) for b in args.batches.split(",")]

    if on_gpu:
        profile = load_profile()
        session, peak_gbps = profile.session_id, profile.peak_bandwidth_gbps
    else:
        # A CPU rehearsal exercises every code path except capture, and its numbers are never
        # published — same contract as T9's gloo rehearsal.
        session, peak_gbps = "rehearsal", 50.0

    prediction = build_prediction(peak_gbps, session_id=session)
    print(f"T11 — fusion vs launch, {' -> '.join(CHAIN_OPS)}, hidden {args.hidden}")
    print(f"session {session}, {peak_gbps:,.1f} GB/s memory roof")
    print(f"predicted crossover at batch {prediction.predicted_crossover_batch:,.0f}")

    modes = [m for m in MODES if on_gpu or not m.endswith("graph")]
    if not on_gpu:
        print("CPU rehearsal: graph modes skipped, numbers not published")

    full = len(CHAIN_OPS)
    per_batch, fusion_speedups, graph_speedups, launches = run_sweep(
        full, batches, args.hidden, device, modes
    )
    rows = _rows(session, full, args.hidden, per_batch, launches, modes)

    print("\n" + "=" * 78)
    print("pre-registered bands")
    print("=" * 78)

    if 1 in per_batch:
        f1 = per_batch[1]["fusion"]
        print(
            f"(1) fusion at batch 1:  {f1:>6.2f}x vs <= {MAX_FUSION_SPEEDUP_AT_BATCH_1}  "
            f"{verdict(f1 <= MAX_FUSION_SPEEDUP_AT_BATCH_1)}"
        )
        if on_gpu:
            g1 = per_batch[1]["graphs"]
            print(
                f"(2) graphs at batch 1:  {g1:>6.2f}x vs >= {MIN_GRAPH_SPEEDUP_AT_BATCH_1}  "
                f"{verdict(g1 >= MIN_GRAPH_SPEEDUP_AT_BATCH_1)}"
            )

    if on_gpu:
        cross = crossover_batch(batches, fusion_speedups, graph_speedups)
        lo, hi = CROSSOVER_BATCH_RANGE
        if cross:
            print(
                f"(3) crossover at batch {cross:>7,.0f} vs {CROSSOVER_BATCH_RANGE}  "
                f"{verdict(lo <= cross <= hi)}"
            )
        else:
            print(f"(3) crossover: fusion never overtook graph capture within {batches}  OUTSIDE")
        rows.append(
            {
                "session_id": session,
                "experiment": "crossover",
                "variant": "chain",
                "x": full,
                "metric": "crossover_batch",
                "value": cross,
            }
        )

        # POST-HOC, and labelled as such: the band above is scored against the crossover predicted
        # from an *assumed* 5 us launch cost, because nothing in this repo had measured one for a
        # plain kernel. This run can measure it. The gap between eager and graph replay at a fixed
        # batch is exactly the launch cost the capture removed, divided by the number of ops:
        #
        #     L = (eager - graph) / ops
        #
        # Feeding that back into the same unchanged formula is what separates "the model is wrong"
        # from "the model's input was wrong". Only the second is a defensible thing to claim, and
        # only if the recomputation is shown rather than described.
        smallest = min(batches)
        measured_launch_us = (per_batch[smallest]["eager"] - per_batch[smallest]["graph"]) / full
        if measured_launch_us > 0:
            remodelled = fusion_crossover_batch(peak_gbps, measured_launch_us, full, args.hidden)
            print(
                f"    launch cost measured at {measured_launch_us:.2f} us/op "
                f"(assumed {prediction.assumed_launch_us:.1f}); the SAME model fed that predicts "
                f"batch {remodelled:,.0f} against a measured {cross:,.0f}"
            )
            for metric, value in (
                ("measured_launch_us", measured_launch_us),
                ("remodelled_crossover_batch", remodelled),
            ):
                rows.append(
                    {
                        "session_id": session,
                        "experiment": "crossover",
                        "variant": "chain",
                        "x": full,
                        "metric": metric,
                        "value": value,
                    }
                )

        # Bands 4 and 5 are scored at the top of the sweep, which is the only place the chain is
        # unambiguously bandwidth-bound. Below the crossover, comparing a launch-bound kernel to a
        # memory roof or to a byte model is comparing it to the wrong thing entirely.
        big = max(batches)
        achieved = fused_bytes(big, args.hidden) / (per_batch[big]["compile"] * 1e-6) / 1e9
        share = achieved / peak_gbps
        print(
            f"(4) fused chain at batch {big}: {achieved:,.1f} GB/s = {share:.1%} of roof vs "
            f">= {MIN_SHARE_OF_MEMORY_ROOF:.0%}  {verdict(share >= MIN_SHARE_OF_MEMORY_ROOF)}"
        )

        # POST-HOC diagnostic, and the band above is NOT rescored against it. The band was
        # registered against `compile`, which was the wrong mode to ask this question of: it runs
        # the fused kernel *and* pays Dynamo's guard evaluation and the eager dispatch path on
        # every call. That overhead is roughly constant, which is why `compile` times barely move
        # across four decades of batch size while `compile_graph` scales with the work.
        #
        # So `compile`'s number is a property of the framework, not of the fuser's output.
        # `compile_graph` runs the same generated kernel with the per-call overhead captured away,
        # which is what "what bandwidth does the fused kernel reach" actually meant.
        graphed = fused_bytes(big, args.hidden) / (per_batch[big]["compile_graph"] * 1e-6) / 1e9
        overhead_us = per_batch[big]["compile"] - per_batch[big]["compile_graph"]
        print(
            f"    the same kernel, graph-replayed: {graphed:,.1f} GB/s = {graphed / peak_gbps:.1%} "
            f"of roof — the band's {share:.1%} includes {overhead_us:.1f} us/call of guard and "
            "dispatch overhead that is not the kernel"
        )
        rows.append(
            {
                "session_id": session,
                "experiment": "mechanism",
                "variant": "chain",
                "x": big,
                "metric": "graphed_fused_gbps",
                "value": graphed,
            }
        )

        predicted_ratio = unfused_bytes(big, args.hidden, full) / fused_bytes(big, args.hidden)
        measured_ratio = per_batch[big]["fusion"]
        agreement = measured_ratio / predicted_ratio if predicted_ratio else 0.0
        print(
            f"(5) fusion at batch {big}: measured {measured_ratio:.2f}x vs byte model "
            f"{predicted_ratio:.2f}x = {agreement:.2f}x  "
            f"{verdict(1 / FUSION_MODEL_TOLERANCE <= agreement <= FUSION_MODEL_TOLERANCE)}"
        )

        # Band 3's control. Each extra chain length is a full sweep, so this roughly doubles the
        # session per length — worth it, because it is the only thing separating a measured
        # property of the hardware from a curve that happened to fit.
        for ops in [int(o) for o in args.chain_lengths.split(",") if o.strip()]:
            ctrl_batch, ctrl_fusion, ctrl_graphs, ctrl_launches = run_sweep(
                ops, batches, args.hidden, device, modes
            )
            ctrl_cross = crossover_batch(batches, ctrl_fusion, ctrl_graphs)
            measured_shift = ctrl_cross / cross if cross and ctrl_cross else 0.0
            # Fusion removes 2k-3 round trips while capture removes k launches, so shortening the
            # chain should push the crossover *later* by exactly this factor. A sharp number to be
            # wrong against, which is the point of running the control at all.
            predicted_shift = (2 * full - 3) / (2 * ops - 3) if 2 * ops > 3 else float("inf")
            print(
                f"    control, {ops} ops: crossover at batch {ctrl_cross:,.0f} — "
                f"{measured_shift:.2f}x the {full}-op result, model says {predicted_shift:.2f}x"
            )
            rows += _rows(session, ops, args.hidden, ctrl_batch, ctrl_launches, modes)
            rows.append(
                {
                    "session_id": session,
                    "experiment": "crossover",
                    "variant": f"chain_ops{ops}",
                    "x": ops,
                    "metric": "crossover_batch",
                    "value": ctrl_cross,
                }
            )

        # The second control, and the one that separates the two explanations the first could not.
        # Same five ops, same byte model, same everything — except the fifth op fuses. If the
        # crossover still barely moves, chain length genuinely does not drive it. If it moves, the
        # first control was measuring incomplete fusion and calling it chain length.
        if args.all_fusing:
            fus_batch, fus_fusion, fus_graphs, fus_launches = run_sweep(
                full, batches, args.hidden, device, modes, fusing=True
            )
            fus_cross = crossover_batch(batches, fus_fusion, fus_graphs)
            print(
                f"    all-fusing 5-op chain: crossover at batch {fus_cross:,.0f} — "
                f"{(fus_cross / cross if cross else 0.0):.2f}x the rotary 5-op result"
            )
            rows += _rows(
                session, full, args.hidden, fus_batch, fus_launches, modes, suffix="_fusing"
            )
            rows.append(
                {
                    "session_id": session,
                    "experiment": "crossover",
                    "variant": "chain_fusing",
                    "x": full,
                    "metric": "crossover_batch",
                    "value": fus_cross,
                }
            )

    append_rows(CSV_PATH, rows)


if __name__ == "__main__":
    _main()
