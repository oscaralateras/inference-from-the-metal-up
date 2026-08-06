"""The cold start, measured on a real checkpoint instead of extrapolated from a synthetic file.

    uv run python -m topics.t10_os_virtual_memory.checkpoint --model-dir /root/models/qwen2.5-7b

The main measurement loads an 8 GiB file of random bytes and extrapolates per-byte rates to
Qwen2.5-7B's 15.23 GB. Per-byte rates are the right thing to generalise, but two things about a
real checkpoint are not captured by a single synthetic file:

    it is four shards with metadata between them, not one contiguous extent
    `safetensors` maps rather than copies — which is precisely the choice the note recommends,
    and recommending it from a measurement of something else is weaker than measuring it

So this loads the real thing, cold, twice: once the way a serving stack does it (map the shard,
move tensors to the device) and once through the copying path the synthetic `read` loader stands
for. Same files, same eviction, same clock.

The end-to-end number this produces is not the sum of the three stage times. It is whatever the
loader actually achieves including any overlap it manages, which is the honest quantity — the
staged sum is an upper bound and the note says so.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from arch_common.results_io import append_rows, read_rows, scalar
from topics.t10_os_virtual_memory.loaders import go_cold
from topics.t10_os_virtual_memory.pipeline import achieved_gbps, tokens_foregone

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "coldstart.csv"
T6_CSV = Path(__file__).resolve().parent.parent / "t06_perf_reasoning" / "results" / "perf.csv"


def shards(model_dir: Path) -> list[Path]:
    """The checkpoint's weight files, in load order."""
    found = sorted(model_dir.glob("*.safetensors"))
    if not found:
        raise FileNotFoundError(f"no *.safetensors under {model_dir}")
    return found


def evict_all(paths: list[Path]) -> str:
    """Drop every shard from the page cache, or refuse to report a cold number."""
    mechanisms = {go_cold(p) for p in paths}
    return "+".join(sorted(mechanisms))


def load_mapped(paths: list[Path], device: torch.device) -> int:
    """The serving path: map each shard, move its tensors to the device.

    `safetensors.load_file` mmaps the file, so the bytes reach the GPU without the intermediate
    userspace copy the `read` path pays. This is the loader the synthetic measurement argued for.
    """
    moved = 0
    for path in paths:
        for tensor in load_file(str(path)).values():
            moved += tensor.numel() * tensor.element_size()
            tensor.to(device, non_blocking=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return moved


def load_copied(paths: list[Path], device: torch.device) -> int:
    """The copying path: read each shard into a host buffer, then transfer that buffer.

    Byte-for-byte the same payload as the mapped path, moved the way `read` moves it — disk into
    the page cache, page cache into a process buffer, process buffer to the device.
    """
    moved = 0
    for path in paths:
        with path.open("rb") as f:
            blob = f.read()
        moved += len(blob)
        torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device, non_blocking=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return moved


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--session", default="", help="hardware session id for the CSV")
    args = parser.parse_args()

    device = torch.device(args.device)
    paths = shards(args.model_dir)
    nbytes = sum(p.stat().st_size for p in paths)

    step_ms = scalar(read_rows(T6_CSV), "decomposition", "measured", "step_time_ms")

    session = args.session
    if not session:
        from arch_common.gpu import load_profile

        session = load_profile().session_id

    print(f"T10 — real checkpoint cold start: {len(paths)} shards, {nbytes / 1e9:.2f} GB")
    print(f"session {session}, T6 decode step {step_ms:.2f} ms\n")

    rows: list[dict[str, object]] = []
    for name, loader in (("mapped", load_mapped), ("copied", load_copied)):
        mechanism = evict_all(paths)
        start = time.perf_counter()
        moved = loader(paths, device)
        seconds = time.perf_counter() - start

        gbps = achieved_gbps(moved, seconds)
        tokens = tokens_foregone(seconds, step_ms)
        print(
            f"  {name:>7}  {seconds:>7.2f} s  {gbps:>6.2f} GB/s  "
            f"{tokens:>8,.0f} tokens  (evicted via {mechanism})"
        )

        rows += [
            {
                "session_id": session,
                "experiment": "checkpoint",
                "variant": name,
                "x": nbytes,
                "metric": metric,
                "value": value,
            }
            for metric, value in (
                ("load_seconds", seconds),
                ("load_gbps", gbps),
                ("tokens_foregone", tokens),
            )
        ]

    append_rows(CSV_PATH, rows)
    print(f"\nwrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    _main()
