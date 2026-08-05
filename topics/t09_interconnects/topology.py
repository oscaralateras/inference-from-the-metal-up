"""The gate that refuses to let a PCIe node publish itself as NVLink.

This is the one failure mode that would quietly invalidate the whole topic. Rented multi-GPU nodes
are not uniform: two GPUs in the same chassis may have no NVLink between them at all, and the
symptom is not an error — it is a plausible-looking bandwidth number roughly 10x too low, which
would land in the lab note as a fact about NVLink. A wrong number published confidently is worse
than no topic, so the check runs first and **aborts** rather than warns.

Two independent checks, because each catches what the other misses:

1. **Declared** — parse `nvidia-smi topo -m` and require an `NV#` link between every pair of GPUs
   in the world. This catches the common case of a node whose GPUs hang off separate PCIe root
   complexes (`SYS`, `NODE`, `PHB`).
2. **Empirical** — actually move a large buffer and check the achieved bandwidth. This catches
   what the matrix cannot: virtualised or mislabelled topology, a link that negotiated down, and
   the case where NCCL declines to use NVLink for its own reasons and silently falls back. The
   matrix is a claim; this is a measurement, and only one of them is evidence.

The empirical check is the load-bearing one. It is deliberately run at a large message size, where
the transfer is bandwidth-bound and the answer is unambiguous — the small-message regime is what
the topic is *about*, and it cannot distinguish a slow link from a fast one at all.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

# Interconnect classes as `nvidia-smi topo -m` reports them. `NV#` is an NVLink bond of # links;
# everything below traverses PCIe or worse and belongs to a different topic than this one.
NVLINK_PATTERN = re.compile(r"^NV\d+$")

# Below this, a large all-reduce is not running over NVLink whatever the matrix says. PCIe 4.0 x16
# tops out near 25 GB/s each way and NVLink starts around 300; anything in between is a
# renegotiated or partially-bonded link, which is equally not the thing being measured. Set well
# clear of both so the check has no opinion in the ambiguous middle.
MIN_NVLINK_BUS_GBPS = 100.0

# Payload for the empirical check. Must be large enough to be firmly bandwidth-bound.
SMOKE_TEST_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Topology:
    """What `nvidia-smi topo -m` claims about this node."""

    matrix: dict[tuple[int, int], str] = field(default_factory=dict)
    raw: str = ""

    def link(self, a: int, b: int) -> str:
        return self.matrix.get((a, b), "?")

    def non_nvlink_pairs(self, world: int) -> list[tuple[int, int, str]]:
        """Every GPU pair in `range(world)` not joined by NVLink, with what it has instead."""
        bad: list[tuple[int, int, str]] = []
        for a in range(world):
            for b in range(a + 1, world):
                link = self.link(a, b)
                if not NVLINK_PATTERN.match(link):
                    bad.append((a, b, link))
        return bad

    def nvlink_width(self, world: int) -> int:
        """Smallest NVLink bond width across the world, e.g. 12 for the `NV12` T5 ran on.

        The minimum rather than the maximum: a ring is only as fast as its narrowest hop, and a
        node with one weak pair would otherwise advertise its best link.
        """
        widths = [
            int(self.link(a, b)[2:])
            for a in range(world)
            for b in range(a + 1, world)
            if NVLINK_PATTERN.match(self.link(a, b))
        ]
        return min(widths) if widths else 0


def parse_topo(text: str) -> Topology:
    """Parse the `nvidia-smi topo -m` matrix into `(a, b) -> link class`.

    Tolerant by design: the command's trailing legend, the CPU/NUMA affinity columns and any
    NIC rows are all ignored, since their formatting varies across driver versions and none of
    them is what the gate is asking about.
    """
    matrix: dict[tuple[int, int], str] = {}
    header: list[int] = []

    for line in text.splitlines():
        cells = line.split()
        if not cells:
            continue

        # The header row starts with the GPU columns and then trails off into `CPU Affinity`,
        # `NUMA Affinity` and sometimes NIC columns, whose names vary by driver version. Take the
        # leading run of `GPU<n>` tokens and stop at the first thing that is not one, rather than
        # requiring the whole row to be GPUs — which is what an earlier version did, and it
        # silently produced an empty matrix on every real node.
        if not header and cells[0] == "GPU0":
            for cell in cells:
                if not (cell.startswith("GPU") and cell[3:].isdigit()):
                    break
                header.append(int(cell[3:]))
            continue

        row_label = cells[0]
        if not header or not row_label.startswith("GPU") or not row_label[3:].isdigit():
            continue

        row = int(row_label[3:])
        for col, value in zip(header, cells[1:], strict=False):
            if col == row or value == "X":
                continue
            matrix[(row, col)] = value
            matrix[(col, row)] = value

    return Topology(matrix=matrix, raw=text)


def read_topo() -> Topology:
    """Run `nvidia-smi topo -m`. Absence of the tool is a hard failure, not a soft one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "could not run `nvidia-smi topo -m`. T9's entire claim is about which wire the "
            "collectives crossed, so a node that cannot report its topology cannot produce a "
            "publishable number here."
        ) from exc
    return parse_topo(out.stdout)


def check_declared(world: int, topo: Topology | None = None) -> Topology:
    """Require NVLink between every pair in the world. Raises with the matrix on failure."""
    topo = topo if topo is not None else read_topo()
    bad = topo.non_nvlink_pairs(world)
    if bad:
        detail = ", ".join(f"GPU{a}-GPU{b}={link}" for a, b, link in bad)
        raise RuntimeError(
            f"topology gate FAILED for world={world}: {detail}.\n"
            "These pairs are not NVLink-connected, so a collective across them measures PCIe. "
            "Destroy this instance and re-rent — do not record results from it, and do not "
            "relabel the topic. `nvidia-smi topo -m` said:\n\n"
            f"{topo.raw}"
        )
    return topo


def check_empirical(measured_bus_gbps: float, world: int) -> None:
    """Require the measured large-message bandwidth to be consistent with an NVLink fabric.

    Deliberately separate from `check_declared` and deliberately second: the matrix can be right
    about the cabling and still wrong about what NCCL chose to do with it.
    """
    if measured_bus_gbps < MIN_NVLINK_BUS_GBPS:
        raise RuntimeError(
            f"topology gate FAILED empirically at world={world}: a "
            f"{SMOKE_TEST_BYTES / 1024**2:.0f} MB all-reduce achieved "
            f"{measured_bus_gbps:,.1f} GB/s bus bandwidth, below the {MIN_NVLINK_BUS_GBPS:,.0f} "
            "GB/s floor. `nvidia-smi topo -m` may claim NVLink, but the traffic is not crossing "
            "it — check NCCL_P2P_DISABLE, container capabilities and MIG state, then re-rent if "
            "unresolved. Numbers from this node would describe PCIe under an NVLink label."
        )


def format_topology(topo: Topology, world: int) -> str:
    """Human-readable summary, written verbatim into `results/topology.txt`.

    Recorded rather than merely checked: the topology is part of the result, and a reader who
    cannot see which fabric produced a bandwidth number has to take it on trust.
    """
    lines = [f"world_size: {world}", f"min_nvlink_width: NV{topo.nvlink_width(world)}", ""]
    for a in range(world):
        for b in range(a + 1, world):
            lines.append(f"GPU{a}-GPU{b}: {topo.link(a, b)}")
    lines += ["", "--- nvidia-smi topo -m ---", topo.raw]
    return "\n".join(lines)
