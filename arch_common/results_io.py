"""One tidy long-format CSV per topic — one row per observation, one contract for every plot.

    session_id,experiment,variant,x,metric,value

`x` is whatever the experiment swept (batch size, sequence length, matrix dimension). `x = 0`
marks a summary row — a single fitted or predicted constant for the whole curve — so summaries and
per-point observations coexist in one file without needing a second one.

`session_id` is stamped from the hardware profile, which is what lets a cross-topic test assert
that T6 and T7 were measured against the same silicon in the same session.

Writes are **idempotent per (experiment, variant) pair**. Keying on experiment alone was a real
bug in T5: re-running one variant silently deleted every sibling variant's rows from the same
experiment, losing data with no error and no warning.
"""

from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ("session_id", "experiment", "variant", "x", "metric", "value")


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Replace rows matching the incoming (experiment, variant) pairs, then append `rows`.

    Idempotent so re-running one experiment refreshes its own numbers rather than stacking a
    second run on top of the first.
    """
    if not rows:
        raise ValueError(
            "refusing to write an empty result set — a run that produced no rows failed"
        )

    missing = [f for r in rows for f in FIELDS if f not in r]
    if missing:
        raise ValueError(f"rows are missing required fields: {sorted(set(missing))}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = {(str(r["experiment"]), str(r["variant"])) for r in rows}

    kept: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open() as f:
            kept = [r for r in csv.DictReader(f) if (r["experiment"], r["variant"]) not in incoming]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in FIELDS})
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})

    experiments = sorted({experiment for experiment, _ in incoming})
    print(
        f"wrote {len(rows)} rows across {len(incoming)} variants "
        f"of {', '.join(experiments)} -> {csv_path}"
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read every observation back. Raises if the experiment has not been run yet."""
    if not csv_path.exists():
        raise FileNotFoundError(f"no results at {csv_path} — run the experiment first")
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def select(
    rows: list[dict[str, str]], experiment: str, variant: str, metric: str
) -> list[tuple[float, float]]:
    """Pull one curve out as sorted `(x, value)` pairs."""
    points = [
        (float(r["x"]), float(r["value"]))
        for r in rows
        if r["experiment"] == experiment and r["variant"] == variant and r["metric"] == metric
    ]
    return sorted(points)


def scalar(rows: list[dict[str, str]], experiment: str, variant: str, metric: str) -> float:
    """Pull a single summary value (an `x = 0` row). Raises if it is absent or ambiguous."""
    hits = [
        float(r["value"])
        for r in rows
        if r["experiment"] == experiment and r["variant"] == variant and r["metric"] == metric
    ]
    if len(hits) != 1:
        raise KeyError(
            f"expected exactly one {experiment}/{variant}/{metric} row, found {len(hits)}"
        )
    return hits[0]
