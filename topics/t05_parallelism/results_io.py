"""Shared long-format CSV for the three T5 experiments.

One tidy file, one row per observation, same five columns for every experiment — so `plot.py`
has a single contract to read and the three experiments can be run and re-run independently.

    experiment,variant,workers,metric,value

`workers = 0` marks a per-curve summary row (a fitted constant) rather than a per-worker
measurement, so summaries and observations coexist without a second file.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "parallelism.csv"
FIELDS = ("experiment", "variant", "workers", "metric", "value")


def append_rows(rows: list[dict[str, object]]) -> None:
    """Replace existing rows matching the (experiment, variant) pairs in `rows`, then append these.

    Idempotent so a re-run refreshes its own numbers rather than stacking a second run on top of
    the first. The replace key is the **(experiment, variant) pair**, not the experiment alone:
    keying on experiment meant that re-running one strategy — say `--strategies ep` — silently
    deleted every other strategy's rows from the same experiment, losing data with no error.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    incoming = {(str(r["experiment"]), str(r["variant"])) for r in rows}

    kept: list[dict[str, str]] = []
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            kept = [r for r in csv.DictReader(f) if (r["experiment"], r["variant"]) not in incoming]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in FIELDS})
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})

    experiments = sorted({e for e, _ in incoming})
    try:
        shown = CSV_PATH.relative_to(Path.cwd())
    except ValueError:  # CSV lives outside cwd (e.g. a tmp dir under test)
        shown = CSV_PATH
    print(f"\nwrote {len(rows)} rows for {experiments} -> {shown}")


def read_rows(experiment: str | None = None) -> list[dict[str, str]]:
    """Read rows back, optionally filtered to one experiment. Empty list if the CSV is absent."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if experiment is None or r["experiment"] == experiment]
