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
    """Replace any existing rows for the experiments present in `rows`, then append these.

    Replace-by-experiment (rather than plain append) keeps the file idempotent: re-running one
    experiment refreshes its own rows and leaves the other two untouched, so a partial re-run
    can never silently produce a CSV holding two different runs of the same experiment.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    incoming = {str(r["experiment"]) for r in rows}

    kept: list[dict[str, str]] = []
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            kept = [r for r in csv.DictReader(f) if r["experiment"] not in incoming]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row[k] for k in FIELDS})
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})

    print(f"\nwrote {len(rows)} rows for {sorted(incoming)} -> {CSV_PATH.relative_to(Path.cwd())}")


def read_rows(experiment: str | None = None) -> list[dict[str, str]]:
    """Read rows back, optionally filtered to one experiment. Empty list if the CSV is absent."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if experiment is None or r["experiment"] == experiment]
