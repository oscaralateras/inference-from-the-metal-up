# Session results

`hardware.json` — this machine's measured ceilings (peak bandwidth, peak TFLOP/s, GEMM sweep,
clocks, session id), written once per session by `python -m arch_common.probe`.

**T6 and T7 both read this file.** Probing separately per topic would give each a slightly
different roof — different thermal state, different clocks, possibly a different pod — and the two
lab notes would quietly disagree with nothing failing to flag it. The `session_id` stamped into
every result row is what `tests/test_distinctness.py` checks to enforce that.

Not committed: it describes one machine at one moment, and is regenerated in seconds.
