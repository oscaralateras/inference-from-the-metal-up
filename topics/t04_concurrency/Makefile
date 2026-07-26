# T4 — Concurrency & synchronization microbenchmark (Rust)
#
# Author/debug anywhere; CANONICAL numbers come from a MULTICORE Linux x86 box — false sharing
# and lock contention only appear with several real cores, and the 64-byte cache-line assumption
# is x86 (Apple Silicon uses 128-byte lines). A 1-core box shows nothing; use 8+ cores.
#
#   make run    -> build (release) + run, writing results/concurrency.csv (+ summary on stderr)
#   make check  -> cargo fmt --check + clippy -D warnings   (what CI runs)
#   make test   -> cargo test
#   make clean  -> cargo clean

run: | results
	cargo run --release --quiet > results/concurrency.csv
	@echo "wrote results/concurrency.csv"

check:
	cargo fmt --check
	cargo clippy --quiet -- -D warnings

test:
	cargo test --quiet

results:
	mkdir -p results

clean:
	cargo clean

.PHONY: run check test clean
