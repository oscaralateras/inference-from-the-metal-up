.PHONY: setup lint format format-check type test ci probe t6 t7 t8 t8-predict t8-ceiling

# GPU topics. Run `make probe` once per pod first — T6, T7 and T8 all read their ceilings from the
# hardware profile it writes, and a cross-topic test asserts they came from the same session.
probe: ; uv run python -m arch_common.probe

# T6 is FIVE commands in this order, not one, and getting the order wrong fails in two different
# ways - one loud, one silent.
#   1. predict     registers the prediction against THIS session's hardware profile. Must come
#                  first: decompose refuses to run against a prediction filed on another machine,
#                  which is the loud failure and the one you want.
#   2. graphs      the default engine configuration. --fresh clears rows from any previous session
#                  so a new pod cannot inherit the old one's numbers.
#   3. eager       a second invocation, because vLLM gets one engine per process: tearing one down
#                  mid-process to build another is fragile.
#   4. decompose   a separate entry point that reads the CSV and writes the error budget.
#   5. plot
#
# The silent failure: running only step 2 on new hardware left step 4's rows behind from the
# previous session, and T8 - which reads T6's weight share - built its prediction from two machines
# at once. append_rows now refuses that outright. This target is how you avoid meeting either.
t6:
	uv run python -m topics.t06_perf_reasoning.predict
	uv run python -m topics.t06_perf_reasoning.measure --mode graphs --fresh
	uv run python -m topics.t06_perf_reasoning.measure --mode eager
	uv run python -m topics.t06_perf_reasoning.decompose
	uv run python -m topics.t06_perf_reasoning.plot

t7: ; uv run python -m topics.t07_roofline.measure && uv run python -m topics.t07_roofline.plot

# T8 — fused int4 GEMV. `t8-predict` runs anywhere and prints the pre-registered prediction
# without touching a GPU; `t8` needs CUDA + Triton and the session's hardware profile.
t8: ; uv run python -m topics.t08_gpu_architecture.measure && uv run python -m topics.t08_gpu_architecture.plot
t8-predict: ; uv run python -m topics.t08_gpu_architecture.measure --skip-kernel
t8-ceiling: ; uv run python -m topics.t08_gpu_architecture.probe_ceiling


setup: ; uv sync
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
type: ; uv run pyright
test: ; uv run pytest
ci: lint format-check type test
