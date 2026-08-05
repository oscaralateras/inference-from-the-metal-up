.PHONY: setup lint format format-check type test ci probe t6 t7 t8 t8-predict t8-ceiling \
        t9 t9-predict t9-tp t9-rehearse

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


# T9 — interconnects. Needs a MULTI-GPU NVLink node; `t9` refuses to run on anything else, and
# that refusal is the point (see topology.py). Order matters here too:
#
#   1. t9-predict   registers the bands before any hardware exists. Runs on a laptop. Do this
#                   first — a band invented after seeing the data is not a band.
#   2. t9-rehearse  the identical harness over gloo on CPU. Exercises every code path except the
#                   NCCL calls, so nothing is debugged at $6/hr. Its numbers are never published.
#   3. t9           topology gate, then the sweep, then the fit, then the plots. On the pod.
#   4. t9-tp        stage 3, the stretch: a real row-parallel layer, scoring band (4).
t9-predict: ; uv run python -m topics.t09_interconnects.predict --write
t9-rehearse:
	uv run python -m topics.t09_interconnects.measure --backend gloo --world-sizes 2,4 \
	    --max-bytes 16777216
	uv run python -m topics.t09_interconnects.tp_matmul --backend gloo --world-sizes 1,2,4 \
	    --hidden 512 --intermediate 2048
t9:
	uv run python -m topics.t09_interconnects.measure --backend nccl --world-sizes 2,4
	uv run python -m topics.t09_interconnects.plot
t9-tp: ; uv run python -m topics.t09_interconnects.tp_matmul --backend nccl --world-sizes 1,2,4

setup: ; uv sync
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
type: ; uv run pyright
test: ; uv run pytest
ci: lint format-check type test
