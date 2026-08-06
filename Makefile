.PHONY: setup lint format format-check type test ci probe t6 t7 t8 t8-predict t8-ceiling \
        t9 t9-predict t9-tp t9-rehearse t9-launch t9-launch-graphs t9-vllm \
        t10 t10-predict t10-rehearse

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
#   5. t9-launch   the causal test for what alpha actually is: capture the collective into a CUDA
#                   graph and see how much of the 35 us survives losing its launches.
#   6. t9-vllm      the end-to-end number, measured rather than modelled. Slowest of the lot --
#                   three engines, one per TP size, each loading a 7B.
#
# World 3 is in the sweep deliberately: with only 2 and 4, the alpha = L + 2(N-1)h decomposition
# has exactly as many equations as unknowns and cannot fail. It is skipped in tp_matmul, where
# 18944 does not divide by 3.
t9:
	uv run python -m topics.t09_interconnects.measure --backend nccl --world-sizes 2,3,4
	uv run python -m topics.t09_interconnects.plot
t9-tp: ; uv run python -m topics.t09_interconnects.tp_matmul --backend nccl --world-sizes 1,2,4
t9-launch: ; uv run python -m topics.t09_interconnects.launch --world-sizes 2,4
# Add --graphs to also capture/replay. Kept opt-in: NCCL inside graph capture deadlocked a
# session once already, and the amortisation sweep answers the question without it.
t9-launch-graphs: ; uv run python -m topics.t09_interconnects.launch --world-sizes 2,4 --graphs
t9-vllm: ; uv run python -m topics.t09_interconnects.vllm_tp --tp 1,2,4

# T10 — cold start. Needs a GPU for stages 3-4 and **root** for the cache drops, which is the one
# requirement that cannot be worked around: a "cold" read on a warm page cache is a DRAM memcpy
# wearing a disk's name, and it is the single most common way a model-load benchmark produces a
# number that will not reproduce on a fresh pod.
#
#   1. t10-predict    files the bands. Laptop, no GPU, no root.
#   2. t10-rehearse   the same harness on a small file, warm only, no GPU. Numbers not published.
#   3. t10            the session: cold and warm x read and mmap, then H2D, then the bands.
t10-predict: ; uv run python -m topics.t10_os_virtual_memory.predict --write
t10-rehearse:
	uv run python -m topics.t10_os_virtual_memory.measure --gib 0.5 --no-cold --no-gpu \
	    --path /tmp/t10_rehearsal.bin
t10:
	uv run python -m topics.t10_os_virtual_memory.measure --gib 8
	uv run python -m topics.t10_os_virtual_memory.plot

setup: ; uv sync
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
type: ; uv run pyright
test: ; uv run pytest
ci: lint format-check type test
