.PHONY: setup lint format format-check type test ci t8 t8-predict

# T8 — fused int4 GEMV. `t8-predict` runs anywhere and prints the pre-registered prediction
# without touching a GPU; `t8` needs CUDA + Triton and the session's hardware profile.
t8: ; uv run python -m topics.t08_gpu_architecture.measure && uv run python -m topics.t08_gpu_architecture.plot
t8-predict: ; uv run python -m topics.t08_gpu_architecture.measure --skip-kernel


setup: ; uv sync
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
type: ; uv run pyright
test: ; uv run pytest
ci: lint format-check type test
