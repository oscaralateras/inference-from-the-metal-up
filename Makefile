.PHONY: setup lint format format-check type test ci

setup: ; uv sync
lint: ; uv run ruff check .
format: ; uv run ruff format .
format-check: ; uv run ruff format --check .
type: ; uv run pyright
test: ; uv run pytest
ci: lint format-check type test
