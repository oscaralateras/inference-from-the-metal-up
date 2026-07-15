# inference-arch-fundamentals

Reproducible artefacts for understanding **LLM inference from the metal up** — one small, *measured*
experiment per computer-architecture topic, each ending in a concrete inference payoff.

Part of a three-repo learning portfolio: this repo explains *why* inference performs as it does.

## Topics

| # | Topic | Headline finding | Inference payoff |
|---|-------|------------------|------------------|
| T1 | Number representation / quantization | _in progress_ | |

*(Rows fill in as each artefact ships — this table is the 30-second view of the whole repo.)*

## Running

```bash
uv sync        # set up the environment (Python 3.12, pinned via .python-version)
make lint      # ruff check
make format    # ruff format
make type      # pyright
make test      # pytest
make ci        # everything CI runs
```

Each artefact lives in `topics/tNN_.../` with its own `README.md` lab note and a `results/` folder.

## License

MIT — see [`LICENSE`](LICENSE).
