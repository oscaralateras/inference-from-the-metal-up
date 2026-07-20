# inference-arch-fundamentals

Reproducible artefacts for understanding **LLM inference from the metal up** — one small, *measured*
experiment per computer-architecture topic, each ending in a concrete inference payoff.

Part of a three-repo learning portfolio: this repo explains *why* inference performs as it does.

## Topics

| # | Topic | Headline finding | Inference payoff |
|---|-------|------------------|------------------|
| T1 | Number representation / quantization | Granularity is decisive at low bit-width: naive per-tensor int4 gives 71% output error (cos 0.79); per-group holds 11% (cos 0.99) — worth ~2–3 bits. | Why production int4 needs per-group scales + why GPTQ/AWQ exist: one outlier channel dominates a single per-tensor scale. |
| T2 | CPU execution & the pipeline | On identical work, fitting the pipeline is worth 1.8×–3.2×: branch prediction 1.76×, ILP 3.20×, SIMD 2.35× (AMD EPYC-Milan). | Decode is slow because it lands on the wrong side of all three: serial token dependency (latency-bound) is why batching exists; branchy sampling misfits wide hardware. |

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
