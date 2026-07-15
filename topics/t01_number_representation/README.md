# T1 — Number representation / floating point / quantization foundation

**Artefact (a01):** a quantizer built from scratch, used to measure **quantization error vs.
bit-width and granularity** on a real LLM weight tensor — and to expose the **activation-outlier
problem** that forces per-channel quantization.

**Status:** scaffold ready — building the quantizer next.

---

## Lab note (fill in as you build)

**Question:** How does quantization error scale with bit-width (int8 vs int4) and granularity
(per-tensor vs per-channel vs per-group), and where do outliers break it?

**Setup:** _tbd — model/weight tensor, quant schemes, metrics, hardware._

**Result:** _tbd — the plot + the one-sentence headline finding._

**Inference payoff:** _tbd — why per-channel/group is essential for LLM weights (the outlier story;
the motivation behind GPTQ/AWQ)._

**What surprised me:** _tbd._

**Caveats & reproduce:** _tbd — `uv run python topics/t01_number_representation/probe.py`; deps/seed
stated._
