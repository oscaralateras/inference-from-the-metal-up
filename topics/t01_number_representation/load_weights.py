"""Load a single real LLM weight tensor for the quantization study.

Pulls one Linear weight matrix from a small open model (SmolLM2-135M) straight from its
safetensors file — without instantiating the full model. Returns it as a float32 numpy
array, which is the input the quantizer will operate on.

Run directly to sanity-check the download and see the tensor's basic stats:

    uv run python topics/t01_number_representation/load_weights.py
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from safetensors import safe_open

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
# A real Linear weight matrix: shape (out_features, in_features). Each ROW is one output
# channel — which is exactly the axis per-channel quantization scales over.
DEFAULT_TENSOR = "model.layers.0.mlp.gate_proj.weight"


def load_linear_weight(
    model_id: str = DEFAULT_MODEL,
    tensor_name: str = DEFAULT_TENSOR,
) -> npt.NDArray[np.float32]:
    """Return one weight tensor from `model_id` as a float32 numpy array.

    Downloads (and caches) the model's `model.safetensors`, then reads a single named
    tensor out of it. Weights stored in bf16/fp16 are up-cast to float32, since the
    quantization study quantizes *from* full precision.
    """
    path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    with safe_open(path, framework="pt") as f:
        available = set(f.keys())
        if tensor_name not in available:
            preview = "\n  ".join(sorted(available)[:20])
            raise KeyError(f"{tensor_name!r} not found in {model_id}. First tensors:\n  {preview}")
        tensor = f.get_tensor(tensor_name)
    return np.asarray(tensor.float().numpy(), dtype=np.float32)


def _main() -> None:
    w = load_linear_weight()
    print(f"tensor:     {DEFAULT_TENSOR}")
    print(f"shape:      {w.shape}  (out_features, in_features)")
    print(f"dtype:      {w.dtype}")
    print(f"min / max:  {w.min():.4f} / {w.max():.4f}")
    print(f"mean |w|:   {np.abs(w).mean():.4f}")
    print(f"max |w|:    {np.abs(w).max():.4f}   <- this value sets the per-tensor scale")


if __name__ == "__main__":
    _main()
