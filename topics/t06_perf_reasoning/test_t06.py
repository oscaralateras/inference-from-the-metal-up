"""Tests for T6. The analytic model must be right before the GPU run, not after."""

from __future__ import annotations

import math

import pytest
import torch

from topics.t06_perf_reasoning.model_math import ModelShape

# Qwen2.5-7B's real config values — the model used for the committed results.
QWEN_7B = {
    "_name_or_path": "Qwen/Qwen2.5-7B",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


@pytest.fixture
def shape() -> ModelShape:
    return ModelShape.from_config(QWEN_7B)


def test_analytic_param_count_matches_the_published_size(shape: ModelShape) -> None:
    """Qwen2.5-7B is 7.6B parameters. The arithmetic must land within 1% of that.

    This is the check that stops `model_math` from being plausible-looking algebra that happens to
    be wrong — every downstream prediction is built on this number.
    """
    assert shape.total_params == pytest.approx(7.6e9, rel=0.01)


def test_flops_per_token_is_twice_the_params_read(shape: ModelShape) -> None:
    """One multiply and one add per weight. The entire derivation of `2P`."""
    assert shape.flops_per_token == 2 * shape.params_read_per_token


def test_input_embedding_is_a_lookup_not_a_matmul(shape: ModelShape) -> None:
    """Decode reads one row of the embedding table, not the whole thing.

    Counting the input embedding as per-token traffic overstates bytes/token by several percent
    and makes the prediction pessimistic for a reason unrelated to the hardware.
    """
    assert shape.params_read_per_token == shape.total_params - shape.embedding_params
    assert shape.params_read_per_token < shape.total_params


def test_bfloat16_decode_intensity_is_one_flop_per_byte(shape: ModelShape) -> None:
    """`2P FLOPs / (P x 2 bytes)` = 1. The number that makes decode memory-bound."""
    assert shape.arithmetic_intensity == pytest.approx(1.0)


def test_float16_and_float32_intensities_follow_two_over_bytes(shape: ModelShape) -> None:
    fp32 = ModelShape.from_config(QWEN_7B, bytes_per_param=4)
    assert fp32.arithmetic_intensity == pytest.approx(0.5)


def test_kv_cache_grows_linearly_with_context(shape: ModelShape) -> None:
    """Doubling the context doubles the KV bytes — this is why long sequences decode slower."""
    assert shape.kv_cache_bytes(2048) == 2 * shape.kv_cache_bytes(1024)
    assert shape.kv_cache_bytes(0) == 0


def test_kv_cache_uses_key_value_heads_not_attention_heads(shape: ModelShape) -> None:
    """Grouped-query attention shares K/V across heads — 4 kv heads here, not 28.

    Using `num_attention_heads` would overstate the cache by 7x on this model. The equivalent
    mistake (a hardcoded head count) was a real bug in T5.
    """
    expected = 2 * 28 * 4 * (3584 // 28) * 1024 * 2
    assert shape.kv_cache_bytes(1024) == expected


def test_weight_traffic_does_not_scale_with_batch(shape: ModelShape) -> None:
    """Weights are read once per step regardless of batch — the whole reason batching works."""
    single = shape.bytes_per_token(512, batch=1)
    batched = shape.bytes_per_token(512, batch=8)
    assert batched < 8 * single


def test_batching_raises_predicted_throughput(shape: ModelShape) -> None:
    """More tokens per weight-read means more tokens per second."""
    at_1 = shape.predicted_tokens_per_sec(1500.0, 512, batch=1)
    at_16 = shape.predicted_tokens_per_sec(1500.0, 512, batch=16)
    assert at_16 > at_1


def test_longer_context_lowers_predicted_throughput(shape: ModelShape) -> None:
    long_ctx = shape.predicted_tokens_per_sec(1500.0, 8192)
    short_ctx = shape.predicted_tokens_per_sec(1500.0, 512)
    assert long_ctx < short_ctx


def test_prediction_is_an_upper_bound_on_the_full_model(shape: ModelShape) -> None:
    """The weights-only figure must exceed the all-bytes figure. If not, a term has a sign error."""
    naive = 1500.0 * 1e9 / shape.weight_bytes_per_token
    assert naive > shape.predicted_tokens_per_sec(1500.0, 512)


def test_config_errors_are_raised_not_guessed() -> None:
    """A missing field must fail loudly. Guessing a head count silently corrupted T5's results."""
    with pytest.raises(KeyError, match="num_attention_heads"):
        ModelShape.from_config({k: v for k, v in QWEN_7B.items() if k != "num_attention_heads"})

    with pytest.raises(ValueError, match="not divisible"):
        ModelShape.from_config({**QWEN_7B, "hidden_size": 3585})


def test_invalid_arguments_are_rejected(shape: ModelShape) -> None:
    with pytest.raises(ValueError, match="seq_len"):
        shape.kv_cache_bytes(-1)
    with pytest.raises(ValueError, match="bandwidth"):
        shape.predicted_tokens_per_sec(0.0, 512)


def test_tied_embeddings_are_counted_once() -> None:
    """A tied LM head shares the embedding matrix — counting it twice inflates a small model."""
    tied = ModelShape.from_config({**QWEN_7B, "tie_word_embeddings": True})
    untied = ModelShape.from_config(QWEN_7B)
    assert untied.total_params - tied.total_params == tied.embedding_params


def test_step_result_metrics_are_self_consistent() -> None:
    """concurrency = throughput x latency. Little's law must fall out by construction."""
    from topics.t06_perf_reasoning.measure import StepResult

    for batch in (1, 8, 32):
        step_ms = 20.0
        result = StepResult(
            step_ms=step_ms,
            tokens_per_sec=batch / (step_ms * 1e-3),
            request_p50_ms=100.0,
            request_p99_ms=150.0,
        )
        assert result.as_metrics()["littles_law_concurrency"] == pytest.approx(float(batch))


def test_step_result_reports_only_declared_metrics() -> None:
    """Every metric T6 emits must be in the vocabulary the distinctness guard enforces."""
    from tests.test_distinctness import T6_METRICS
    from topics.t06_perf_reasoning.measure import StepResult

    emitted = set(StepResult(1.0, 1.0, 1.0, 1.0).as_metrics())
    assert emitted <= T6_METRICS, f"undeclared metrics: {emitted - T6_METRICS}"


def test_differenced_step_time_cancels_prefill() -> None:
    """The differencing identity the measurement rests on, checked on synthetic timings."""
    from topics.t06_perf_reasoning.measure import LONG_TOKENS, SHORT_TOKENS

    prefill_s, true_step_s = 0.400, 0.020
    long_s = prefill_s + LONG_TOKENS * true_step_s
    short_s = prefill_s + SHORT_TOKENS * true_step_s

    recovered = (long_s - short_s) / (LONG_TOKENS - SHORT_TOKENS)
    assert recovered == pytest.approx(true_step_s)

    # Dividing a single call by its token count would fold prefill in and overstate the step.
    naive = long_s / LONG_TOKENS
    assert naive > recovered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_analytic_params_match_the_real_model_weights() -> None:
    """The strongest check available: the arithmetic against the weights themselves.

    Skipped without a GPU and run as part of the paid session — an analytic parameter count that
    disagrees with the real model invalidates every number in the topic.
    """
    from transformers import AutoModelForCausalLM

    from topics.t06_perf_reasoning.measure import DEFAULT_MODEL, shape_from_model

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    actual = sum(p.numel() for p in model.parameters())
    analytic = shape_from_model(DEFAULT_MODEL, bytes_per_param=2).total_params
    assert math.isclose(analytic, actual, rel_tol=0.005)
