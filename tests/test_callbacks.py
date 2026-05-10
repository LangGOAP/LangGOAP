"""Unit tests for ``langgoap.callbacks.CostAccumulator``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from langgoap.callbacks import DEFAULT_COST_PER_1K_TOKENS, CostAccumulator


def _legacy_response(
    prompt_tokens: int, completion_tokens: int, model_name: str = "gpt-4o-mini"
) -> SimpleNamespace:
    """Build a v0.3-style LLMResult with ``llm_output.token_usage``."""
    return SimpleNamespace(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model_name,
        },
    )


def _modern_response(
    input_tokens: int, output_tokens: int, model_name: str = "gpt-4o-mini"
) -> SimpleNamespace:
    """Build a response with ``usage_metadata`` on the AIMessage."""
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={"model_name": model_name},
    )
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(generations=[[generation]], llm_output=None)


class TestAccumulationFromLegacyShape:
    def test_first_call_initialises_keys(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        acc.on_llm_end(_legacy_response(prompt_tokens=1000, completion_tokens=500))
        # gpt-4o-mini: 1000 in * 0.00015/1k + 500 out * 0.0006/1k
        # = 0.000150 + 0.000300 = 0.000450
        assert ws["total_cost_usd"] == pytest.approx(0.00045)
        assert ws["total_tokens"] == 1500
        assert ws["llm_call_count"] == 1

    def test_subsequent_calls_accumulate(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        acc.on_llm_end(_legacy_response(prompt_tokens=1000, completion_tokens=500))
        acc.on_llm_end(_legacy_response(prompt_tokens=2000, completion_tokens=1000))
        assert ws["llm_call_count"] == 2
        assert ws["total_tokens"] == 4500
        # Both calls priced as gpt-4o-mini.
        expected = (1000 * 0.00015 + 500 * 0.0006) / 1000 + (
            2000 * 0.00015 + 1000 * 0.0006
        ) / 1000
        assert ws["total_cost_usd"] == pytest.approx(expected)

    def test_existing_keys_are_extended_not_overwritten(self) -> None:
        ws: dict[str, Any] = {
            "total_cost_usd": 5.00,
            "total_tokens": 9000,
            "llm_call_count": 7,
        }
        acc = CostAccumulator(ws)
        acc.on_llm_end(_legacy_response(prompt_tokens=1000, completion_tokens=500))
        assert ws["total_cost_usd"] > 5.00
        assert ws["total_tokens"] == 10500
        assert ws["llm_call_count"] == 8


class TestAccumulationFromModernShape:
    def test_usage_metadata_path(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        acc.on_llm_end(_modern_response(input_tokens=2000, output_tokens=1000))
        # gpt-4o-mini pricing.
        expected = (2000 * 0.00015 + 1000 * 0.0006) / 1000
        assert ws["total_cost_usd"] == pytest.approx(expected)
        assert ws["total_tokens"] == 3000
        assert ws["llm_call_count"] == 1


class TestPricingResolution:
    def test_substring_match_handles_versioned_model(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        # Versioned: "gpt-4o-mini-2024-07-18" — should match "gpt-4o-mini".
        acc.on_llm_end(
            _legacy_response(
                prompt_tokens=1000,
                completion_tokens=500,
                model_name="gpt-4o-mini-2024-07-18",
            )
        )
        expected = (1000 * 0.00015 + 500 * 0.0006) / 1000
        assert ws["total_cost_usd"] == pytest.approx(expected)

    def test_unknown_model_uses_default_prices(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(
            ws,
            default_input_price=0.001,
            default_output_price=0.002,
        )
        acc.on_llm_end(
            _legacy_response(
                prompt_tokens=1000,
                completion_tokens=500,
                model_name="some-future-model",
            )
        )
        # 1000 * 0.001/1k + 500 * 0.002/1k = 0.001 + 0.001 = 0.002
        assert ws["total_cost_usd"] == pytest.approx(0.002)

    def test_user_pricing_overrides_default(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(
            ws,
            cost_per_1k_tokens={"gpt-4o-mini": (1.0, 2.0)},  # absurd test prices
        )
        acc.on_llm_end(_legacy_response(prompt_tokens=1000, completion_tokens=500))
        # 1000 * 1.0/1k + 500 * 2.0/1k = 1.0 + 1.0 = 2.0
        assert ws["total_cost_usd"] == pytest.approx(2.0)


class TestRobustness:
    def test_missing_token_usage_does_not_crash(self) -> None:
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        empty_response = SimpleNamespace(generations=[], llm_output={})
        acc.on_llm_end(empty_response)
        # Cost zero (no tokens) but the call counter still ticks.
        assert ws["total_cost_usd"] == 0.0
        assert ws["total_tokens"] == 0
        assert ws["llm_call_count"] == 1

    def test_accepts_kwargs_per_basecallbackhandler_contract(self) -> None:
        """``on_llm_end`` is invoked by LangChain with run_id / parent_run_id
        kwargs; the handler must accept them without raising."""
        ws: dict[str, Any] = {}
        acc = CostAccumulator(ws)
        from uuid import uuid4

        acc.on_llm_end(
            _legacy_response(prompt_tokens=10, completion_tokens=5),
            run_id=uuid4(),
            parent_run_id=uuid4(),
            tags=["whatever"],
        )
        assert ws["llm_call_count"] == 1


class TestDefaultPricingTable:
    @pytest.mark.parametrize("model", list(DEFAULT_COST_PER_1K_TOKENS.keys()))
    def test_known_models_have_two_positive_prices(self, model: str) -> None:
        in_price, out_price = DEFAULT_COST_PER_1K_TOKENS[model]
        assert in_price > 0
        assert out_price > 0
