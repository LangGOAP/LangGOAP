"""LangChain callback adapters for LangGOAP cost / token accounting.

The :class:`~langgoap.termination.MaxCostPolicy`,
:class:`~langgoap.termination.MaxTokensPolicy`, and
:class:`~langgoap.termination.MaxLLMCallsPolicy` policies read three
keys off ``world_state``:

- ``total_cost_usd`` — running USD cost across LLM calls
- ``total_tokens`` — running total of input + output tokens
- ``llm_call_count`` — running count of LLM invocations

This module provides :class:`CostAccumulator`, a
``BaseCallbackHandler`` that populates those keys automatically when
attached to a LangChain LLM client.  Drop it on a ``ChatOpenAI`` /
``ChatAnthropic`` / etc., bind the handler to the same dict you pass
as ``world_state``, and the termination policies see real numbers
without any further wiring.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# (input_per_1k, output_per_1k) USD pricing for LangChain models.
# Pulled from publicly published pricing snapshots; users override per
# call via ``cost_per_1k_tokens=...``.
DEFAULT_COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o1": (0.015, 0.06),
    "o1-mini": (0.003, 0.012),
    "o3-mini": (0.0011, 0.0044),
    # Anthropic
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.001, 0.005),
    "claude-3-opus": (0.015, 0.075),
    "claude-sonnet-4": (0.003, 0.015),
    "claude-opus-4": (0.015, 0.075),
}


class CostAccumulator(BaseCallbackHandler):
    """Accumulate LLM cost / tokens / call count into a ``world_state`` dict.

    Wire it on the LLM you call from your action's ``execute`` /
    ``aexecute``::

        from langgoap import CostAccumulator
        from langchain_openai import ChatOpenAI

        def research(world_state: dict) -> dict:
            llm = ChatOpenAI(
                model='gpt-4o-mini',
                temperature=0,
                callbacks=[CostAccumulator(world_state)],
            )
            response = llm.invoke('summarise these sources: ...')
            return {'summary': response.content}

    After the call, ``world_state['total_cost_usd']``,
    ``['total_tokens']``, and ``['llm_call_count']`` carry the running
    totals — exactly what :class:`MaxCostPolicy` /
    :class:`MaxTokensPolicy` / :class:`MaxLLMCallsPolicy` consume.

    Args:
        world_state: The dict to write totals into.  Mutated in place
            so the policies see live values across action invocations.
        cost_per_1k_tokens: Optional pricing override (``model_name``
            substring → ``(input_per_1k, output_per_1k)`` in USD).
            Merged with :data:`DEFAULT_COST_PER_1K_TOKENS` — your
            entries win on collision.
        default_input_price: Used when no model-specific entry matches.
            Defaults to ``0.0`` (unknown models contribute zero cost).
        default_output_price: As above.
    """

    def __init__(
        self,
        world_state: dict[str, Any],
        *,
        cost_per_1k_tokens: Mapping[str, tuple[float, float]] | None = None,
        default_input_price: float = 0.0,
        default_output_price: float = 0.0,
    ) -> None:
        self._world_state = world_state
        self._pricing: dict[str, tuple[float, float]] = {
            **DEFAULT_COST_PER_1K_TOKENS,
            **(dict(cost_per_1k_tokens) if cost_per_1k_tokens else {}),
        }
        self._default_input_price = default_input_price
        self._default_output_price = default_output_price

    def _resolve_pricing(self, model_name: str | None) -> tuple[float, float]:
        if not model_name:
            return self._default_input_price, self._default_output_price
        # Exact match wins.
        if model_name in self._pricing:
            return self._pricing[model_name]
        # Substring match handles versioned variants ("gpt-4o-mini-2024-07-18"
        # → "gpt-4o-mini").  Longest match wins so "gpt-4o-mini" beats
        # "gpt-4o" for "gpt-4o-mini-2024-07-18".
        candidates = sorted(
            (k for k in self._pricing if k in model_name),
            key=len,
            reverse=True,
        )
        if candidates:
            return self._pricing[candidates[0]]
        return self._default_input_price, self._default_output_price

    def _extract_usage(self, response: Any) -> tuple[int, int, str | None]:
        """Pull (input_tokens, output_tokens, model_name) from an LLMResult.

        Handles both shapes LangChain produces:
        - Legacy: ``response.llm_output['token_usage']`` and ``['model_name']``
        - Modern: ``response.generations[0][0].message.usage_metadata`` and
          ``response_metadata['model_name']``
        """
        # Modern shape first.
        try:
            generations = getattr(response, "generations", None)
            if generations:
                gen = generations[0][0]
                msg = getattr(gen, "message", None)
                if msg is not None:
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        meta = getattr(msg, "response_metadata", {}) or {}
                        return (
                            int(usage.get("input_tokens", 0)),
                            int(usage.get("output_tokens", 0)),
                            meta.get("model_name") or meta.get("model"),
                        )
        except Exception:  # pragma: no cover - defensive
            logger.debug("usage_metadata extraction failed", exc_info=True)

        # Legacy shape.
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {}) or {}
        return (
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            llm_output.get("model_name") or llm_output.get("model"),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        input_tokens, output_tokens, model_name = self._extract_usage(response)
        in_price, out_price = self._resolve_pricing(model_name)
        cost = (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price

        ws = self._world_state
        ws["total_cost_usd"] = float(ws.get("total_cost_usd", 0.0)) + cost
        ws["total_tokens"] = (
            int(ws.get("total_tokens", 0)) + input_tokens + output_tokens
        )
        ws["llm_call_count"] = int(ws.get("llm_call_count", 0)) + 1


__all__ = [
    "DEFAULT_COST_PER_1K_TOKENS",
    "CostAccumulator",
]
