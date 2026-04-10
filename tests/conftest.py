"""Shared test fixtures and helpers for LangGoap tests."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from langgoap.actions import ActionSpec
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


class FakeStructuredModel(BaseChatModel):
    """Minimal BaseChatModel for testing structured output.

    Returns a pre-configured response from ``with_structured_output()``.
    Both sync and async paths work via ``RunnableLambda``.

    Args:
        response: The object that ``with_structured_output()`` will return.
        expected_schema: If provided, ``with_structured_output()`` will assert
            that the requested schema matches.  Use this in tests that want to
            verify the caller is asking for the right type.
    """

    response: Any = None
    expected_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        if self.expected_schema is not None and schema is not self.expected_schema:
            raise AssertionError(
                f"with_structured_output called with schema={schema!r}, "
                f"expected {self.expected_schema!r}"
            )
        resp = self.response
        return RunnableLambda(lambda x: resp)


def make_action(
    name: str,
    pre: dict[str, Any] | None = None,
    eff: dict[str, Any] | None = None,
    cost: float = 1.0,
    execute: Any = None,
    max_retries: int = 0,
    resources: dict[str, float] | None = None,
    duration: timedelta | None = None,
) -> ActionSpec:
    """Create an ActionSpec with convenient defaults for testing."""
    return ActionSpec(
        name=name,
        preconditions=pre or {},
        effects=eff or {},
        cost=cost,
        execute=execute,
        max_retries=max_retries,
        resources=resources,
        duration=duration,
    )


def make_plan(*actions: ActionSpec) -> Plan:
    """Create a Plan from actions, computing expected states automatically."""
    states = []
    state = PlanningState.from_dict({})
    for a in actions:
        state = state.apply(a.effects)
        states.append(state)
    return Plan(
        actions=actions,
        expected_states=tuple(states),
        total_cost=sum(a.get_cost() for a in actions),
    )
