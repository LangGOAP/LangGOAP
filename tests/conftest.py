"""Shared test fixtures and helpers for LangGoap tests."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load .env file for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

from langgoap.actions import ActionSpec
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from langgoap.testing import FakeStructuredModel

__all__ = ["FakeStructuredModel", "make_action", "make_plan"]


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
