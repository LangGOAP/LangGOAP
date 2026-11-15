"""Condition resolvers through the full GOAP loop (issue #32).

Resolved values must be written back into the executed ``world_state`` (so
the observer's deviation check does not fire after every step), and
resolvers must only be consulted for keys no action can produce: asking a
resolver about effect keys or the goal itself lets it declare the goal
already achieved.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.conditions import FunctionalConditionResolver


def _chain() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="prep",
            preconditions={"ready": True},
            effects={"x": True},
            execute=lambda ws: {"x": True},
        ),
        ActionSpec(
            name="mid",
            preconditions={"x": True},
            effects={"y": True},
            execute=lambda ws: {"y": True},
        ),
        ActionSpec(
            name="fin",
            preconditions={"y": True},
            effects={"done": True},
            execute=lambda ws: {"done": True},
        ),
    ]


class _Recorder:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, key: str, world_state: dict[str, Any]) -> Any:
        self.asked.append(key)
        return True if key == "ready" else None


class TestResolverResultsPersist:
    def test_resolved_key_is_persisted_and_no_spurious_replan(self) -> None:
        recorder = _Recorder()
        graph = GoapGraph(
            _chain(),
            resolvers=[FunctionalConditionResolver(name="probe", fn=recorder)],
        )
        result = graph.invoke(goal=GoalSpec(conditions={"done": True}), world_state={})
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0
        assert result["world_state"]["ready"] is True

    def test_resolvers_are_only_asked_about_external_preconditions(self) -> None:
        recorder = _Recorder()
        graph = GoapGraph(
            _chain(),
            resolvers=[FunctionalConditionResolver(name="probe", fn=recorder)],
        )
        graph.invoke(goal=GoalSpec(conditions={"done": True}), world_state={})
        # ``x`` / ``y`` are produced by actions and ``done`` is the goal;
        # none of them is a question for a resolver.
        assert set(recorder.asked) == {"ready"}

    @pytest.mark.asyncio
    async def test_async_path_matches(self) -> None:
        recorder = _Recorder()
        graph = GoapGraph(
            _chain(),
            resolvers=[FunctionalConditionResolver(name="probe", fn=recorder)],
        )
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 0
        assert result["world_state"]["ready"] is True
        assert set(recorder.asked) == {"ready"}
