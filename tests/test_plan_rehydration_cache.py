"""A plan rebuilt from the state channel is remembered for that channel value (#92).

Every node rehydrates the state. Rebinding the plan's actions and rebuilding its
metadata cost a ``Plan.from_dict`` per node; LangGraph hands the same channel
object to every node until the planner writes a new plan, so the rebuilt plan is
kept for that object and that action list.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph, plan
from langgoap.graph.state import plan_from_state
from langgoap.planner.types import Plan


def chain(n: int) -> list[ActionSpec]:
    return [
        ActionSpec(
            f"a{i}",
            {f"s{i}": True},
            {f"s{i + 1}": True},
            execute=lambda ws, i=i: {f"s{i + 1}": True},
        )
        for i in range(n)
    ]


def start(n: int) -> dict[str, Any]:
    return {"s0": True, **{f"s{i}": False for i in range(1, n + 1)}}


def channel_value(actions: list[ActionSpec]) -> dict[str, Any]:
    n = len(actions)
    found = plan(start(n), GoalSpec(conditions={f"s{n}": True}), actions)
    assert found is not None
    return found.to_dict()


def test_the_same_channel_value_rehydrates_once() -> None:
    actions = chain(3)
    data = channel_value(actions)
    first = plan_from_state(data, actions)
    assert first is not None
    assert plan_from_state(data, actions) is first
    assert first.actions[0] is actions[0]  # callables rebound to the live specs


def test_an_equal_copy_is_rebuilt() -> None:
    actions = chain(3)
    data = channel_value(actions)
    first = plan_from_state(data, actions)
    again = plan_from_state(dict(data), actions)
    assert again is not first
    assert again == first


def test_lists_and_mappings_over_the_same_specs_share_the_plan() -> None:
    # The executor and the observer each hold their own list of the specs.
    actions = chain(3)
    data = channel_value(actions)
    first = plan_from_state(data, actions)
    assert plan_from_state(data, list(actions)) is first
    assert plan_from_state(data, {a.name: a for a in actions}) is first


def test_a_different_action_list_rebinds() -> None:
    actions, other = chain(3), chain(3)
    data = channel_value(actions)
    bound = plan_from_state(data, actions)
    rebound = plan_from_state(data, other)
    assert bound is not None and rebound is not None
    assert rebound is not bound
    assert rebound.actions[0] is other[0]


def test_a_channel_value_whose_actions_changed_is_rebuilt() -> None:
    actions = chain(3)
    data = channel_value(actions)
    first = plan_from_state(data, actions)
    data["actions"] = data["actions"][:2]
    shorter = plan_from_state(data, actions)
    assert shorter is not first
    assert len(shorter.actions) == 2


def test_a_run_rebuilds_the_plan_once_per_planning_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = Plan.from_dict

    def counting(cls: type[Plan], data: Any, *, actions: Any = None) -> Plan:
        nonlocal calls
        calls += 1
        return original(data, actions=actions)

    monkeypatch.setattr(Plan, "from_dict", classmethod(counting))
    actions = chain(12)
    result = GoapGraph(actions).invoke(GoalSpec(conditions={"s12": True}), start(12))
    assert result["status"] == "goal_achieved"
    assert calls == 1
