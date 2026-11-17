"""The observer's deviation check compares planning keys directly (#93).

It used to build two ``PlanningState`` objects per executed step. The routing
must not change: only planning keys count, a key missing from the world differs
from one that is present, and an unhashable planning value counts as missing,
as it did when ``PlanningState.from_dict`` dropped it.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec
from langgoap.graph.nodes.observer import GoapObserver
from langgoap.planner.types import Plan
from langgoap.state import PlanningState

ACTIONS = [
    ActionSpec("warm", {"x": False}, {"x": True}),
    ActionSpec("finish", {"x": True}, {"z": True}),
]
GOAL = GoalSpec(conditions={"z": True})


def two_step_plan() -> Plan:
    """A hand-built plan whose expected states carry a non-planning key."""
    return Plan.from_dict(
        {
            "actions": [{"name": a.name} for a in ACTIONS],
            "expected_states": [
                {"x": True, "z": False, "extra": 1},
                {"x": True, "z": True, "extra": 1},
            ],
            "total_cost": 2.0,
        },
        actions=ACTIONS,
    )


def route(world: dict[str, Any], plan: Plan | None = None) -> Any:
    observer = GoapObserver(ACTIONS)
    return observer(
        {
            "goal": GOAL,
            "world_state": world,
            "plan": plan if plan is not None else two_step_plan(),
            "current_step": 1,
            "status": "",
            "execution_history": [],
        }
    )


def test_matching_planning_keys_continue_the_plan() -> None:
    assert route({"x": True, "z": False, "docs": [1, 2]}).goto == "executor"


def test_a_changed_planning_key_replans() -> None:
    cmd = route({"x": False, "z": False})
    assert cmd.goto == "planner"
    assert cmd.update["replan_reason"] == "state_deviation"


def test_a_missing_planning_key_differs_from_a_present_one() -> None:
    assert route({"z": False}).goto == "planner"


def test_non_planning_keys_are_ignored_on_both_sides() -> None:
    # The expected state carries ``extra``; the world lacks it and carries ``docs``.
    assert route({"x": True, "z": False, "docs": "many"}).goto == "executor"


def test_an_unhashable_planning_value_counts_as_missing() -> None:
    assert route({"x": [True], "z": False}).goto == "planner"


def test_the_check_builds_no_planning_state(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = two_step_plan()
    plan.expected_states[0]  # the lazy entry is materialised once, before the check

    def boom(*_args: Any, **_kwargs: Any) -> PlanningState:
        raise AssertionError("the deviation check built a PlanningState")

    monkeypatch.setattr(PlanningState, "from_dict", classmethod(boom))
    assert route({"x": True, "z": False}, plan).goto == "executor"
