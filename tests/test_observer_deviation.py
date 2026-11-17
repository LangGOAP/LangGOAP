"""The observer's deviation check: what counts as a deviation.

Only planning keys are compared (those in action preconditions and effects
and in the goal), a key missing from the world differs from one that is
present, and an unhashable value under a planning key counts as missing.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec
from langgoap.graph.nodes.observer import GoapObserver
from langgoap.planner.types import Plan

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


def route(world: dict[str, Any]) -> Any:
    """The observer's command after the first step with *world* observed."""
    observer = GoapObserver(ACTIONS)
    return observer(
        {
            "goal": GOAL,
            "world_state": world,
            "plan": two_step_plan(),
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
