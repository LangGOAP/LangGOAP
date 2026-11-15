"""Declared effects are verified by default when ``execute`` returns a dict
(issue #31).

A dict returned from ``execute`` is authoritative for the world state.  If it
never sets a declared effect key, the action was a no-op with respect to the
plan, so the step is an ``action_failed`` (blacklist, replan) instead of a
recorded success that loops the planner into the same action until
``max_replans``.  Values may differ from the declaration (placeholders for
runtime numbers); value drift is the observer's deviation check to handle.  Only user-returned dicts are checked: effects applied from
the declaration or sampled from a ``TransitionModel`` are the runtime's own
view and are judged by the observer's deviation check instead.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.graph.nodes import GoapExecutor, ParallelGoapExecutor
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


def _plan(*actions: ActionSpec) -> Plan:
    states = []
    state = PlanningState.from_dict({})
    for a in actions:
        state = state.apply(a.get_effects(state.to_dict()))
        states.append(state)
    return Plan(
        actions=actions, expected_states=tuple(states), total_cost=float(len(actions))
    )


class TestNoOpReturnsAreFailures:
    def test_partial_dict_fails_the_action_instead_of_looping(self) -> None:
        calls = {"n": 0}

        def forgetful(ws: dict[str, Any]) -> dict[str, Any]:
            calls["n"] += 1
            return {"log": "ran"}  # never sets the declared ``done``

        actions = [ActionSpec(name="finish", effects={"done": True}, execute=forgetful)]
        result = GoapGraph(actions).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "no_plan"
        assert calls["n"] == 1, "the action is blacklisted after one visible failure"
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) == 1
        assert "done" in (failures[0].error or "")
        assert result["blacklisted_actions"] == ["finish"]
        assert result["replan_count"] == 1

    def test_different_value_for_a_declared_effect_is_accepted(self) -> None:
        # ``{"count": 0}`` is a planning placeholder; the runtime number wins
        # and any drift from the plan's expectation is the observer's job.
        actions = [
            ActionSpec(
                name="count", effects={"count": 0}, execute=lambda ws: {"count": 2}
            )
        ]
        executor = GoapExecutor()
        result = executor(
            {"world_state": {}, "plan": _plan(*actions), "current_step": 0}
        )
        assert result.get("status") != "action_failed"
        assert result["world_state"] == {"count": 2}
        assert result["execution_history"][0]["success"] is True

    def test_missing_key_rolls_back_and_names_it(self) -> None:
        actions = [
            ActionSpec(
                name="finish", effects={"done": True}, execute=lambda ws: {"log": "ran"}
            )
        ]
        executor = GoapExecutor()
        result = executor(
            {"world_state": {"prior": 1}, "plan": _plan(*actions), "current_step": 0}
        )
        assert result["status"] == "action_failed"
        assert result["world_state"] == {"prior": 1}, "rejected effects are rolled back"
        entry = result["execution_history"][0]
        assert entry["success"] is False
        assert "'done'" in (entry["error"] or "")
        assert entry["state_after"] == {"prior": 1, "log": "ran"}

    def test_matching_dict_is_a_success(self) -> None:
        actions = [
            ActionSpec(
                name="finish",
                effects={"done": True},
                execute=lambda ws: {"done": True, "log": "ok"},
            )
        ]
        result = GoapGraph(actions).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["log"] == "ok"

    def test_none_return_applies_declared_effects(self) -> None:
        seen: list[str] = []
        actions = [
            ActionSpec(
                name="finish",
                effects={"done": True},
                execute=lambda ws: seen.append("ran"),
            )
        ]
        result = GoapGraph(actions).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved" and seen == ["ran"]

    def test_explicit_validator_still_decides(self) -> None:
        actions = [
            ActionSpec(
                name="finish",
                effects={"done": True},
                execute=lambda ws: {"log": "ran"},
                effect_validator=lambda pre, post: True,  # opt out per action
            )
        ]
        executor = GoapExecutor()
        result = executor(
            {"world_state": {}, "plan": _plan(*actions), "current_step": 0}
        )
        assert result.get("status") != "action_failed"
        assert result["execution_history"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_async_partial_dict_fails_the_action(self) -> None:
        async def forgetful(ws: dict[str, Any]) -> dict[str, Any]:
            return {"log": "ran"}

        actions = [
            ActionSpec(name="finish", effects={"done": True}, aexecute=forgetful)
        ]
        result = await GoapGraph(actions).ainvoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "no_plan"
        assert result["blacklisted_actions"] == ["finish"]

    def test_parallel_executor_applies_the_same_rule(self) -> None:
        actions = (
            ActionSpec(name="a", effects={"a": True}, execute=lambda ws: {"a": True}),
            ActionSpec(
                name="b", effects={"b": True}, execute=lambda ws: {"log": "ran"}
            ),
        )
        executor = ParallelGoapExecutor()
        result = executor(
            {"world_state": {}, "plan": _plan(*actions), "current_step": 0}
        )
        assert result["status"] == "action_failed"
        names = {h["action_name"]: h["success"] for h in result["execution_history"]}
        assert names["b"] is False


class _SlipModel:
    divergence_policy: Any = None

    def expected(
        self, state: Mapping[str, Any], action: ActionSpec
    ) -> Mapping[str, Any]:
        return action.get_effects(dict(state))

    def sample(
        self, state: Mapping[str, Any], action: ActionSpec, rng: random.Random
    ) -> Mapping[str, Any]:
        return {"slipped": True}  # never produces the declared effect


class TestRuntimeOwnedEffectsAreNotValidated:
    def test_transition_model_sample_is_not_an_action_failure(self) -> None:
        action = ActionSpec(name="move", effects={"moved": True})  # no execute
        executor = GoapExecutor(transition_model=_SlipModel(), rng=random.Random(0))
        result = executor({"world_state": {}, "plan": _plan(action), "current_step": 0})
        assert result.get("status") != "action_failed"
        assert result["world_state"] == {"slipped": True}
        assert result["execution_history"][0]["success"] is True
