"""``effects="merged"``: a returned dict adds to the declared effects (issue #97).

The declared effects are applied first and whatever ``execute`` returns is
merged on top, so an action returns only what it adds or what differs.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph


def _ship(world_state: dict[str, Any]) -> dict[str, Any]:
    return {"shipping_method": "express"}  # the declared effect is not restated


def _actions(execute: Any = _ship) -> list[ActionSpec]:
    return [
        ActionSpec(
            "ship", {"ordered": True}, {"scheduled": True}, execute=execute, cost=1.0
        ),
        ActionSpec("ship_by_hand", {"ordered": True}, {"scheduled": True}, cost=9.0),
    ]


GOAL = GoalSpec(conditions={"scheduled": True})
WORLD = {"ordered": True}


class TestMerged:
    def test_extra_facts_are_merged_onto_the_declared_effects(self) -> None:
        result = GoapGraph(_actions(), effects="merged").invoke(GOAL, dict(WORLD))
        assert result["status"] == "goal_achieved"
        assert [r.action_name for r in result["execution_history"]] == ["ship"]
        assert result["world_state"]["scheduled"] is True
        assert result["world_state"]["shipping_method"] == "express"

    def test_a_returned_value_for_a_declared_key_wins(self) -> None:
        """The action may still report that the world did not follow; as in the
        default mode that is a deviation the observer answers with a re-plan."""
        result = GoapGraph(
            _actions(lambda ws: {"scheduled": False}), effects="merged"
        ).invoke(GOAL, dict(WORLD))
        first = result["execution_history"][0]
        assert first.action_name == "ship" and first.success is True
        assert first.state_after["scheduled"] is False
        assert result["replan_count"] >= 1

    def test_none_still_applies_the_declaration(self) -> None:
        result = GoapGraph(_actions(lambda ws: None), effects="merged").invoke(
            GOAL, dict(WORLD)
        )
        assert result["world_state"]["scheduled"] is True

    async def test_async_path_merges_too(self) -> None:
        async def ship(world_state: dict[str, Any]) -> dict[str, Any]:
            return {"shipping_method": "express"}

        actions = [
            ActionSpec("ship", {"ordered": True}, {"scheduled": True}, aexecute=ship)
        ]
        result = await GoapGraph(actions, effects="merged").ainvoke(GOAL, dict(WORLD))
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["shipping_method"] == "express"

    def test_parallel_executor_merges_too(self) -> None:
        graph = GoapGraph(_actions(), effects="merged", executor="parallel")
        result = graph.invoke(GOAL, dict(WORLD))
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["shipping_method"] == "express"

    def test_no_sensors_are_needed(self) -> None:
        GoapGraph(_actions(), effects="merged")  # does not raise

    def test_an_explicit_effect_validator_still_decides(self) -> None:
        actions = [
            ActionSpec(
                "ship",
                {"ordered": True},
                {"scheduled": True},
                execute=_ship,
                effect_validator=lambda pre, post: False,
            ),
            ActionSpec(
                "ship_by_hand", {"ordered": True}, {"scheduled": True}, cost=9.0
            ),
        ]
        result = GoapGraph(actions, effects="merged").invoke(GOAL, dict(WORLD))
        assert result["execution_history"][0].success is False


class TestTheDefaultIsUnchanged:
    def test_an_incomplete_dict_still_fails_the_action(self) -> None:
        """Issue #31: under the default mode a returned dict must set every declared key."""
        result = GoapGraph(_actions()).invoke(GOAL, dict(WORLD))
        first = result["execution_history"][0]
        assert first.action_name == "ship" and first.success is False

    def test_the_mode_is_validated(self) -> None:
        with pytest.raises(ValueError, match="merged"):
            GoapGraph(_actions(), effects="nope")  # type: ignore[arg-type]
