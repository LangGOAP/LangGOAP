"""Failing tests for :meth:`GoalSpec.per_entity`.

``per_entity`` is the ergonomic factory that lets callers express
"one sub-goal per adversarial entity" — e.g. per-ghost flee goals
in langgoap-pacman's ``directional-chase`` scenario — without
hand-building a :class:`MultiGoal` each tick.

The factory consumes a flat list of entity ids and a condition
template keyed/valued with ``{entity}`` placeholders, and returns
a :class:`MultiGoal` whose children are flat :class:`GoalSpec`
instances.  Nested MultiGoal (HTN-style) is explicitly out of
scope; the factory's output must remain valid input to the
existing planner without any other core changes.
"""

from __future__ import annotations

import pytest

from langgoap.goals import GoalSpec, MultiGoal
from langgoap.types import ReplanStrategy


class TestPerEntityFormatting:
    def test_formats_condition_keys(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["alpha", "beta"],
            conditions={"safe_from_{entity}": True},
        )
        assert isinstance(mg, MultiGoal)
        assert len(mg.goals) == 2
        assert dict(mg.goals[0].conditions) == {"safe_from_alpha": True}
        assert dict(mg.goals[1].conditions) == {"safe_from_beta": True}

    def test_formats_string_values(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["a", "b"],
            conditions={"target": "entity_{entity}_escaped"},
        )
        assert dict(mg.goals[0].conditions) == {"target": "entity_a_escaped"}
        assert dict(mg.goals[1].conditions) == {"target": "entity_b_escaped"}

    def test_passes_through_non_string_values(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["x"],
            conditions={
                "k_{entity}": 42,
                "flag_{entity}": True,
                "tuple_{entity}": (1, 2, 3),
            },
        )
        got = dict(mg.goals[0].conditions)
        assert got == {"k_x": 42, "flag_x": True, "tuple_x": (1, 2, 3)}

    def test_preserves_entity_order(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["c", "a", "b"],
            conditions={"safe_from_{entity}": True},
        )
        keys = [list(g.conditions.keys())[0] for g in mg.goals]
        assert keys == ["safe_from_c", "safe_from_a", "safe_from_b"]


class TestPerEntityMode:
    def test_defaults_to_any_mode(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["a"], conditions={"safe_from_{entity}": True}
        )
        assert mg.mode == "any"

    def test_respects_explicit_sequential_mode(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["a", "b"],
            conditions={"safe_from_{entity}": True},
            mode="sequential",
        )
        assert mg.mode == "sequential"


class TestPerEntityGoalKwargs:
    def test_forwards_priority_and_max_replans_to_every_child(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["a", "b"],
            conditions={"safe_from_{entity}": True},
            priority=7,
            max_replans=3,
        )
        for child in mg.goals:
            assert child.priority == 7
            assert child.max_replans == 3

    def test_forwards_replan_strategy_to_every_child(self) -> None:
        mg = GoalSpec.per_entity(
            entity_ids=["a"],
            conditions={"safe_from_{entity}": True},
            replan_strategy=ReplanStrategy.EVERY_ACTION,
        )
        assert mg.goals[0].replan_strategy is ReplanStrategy.EVERY_ACTION


class TestPerEntityValidation:
    def test_rejects_empty_entity_list(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GoalSpec.per_entity(entity_ids=[], conditions={"safe_from_{entity}": True})

    def test_rejects_conditions_without_entity_placeholder(self) -> None:
        # A template with NO ``{entity}`` placeholder in any key would
        # produce identical GoalSpecs for every entity — almost
        # certainly a caller mistake.  Fail loudly.
        with pytest.raises(ValueError, match="entity"):
            GoalSpec.per_entity(entity_ids=["a", "b"], conditions={"static_key": True})


class TestPerEntityPlannerIntegration:
    """The factory's output must be consumable by the existing planner."""

    def test_any_mode_picks_cheapest_applicable_branch(self) -> None:
        from langgoap import ActionSpec, GoapGraph

        actions = [
            ActionSpec(
                name="rescue_a",
                preconditions={},
                effects={"safe_from_a": True},
                cost=1.0,
            ),
            ActionSpec(
                name="rescue_b",
                preconditions={},
                effects={"safe_from_b": True},
                cost=5.0,
            ),
        ]
        mg = GoalSpec.per_entity(
            entity_ids=["a", "b"],
            conditions={"safe_from_{entity}": True},
        )
        graph = GoapGraph(actions).compile()
        result = graph.invoke({"world_state": {}, "goal": mg})
        assert result["status"] == "goal_achieved"
        # The cheaper "rescue_a" branch should win on cost parity with any-mode.
        names = [r.action_name for r in result["execution_history"]]
        assert "rescue_a" in names
