"""Failing tests for sequential ``MultiGoal`` preflight + fallback.

Two structural improvements to the sequential sub-goal dispatch,
uncovered by the Phase 2 per-entity Pac-Man experiment
(the internal "2026-04-20-per-entity-goals" design note (not in the repository)):

1. **Preflight** — when the current sub-goal is already satisfied
   in the sensed world state, advance past it **inside
   ``_plan_sequential``** rather than paying for an
   empty-A*-plan → observer-advance round-trip.  The next CS188
   tick and the per-tick replan cadence multiply that overhead
   across the full game length.

2. **Infeasible-sub-goal fallback** — when A* cannot find a plan
   for the current sub-goal (e.g. per-ghost escape is physically
   unreachable inside one planning invocation), advance to the
   next sub-goal and try again.  Since 0.2.0 this is the explicit
   ``MultiGoal(on_infeasible="skip")`` opt-in (issue #35): the
   default is strict, matching the ``MultiGoal`` docstring, and a
   skipping run records ``skipped_subgoals`` and finishes with
   ``status="partial"`` rather than claiming ``goal_achieved``.
"""

from __future__ import annotations

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy
from langgoap.goals import MultiGoal

# ---------------------------------------------------------------------------
# Preflight: already-satisfied sub-goals are skipped in one shot
# ---------------------------------------------------------------------------


class TestSequentialPreflight:
    def test_preflight_skips_trivially_satisfied_prefix(self) -> None:
        """A prefix of satisfied sub-goals must not cost an A* call each."""
        actions = [
            ActionSpec(
                name="eat",
                preconditions={"hungry": True},
                effects={"food_in_belly": True, "hungry": False},
                cost=1.0,
            ),
        ]
        graph = GoapGraph(actions).compile()
        # Two prefix sub-goals are already True in the world state; the
        # terminal food-in-belly sub-goal drives the actual planning.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"safe_a": True}),
                GoalSpec(conditions={"safe_b": True}),
                GoalSpec(conditions={"food_in_belly": True}),
            ),
            mode="sequential",
        )
        world = {"safe_a": True, "safe_b": True, "hungry": True}
        result = graph.invoke({"world_state": world, "goal": mg})
        assert result["status"] == "goal_achieved"
        # current_subgoal_index must jump past the satisfied prefix.
        assert result["current_subgoal_index"] >= 2
        names = [r["action_name"] for r in result["execution_history"]]
        assert names == ["eat"]


# ---------------------------------------------------------------------------
# Fallback: an infeasible sub-goal falls through to the next
# ---------------------------------------------------------------------------


class TestSequentialStrictByDefault:
    """A sequential ``MultiGoal`` fails on an unreachable sub-goal (issue #35).

    The docstring promises "goals[0] to completion, then goals[1]"; a
    skipped stage reported as ``goal_achieved`` is a silent lie.  Skipping
    is an explicit opt-in (``on_infeasible="skip"``) that records what was
    skipped and finishes ``partial``.
    """

    def _actions(self) -> list[ActionSpec]:
        return [
            ActionSpec(
                name="eat",
                preconditions={},
                effects={"food_in_belly": True},
                cost=1.0,
            ),
        ]

    def _goals(self) -> tuple[GoalSpec, GoalSpec]:
        return (
            GoalSpec(conditions={"unreachable_flag": True}),
            GoalSpec(conditions={"food_in_belly": True}),
        )

    def test_default_is_strict_and_reports_no_plan(self) -> None:
        mg = MultiGoal(goals=self._goals(), mode="sequential")
        assert mg.on_infeasible == "fail"
        result = (
            GoapGraph(self._actions()).compile().invoke({"world_state": {}, "goal": mg})
        )
        assert result["status"] == "no_plan"
        assert result["execution_history"] == []
        assert result.get("current_subgoal_index", 0) == 0
        assert (
            "unreachable_flag"
            in result["no_plan_explanation"]["unreachable_conditions"]
        )

    def test_skip_opt_in_records_the_skip_and_finishes_partial(self) -> None:
        mg = MultiGoal(goals=self._goals(), mode="sequential", on_infeasible="skip")
        result = (
            GoapGraph(self._actions()).compile().invoke({"world_state": {}, "goal": mg})
        )
        assert result["status"] == "partial"
        assert [r["action_name"] for r in result["execution_history"]] == ["eat"]
        assert result["skipped_subgoals"] == [0]
        assert result["world_state"]["food_in_belly"] is True

    def test_skip_with_nothing_skipped_is_goal_achieved(self) -> None:
        mg = MultiGoal(
            goals=(GoalSpec(conditions={"food_in_belly": True}),),
            mode="sequential",
            on_infeasible="skip",
        )
        result = (
            GoapGraph(self._actions()).compile().invoke({"world_state": {}, "goal": mg})
        )
        assert result["status"] == "goal_achieved"
        assert result.get("skipped_subgoals", []) == []

    def test_all_infeasible_subgoals_yields_no_plan_status(self) -> None:
        for policy in ("fail", "skip"):
            mg = MultiGoal(
                goals=(
                    GoalSpec(conditions={"impossible_a": True}),
                    GoalSpec(conditions={"impossible_b": True}),
                ),
                mode="sequential",
                on_infeasible=policy,  # type: ignore[arg-type]
            )
            result = (
                GoapGraph(self._actions())
                .compile()
                .invoke({"world_state": {}, "goal": mg})
            )
            assert result.get("status") == "no_plan", policy

    def test_invalid_policy_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="on_infeasible"):
            MultiGoal(goals=self._goals(), on_infeasible="maybe")  # type: ignore[arg-type]

    def test_policy_round_trips_through_the_codec(self) -> None:
        mg = MultiGoal(goals=self._goals(), mode="sequential", on_infeasible="skip")
        assert MultiGoal.from_dict(mg.to_dict()) == mg
        assert mg.to_dict()["on_infeasible"] == "skip"


class TestNoRegressionOnOtherModes:
    def test_any_mode_still_picks_cheapest_feasible(self) -> None:
        """``mode='any'`` must keep its existing semantics."""
        actions = [
            ActionSpec(
                name="cheap_a",
                preconditions={},
                effects={"a": True},
                cost=1.0,
            ),
            ActionSpec(
                name="costly_b",
                preconditions={},
                effects={"b": True},
                cost=10.0,
            ),
        ]
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}),
                GoalSpec(conditions={"b": True}),
            ),
            mode="any",
        )
        graph = GoapGraph(actions).compile()
        result = graph.invoke({"world_state": {}, "goal": mg})
        names = [r["action_name"] for r in result["execution_history"]]
        assert "cheap_a" in names
        assert "costly_b" not in names
