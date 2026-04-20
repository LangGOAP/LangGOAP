"""Failing tests for sequential ``MultiGoal`` preflight + fallback.

Two structural improvements to the sequential sub-goal dispatch,
uncovered by the Phase 2 per-entity Pac-Man experiment
(``research/experiments/2026-04-20-per-entity-goals.md``):

1. **Preflight** — when the current sub-goal is already satisfied
   in the sensed world state, advance past it **inside
   ``_resolve_sequential_subgoal``** rather than paying for an
   empty-A*-plan → observer-advance round-trip.  The next CS188
   tick and the per-tick replan cadence multiply that overhead
   across the full game length.

2. **Infeasible-sub-goal fallback** — when A* cannot find a plan
   for the current sub-goal (e.g. per-ghost escape is physically
   unreachable inside one planning invocation), advance to the
   next sub-goal and try again.  "Sequential" becomes
   *best-effort sequential*: the planner still prefers lower
   sub-goals but doesn't block the whole invocation on an
   unreachable one.  Directly enables the per-entity Pac-Man
   agent to degrade gracefully when flee is insufficient instead
   of collapsing to ``STOP``.
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
        names = [r.action_name for r in result["execution_history"]]
        assert names == ["eat"]


# ---------------------------------------------------------------------------
# Fallback: an infeasible sub-goal falls through to the next
# ---------------------------------------------------------------------------


class TestSequentialFallback:
    def test_infeasible_subgoal_falls_through_to_next(self) -> None:
        """Unreachable sub-goal must not wedge the whole invocation."""
        actions = [
            ActionSpec(
                name="eat",
                preconditions={},
                effects={"food_in_belly": True},
                cost=1.0,
            ),
        ]
        # ``unreachable_flag`` has no action that establishes it and is
        # not already True in the world state.  Strict-sequential today
        # would fail the whole plan; best-effort sequential must skip
        # it and reach the food goal.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"unreachable_flag": True}),
                GoalSpec(conditions={"food_in_belly": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(actions).compile()
        result = graph.invoke({"world_state": {}, "goal": mg})
        assert result["status"] == "goal_achieved"
        names = [r.action_name for r in result["execution_history"]]
        assert names == ["eat"]
        # The unreachable sub-goal was skipped, not satisfied.
        assert result["current_subgoal_index"] >= 1

    def test_all_infeasible_subgoals_yields_no_plan_status(self) -> None:
        """When every sub-goal is unreachable, the invocation fails cleanly."""
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}),
                GoalSpec(conditions={"b": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(actions=[]).compile()
        result = graph.invoke({"world_state": {}, "goal": mg})
        # Every sub-goal is infeasible and no action establishes either.
        # The previous strict-sequential behaviour emitted ``no_plan``;
        # best-effort sequential preserves that terminal signal after
        # exhausting the fallback chain.
        assert result.get("status") == "no_plan"


# ---------------------------------------------------------------------------
# Regression: non-sequential modes and single-goal paths unaffected
# ---------------------------------------------------------------------------


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
        names = [r.action_name for r in result["execution_history"]]
        assert "cheap_a" in names
        assert "costly_b" not in names
