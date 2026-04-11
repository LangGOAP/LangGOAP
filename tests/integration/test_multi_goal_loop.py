"""Integration tests for ``MultiGoal`` through ``GoapGraph``.

Sequential mode chains sub-goals: the world state produced by one
sub-goal becomes the starting state for the next.  ``any`` mode plans
all sub-goals independently and picks the lowest-score feasible plan.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.builder import GoapGraph
from langgoap.graph.nodes import GoapObserver
from langgoap.graph.state import GoapState


def _actions() -> list[ActionSpec]:
    """Linear action pipeline: gather → process → publish.

    Each action unlocks the next and adds a single fact to the world
    state.  Sub-goals in sequential mode hand off via these facts.
    """
    return [
        ActionSpec(name="gather", preconditions={}, effects={"data": True}, cost=1.0),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"processed": True},
            cost=1.0,
        ),
        ActionSpec(
            name="publish",
            preconditions={"processed": True},
            effects={"published": True},
            cost=1.0,
        ),
    ]


class TestSequentialMode:
    def test_two_subgoals_chain_world_state(self) -> None:
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("data") is True
        assert result["world_state"].get("processed") is True
        assert result["world_state"].get("published") is True

    def test_already_satisfied_first_subgoal_advances(self) -> None:
        # First sub-goal is already satisfied in the starting world
        # state → observer should advance immediately.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={"data": True})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("published") is True

    @pytest.mark.asyncio
    async def test_sequential_mode_async(self) -> None:
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(_actions())
        result = await graph.ainvoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("published") is True


class TestAnyMode:
    def test_any_mode_picks_cheapest_feasible(self) -> None:
        # Two competing sub-goals.  "data" is reachable with one action
        # (gather, cost 1.0); "published" costs 3.0 (gather + process +
        # publish).  "any" mode should satisfy the cheaper one first.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="any",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        # Only the cheap sub-goal's actions should have run.
        assert result["world_state"].get("data") is True
        assert "published" not in result["world_state"]

    def test_any_mode_falls_back_to_reachable_subgoal(self) -> None:
        # First sub-goal is unreachable; second should still satisfy.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"impossible": True}),
                GoalSpec(conditions={"data": True}),
            ),
            mode="any",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("data") is True

    def test_any_mode_all_unreachable_fails(self) -> None:
        # M1: every sub-goal is unreachable → terminal failure.
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"impossible_a": True}),
                GoalSpec(conditions={"impossible_b": True}),
            ),
            mode="any",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "no_plan"


class TestSingleElementMultiGoal:
    # L1: a one-element MultiGoal is a valid degenerate composite.
    def test_sequential_single_element(self) -> None:
        mg = MultiGoal(
            goals=(GoalSpec(conditions={"data": True}),),
            mode="sequential",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("data") is True

    def test_any_single_element(self) -> None:
        mg = MultiGoal(
            goals=(GoalSpec(conditions={"data": True}),),
            mode="any",
        )
        graph = GoapGraph(_actions())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("data") is True


class TestSubgoalAdvanceCommand:
    """H1 + H2 regression: the sub-goal advance must reset per-sub-goal
    accounting so ``max_replans`` is per-sub-goal and a blacklist built
    up while working on sub-goal *i* does not leak into sub-goal *i+1*.
    These bugs are hard to expose end-to-end because ``astar_plan`` has
    an internal fallback that unblacklists on unreachability, so the
    most direct way to prove the fix is to call ``_route`` with a
    pre-baked state and inspect the emitted ``Command.update``.
    """

    def test_advance_resets_replan_count_blacklist_and_failure_counts(
        self,
    ) -> None:
        observer = GoapObserver(_actions())
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        # Pre-satisfy the first sub-goal and plant stale per-sub-goal
        # accounting that must NOT carry over.
        state: GoapState = cast(
            GoapState,
            {
                "goal": mg,
                "world_state": {"data": True},
                "current_subgoal_index": 0,
                "replan_count": 3,
                "blacklisted_actions": ["poison"],
                "action_failure_counts": {"poison": 4},
                "status": "",
            },
        )
        cmd = observer._route(state)
        assert cmd.goto == "planner"
        update: dict[str, Any] = cmd.update or {}
        assert update["current_subgoal_index"] == 1
        assert update["replan_reason"] == "subgoal_achieved"
        # H1: per-sub-goal replan budget, not cumulative.
        assert update["replan_count"] == 0
        # H2: sub-goal i's blacklist must not poison sub-goal i+1.
        assert update["blacklisted_actions"] == []
        assert update["action_failure_counts"] == {}
        # plan/current_step wiped so the next sub-goal starts clean.
        assert update["plan"] is None
        assert update["current_step"] == 0

    def test_last_subgoal_satisfied_routes_to_end(self) -> None:
        # Sanity: when the final sequential sub-goal is satisfied we
        # still emit goal_achieved (and do not attempt another advance).
        observer = GoapObserver(_actions())
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        state: GoapState = cast(
            GoapState,
            {
                "goal": mg,
                "world_state": {
                    "data": True,
                    "processed": True,
                    "published": True,
                },
                "current_subgoal_index": 1,
                "status": "",
            },
        )
        cmd = observer._route(state)
        # LangGraph END is a sentinel string; check the goal_achieved
        # status to avoid importing the sentinel just for an equality.
        assert (cmd.update or {}).get("status") == "goal_achieved"


class TestSequentialReplanBudget:
    """M2: replanning inside a sub-goal must consume only that
    sub-goal's ``max_replans`` budget, not a cumulative one.  Without
    the H1 fix, a replan in sub-goal 0 permanently reduces sub-goal
    1's budget.
    """

    def test_per_subgoal_replan_budget(self) -> None:
        # Two sub-goals, each with max_replans=1.  Each sub-goal's
        # critical action fails on its first attempt and succeeds on
        # its second, consuming exactly one replan per sub-goal.
        first_attempts = {"n": 0}
        second_attempts = {"n": 0}

        def first_exec(ws: dict[str, Any]) -> dict[str, Any]:
            first_attempts["n"] += 1
            if first_attempts["n"] == 1:
                raise RuntimeError("transient failure A")
            return {"a_done": True}

        def second_exec(ws: dict[str, Any]) -> dict[str, Any]:
            second_attempts["n"] += 1
            if second_attempts["n"] == 1:
                raise RuntimeError("transient failure B")
            return {"b_done": True}

        actions = [
            ActionSpec(
                name="do_a",
                preconditions={},
                effects={"a_done": True},
                cost=1.0,
                execute=first_exec,
                max_retries=1,
            ),
            ActionSpec(
                name="do_b",
                preconditions={},
                effects={"b_done": True},
                cost=1.0,
                execute=second_exec,
                max_retries=1,
            ),
        ]
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"a_done": True}, max_replans=1),
                GoalSpec(conditions={"b_done": True}, max_replans=1),
            ),
            mode="sequential",
        )
        graph = GoapGraph(actions)
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("a_done") is True
        assert result["world_state"].get("b_done") is True
        # Each action was executed twice: one failure, one success.
        assert first_attempts["n"] == 2
        assert second_attempts["n"] == 2


class TestTracerSeesEffectiveSubgoal:
    """Regression: the planner must hand tracers the *effective*
    sub-goal in sequential mode, not the raw ``MultiGoal``.  A tracer
    reading ``goal.conditions.keys()`` would otherwise crash on every
    plan invocation, and the crash would be silently swallowed by
    ``_safe_tracer_call`` — the worst kind of observability bug.
    """

    def test_sequential_mode_feeds_subgoals_to_on_plan_start(self) -> None:
        plan_start_goals: list[Any] = []

        class Recorder:
            # Intentionally NOT a NullTracer subclass so any missing
            # hook surfaces loudly rather than being silently absorbed.
            def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
                # Must not raise — if ``goal`` is a ``MultiGoal`` the
                # ``.conditions`` attribute does not exist and the
                # tracer call is swallowed by ``_safe_tracer_call``.
                plan_start_goals.append(tuple(sorted(goal.conditions.keys())))

            def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
                pass

            def on_plan_failed(self, reason: str, duration_ms: float) -> None:
                pass

            def on_action_start(self, action: Any, state: Any) -> None:
                pass

            def on_action_complete(self, result: Any) -> None:
                pass

            def on_replan(self, reason: str, new_plan: Any) -> None:
                pass

            def on_goal_achieved(self, final_state: Any) -> None:
                pass

            async def aon_plan_start(
                self, goal: Any, state: Any, strategy_name: str
            ) -> None:
                self.on_plan_start(goal, state, strategy_name)

            async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
                pass

            async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
                pass

            async def aon_action_start(self, action: Any, state: Any) -> None:
                pass

            async def aon_action_complete(self, result: Any) -> None:
                pass

            async def aon_replan(self, reason: str, new_plan: Any) -> None:
                pass

            async def aon_goal_achieved(self, final_state: Any) -> None:
                pass

        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="sequential",
        )
        graph = GoapGraph(_actions(), tracer=Recorder())
        result = graph.invoke(goal=mg, world_state={})
        assert result["status"] == "goal_achieved"
        # Exactly one on_plan_start per sub-goal with the correct keys.
        assert plan_start_goals == [("data",), ("published",)]

    def test_any_mode_feeds_raw_multigoal_to_on_plan_start(self) -> None:
        # ``any`` mode enumerates sub-goals inside ``_plan_core`` and
        # has no single effective sub-goal at the time ``on_plan_start``
        # fires, so the tracer receives the ``MultiGoal`` itself.
        # Tracers that care about ``any`` mode should branch on
        # ``isinstance(goal, MultiGoal)``.
        observed_goals: list[Any] = []

        class MGTracer:
            def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
                observed_goals.append(goal)

            def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
                pass

            def on_plan_failed(self, reason: str, duration_ms: float) -> None:
                pass

            def on_action_start(self, action: Any, state: Any) -> None:
                pass

            def on_action_complete(self, result: Any) -> None:
                pass

            def on_replan(self, reason: str, new_plan: Any) -> None:
                pass

            def on_goal_achieved(self, final_state: Any) -> None:
                pass

            async def aon_plan_start(
                self, goal: Any, state: Any, strategy_name: str
            ) -> None:
                self.on_plan_start(goal, state, strategy_name)

            async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
                pass

            async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
                pass

            async def aon_action_start(self, action: Any, state: Any) -> None:
                pass

            async def aon_action_complete(self, result: Any) -> None:
                pass

            async def aon_replan(self, reason: str, new_plan: Any) -> None:
                pass

            async def aon_goal_achieved(self, final_state: Any) -> None:
                pass

        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"data": True}),
                GoalSpec(conditions={"published": True}),
            ),
            mode="any",
        )
        graph = GoapGraph(_actions(), tracer=MGTracer())
        graph.invoke(goal=mg, world_state={})
        # At least one on_plan_start fired; the first (and only) goal
        # handed to the tracer is the raw MultiGoal.
        assert len(observed_goals) >= 1
        assert isinstance(observed_goals[0], MultiGoal)
