"""Wrapper strategies forward the planner's tracer; callables see the context.

#63 made ``GoapPlanner`` bind its tracer into the strategy's inner search.
The audit found only A* and the two-phase pipeline asserted; the lazy,
anytime and repair wrappers delegate through ``rebind_inner_tracer`` and
are pinned here.  #30 kept non-planning keys as ``PlanningState.context``
so cost functions and dynamic effects can read them; A* and MCTS were
asserted, the pipeline, repair and utility planners were not.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.planner.repair import RepairStrategy
from langgoap.planner.strategy import (
    AnytimePlanningStrategy,
    AStarStrategy,
    LazyDecompositionStrategy,
    TwoPhasePipelineStrategy,
)
from langgoap.planner.utility import UtilityStrategy
from langgoap.state import PlanningState
from langgoap.tracing.null import NullTracer


class _Recorder(NullTracer):
    def __init__(self) -> None:
        self.expansions: list[int] = []
        self.plans: list[Any] = []

    def on_search_expand(self, node_id: int, *args: Any, **kwargs: Any) -> None:
        self.expansions.append(node_id)

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.plans.append(plan)


def _chain() -> list[ActionSpec]:
    return [
        ActionSpec(name="a", effects={"a": True}, execute=lambda ws: {"a": True}),
        ActionSpec(
            name="b",
            preconditions={"a": True},
            effects={"b": True},
            execute=lambda ws: {"b": True},
        ),
    ]


GOAL = GoalSpec(conditions={"b": True})


class TestWrapperStrategiesForwardTheTracer:
    def _expansions_with(self, strategy: Any) -> int:
        recorder = _Recorder()
        result = GoapGraph(
            _chain(), strategy=strategy, tracer=recorder, record_expansions=True
        ).invoke(goal=GOAL, world_state={})
        assert result["status"] == "goal_achieved"
        assert recorder.plans, "the planner's own hooks fired"
        return len(recorder.expansions)

    def test_lazy_decomposition(self) -> None:
        assert self._expansions_with(LazyDecompositionStrategy(lookahead=1)) > 0

    def test_anytime(self) -> None:
        assert self._expansions_with(AnytimePlanningStrategy(time_budget_ms=5_000)) > 0

    def test_repair(self) -> None:
        assert self._expansions_with(RepairStrategy()) > 0

    def test_a_wrapper_keeps_an_inner_tracer_it_was_built_with(self) -> None:
        own = _Recorder()
        planner_tracer = _Recorder()
        strategy = LazyDecompositionStrategy(
            inner=AStarStrategy(tracer=own, record_expansions=True)
        )
        GoapGraph(
            _chain(), strategy=strategy, tracer=planner_tracer, record_expansions=True
        ).invoke(goal=GOAL, world_state={})
        assert own.expansions, "the inner search reports to the tracer it was given"
        assert planner_tracer.expansions == []
        assert planner_tracer.plans, "the planner's own hooks still fire"

    def test_two_phase_pipeline_forwards_too(self) -> None:
        assert self._expansions_with(TwoPhasePipelineStrategy()) > 0


# --------------------------------------------------------------------------- #30 context


def _tiered() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="basic",
            effects={"done": True},
            cost=5.0,
            execute=lambda ws: {"done": True},
        ),
        ActionSpec(
            name="premium",
            effects={"done": True},
            cost=lambda ws: 1.0 if ws.get("vip") else 100.0,
            execute=lambda ws: {"done": True},
        ),
    ]


DONE = GoalSpec(conditions={"done": True})


class TestCallablesSeeTheContextEverywhere:
    def test_pipeline(self) -> None:
        start = PlanningState.from_dict({"vip": True, "done": False}, keys={"done"})
        assert "vip" in start.as_context() and "vip" not in dict(start.conditions)
        plan = pipeline_plan(start, DONE, _tiered())
        assert plan is not None and plan.action_names == ["premium"]
        plan = pipeline_plan(
            PlanningState.from_dict({"done": False}, keys={"done"}), DONE, _tiered()
        )
        assert plan is not None and plan.action_names == ["basic"]

    def test_repair_strategy_in_the_graph(self) -> None:
        vip = GoapGraph(_tiered(), strategy=RepairStrategy()).invoke(
            goal=DONE, world_state={"vip": True}
        )
        assert vip["plan"].action_names == ["premium"]
        plain = GoapGraph(_tiered(), strategy=RepairStrategy()).invoke(
            goal=DONE, world_state={}
        )
        assert plain["plan"].action_names == ["basic"]

    def test_utility_strategy(self) -> None:
        start = PlanningState.from_dict({"vip": True, "done": False}, keys={"done"})
        plan = UtilityStrategy().plan(start, DONE, _tiered())
        assert plan is not None and plan.action_names == ["premium"]
        plan = UtilityStrategy().plan(
            PlanningState.from_dict({"done": False}, keys={"done"}), DONE, _tiered()
        )
        assert plan is not None and plan.action_names == ["basic"]

    def test_lazy_and_anytime_wrappers_in_the_graph(self) -> None:
        for strategy in (
            LazyDecompositionStrategy(),
            AnytimePlanningStrategy(time_budget_ms=5_000),
        ):
            vip = GoapGraph(_tiered(), strategy=strategy).invoke(
                goal=DONE, world_state={"vip": True}
            )
            assert vip["plan"].action_names == ["premium"], type(strategy).__name__
