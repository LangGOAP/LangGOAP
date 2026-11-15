"""Plans restored from a checkpoint are usable (issue #39).

Callable ``ActionSpec`` fields cannot be persisted.  The old serializer
turned a callable ``cost`` into ``None`` on restore, which crashed the
renderers and ``RepairStrategy`` on the restored plan.  With the JSON-native
state the plan is rebound to the live action list on rehydration, and a
plan rebuilt without live actions degrades to the documented defaults
instead of violating the ``ActionSpec`` contract.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.planner.repair import RepairStrategy
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="gather",
            effects={"data": True},
            cost=lambda ws: 2.0,
            execute=lambda ws: {"data": True},
        ),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects=lambda ws: {"done": True},
            effect_keys=frozenset({"done"}),
            cost=1.0,
            execute=lambda ws: {"done": True},
        ),
    ]


def _paused_run() -> tuple[GoapGraph, Any, dict[str, Any], dict[str, Any]]:
    actions = _actions()
    graph = GoapGraph(actions)
    compiled = graph.compile(checkpointer=MemorySaver(), interrupt_before=["executor"])
    config = {"configurable": {"thread_id": "t-restore"}}
    goal = GoalSpec(conditions={"done": True})
    compiled.invoke(graph.initial_state(goal), config=config)
    return graph, compiled, config, compiled.get_state(config).values


class TestRestoredPlans:
    def test_rehydrated_plan_keeps_live_callables(self) -> None:
        graph, _, _, values = _paused_run()
        plan = graph.rehydrate(values)["plan"]
        assert plan is not None
        assert plan.actions[0] is graph.actions[0]
        assert callable(plan.actions[0].cost)
        assert plan.actions[1].has_dynamic_effects

    def test_rehydrated_plan_renders(self) -> None:
        graph, _, _, values = _paused_run()
        plan = graph.rehydrate(values)["plan"]
        assert plan is not None
        assert "gather" in plan.to_mermaid()
        assert "gather" in plan.to_dot()
        assert "gather" in plan.to_ascii()

    def test_rehydrated_plan_can_be_repaired(self) -> None:
        graph, _, _, values = _paused_run()
        plan = graph.rehydrate(values)["plan"]
        repaired = RepairStrategy().plan(
            PlanningState.from_dict({}),
            GoalSpec(conditions={"done": True}),
            graph.actions,
            prior_plan=plan,
            current_step=0,
        )
        assert repaired is not None
        assert repaired.action_names == ["gather", "process"]

    def test_plan_without_live_actions_degrades_to_defaults(self) -> None:
        _, _, _, values = _paused_run()
        plan = Plan.from_dict(values["plan"])
        assert plan.actions[0].cost == 1.0
        assert plan.actions[0].execute is None
        assert plan.actions[1].has_dynamic_effects
        assert "process" in plan.to_ascii()

    def test_resume_executes_the_restored_plan(self) -> None:
        _, compiled, config, _ = _paused_run()
        compiled.invoke(None, config=config)
        result = compiled.invoke(None, config=config)
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["done"] is True
