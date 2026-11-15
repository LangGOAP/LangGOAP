"""``GoapGraph(executor="parallel")`` wires ``ParallelGoapExecutor`` (issue #49).

Using the parallel executor used to require assembling the three nodes by
hand.  The option is also accepted by the low-code on-ramps.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import tool

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.integrations import GoapSubgraph, create_goap_agent


def _diamond() -> list[ActionSpec]:
    return [
        ActionSpec(name="a", effects={"a": True}, execute=lambda ws: {"a": True}),
        ActionSpec(name="b", effects={"b": True}, execute=lambda ws: {"b": True}),
        ActionSpec(
            name="c",
            preconditions={"a": True, "b": True},
            effects={"done": True},
            execute=lambda ws: {"done": True},
        ),
    ]


def _executor_updates(graph: GoapGraph) -> list[int]:
    """Number of history entries written by each executor node invocation."""
    compiled = graph.compile()
    sizes: list[int] = []
    for chunk in compiled.stream(
        graph.initial_state(GoalSpec(conditions={"done": True})), stream_mode="updates"
    ):
        if "executor" in chunk:
            sizes.append(len(chunk["executor"].get("execution_history", [])))
    return sizes


class TestExecutorOption:
    def test_default_is_sequential_one_action_per_step(self) -> None:
        assert _executor_updates(GoapGraph(_diamond())) == [1, 1, 1]

    def test_parallel_runs_independent_actions_as_one_wave(self) -> None:
        assert _executor_updates(GoapGraph(_diamond(), executor="parallel")) == [2, 1]

    def test_parallel_run_reaches_the_goal(self) -> None:
        result = GoapGraph(_diamond(), executor="parallel").invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert [h.action_name for h in result["execution_history"]] == ["a", "b", "c"]

    def test_unknown_executor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="executor"):
            GoapGraph(_diamond(), executor="turbo")  # type: ignore[arg-type]

    def test_on_ramps_accept_the_option(self) -> None:
        @tool
        def gather() -> str:
            """Gather."""
            return "ok"

        agent = create_goap_agent(
            tools=[gather],
            goal=GoalSpec(conditions={"has_data": True}),
            effects={"gather": {"has_data": True}},
            executor="parallel",
        )
        assert (
            agent.invoke({"goal": agent.goap_goal, "world_state": {}})["status"]
            == "goal_achieved"
        )
        sub = GoapSubgraph(
            _diamond(), GoalSpec(conditions={"done": True}), executor="parallel"
        )
        assert (
            sub.compile().invoke({"world_state": {}, "goal": sub.goal})["status"]
            == "goal_achieved"
        )
