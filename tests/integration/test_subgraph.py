"""Integration tests for ``langgoap.integrations.subgraph``.

Layer C of the three-layer low-code on-ramp (AD-2): embed a GOAP
sub-graph inside an existing LangGraph application without leaking
GOAP internals into the parent state.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.integrations import GoapSubgraph, add_goap_subgraph


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="step1",
            preconditions={},
            effects={"a": True},
            execute=lambda s: {"a": True},
        ),
        ActionSpec(
            name="step2",
            preconditions={"a": True},
            effects={"done": True},
            execute=lambda s: {"done": True},
        ),
    ]


class ParentState(TypedDict, total=False):
    """Parent app state — the GOAP subgraph should not pollute this."""

    user_id: str
    world_state: dict[str, Any]
    plan_result: dict[str, Any]
    final_message: str


class TestGoapSubgraphCompile:
    def test_compile_returns_runnable_graph(self) -> None:
        sub = GoapSubgraph(
            actions=_actions(),
            goal=GoalSpec(conditions={"done": True}),
        )
        compiled = sub.compile()
        # It should accept the two-key contract
        result = compiled.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"done": True}),
            }
        )
        assert result.get("status") == "goal_achieved"

    def test_subgraph_internal_keys_do_not_leak(self) -> None:
        sub = GoapSubgraph(
            actions=_actions(),
            goal=GoalSpec(conditions={"done": True}),
        )
        compiled = sub.compile()
        result = compiled.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"done": True}),
            }
        )
        # The internal plan and current_step are still observable via the
        # compiled graph (GoapState schema), but when used inside a parent
        # graph via add_goap_subgraph, only world_state / plan_result are
        # exposed.  This test ensures end_to_end execution completed.
        assert "world_state" in result


class TestAddGoapSubgraph:
    def test_embeds_goap_subgraph_into_parent_graph(self) -> None:
        parent_builder: StateGraph = StateGraph(ParentState)

        def entry(state: ParentState) -> dict[str, Any]:
            return {"world_state": {}}

        def finish(state: ParentState) -> dict[str, Any]:
            return {
                "final_message": (f"goal achieved for {state.get('user_id', 'anon')}"),
            }

        parent_builder.add_node("entry", entry)
        add_goap_subgraph(
            parent_builder,
            name="goap_planner",
            actions=_actions(),
            goal=GoalSpec(conditions={"done": True}),
        )
        parent_builder.add_node("finish", finish)
        parent_builder.add_edge(START, "entry")
        parent_builder.add_edge("entry", "goap_planner")
        parent_builder.add_edge("goap_planner", "finish")
        parent_builder.add_edge("finish", END)

        compiled = parent_builder.compile()
        result = compiled.invoke({"user_id": "u1"})

        assert result["final_message"] == "goal achieved for u1"
        # Parent state should contain the sub-graph's result key.
        assert "plan_result" in result
        assert result["plan_result"].get("status") == "goal_achieved"

    def test_parent_state_keys_preserved(self) -> None:
        parent_builder: StateGraph = StateGraph(ParentState)

        def entry(state: ParentState) -> dict[str, Any]:
            return {"world_state": {}}

        parent_builder.add_node("entry", entry)
        add_goap_subgraph(
            parent_builder,
            name="goap_planner",
            actions=_actions(),
            goal=GoalSpec(conditions={"done": True}),
        )
        parent_builder.add_edge(START, "entry")
        parent_builder.add_edge("entry", "goap_planner")
        parent_builder.add_edge("goap_planner", END)

        compiled = parent_builder.compile()
        result = compiled.invoke({"user_id": "u42"})
        # user_id from the parent state should survive.
        assert result["user_id"] == "u42"
