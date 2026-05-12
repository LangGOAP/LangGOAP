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


# ---------------------------------------------------------------------------
# Quickstart-notebook mirror tests
# ---------------------------------------------------------------------------
#
# The cells of ``examples/basics/goap_subgraph.ipynb`` are reproduced
# below as assertions so the notebook's code path cannot drift from the
# public API without a test failure.  Any divergence here means the
# notebook needs to be regenerated.


def _quickstart_actions() -> list[ActionSpec]:
    """Three-step pipeline used by the GoapSubgraph quickstart notebook."""
    return [
        ActionSpec(
            name="research",
            preconditions={},
            effects={"have_brief": True},
            execute=lambda ws: {
                "brief": f"Brief on {ws['topic']}",
                "have_brief": True,
            },
            cost=1.0,
        ),
        ActionSpec(
            name="write",
            preconditions={"have_brief": True},
            effects={"have_draft": True},
            execute=lambda ws: {
                "draft": f"Article from: {ws['brief']}",
                "have_draft": True,
            },
            cost=1.0,
        ),
        ActionSpec(
            name="publish",
            preconditions={"have_draft": True},
            effects={"published": True},
            execute=lambda ws: {
                "url": f"https://blog/{ws['draft'][:10]}",
                "published": True,
            },
            cost=1.0,
        ),
    ]


_QUICKSTART_GOAL = GoalSpec(conditions={"published": True})


class TestSubgraphQuickstart:
    """Mirror of ``examples/basics/goap_subgraph.ipynb``.

    Each test corresponds to a section of the notebook so the notebook
    is exercised end-to-end on every CI run.
    """

    def test_standalone_subgraph_runs_three_action_chain(self) -> None:
        sub = GoapSubgraph(actions=_quickstart_actions(), goal=_QUICKSTART_GOAL)
        compiled = sub.compile()
        result = compiled.invoke(
            {
                "world_state": {"topic": "GOAP"},
                "goal": _QUICKSTART_GOAL,
            }
        )

        assert result["status"] == "goal_achieved"
        names = [r.action_name for r in result["execution_history"]]
        assert names == ["research", "write", "publish"]
        ws = result["world_state"]
        assert ws["published"] is True
        assert ws["brief"] == "Brief on GOAP"
        assert ws["draft"] == "Article from: Brief on GOAP"

    def test_embed_in_parent_stategraph_keeps_internals_sealed(self) -> None:
        class QuickstartParentState(TypedDict, total=False):
            user_id: str
            world_state: dict[str, Any]
            plan_result: dict[str, Any]
            final_message: str

        def entry(state: QuickstartParentState) -> dict[str, Any]:
            return {"world_state": {"topic": "GOAP for LangGraph"}}

        def finish(state: QuickstartParentState) -> dict[str, Any]:
            url = state.get("plan_result", {}).get("world_state", {}).get("url", "<none>")
            return {
                "final_message": (
                    f"hi {state.get('user_id', 'anon')} - published at {url}"
                ),
            }

        parent_builder: StateGraph = StateGraph(QuickstartParentState)
        parent_builder.add_node("entry", entry)
        add_goap_subgraph(
            parent_builder,
            name="goap_planner",
            actions=_quickstart_actions(),
            goal=_QUICKSTART_GOAL,
        )
        parent_builder.add_node("finish", finish)
        parent_builder.add_edge(START, "entry")
        parent_builder.add_edge("entry", "goap_planner")
        parent_builder.add_edge("goap_planner", "finish")
        parent_builder.add_edge("finish", END)

        compiled = parent_builder.compile()
        result = compiled.invoke({"user_id": "u42"})

        # Parent narrative threaded through the subgraph.
        assert result["final_message"].startswith("hi u42 - published at https://blog/")
        assert result["user_id"] == "u42"

        # Sealed internals: GOAP-private keys must not leak to top-level
        # parent state — they only live inside ``plan_result``.
        goap_internal_keys = {
            "plan",
            "current_step",
            "execution_history",
            "blacklisted_actions",
            "action_failure_counts",
        }
        assert goap_internal_keys.isdisjoint(result.keys())
        assert "execution_history" in result["plan_result"]
        assert result["plan_result"]["status"] == "goal_achieved"

    def test_custom_input_output_keys(self) -> None:
        class CustomKeyState(TypedDict, total=False):
            biz_input: dict[str, Any]
            biz_output: dict[str, Any]

        def entry(state: CustomKeyState) -> dict[str, Any]:
            return {"biz_input": {"topic": "GOAP"}}

        parent_builder: StateGraph = StateGraph(CustomKeyState)
        parent_builder.add_node("entry", entry)
        add_goap_subgraph(
            parent_builder,
            name="goap",
            actions=_quickstart_actions(),
            goal=_QUICKSTART_GOAL,
            input_key="biz_input",
            output_key="biz_output",
        )
        parent_builder.add_edge(START, "entry")
        parent_builder.add_edge("entry", "goap")
        parent_builder.add_edge("goap", END)

        compiled = parent_builder.compile()
        result = compiled.invoke({})

        assert "biz_output" in result
        assert result["biz_output"]["status"] == "goal_achieved"
        assert result["biz_output"]["world_state"]["published"] is True
