"""Integration tests for the full GOAP planning-execution loop.

These tests verify end-to-end behavior: building a GoapGraph,
invoking it with a goal and world state, and checking the result.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph

from langgoap.actions import ActionSpec
from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.graph.nodes import GoapExecutor, GoapObserver, GoapPlanner
from langgoap.graph.state import GoapState, successful_action_names
from langgoap.types import ReplanStrategy
from tests.conftest import make_action as _action

# ---------------------------------------------------------------------------
# Report pipeline: gather → clean → analyze → write_report
# ---------------------------------------------------------------------------


def _report_pipeline_actions() -> list[ActionSpec]:
    """4-action pipeline: gather data → clean → analyze → write report."""
    return [
        _action("gather_data", eff={"has_raw_data": True}),
        _action("clean_data", pre={"has_raw_data": True}, eff={"has_clean_data": True}),
        _action(
            "analyze_data",
            pre={"has_clean_data": True},
            eff={"has_analysis": True},
        ),
        _action(
            "write_report",
            pre={"has_analysis": True},
            eff={"report_complete": True},
        ),
    ]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullGoapLoop:
    def test_achieves_goal(self) -> None:
        """Full loop: 4-action pipeline achieves report_complete."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(conditions={"report_complete": True}),
                "world_state": {},
            }
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["report_complete"] is True
        assert result["world_state"]["has_analysis"] is True

    def test_builder_matches_manual_graph(self) -> None:
        """GoapGraph produces the same result as manually constructed graph."""
        actions = _report_pipeline_actions()

        # Builder
        builder_result = (
            GoapGraph(actions=actions)
            .compile()
            .invoke(
                {
                    "goal": GoalSpec(conditions={"report_complete": True}),
                    "world_state": {},
                }
            )
        )

        # Manual
        manual = StateGraph(GoapState)
        manual.add_node("planner", GoapPlanner(actions))
        manual.add_node("executor", GoapExecutor())
        manual.add_node("observer", GoapObserver(actions))
        manual.add_edge(START, "planner")
        manual.add_edge("planner", "executor")
        manual.add_edge("executor", "observer")
        manual_result = manual.compile().invoke(
            {
                "goal": GoalSpec(conditions={"report_complete": True}),
                "world_state": {},
            }
        )

        assert builder_result["status"] == manual_result["status"]
        assert builder_result["world_state"] == manual_result["world_state"]

    def test_replans_on_state_deviation(self) -> None:
        """When execution produces unexpected state, observer triggers replan."""
        call_count = 0

        def flaky_gather(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: unexpected side effect
                return {"has_raw_data": True, "has_clean_data": False}
            return {"has_raw_data": True}

        actions = [
            ActionSpec(
                name="gather_data",
                effects={"has_raw_data": True},
                execute=flaky_gather,
            ),
            _action(
                "clean_data", pre={"has_raw_data": True}, eff={"has_clean_data": True}
            ),
            _action(
                "analyze_data", pre={"has_clean_data": True}, eff={"has_analysis": True}
            ),
            _action(
                "write_report",
                pre={"has_analysis": True},
                eff={"report_complete": True},
            ),
        ]
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(
                    conditions={"report_complete": True},
                    policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION),
                ),
                "world_state": {},
            }
        )

        assert result["status"] == "goal_achieved"
        # Initial plan does not count; replan_count >= 1 means at least one
        # genuine replan cycle occurred due to the state deviation.
        assert result["replan_count"] >= 1

    def test_handles_action_failure_with_replanning(self) -> None:
        """When an action raises an exception, observer triggers replan."""
        fail_count = 0

        def sometimes_fails(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal fail_count
            fail_count += 1
            if fail_count == 1:
                raise RuntimeError("transient failure")
            return {"has_raw_data": True}

        actions = [
            ActionSpec(
                name="gather_data",
                effects={"has_raw_data": True},
                execute=sometimes_fails,
            ),
            _action(
                "clean_data", pre={"has_raw_data": True}, eff={"has_clean_data": True}
            ),
            _action("write_report", pre={"has_clean_data": True}, eff={"done": True}),
        ]
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(conditions={"done": True}),
                "world_state": {},
            }
        )

        assert result["status"] == "goal_achieved"
        # Should have at least one failed execution in history
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1

    def test_unreachable_goal_fails_gracefully(self) -> None:
        """When no action can reach the goal, planner returns no_plan."""
        actions = [_action("useless", eff={"x": True})]
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(conditions={"impossible": True}),
                "world_state": {},
            }
        )

        assert result["status"] == "no_plan"

    def test_every_action_replan_strategy(self) -> None:
        """EVERY_ACTION strategy replans after each action."""
        actions = [
            _action("step1", eff={"a": True}),
            _action("step2", pre={"a": True}, eff={"b": True}),
        ]
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(
                    conditions={"b": True},
                    policy=GoalPolicy(replan_strategy=ReplanStrategy.EVERY_ACTION),
                ),
                "world_state": {},
            }
        )

        assert result["status"] == "goal_achieved"
        # With EVERY_ACTION on a 2-step plan: the first plan doesn't count,
        # the second step triggers 1 genuine replan → replan_count >= 1.
        assert result["replan_count"] >= 1

    def test_execution_history_fully_populated(self) -> None:
        """execution_history contains a record for every action executed."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "goal": GoalSpec(conditions={"report_complete": True}),
                "world_state": {},
            }
        )

        history = result["execution_history"]
        assert len(history) >= 4
        action_names = [h.action_name for h in history if h.success]
        assert "gather_data" in action_names
        assert "clean_data" in action_names
        assert "analyze_data" in action_names
        assert "write_report" in action_names

    def test_graph_invoke_convenience_method(self) -> None:
        """GoapGraph.invoke() is a single-shot compile+invoke shorthand."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)

        result = graph.invoke(
            goal=GoalSpec(conditions={"report_complete": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"].get("report_complete") is True

    def test_never_strategy_stops_on_action_failure_end_to_end(self) -> None:
        """NEVER strategy terminates with 'failed' status on action failure."""
        call_count = 0

        def always_fails(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        actions = [
            ActionSpec(name="bad_action", effects={"done": True}, execute=always_fails)
        ]
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={"done": True},
                policy=GoalPolicy(replan_strategy=ReplanStrategy.NEVER),
            ),
        )

        assert result["status"] == "failed"
        # NEVER strategy: action executed exactly once, no retries
        assert call_count == 1

    def test_full_loop_with_rich_world_state(self) -> None:
        """End-to-end: world_state with unhashable values (lists, dicts)."""

        def retrieve_docs(ws: dict[str, Any]) -> dict[str, Any]:
            return {
                "has_documents": True,
                "documents": [{"title": "doc1"}, {"title": "doc2"}],
            }

        def generate_answer(ws: dict[str, Any]) -> dict[str, Any]:
            docs = ws.get("documents", [])
            return {
                "answer_ready": True,
                "answer": f"Based on {len(docs)} documents",
            }

        actions = [
            ActionSpec(
                name="retrieve",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                execute=retrieve_docs,
            ),
            ActionSpec(
                name="generate",
                preconditions={"has_documents": True},
                effects={"answer_ready": True},
                execute=generate_answer,
            ),
        ]
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={
                "has_question": True,
                "question": "What is GOAP?",
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["answer_ready"] is True
        assert result["world_state"]["documents"] == [
            {"title": "doc1"},
            {"title": "doc2"},
        ]
        assert "Based on 2 documents" in result["world_state"]["answer"]

    def test_successful_action_names_utility(self) -> None:
        """successful_action_names() extracts names of successful actions."""
        actions = _report_pipeline_actions()
        graph = GoapGraph(actions=actions)
        result = graph.invoke(
            goal=GoalSpec(conditions={"report_complete": True}),
            world_state={},
        )
        names = successful_action_names(result)
        assert "gather_data" in names
        assert "clean_data" in names
        assert "analyze_data" in names
        assert "write_report" in names

    def test_max_replans_prevents_infinite_loop(self) -> None:
        """max_replans guard terminates a persistently-failing replan cycle."""
        fail_count = 0

        def always_fails(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("permanent failure")

        actions = [
            ActionSpec(name="bad_action", effects={"done": True}, execute=always_fails)
        ]
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"done": True}, policy=GoalPolicy(max_replans=2)),
        )

        assert result["status"] == "failed"
        assert "max_replans_exceeded" in result.get("replan_reason", "")
        # Action attempted: initial + up to max_replans retries
        assert fail_count <= 3
