"""Integration tests for GOAPified Plan-and-Execute.

The original LangGraph Plan-and-Execute uses an LLM to generate a text plan
(list of strings), executes each step with a ReAct agent, and optionally
replans. In the GOAPified version, GOAP's A* planner replaces the LLM planner
— producing a formal, verifiable plan with typed preconditions and effects
instead of freeform LLM-generated text.

Key advantages over the original:
- Formal planning: plans are verified by A* to be achievable
- No hallucinated steps: every action has real preconditions and effects
- Automatic replanning via observer (not LLM re-prompting)
- Cost optimization: A* minimizes total action cost

Reference: research/repos/langgraph/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
"""

from __future__ import annotations

from typing import Any

import pytest
from tutorial_examples.plan_and_execute import (
    compose_response,
    extract_facts,
    five_step_research_actions,
    plan_and_execute_actions,
    search_web,
)

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy


class TestPlanAndExecuteGoapified:
    """GOAPified Plan-and-Execute: A* replaces LLM text plans."""

    def test_formal_plan_discovery(self) -> None:
        """A* planner discovers search → extract → compose sequence."""
        actions = plan_and_execute_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"response_ready": True}),
            world_state={
                "has_task": True,
                "task": "What is the hometown of the 2024 Australian Open winner?",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["response_ready"] is True
        assert "San Candido" in ws["response"]
        assert "Jannik Sinner" in ws["response"]

        # Verify correct action sequence
        history = result["execution_history"]
        successful = [h.action_name for h in history if h.success]
        assert successful == ["search_web", "extract_facts", "compose_response"]

    def test_execution_history_fully_populated(self) -> None:
        """Every step in the plan produces an execution history entry."""
        actions = plan_and_execute_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"response_ready": True}),
            world_state={"has_task": True},
        )

        history = result["execution_history"]
        assert len(history) == 3
        for entry in history:
            assert entry.success is True
            assert entry.action_name in (
                "search_web",
                "extract_facts",
                "compose_response",
            )
            assert entry.state_before is not None
            assert entry.state_after is not None

    def test_replanning_on_step_failure(self) -> None:
        """When a step fails, observer triggers replanning."""
        call_count = {"search": 0}

        def flaky_search(ws: dict[str, Any]) -> dict[str, Any]:
            call_count["search"] += 1
            if call_count["search"] == 1:
                raise RuntimeError("Search API timeout")
            return search_web(ws)

        actions = [
            ActionSpec(
                name="search_web",
                preconditions={"has_task": True},
                effects={"has_search_results": True},
                cost=1.0,
                execute=flaky_search,
            ),
            ActionSpec(
                name="extract_facts",
                preconditions={"has_search_results": True},
                effects={"has_extracted_facts": True},
                execute=extract_facts,
            ),
            ActionSpec(
                name="compose_response",
                preconditions={"has_extracted_facts": True},
                effects={"response_ready": True},
                execute=compose_response,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"response_ready": True}),
            world_state={"has_task": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert call_count["search"] >= 2
        # First attempt failed, second succeeded
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1

    def test_never_replan_stops_on_failure(self) -> None:
        """NEVER strategy terminates immediately on step failure."""

        def always_fails(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("permanent failure")

        actions = [
            ActionSpec(
                name="search_web",
                preconditions={"has_task": True},
                effects={"has_search_results": True},
                execute=always_fails,
            ),
            ActionSpec(
                name="extract_facts",
                preconditions={"has_search_results": True},
                effects={"has_extracted_facts": True},
                execute=extract_facts,
            ),
            ActionSpec(
                name="compose_response",
                preconditions={"has_extracted_facts": True},
                effects={"response_ready": True},
                execute=compose_response,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={"response_ready": True},
                replan_strategy=ReplanStrategy.NEVER,
            ),
            world_state={"has_task": True},
        )

        assert result["status"] == "failed"
        assert result["replan_count"] == 0

    def test_multi_step_research_pipeline(self) -> None:
        """Longer pipeline: search → verify → analyze → draft → finalize."""
        actions = five_step_research_actions()

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"report_complete": True}),
            world_state={"has_task": True, "task": "Research AI planning"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["report_complete"] is True
        assert "[FINAL]" in ws["final_report"]

        # Verify all 5 steps executed in order
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["search", "verify", "analyze", "draft", "finalize"]

    def test_rich_state_flows_through_steps(self) -> None:
        """Rich execution data (lists, nested dicts) flows between actions."""
        actions = plan_and_execute_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"response_ready": True}),
            world_state={
                "has_task": True,
                "question": "Australian Open winner hometown",
                "metadata": {"session_id": "abc123"},
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        # Rich data preserved
        assert isinstance(ws["search_results"], list)
        assert isinstance(ws["facts"], list)
        assert ws["metadata"] == {"session_id": "abc123"}
