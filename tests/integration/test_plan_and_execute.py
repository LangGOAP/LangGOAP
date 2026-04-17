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

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Deterministic stubs for GOAP mechanics tests.
# These mirror the original tutorial_examples functions but are self-contained
# so the tutorial module can evolve to require a real LLM without breaking
# planning-focused tests.
# ---------------------------------------------------------------------------


def _search_web(ws: dict[str, Any]) -> dict[str, Any]:
    query = ws.get("question", ws.get("task", ""))
    return {
        "has_search_results": True,
        "search_results": [
            {
                "content": f"Search result for '{query}': Jannik Sinner won the 2024 Australian Open."
            },
            {"content": "Sinner is from San Candido, South Tyrol, Italy."},
        ],
    }


def _extract_facts(ws: dict[str, Any]) -> dict[str, Any]:
    results = ws.get("search_results", [])
    facts = [r["content"] for r in results]
    return {
        "has_extracted_facts": True,
        "facts": facts,
        "winner": "Jannik Sinner",
        "hometown": "San Candido, South Tyrol, Italy",
    }


def _compose_response(ws: dict[str, Any]) -> dict[str, Any]:
    winner = ws.get("winner", "unknown")
    hometown = ws.get("hometown", "unknown")
    return {
        "response_ready": True,
        "response": f"The hometown of the 2024 Australian Open winner ({winner}) is {hometown}.",
    }


def _plan_and_execute_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="search_web",
            preconditions={"has_task": True},
            effects={"has_search_results": True},
            cost=1.0,
            execute=_search_web,
        ),
        ActionSpec(
            name="extract_facts",
            preconditions={"has_search_results": True},
            effects={"has_extracted_facts": True},
            cost=1.0,
            execute=_extract_facts,
        ),
        ActionSpec(
            name="compose_response",
            preconditions={"has_extracted_facts": True},
            effects={"response_ready": True},
            cost=1.0,
            execute=_compose_response,
        ),
    ]


def _pipeline_search(ws: dict[str, Any]) -> dict[str, Any]:
    return {"has_raw_data": True, "raw_data": ["fact1", "fact2", "fact3"]}


def _pipeline_verify(ws: dict[str, Any]) -> dict[str, Any]:
    data = ws.get("raw_data", [])
    return {"has_verified_data": True, "verified_data": data[:2]}


def _pipeline_analyze(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_analysis": True,
        "analysis": "Comprehensive analysis of verified facts.",
    }


def _pipeline_draft(ws: dict[str, Any]) -> dict[str, Any]:
    analysis = ws.get("analysis", "")
    return {"has_draft": True, "draft": f"Draft report: {analysis}"}


def _pipeline_finalize(ws: dict[str, Any]) -> dict[str, Any]:
    draft = ws.get("draft", "")
    return {"report_complete": True, "final_report": f"[FINAL] {draft}"}


def _five_step_research_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="search",
            preconditions={"has_task": True},
            effects={"has_raw_data": True},
            execute=_pipeline_search,
        ),
        ActionSpec(
            name="verify",
            preconditions={"has_raw_data": True},
            effects={"has_verified_data": True},
            execute=_pipeline_verify,
        ),
        ActionSpec(
            name="analyze",
            preconditions={"has_verified_data": True},
            effects={"has_analysis": True},
            execute=_pipeline_analyze,
        ),
        ActionSpec(
            name="draft",
            preconditions={"has_analysis": True},
            effects={"has_draft": True},
            execute=_pipeline_draft,
        ),
        ActionSpec(
            name="finalize",
            preconditions={"has_draft": True},
            effects={"report_complete": True},
            execute=_pipeline_finalize,
        ),
    ]


class TestPlanAndExecuteGoapified:
    """GOAPified Plan-and-Execute: A* replaces LLM text plans."""

    def test_formal_plan_discovery(self) -> None:
        """A* planner discovers search → extract → compose sequence."""
        actions = _plan_and_execute_actions()
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
        actions = _plan_and_execute_actions()
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
            return _search_web(ws)

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
                execute=_extract_facts,
            ),
            ActionSpec(
                name="compose_response",
                preconditions={"has_extracted_facts": True},
                effects={"response_ready": True},
                execute=_compose_response,
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
                execute=_extract_facts,
            ),
            ActionSpec(
                name="compose_response",
                preconditions={"has_extracted_facts": True},
                effects={"response_ready": True},
                execute=_compose_response,
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
        actions = _five_step_research_actions()

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
        actions = _plan_and_execute_actions()
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


# ---------------------------------------------------------------------------
# Real LLM integration tests — run with ``uv run pytest -m api``
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestPlanAndExecuteWithLLM:
    """Exercises the LLM-powered tutorial_examples factories end-to-end."""

    @pytest.fixture
    def llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def test_three_step_pipeline_with_llm(self, llm: Any) -> None:
        """LLM-powered search → extract → compose reaches goal_achieved."""
        from tutorial_examples.plan_and_execute import plan_and_execute_actions

        actions = plan_and_execute_actions(llm)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"response_ready": True}),
            world_state={
                "has_task": True,
                "question": "What is the hometown of the 2024 Australian Open winner?",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["response_ready"] is True
        assert isinstance(ws["response"], str)
        assert len(ws["response"]) > 10

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["search_web", "extract_facts", "compose_response"]

    def test_five_step_pipeline_with_llm(self, llm: Any) -> None:
        """LLM-powered 5-step pipeline reaches goal_achieved with [FINAL]."""
        from tutorial_examples.plan_and_execute import five_step_research_actions

        actions = five_step_research_actions(llm)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"report_complete": True}),
            world_state={
                "has_task": True,
                "task": "Goal-oriented action planning for AI agents",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["report_complete"] is True
        assert "[FINAL]" in ws["final_report"]

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["search", "verify", "analyze", "draft", "finalize"]
