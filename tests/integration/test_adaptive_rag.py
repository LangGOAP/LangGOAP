"""Integration tests for GOAPified Adaptive RAG.

The original LangGraph Adaptive RAG uses hardcoded conditional edges
(route_question, decide_to_generate, grade_generation_v_documents_and_question)
to route between nodes. In the GOAPified version, these routing decisions are
replaced by GOAP A* planning — the planner discovers the optimal action sequence
automatically, and the observer triggers replanning when execution deviates.

Reference: research/repos/langgraph/examples/rag/langgraph_adaptive_rag.ipynb
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Mock execute functions (deterministic, no LLM calls)
# ---------------------------------------------------------------------------


def _retrieve_documents(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate vector store retrieval."""
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Relevant info about {question}", "source": "vectorstore"},
        {"page_content": "Background context", "source": "vectorstore"},
    ]
    return {"has_documents": True, "documents": docs}


def _web_search(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate web search via Tavily-like tool."""
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Web result for: {question}", "source": "web"},
    ]
    return {"has_documents": True, "documents": docs}


def _grade_documents_all_relevant(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate grading: all documents pass relevance check."""
    docs = ws.get("documents", [])
    return {"has_relevant_documents": True, "relevant_documents": docs}


def _grade_documents_none_relevant(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate grading: no documents pass relevance check (triggers replan)."""
    return {"has_relevant_documents": False, "relevant_documents": []}


def _generate_answer(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate RAG generation that produces a grounded, useful answer."""
    docs = ws.get("relevant_documents", ws.get("documents", []))
    content = "; ".join(d.get("page_content", "") for d in docs)
    return {
        "answer_ready": True,
        "generation": f"Generated answer from {len(docs)} docs: {content}",
    }


def _transform_query(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate question rewriting for better retrieval."""
    question = ws.get("question", "")
    return {
        "has_question": True,
        "question": f"[rewritten] {question}",
        "query_transformed": True,
    }


# ---------------------------------------------------------------------------
# Action factories
# ---------------------------------------------------------------------------


def _adaptive_rag_actions(
    *,
    grade_fn: Any = _grade_documents_all_relevant,
    retrieve_cost: float = 1.0,
    web_cost: float = 2.0,
) -> list[ActionSpec]:
    """Build the standard set of Adaptive RAG GOAP actions."""
    return [
        ActionSpec(
            name="retrieve_documents",
            preconditions={"has_question": True},
            effects={"has_documents": True},
            cost=retrieve_cost,
            execute=_retrieve_documents,
        ),
        ActionSpec(
            name="web_search",
            preconditions={"has_question": True},
            effects={"has_documents": True},
            cost=web_cost,
            execute=_web_search,
        ),
        ActionSpec(
            name="grade_documents",
            preconditions={"has_documents": True},
            effects={"has_relevant_documents": True},
            cost=1.0,
            execute=grade_fn,
        ),
        ActionSpec(
            name="generate_answer",
            preconditions={"has_relevant_documents": True},
            effects={"answer_ready": True},
            cost=1.0,
            execute=_generate_answer,
        ),
        ActionSpec(
            name="transform_query",
            preconditions={"has_question": True},
            effects={"query_transformed": True},
            cost=1.5,
            execute=_transform_query,
        ),
    ]


class TestAdaptiveRagGoapified:
    """GOAPified Adaptive RAG: GOAP planner replaces hardcoded routing."""

    def test_happy_path_vectorstore_retrieval(self) -> None:
        """Planner discovers retrieve → grade → generate path (lowest cost)."""
        actions = _adaptive_rag_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={
                "has_question": True,
                "question": "What are agent memory types?",
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["answer_ready"] is True
        assert "Generated answer" in result["world_state"]["generation"]

        # Verify the plan chose vectorstore retrieval (cost=1) over web search (cost=2)
        history = result["execution_history"]
        action_names = [h.action_name for h in history if h.success]
        assert "retrieve_documents" in action_names
        assert "grade_documents" in action_names
        assert "generate_answer" in action_names

    def test_web_search_when_vectorstore_unavailable(self) -> None:
        """When only web_search is available, planner routes through it."""
        actions = [
            ActionSpec(
                name="web_search",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                execute=_web_search,
            ),
            ActionSpec(
                name="grade_documents",
                preconditions={"has_documents": True},
                effects={"has_relevant_documents": True},
                execute=_grade_documents_all_relevant,
            ),
            ActionSpec(
                name="generate_answer",
                preconditions={"has_relevant_documents": True},
                effects={"answer_ready": True},
                execute=_generate_answer,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={"has_question": True, "question": "2024 NFL draft picks"},
        )

        assert result["status"] == "goal_achieved"
        history = result["execution_history"]
        action_names = [h.action_name for h in history if h.success]
        assert "web_search" in action_names
        assert "retrieve_documents" not in action_names
        assert "Web result" in result["world_state"]["generation"]

    def test_planner_prefers_lower_cost_retrieval(self) -> None:
        """When both paths are available, planner picks vectorstore (lower cost)."""
        actions = _adaptive_rag_actions(retrieve_cost=1.0, web_cost=5.0)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={"has_question": True, "question": "agent architectures"},
        )

        assert result["status"] == "goal_achieved"
        history = result["execution_history"]
        successful = [h.action_name for h in history if h.success]
        assert successful[0] == "retrieve_documents"

    def test_replanning_on_grade_failure(self) -> None:
        """When grading finds no relevant docs, observer triggers replan.

        The grade_documents action sets has_relevant_documents=False,
        deviating from the planner's expected has_relevant_documents=True.
        The observer detects this deviation and routes back to the planner,
        which finds an alternative path.
        """
        call_count = {"grade": 0}

        def grade_then_succeed(ws: dict[str, Any]) -> dict[str, Any]:
            call_count["grade"] += 1
            if call_count["grade"] == 1:
                # First call: no relevant docs (causes deviation)
                return {"has_relevant_documents": False, "relevant_documents": []}
            # Subsequent calls: docs are relevant (after query transform + re-retrieve)
            docs = ws.get("documents", [])
            return {"has_relevant_documents": True, "relevant_documents": docs}

        actions = _adaptive_rag_actions(grade_fn=grade_then_succeed)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={"answer_ready": True},
                replan_strategy=ReplanStrategy.ON_DEVIATION,
            ),
            world_state={"has_question": True, "question": "obscure topic"},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert call_count["grade"] >= 2

    def test_action_failure_triggers_replan(self) -> None:
        """When an action raises an exception, observer triggers replan."""
        call_count = {"retrieve": 0}

        def flaky_retrieve(ws: dict[str, Any]) -> dict[str, Any]:
            call_count["retrieve"] += 1
            if call_count["retrieve"] == 1:
                raise ConnectionError("Vector store unavailable")
            return _retrieve_documents(ws)

        actions = [
            ActionSpec(
                name="retrieve_documents",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                cost=1.0,
                execute=flaky_retrieve,
            ),
            ActionSpec(
                name="web_search",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                cost=2.0,
                execute=_web_search,
            ),
            ActionSpec(
                name="grade_documents",
                preconditions={"has_documents": True},
                effects={"has_relevant_documents": True},
                execute=_grade_documents_all_relevant,
            ),
            ActionSpec(
                name="generate_answer",
                preconditions={"has_relevant_documents": True},
                effects={"answer_ready": True},
                execute=_generate_answer,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={"has_question": True, "question": "agent memory"},
        )

        assert result["status"] == "goal_achieved"
        # Should have at least one failure in history
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1
        assert "Vector store unavailable" in failures[0].error

    def test_rich_world_state_preserved_through_pipeline(self) -> None:
        """Documents (lists of dicts) flow through the pipeline correctly.

        This tests the two-tier state separation: planning uses boolean flags
        while execution context carries rich data (unhashable document lists).
        """
        actions = _adaptive_rag_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={
                "has_question": True,
                "question": "What is GOAP planning?",
                "user_id": "test-user-123",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        # Rich data preserved through pipeline
        assert isinstance(ws["documents"], list)
        assert len(ws["documents"]) > 0
        assert isinstance(ws["relevant_documents"], list)
        # Non-planning metadata preserved
        assert ws["user_id"] == "test-user-123"

    def test_no_plan_when_goal_unreachable(self) -> None:
        """When no action chain can reach the goal, planner reports no_plan."""
        actions = [
            ActionSpec(
                name="retrieve_documents",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                execute=_retrieve_documents,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={"has_question": True},
        )

        assert result["status"] == "no_plan"
