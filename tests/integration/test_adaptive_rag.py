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

from langgoap import ActionSpec, GoalPolicy, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Deterministic stubs for GOAP mechanics tests.
# These mirror the original tutorial_examples functions but are self-contained
# so the tutorial module can evolve to require a real LLM without breaking
# planning-focused tests.
# ---------------------------------------------------------------------------


def _retrieve_documents(ws: dict[str, Any]) -> dict[str, Any]:
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Relevant info about {question}", "source": "vectorstore"},
        {"page_content": "Background context", "source": "vectorstore"},
    ]
    return {"has_documents": True, "documents": docs}


def _web_search(ws: dict[str, Any]) -> dict[str, Any]:
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Web result for: {question}", "source": "web"},
    ]
    return {"has_documents": True, "documents": docs}


def _grade_documents(ws: dict[str, Any]) -> dict[str, Any]:
    docs = ws.get("documents", [])
    return {"has_relevant_documents": True, "relevant_documents": docs}


def _generate_answer(ws: dict[str, Any]) -> dict[str, Any]:
    docs = ws.get("relevant_documents", ws.get("documents", []))
    content = "; ".join(d.get("page_content", "") for d in docs)
    return {
        "answer_ready": True,
        "generation": f"Generated answer from {len(docs)} docs: {content}",
    }


def _adaptive_rag_actions(
    *,
    grade_fn: Any = None,
    retrieve_cost: float = 1.0,
    web_cost: float = 2.0,
) -> list[ActionSpec]:
    if grade_fn is None:
        grade_fn = _grade_documents
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
                execute=_grade_documents,
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
        which re-plans from the current partial state.  On the second attempt
        the grade function succeeds, allowing the pipeline to complete.
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
                policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION),
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
                execute=_grade_documents,
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


# ---------------------------------------------------------------------------
# Real LLM integration tests — run with ``uv run pytest -m api``
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestAdaptiveRagWithLLM:
    """Exercises LLM-powered Adaptive RAG actions end-to-end."""

    @pytest.fixture
    def llm(self) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @staticmethod
    def _build_actions(
        llm: Any, *, validate_hallucinations: bool = True
    ) -> list[ActionSpec]:
        from langchain_core.messages import HumanMessage, SystemMessage

        def retrieve_documents(ws: dict[str, Any]) -> dict[str, Any]:
            question = ws.get("question", "")
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a document retrieval system. Given a question, generate 2 relevant document excerpts. Return each on its own line, prefixed with '- '."
                    ),
                    HumanMessage(content=f"Retrieve documents for: {question}"),
                ]
            )
            lines = [
                l.strip().lstrip("- ").strip()
                for l in response.content.strip().split("\n")
                if l.strip() and l.strip() != "-"
            ]
            return {
                "has_documents": True,
                "documents": [
                    {"page_content": l, "source": "vectorstore"} for l in lines if l
                ],
            }

        def web_search(ws: dict[str, Any]) -> dict[str, Any]:
            question = ws.get("question", "")
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a web search tool. Given a question, provide 1-2 concise, factual web search results. Return each on its own line, prefixed with '- '."
                    ),
                    HumanMessage(content=f"Web search for: {question}"),
                ]
            )
            lines = [
                l.strip().lstrip("- ").strip()
                for l in response.content.strip().split("\n")
                if l.strip() and l.strip() != "-"
            ]
            return {
                "has_documents": True,
                "documents": [{"page_content": l, "source": "web"} for l in lines if l],
            }

        def grade_documents(ws: dict[str, Any]) -> dict[str, Any]:
            docs, question, relevant = (
                ws.get("documents", []),
                ws.get("question", ""),
                [],
            )
            for doc in docs:
                r = llm.invoke(
                    [
                        SystemMessage(
                            content="You are a document relevance grader. Respond with EXACTLY 'yes' or 'no'."
                        ),
                        HumanMessage(
                            content=f"Question: {question}\nDocument: {doc.get('page_content', '')}\nRelevant?"
                        ),
                    ]
                )
                if "yes" in r.content.strip().lower():
                    relevant.append(doc)
            return {
                "has_relevant_documents": len(relevant) > 0,
                "relevant_documents": relevant,
            }

        def generate_answer(ws: dict[str, Any]) -> dict[str, Any]:
            docs = ws.get("relevant_documents", ws.get("documents", []))
            question, context = ws.get("question", ""), "\n\n".join(
                d.get("page_content", str(d)) for d in docs
            )
            r = llm.invoke(
                [
                    SystemMessage(
                        content="You are a RAG assistant. Generate a concise, grounded answer using only the provided documents."
                    ),
                    HumanMessage(
                        content=f"Question: {question}\nDocuments:\n{context}\nGenerate an answer:"
                    ),
                ]
            )
            return {"answer_ready": True, "generation": r.content.strip()}

        validator = None
        if validate_hallucinations:

            def validator(pre: dict[str, Any], post: dict[str, Any]) -> bool:
                gen = post.get("generation", "")
                docs = pre.get("relevant_documents", pre.get("documents", []))
                if not gen or not docs:
                    return False
                ctx = "\n".join(d.get("page_content", str(d)) for d in docs)
                r = llm.invoke(
                    [
                        SystemMessage(
                            content="You are a hallucination grader. Respond 'yes' if grounded, 'no' otherwise."
                        ),
                        HumanMessage(
                            content=f"Documents:\n{ctx}\nAnswer: {gen}\nGrounded?"
                        ),
                    ]
                )
                return "yes" in r.content.strip().lower()

        return [
            ActionSpec(
                name="retrieve_documents",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                cost=1.0,
                execute=retrieve_documents,
            ),
            ActionSpec(
                name="web_search",
                preconditions={"has_question": True},
                effects={"has_documents": True},
                cost=2.0,
                execute=web_search,
            ),
            ActionSpec(
                name="grade_documents",
                preconditions={"has_documents": True},
                effects={"has_relevant_documents": True},
                cost=1.0,
                execute=grade_documents,
            ),
            ActionSpec(
                name="generate_answer",
                preconditions={"has_relevant_documents": True},
                effects={"answer_ready": True},
                cost=1.0,
                execute=generate_answer,
                effect_validator=validator,
            ),
        ]

    def test_full_pipeline_with_llm(self, llm: Any) -> None:
        """LLM-powered retrieve -> grade -> generate reaches goal_achieved."""
        actions = self._build_actions(llm)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={
                "has_question": True,
                "question": "What are the main types of agent memory?",
            },
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["answer_ready"] is True
        assert isinstance(ws["generation"], str)
        assert len(ws["generation"]) > 20

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == [
            "retrieve_documents",
            "grade_documents",
            "generate_answer",
        ]

    def test_hallucination_validator_with_llm(self, llm: Any) -> None:
        """Effect validator runs LLM hallucination check on generated answer."""
        actions = self._build_actions(llm, validate_hallucinations=True)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"answer_ready": True}),
            world_state={
                "has_question": True,
                "question": "What is GOAP planning in game AI?",
            },
        )

        assert result["status"] == "goal_achieved"
        assert isinstance(result["world_state"]["generation"], str)
