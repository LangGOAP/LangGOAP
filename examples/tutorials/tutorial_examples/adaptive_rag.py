"""Shared execute functions for the GOAPified Adaptive RAG tutorial.

Imported by:
- ``tests/integration/test_adaptive_rag.py``
- ``examples/tutorials/adaptive_rag_goapified.ipynb``

Note: The original Adaptive RAG also includes a ``transform_query`` step that
rewrites the question after failed document grading.  Full query-transform
routing requires action blacklisting (so the planner stops re-selecting the
same failing retrieval path), which is planned for a future LangGoap release.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec


def retrieve_documents(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate vector store retrieval."""
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Relevant info about {question}", "source": "vectorstore"},
        {"page_content": "Background context", "source": "vectorstore"},
    ]
    return {"has_documents": True, "documents": docs}


def web_search(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate web search via a Tavily-like tool."""
    question = ws.get("question", "")
    docs = [
        {"page_content": f"Web result for: {question}", "source": "web"},
    ]
    return {"has_documents": True, "documents": docs}


def grade_documents(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate grading: all documents pass relevance check (happy path)."""
    docs = ws.get("documents", [])
    return {"has_relevant_documents": True, "relevant_documents": docs}


def grade_documents_none_relevant(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate grading: no documents pass (triggers deviation / replan)."""
    return {"has_relevant_documents": False, "relevant_documents": []}


def generate_answer(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate RAG generation that produces a grounded, useful answer."""
    docs = ws.get("relevant_documents", ws.get("documents", []))
    content = "; ".join(d.get("page_content", "") for d in docs)
    return {
        "answer_ready": True,
        "generation": f"Generated answer from {len(docs)} docs: {content}",
    }


def adaptive_rag_actions(
    *,
    grade_fn: Any = grade_documents,
    retrieve_cost: float = 1.0,
    web_cost: float = 2.0,
) -> list[ActionSpec]:
    """Build the standard Adaptive RAG GOAP action set.

    Args:
        grade_fn: Execute callable for the grade_documents action.  Swap in
            ``grade_documents_none_relevant`` to simulate grading failure.
        retrieve_cost: Cost of vectorstore retrieval (prefer lower values).
        web_cost: Cost of web search (use higher value to make it a fallback).
    """
    return [
        ActionSpec(
            name="retrieve_documents",
            preconditions={"has_question": True},
            effects={"has_documents": True},
            cost=retrieve_cost,
            execute=retrieve_documents,
        ),
        ActionSpec(
            name="web_search",
            preconditions={"has_question": True},
            effects={"has_documents": True},
            cost=web_cost,
            execute=web_search,
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
            execute=generate_answer,
        ),
    ]
