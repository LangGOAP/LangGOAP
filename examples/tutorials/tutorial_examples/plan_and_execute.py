"""Shared execute functions for the GOAPified Plan-and-Execute tutorial.

Imported by:
- ``tests/integration/test_plan_and_execute.py``
- ``examples/tutorials/plan_and_execute_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec


# ---------------------------------------------------------------------------
# 3-step research pipeline: search → extract → compose
# Scenario: "What is the hometown of the 2024 Australian Open winner?"
# ---------------------------------------------------------------------------


def search_web(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate web search for information."""
    query = ws.get("question", ws.get("task", ""))
    return {
        "has_search_results": True,
        "search_results": [
            {"content": f"Search result for '{query}': Jannik Sinner won the 2024 Australian Open."},
            {"content": "Sinner is from San Candido, South Tyrol, Italy."},
        ],
    }


def extract_facts(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate extracting structured facts from search results."""
    results = ws.get("search_results", [])
    facts = [r["content"] for r in results]
    return {
        "has_extracted_facts": True,
        "facts": facts,
        "winner": "Jannik Sinner",
        "hometown": "San Candido, South Tyrol, Italy",
    }


def compose_response(ws: dict[str, Any]) -> dict[str, Any]:
    """Simulate composing a final response from extracted facts."""
    winner = ws.get("winner", "unknown")
    hometown = ws.get("hometown", "unknown")
    return {
        "response_ready": True,
        "response": f"The hometown of the 2024 Australian Open winner ({winner}) is {hometown}.",
    }


def plan_and_execute_actions() -> list[ActionSpec]:
    """Standard 3-step Plan-and-Execute action set."""
    return [
        ActionSpec(
            name="search_web",
            preconditions={"has_task": True},
            effects={"has_search_results": True},
            cost=1.0,
            execute=search_web,
        ),
        ActionSpec(
            name="extract_facts",
            preconditions={"has_search_results": True},
            effects={"has_extracted_facts": True},
            cost=1.0,
            execute=extract_facts,
        ),
        ActionSpec(
            name="compose_response",
            preconditions={"has_extracted_facts": True},
            effects={"response_ready": True},
            cost=1.0,
            execute=compose_response,
        ),
    ]


# ---------------------------------------------------------------------------
# 5-step research pipeline: search → verify → analyze → draft → finalize
# Demonstrates that A* handles arbitrarily long pipelines.
# ---------------------------------------------------------------------------


def pipeline_search(ws: dict[str, Any]) -> dict[str, Any]:
    """Step 1: raw data collection."""
    return {"has_raw_data": True, "raw_data": ["fact1", "fact2", "fact3"]}


def pipeline_verify(ws: dict[str, Any]) -> dict[str, Any]:
    """Step 2: verification and filtering."""
    data = ws.get("raw_data", [])
    return {"has_verified_data": True, "verified_data": data[:2]}


def pipeline_analyze(ws: dict[str, Any]) -> dict[str, Any]:
    """Step 3: analysis."""
    return {"has_analysis": True, "analysis": "Comprehensive analysis of verified facts."}


def pipeline_draft(ws: dict[str, Any]) -> dict[str, Any]:
    """Step 4: drafting."""
    analysis = ws.get("analysis", "")
    return {"has_draft": True, "draft": f"Draft report: {analysis}"}


def pipeline_finalize(ws: dict[str, Any]) -> dict[str, Any]:
    """Step 5: finalization."""
    draft = ws.get("draft", "")
    return {"report_complete": True, "final_report": f"[FINAL] {draft}"}


def five_step_research_actions() -> list[ActionSpec]:
    """Build the 5-step research pipeline action set."""
    return [
        ActionSpec(
            name="search",
            preconditions={"has_task": True},
            effects={"has_raw_data": True},
            execute=pipeline_search,
        ),
        ActionSpec(
            name="verify",
            preconditions={"has_raw_data": True},
            effects={"has_verified_data": True},
            execute=pipeline_verify,
        ),
        ActionSpec(
            name="analyze",
            preconditions={"has_verified_data": True},
            effects={"has_analysis": True},
            execute=pipeline_analyze,
        ),
        ActionSpec(
            name="draft",
            preconditions={"has_analysis": True},
            effects={"has_draft": True},
            execute=pipeline_draft,
        ),
        ActionSpec(
            name="finalize",
            preconditions={"has_draft": True},
            effects={"report_complete": True},
            execute=pipeline_finalize,
        ),
    ]
