"""Shared execute functions for the GOAPified Hierarchical Agent Teams tutorial.

Imported by:
- ``tests/integration/test_hierarchical_teams.py``
- ``examples/tutorials/hierarchical_agent_teams_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec


# ---------------------------------------------------------------------------
# Research Team
# ---------------------------------------------------------------------------


def search_agent(ws: dict[str, Any]) -> dict[str, Any]:
    """Search agent (TavilySearch tool)."""
    topic = ws.get("topic", ws.get("task", "AI agents"))
    return {
        "has_search_results": True,
        "search_results": [
            f"Key finding about {topic}: autonomous agent systems use planning.",
            f"Recent development in {topic}: LLM-based planning is emerging.",
        ],
    }


def web_scraper_agent(ws: dict[str, Any]) -> dict[str, Any]:
    """Web scraper agent (WebBaseLoader tool)."""
    results = ws.get("search_results", [])
    return {
        "has_detailed_content": True,
        "detailed_content": (
            f"Detailed analysis based on {len(results)} search results. "
            "AI agents combine planning, memory, and tool use for autonomous task completion."
        ),
    }


# ---------------------------------------------------------------------------
# Writing Team
# ---------------------------------------------------------------------------


def note_taker_agent(ws: dict[str, Any]) -> dict[str, Any]:
    """Note taker agent (create_outline tool)."""
    return {
        "has_outline": True,
        "outline": [
            "1. Introduction to AI Agents",
            "2. Planning and Memory Systems",
            "3. Tool Use and Integration",
            "4. Current Challenges",
            "5. Future Directions",
        ],
    }


def doc_writer_agent(ws: dict[str, Any]) -> dict[str, Any]:
    """Doc writer agent (write_document tool)."""
    outline = ws.get("outline", [])
    return {
        "has_document": True,
        "document": (
            f"Report based on {len(outline)}-point outline.\n"
            "AI agents are systems that combine LLM capabilities with planning, "
            "memory, and tool use to autonomously complete complex tasks."
        ),
    }


def chart_generator_agent(ws: dict[str, Any]) -> dict[str, Any]:
    """Chart generator agent (Python REPL tool)."""
    return {
        "has_visualization": True,
        "visualization": "chart_ai_agents_timeline.png",
    }


# ---------------------------------------------------------------------------
# Action factory
# ---------------------------------------------------------------------------


def full_team_actions() -> list[ActionSpec]:
    """All agents from both research and writing teams."""
    return [
        # Research team
        ActionSpec(
            name="search_agent",
            preconditions={"has_topic": True},
            effects={"has_search_results": True},
            cost=1.0,
            execute=search_agent,
        ),
        ActionSpec(
            name="web_scraper_agent",
            preconditions={"has_search_results": True},
            effects={"has_detailed_content": True},
            cost=2.0,
            execute=web_scraper_agent,
        ),
        # Writing team
        ActionSpec(
            name="note_taker_agent",
            preconditions={"has_detailed_content": True},
            effects={"has_outline": True},
            cost=1.0,
            execute=note_taker_agent,
        ),
        ActionSpec(
            name="doc_writer_agent",
            preconditions={"has_outline": True},
            effects={"has_document": True},
            cost=2.0,
            execute=doc_writer_agent,
        ),
        ActionSpec(
            name="chart_generator_agent",
            preconditions={"has_detailed_content": True},
            effects={"has_visualization": True},
            cost=1.5,
            execute=chart_generator_agent,
        ),
    ]
