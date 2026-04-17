r"""LLM-powered execute functions for the GOAPified Hierarchical Agent Teams tutorial.

The original LangGraph Hierarchical Agent Teams uses LLM-based supervisors
to route between team members (research team, writing team).  In the
GOAPified version:

- A\* replaces the 3 supervisor LLMs (top-level, research, writing).
- Agent capabilities become ``ActionSpec`` preconditions and effects.
- The planner automatically selects and orders agents.
- Each worker agent retains **real LLM intelligence** for execution.

Research team:

- ``search_agent`` — uses a search tool or LLM to find information.
- ``web_scraper_agent`` — LLM summarizes and analyzes source material.

Writing team:

- ``note_taker_agent`` — LLM creates a structured outline.
- ``doc_writer_agent`` — LLM writes a full report from the outline.
- ``chart_generator_agent`` — LLM generates a chart description.

All factories require a ``BaseChatModel``.

Imported by:

- ``examples/tutorials/hierarchical_agent_teams_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec

_Execute = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Research Team
# ---------------------------------------------------------------------------


def _make_search_agent(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """Search agent — finds information about a topic."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        topic = ws.get("topic", ws.get("task", "AI agents"))

        if search_tool is not None:
            raw = search_tool.invoke(topic)
            items = raw if isinstance(raw, list) else [raw]
            results = [
                r.get("content", r.get("snippet", str(r))) for r in items
            ]
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a research agent with web search "
                            "capabilities. Given a topic, provide 2-3 "
                            "concise, factual search results that cover the "
                            "key aspects. Return each result on its own line."
                        )
                    ),
                    HumanMessage(
                        content=f"Search for information about: {topic}"
                    ),
                ]
            )
            results = [
                line.strip().lstrip("- ").lstrip("0123456789.").strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
        return {"has_search_results": True, "search_results": results}

    return execute


def _make_web_scraper_agent(llm: BaseChatModel) -> _Execute:
    """Web scraper agent — summarizes and deepens search results."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        results = ws.get("search_results", [])
        topic = ws.get("topic", ws.get("task", ""))
        results_text = "\n".join(
            f"- {r}" if isinstance(r, str) else f"- {r}"
            for r in results
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a web scraping and content analysis agent. "
                        "Given search results about a topic, synthesize the "
                        "information into a detailed analysis paragraph. "
                        "Include specific details and insights."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Search results:\n{results_text}\n\n"
                        "Provide a detailed analysis:"
                    )
                ),
            ]
        )
        return {
            "has_detailed_content": True,
            "detailed_content": response.content.strip(),
        }

    return execute


# ---------------------------------------------------------------------------
# Writing Team
# ---------------------------------------------------------------------------


def _make_note_taker_agent(llm: BaseChatModel) -> _Execute:
    """Note taker agent — creates a structured outline from content."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        content = ws.get("detailed_content", "")
        topic = ws.get("topic", ws.get("task", ""))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a note-taking agent. Given detailed content "
                        "about a topic, create a structured outline with "
                        "exactly 5 numbered points. Each point should be a "
                        "concise section heading. Return ONLY the 5 points, "
                        "one per line, formatted as '1. Title'."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Content:\n{content}\n\n"
                        "Create a 5-point outline:"
                    )
                ),
            ]
        )
        lines = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]
        return {"has_outline": True, "outline": lines}

    return execute


def _make_doc_writer_agent(llm: BaseChatModel) -> _Execute:
    """Doc writer agent — writes a full report from an outline."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        outline = ws.get("outline", [])
        content = ws.get("detailed_content", "")
        topic = ws.get("topic", ws.get("task", ""))
        outline_text = "\n".join(outline)

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a document writing agent. Given an outline "
                        "and source content about a topic, write a concise "
                        "report that covers all outline points. The report "
                        "should be well-structured and informative."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Outline:\n{outline_text}\n\n"
                        f"Source content:\n{content}\n\n"
                        "Write the report:"
                    )
                ),
            ]
        )
        return {"has_document": True, "document": response.content.strip()}

    return execute


def _make_chart_generator_agent(llm: BaseChatModel) -> _Execute:
    """Chart generator agent — generates a chart description from content."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        content = ws.get("detailed_content", "")
        topic = ws.get("topic", ws.get("task", ""))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data visualization agent. Given content "
                        "about a topic, describe a chart or visualization "
                        "that would effectively communicate the key data "
                        "points. Include chart type, axes, and data points."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Content:\n{content}\n\n"
                        "Describe an appropriate visualization:"
                    )
                ),
            ]
        )
        return {
            "has_visualization": True,
            "visualization": response.content.strip(),
        }

    return execute


# ---------------------------------------------------------------------------
# Action factory
# ---------------------------------------------------------------------------


def full_team_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """All agents from both research and writing teams.

    Args:
        llm: Language model powering every worker agent.
        search_tool: Optional LangChain tool for the search agent.
    """
    return [
        # Research team
        ActionSpec(
            name="search_agent",
            preconditions={"has_topic": True},
            effects={"has_search_results": True},
            cost=1.0,
            execute=_make_search_agent(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="web_scraper_agent",
            preconditions={"has_search_results": True},
            effects={"has_detailed_content": True},
            cost=2.0,
            execute=_make_web_scraper_agent(llm),
        ),
        # Writing team
        ActionSpec(
            name="note_taker_agent",
            preconditions={"has_detailed_content": True},
            effects={"has_outline": True},
            cost=1.0,
            execute=_make_note_taker_agent(llm),
        ),
        ActionSpec(
            name="doc_writer_agent",
            preconditions={"has_outline": True},
            effects={"has_document": True},
            cost=2.0,
            execute=_make_doc_writer_agent(llm),
        ),
        ActionSpec(
            name="chart_generator_agent",
            preconditions={"has_detailed_content": True},
            effects={"has_visualization": True},
            cost=1.5,
            execute=_make_chart_generator_agent(llm),
        ),
    ]
