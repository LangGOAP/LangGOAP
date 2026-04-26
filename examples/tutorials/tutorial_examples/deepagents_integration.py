"""LLM-powered content-analysis actions for the DeepAgents integration tutorial.

Four actions form a sequential pipeline:

1. ``extract_topics``       — LLM extracts key topics from a document.
2. ``analyze_sentiment``    — LLM analyzes sentiment and tone.
3. ``generate_summary``     — LLM produces an executive summary.
4. ``write_recommendations``— LLM writes actionable recommendations.

All four call a real ``BaseChatModel`` — GOAP replaces the orchestration,
not the intelligence inside the actions.

Imported by:

- ``examples/tutorials/deepagents_integration.ipynb``
- ``tests/integration/test_deepagents_integration.py`` (via FakeStructuredModel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec, GoalSpec

from .data.deepagents_instance import SAMPLE_DOCUMENT


@dataclass
class ContentWorkspace:
    """Mutable workspace threaded through the GOAP world state."""

    document: str = ""
    topics: list[str] = field(default_factory=list)
    sentiment: str = ""
    summary: str = ""
    recommendations: str = ""


# ---------------------------------------------------------------------------
# Action factories
# ---------------------------------------------------------------------------


def _make_extract_topics(llm: BaseChatModel) -> Any:
    """LLM extracts key topics from the document."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        doc = ws.get("document", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract the 3-5 most important topics from the "
                        "document below. Return each topic on its own line, "
                        "prefixed with '- '. Be specific and concise."
                    )
                ),
                HumanMessage(content=doc),
            ]
        )
        topics = [
            line.strip().lstrip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]
        return {"has_topics": True, "topics": topics}

    return execute


def _make_analyze_sentiment(llm: BaseChatModel) -> Any:
    """LLM analyzes sentiment and tone of the document."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        doc = ws.get("document", "")
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Analyze the overall sentiment and tone of the "
                        "document below. Describe the sentiment in 2-3 "
                        "sentences. Mention whether the tone is optimistic, "
                        "cautious, neutral, critical, etc."
                    )
                ),
                HumanMessage(content=doc),
            ]
        )
        return {"has_sentiment": True, "sentiment": response.content.strip()}

    return execute


def _make_generate_summary(llm: BaseChatModel) -> Any:
    """LLM generates an executive summary combining topics and sentiment."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        doc = ws.get("document", "")
        topics = ws.get("topics", [])
        sentiment = ws.get("sentiment", "")
        topics_text = ", ".join(topics) if topics else "not yet extracted"

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a concise executive summary (3-4 sentences) "
                        "of the document. Incorporate the identified topics "
                        "and sentiment analysis. Be factual and professional."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Document:\n{doc}\n\n"
                        f"Key topics: {topics_text}\n"
                        f"Sentiment: {sentiment}\n\n"
                        "Write the executive summary:"
                    )
                ),
            ]
        )
        return {"has_summary": True, "summary": response.content.strip()}

    return execute


def _make_write_recommendations(llm: BaseChatModel) -> Any:
    """LLM writes actionable recommendations based on the analysis."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        topics = ws.get("topics", [])
        sentiment = ws.get("sentiment", "")
        summary = ws.get("summary", "")
        topics_text = ", ".join(topics) if topics else "general"

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Based on the analysis below, write 3-4 actionable "
                        "recommendations. Each should be on its own line, "
                        "prefixed with a number and period (e.g. '1. ...'). "
                        "Be specific and practical."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topics: {topics_text}\n"
                        f"Sentiment: {sentiment}\n"
                        f"Summary: {summary}\n\n"
                        "Write actionable recommendations:"
                    )
                ),
            ]
        )
        return {
            "has_recommendations": True,
            "recommendations": response.content.strip(),
        }

    return execute


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def content_analysis_actions(llm: BaseChatModel) -> list[ActionSpec]:
    """Build the content-analysis action set.

    Args:
        llm: Language model for extraction, analysis, and generation.

    Returns:
        Four actions forming a sequential pipeline.
    """
    return [
        ActionSpec(
            name="extract_topics",
            preconditions={"has_document": True},
            effects={"has_topics": True},
            execute=_make_extract_topics(llm),
        ),
        ActionSpec(
            name="analyze_sentiment",
            preconditions={"has_document": True},
            effects={"has_sentiment": True},
            execute=_make_analyze_sentiment(llm),
        ),
        ActionSpec(
            name="generate_summary",
            preconditions={"has_topics": True, "has_sentiment": True},
            effects={"has_summary": True},
            execute=_make_generate_summary(llm),
        ),
        ActionSpec(
            name="write_recommendations",
            preconditions={"has_summary": True},
            effects={"has_recommendations": True},
            execute=_make_write_recommendations(llm),
        ),
    ]


def content_analysis_goal() -> GoalSpec:
    """Return the goal for a complete content analysis."""
    return GoalSpec(conditions={"has_recommendations": True})
