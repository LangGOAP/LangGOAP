r"""LLM-powered execute functions for the GOAPified Plan-and-Execute tutorial.

The original LangGraph Plan-and-Execute uses an LLM to generate text plans
(list of strings), executes each step with a ReAct agent, and optionally
replans.  In the GOAPified version:

- A\* planner replaces the LLM planner (formal, verifiable plans).
- The observer replaces the LLM replanner (automatic on deviation).
- Each action retains **real LLM intelligence** for execution.

Two pipelines are provided:

1. **3-step research**: ``search_web`` |rarr| ``extract_facts`` |rarr|
   ``compose_response`` — the canonical Plan-and-Execute workflow.
2. **5-step deep research**: ``search`` |rarr| ``verify`` |rarr|
   ``analyze`` |rarr| ``draft`` |rarr| ``finalize`` — demonstrates that
   A\* handles arbitrarily long LLM-powered pipelines.

Both factories require a ``BaseChatModel`` so the actions use real LLM
calls for extraction, analysis, and composition.

Imported by:

- ``examples/tutorials/plan_and_execute_goapified.ipynb``
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgoap import ActionSpec

# Type alias for ActionSpec execute callables.
_Execute = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# 3-step research pipeline: search -> extract -> compose
# ---------------------------------------------------------------------------


def _make_search_web(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """Create a ``search_web`` execute function.

    When *search_tool* is provided (e.g. ``TavilySearchResults``), the
    action invokes it directly.  Otherwise the LLM generates research
    notes in lieu of a real web search — useful when no API key is
    available but the tutorial still needs to run end-to-end.
    """

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        query = ws.get("question", ws.get("task", ""))

        if search_tool is not None:
            raw = search_tool.invoke(query)
            items = raw if isinstance(raw, list) else [raw]
            results = [
                {"content": r.get("content", r.get("snippet", str(r)))}
                for r in items
            ]
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a research assistant. Given a question, "
                            "provide 2-3 concise, factual search results that "
                            "would help answer it. Return each result on its "
                            "own line, prefixed with '- '."
                        )
                    ),
                    HumanMessage(
                        content=f"Find information to answer: {query}"
                    ),
                ]
            )
            lines = [
                line.strip().lstrip("- ").strip()
                for line in response.content.strip().split("\n")
                if line.strip() and line.strip() != "-"
            ]
            results = [{"content": line} for line in lines if line]

        return {"has_search_results": True, "search_results": results}

    return execute


def _make_extract_facts(llm: BaseChatModel) -> _Execute:
    """Create an ``extract_facts`` execute function powered by *llm*."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        results = ws.get("search_results", [])
        results_text = "\n".join(
            r.get("content", str(r)) for r in results
        )
        question = ws.get("question", ws.get("task", ""))

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a fact extraction assistant. Given search "
                        "results and a question, extract the key facts needed "
                        "to answer the question. Return each fact on its own "
                        "line, prefixed with '- '. Be concise and factual."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Search results:\n{results_text}\n\n"
                        "Extract the key facts:"
                    )
                ),
            ]
        )
        text = response.content.strip()
        facts = [
            line.strip().lstrip("- ").strip()
            for line in text.split("\n")
            if line.strip() and line.strip() != "-"
        ]
        return {
            "has_extracted_facts": True,
            "facts": facts,
            "extracted_text": text,
        }

    return execute


def _make_compose_response(llm: BaseChatModel) -> _Execute:
    """Create a ``compose_response`` execute function powered by *llm*."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        facts = ws.get("facts", [])
        question = ws.get("question", ws.get("task", ""))
        facts_text = (
            "\n".join(f"- {f}" for f in facts)
            if facts
            else ws.get("extracted_text", "No facts available.")
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant. Given extracted facts "
                        "and a question, compose a clear, concise answer. "
                        "Answer the question directly in one or two sentences."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Facts:\n{facts_text}\n\n"
                        "Compose a clear answer:"
                    )
                ),
            ]
        )
        return {
            "response_ready": True,
            "response": response.content.strip(),
        }

    return execute


def plan_and_execute_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """Build the 3-step Plan-and-Execute action set.

    Args:
        llm: Language model for fact extraction and response composition.
        search_tool: Optional LangChain tool (e.g. ``TavilySearchResults``).
            When provided, ``search_web`` invokes it for real web search.
            When omitted, the LLM generates research notes instead.
    """
    return [
        ActionSpec(
            name="search_web",
            preconditions={"has_task": True},
            effects={"has_search_results": True},
            cost=1.0,
            execute=_make_search_web(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="extract_facts",
            preconditions={"has_search_results": True},
            effects={"has_extracted_facts": True},
            cost=1.0,
            execute=_make_extract_facts(llm),
        ),
        ActionSpec(
            name="compose_response",
            preconditions={"has_extracted_facts": True},
            effects={"response_ready": True},
            cost=1.0,
            execute=_make_compose_response(llm),
        ),
    ]


# ---------------------------------------------------------------------------
# 5-step research pipeline: search -> verify -> analyze -> draft -> finalize
# Demonstrates that A* handles arbitrarily long LLM-powered pipelines.
# ---------------------------------------------------------------------------


def _make_pipeline_search(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> _Execute:
    """Step 1: collect raw data via web search or LLM research."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        task = ws.get("task", "")

        if search_tool is not None:
            raw = search_tool.invoke(task)
            items = raw if isinstance(raw, list) else [raw]
            data = [
                r.get("content", r.get("snippet", str(r))) for r in items
            ]
        else:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a research assistant. Provide 3-5 "
                            "concise factual points about the given topic. "
                            "Return each point on its own line."
                        )
                    ),
                    HumanMessage(content=f"Research topic: {task}"),
                ]
            )
            data = [
                line.strip().lstrip("- ").lstrip("0123456789.").strip()
                for line in response.content.strip().split("\n")
                if line.strip()
            ]
        return {"has_raw_data": True, "raw_data": data}

    return execute


def _make_pipeline_verify(llm: BaseChatModel) -> _Execute:
    """Step 2: LLM filters raw data for accuracy and relevance."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        data = ws.get("raw_data", [])
        data_text = "\n".join(f"- {item}" for item in data)

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a fact-verification assistant. Review the "
                        "following data points and return ONLY the ones that "
                        "appear factually accurate and relevant. Return each "
                        "verified point on its own line, prefixed with '- '."
                    )
                ),
                HumanMessage(content=f"Data points to verify:\n{data_text}"),
            ]
        )
        verified = [
            line.strip().lstrip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]
        return {"has_verified_data": True, "verified_data": verified}

    return execute


def _make_pipeline_analyze(llm: BaseChatModel) -> _Execute:
    """Step 3: LLM synthesizes verified data into an analysis."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        data = ws.get("verified_data", [])
        data_text = "\n".join(f"- {item}" for item in data)

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an analytical assistant. Given verified "
                        "facts, write a concise analytical summary that "
                        "synthesizes the key insights. Keep it to 2-3 "
                        "sentences."
                    )
                ),
                HumanMessage(
                    content=f"Verified facts:\n{data_text}\n\nAnalysis:"
                ),
            ]
        )
        return {
            "has_analysis": True,
            "analysis": response.content.strip(),
        }

    return execute


def _make_pipeline_draft(llm: BaseChatModel) -> _Execute:
    """Step 4: LLM drafts a report from the analysis."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        analysis = ws.get("analysis", "")
        task = ws.get("task", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a report writer. Given an analysis and the "
                        "original research topic, draft a short report "
                        "(3-5 sentences) that presents the findings clearly."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {task}\n\n"
                        f"Analysis:\n{analysis}\n\n"
                        "Draft report:"
                    )
                ),
            ]
        )
        return {"has_draft": True, "draft": response.content.strip()}

    return execute


def _make_pipeline_finalize(llm: BaseChatModel) -> _Execute:
    """Step 5: LLM polishes the draft into a final report."""

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        draft = ws.get("draft", "")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an editor. Polish the following draft report "
                        "into a final version. Fix any grammar issues, "
                        "improve clarity, and add a brief concluding sentence. "
                        "Prefix the final output with '[FINAL] '."
                    )
                ),
                HumanMessage(content=f"Draft to finalize:\n{draft}"),
            ]
        )
        text = response.content.strip()
        if not text.startswith("[FINAL]"):
            text = f"[FINAL] {text}"
        return {"report_complete": True, "final_report": text}

    return execute


def five_step_research_actions(
    llm: BaseChatModel,
    *,
    search_tool: Any | None = None,
) -> list[ActionSpec]:
    """Build the 5-step research pipeline action set.

    Args:
        llm: Language model powering every step.
        search_tool: Optional LangChain tool for step 1.
    """
    return [
        ActionSpec(
            name="search",
            preconditions={"has_task": True},
            effects={"has_raw_data": True},
            execute=_make_pipeline_search(llm, search_tool=search_tool),
        ),
        ActionSpec(
            name="verify",
            preconditions={"has_raw_data": True},
            effects={"has_verified_data": True},
            execute=_make_pipeline_verify(llm),
        ),
        ActionSpec(
            name="analyze",
            preconditions={"has_verified_data": True},
            effects={"has_analysis": True},
            execute=_make_pipeline_analyze(llm),
        ),
        ActionSpec(
            name="draft",
            preconditions={"has_analysis": True},
            effects={"has_draft": True},
            execute=_make_pipeline_draft(llm),
        ),
        ActionSpec(
            name="finalize",
            preconditions={"has_draft": True},
            effects={"report_complete": True},
            execute=_make_pipeline_finalize(llm),
        ),
    ]
