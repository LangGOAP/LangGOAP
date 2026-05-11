"""The 5 real tools used by all four screencast versions.

Each call is metered so :class:`CostMeter` reports an honest USD
total. The tools are returned as both raw callables (used by the
hand-wired LangGraph and the LangGOAP planner) and ``BaseTool``
instances (used by ``create_react_agent``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from examples.screencast.research_agent.shared.cost_meter import CostMeter
from examples.screencast.research_agent.shared.tickers import TICKER_HINTS


class TavilyUnauthorized(RuntimeError):
    """Raised when ``TAVILY_API_KEY`` is missing or revoked."""


@dataclass
class ResearchTools:
    """Bundle of callables + ``BaseTool``s shared by all 4 versions."""

    callables: dict[str, Callable[..., Any]]
    langchain_tools: list[BaseTool]


def build_research_tools(
    *,
    llm: ChatOpenAI,
    meter: CostMeter,
    offline_hints: dict[str, dict[str, Any]] | None = None,
) -> ResearchTools:
    """Build the 5 research tools wired to a metered LLM.

    When ``offline_hints`` is provided, search tools synthesize a
    snippet from the hint table instead of hitting the network. The
    mocked integration test uses this; production paths leave it
    ``None`` and let Tavily / DuckDuckGo answer.
    """

    def _tavily_search(ticker: str) -> str:
        """Search Tavily for the latest quarterly earnings of ``ticker``.

        Raises ``TavilyUnauthorized`` when no API key is set so the
        screencast can simulate mid-run revocation by popping
        ``TAVILY_API_KEY`` from the environment.
        """
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise TavilyUnauthorized("TAVILY_API_KEY missing or revoked")
        meter.record_tavily_call()
        if offline_hints is not None:
            return _hint_snippet(ticker, offline_hints)
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        resp = client.search(
            query=f"{ticker} latest quarterly earnings EPS consensus beat miss",
            max_results=3,
            search_depth="basic",
        )
        return _flatten_tavily(resp)

    def _duckduckgo_search(ticker: str) -> str:
        """Search DuckDuckGo (free) for the latest earnings of ``ticker``."""
        meter.record_ddg_call()
        if offline_hints is not None:
            return _hint_snippet(ticker, offline_hints)
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(
                ddgs.text(
                    f"{ticker} latest quarterly earnings EPS consensus",
                    max_results=3,
                )
            )
        return _flatten_ddg(hits)

    def _extract_earnings(ticker: str, snippet: str) -> dict[str, Any]:
        """LLM-backed structured extraction of EPS + consensus from a snippet."""
        prompt = (
            f"Extract the most recent quarterly earnings for {ticker} from "
            f"the snippet below. Respond as compact JSON with keys: "
            f"eps_reported (float), eps_consensus (float), quarter (str). "
            f"If a value is unknown, use null. Snippet:\n{snippet}"
        )
        resp = llm.invoke(prompt)
        return _parse_json_loose(resp.content) or {
            "eps_reported": None,
            "eps_consensus": None,
            "quarter": None,
        }

    def _compare_to_consensus(ticker: str, earnings: dict[str, Any]) -> dict[str, Any]:
        """Flag whether ``ticker``'s EPS beat or missed consensus."""
        prompt = (
            f"For {ticker} with reported EPS {earnings.get('eps_reported')} "
            f"vs consensus {earnings.get('eps_consensus')}, classify the "
            f"result as one of: 'beat', 'miss', 'inline', 'unknown'. "
            f'Respond as compact JSON: {{"verdict": "...", "delta": <float>}}.'
        )
        resp = llm.invoke(prompt)
        parsed = _parse_json_loose(resp.content) or {"verdict": "unknown", "delta": 0.0}
        return {**earnings, **parsed, "ticker": ticker}

    def _synthesize_summary(findings: list[dict[str, Any]]) -> str:
        """Produce a one-paragraph cohort summary from per-ticker findings."""
        payload = json.dumps(findings, default=str)
        prompt = (
            "Write a one-paragraph (<=120 word) summary of this cohort's "
            "most recent quarterly earnings. Call out how many beat vs "
            "missed consensus and highlight 1-2 standouts. Findings:\n"
            f"{payload}"
        )
        resp = llm.invoke(prompt)
        return str(resp.content)

    callables = {
        "tavily_search": _tavily_search,
        "duckduckgo_search": _duckduckgo_search,
        "extract_earnings": _extract_earnings,
        "compare_to_consensus": _compare_to_consensus,
        "synthesize_summary": _synthesize_summary,
    }
    return ResearchTools(
        callables=callables,
        langchain_tools=_wrap_as_langchain_tools(callables),
    )


def _wrap_as_langchain_tools(
    callables: dict[str, Callable[..., Any]],
) -> list[BaseTool]:
    """Wrap the raw callables as ``@tool`` instances for ``create_react_agent``."""

    @tool
    def tavily_search(ticker: str) -> str:
        """Search Tavily (paid, primary) for ``ticker``'s latest earnings."""
        return callables["tavily_search"](ticker)

    @tool
    def duckduckgo_search(ticker: str) -> str:
        """Search DuckDuckGo (free, fallback) for ``ticker``'s latest earnings."""
        return callables["duckduckgo_search"](ticker)

    @tool
    def extract_earnings(ticker: str, snippet: str) -> str:
        """Extract EPS and consensus from a search snippet. Returns JSON."""
        return json.dumps(callables["extract_earnings"](ticker, snippet))

    @tool
    def compare_to_consensus(ticker: str, earnings_json: str) -> str:
        """Flag beat / miss vs consensus. Returns JSON."""
        earnings = json.loads(earnings_json) if earnings_json else {}
        return json.dumps(callables["compare_to_consensus"](ticker, earnings))

    @tool
    def synthesize_summary(findings_json: str) -> str:
        """Write a one-paragraph cohort summary from a JSON list of findings."""
        findings = json.loads(findings_json) if findings_json else []
        return callables["synthesize_summary"](findings)

    return [
        tavily_search,
        duckduckgo_search,
        extract_earnings,
        compare_to_consensus,
        synthesize_summary,
    ]


def _hint_snippet(ticker: str, hints: dict[str, dict[str, Any]]) -> str:
    hint = hints.get(ticker) or TICKER_HINTS.get(ticker)
    if not hint:
        return f"No data available for {ticker}."
    return (
        f"{ticker} ({hint['company']}) reported {hint['quarter']} EPS of "
        f"${hint['eps_reported']:.2f} vs analyst consensus of "
        f"${hint['eps_consensus']:.2f}."
    )


def _flatten_tavily(resp: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in resp.get("results", [])[:3]:
        title = item.get("title") or ""
        content = item.get("content") or ""
        parts.append(f"{title}: {content}")
    return " | ".join(parts) or str(resp.get("answer") or "")


def _flatten_ddg(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for hit in hits[:3]:
        title = hit.get("title") or ""
        body = hit.get("body") or hit.get("snippet") or ""
        parts.append(f"{title}: {body}")
    return " | ".join(parts)


def _parse_json_loose(text: Any) -> dict[str, Any] | None:
    """Tolerate code fences and stray prose around a JSON object."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
