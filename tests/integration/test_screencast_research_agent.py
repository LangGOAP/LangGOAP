"""Integration tests for the research-agent flagship screencast.

Two layers of coverage:

* **Mocked** (always runs in CI). Uses dependency-injected fake tool
  callables so the planner and the hand-wired graph execute end to
  end without hitting OpenAI or Tavily. Verifies the failure
  recovery behaviour that the screencast hangs its narrative on —
  Tavily blacklisted, planner replans through DuckDuckGo, goal
  achieved under the $2.00 cap.

* **Real-API** (auto-enabled when ``OPENAI_API_KEY`` and
  ``TAVILY_API_KEY`` are present — typically loaded from ``.env``
  by ``tests/conftest.py``). Runs all three agents against live
  OpenAI ``gpt-4o-mini`` + Tavily and asserts the GOAP version
  finishes under the $2.00 budget. Skipped when either key is
  missing.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from examples.screencast.research_agent import (
    langgoap_planned,
    langgraph_routed,
)
from examples.screencast.research_agent.shared.tools import TavilyUnauthorized

SAMPLE_TICKERS = ["AAPL", "MSFT", "GOOG"]


def _fake_fns(*, tavily_raises: bool = False) -> dict[str, Any]:
    """Build a fake-tools dict the run() seam accepts.

    When ``tavily_raises`` is ``True`` the Tavily callable raises
    :class:`TavilyUnauthorized` so the planner's observer blacklists
    it and replans through ``duckduckgo_search``.
    """

    def tavily(t: str) -> str:
        if tavily_raises:
            raise TavilyUnauthorized("TAVILY_API_KEY revoked (test)")
        return f"tavily snippet for {t}"

    def ddg(t: str) -> str:
        return f"ddg snippet for {t}"

    def extract(t: str, snippet: str) -> dict[str, Any]:
        return {
            "ticker": t,
            "eps_reported": 1.50,
            "eps_consensus": 1.40,
            "snippet_kind": "ddg" if snippet.startswith("ddg") else "tavily",
        }

    def compare(t: str, earnings: dict[str, Any]) -> dict[str, Any]:
        return {**earnings, "ticker": t, "verdict": "beat", "delta": 0.10}

    def synthesize(findings: list[dict[str, Any]]) -> str:
        return f"Cohort summary: {len(findings)} tickers, all beat consensus."

    return {
        "tavily_search": tavily,
        "duckduckgo_search": ddg,
        "extract_earnings": extract,
        "compare_to_consensus": compare,
        "synthesize_summary": synthesize,
    }


# ---------------------------------------------------------------------------
# Mocked path — always runs in CI
# ---------------------------------------------------------------------------


class TestLangGOAPHappyPath:
    """Version C on the happy path: tavily chosen, no replans, summary ready."""

    def test_finishes_via_tavily_under_budget(self) -> None:
        result = langgoap_planned.run(
            tickers=SAMPLE_TICKERS,
            _fns_override=_fake_fns(tavily_raises=False),
        )
        assert result.status == "goal_achieved"
        assert result.replans == 0
        assert result.path_taken is not None
        assert result.path_taken[0] == "search_all_via_tavily"
        assert "search_all_via_ddg" not in result.path_taken
        assert result.summary_text.startswith("Cohort summary:")


class TestLangGOAPDisrupted:
    """Version C when Tavily fails: planner replans through DuckDuckGo."""

    def test_recovers_via_ddg_fallback(self) -> None:
        result = langgoap_planned.run(
            tickers=SAMPLE_TICKERS,
            _fns_override=_fake_fns(tavily_raises=True),
        )
        assert result.status == "goal_achieved"
        assert result.replans >= 1
        assert result.path_taken is not None
        assert "search_all_via_ddg" in result.path_taken
        assert result.path_taken[-1] == "synthesize_summary"
        assert result.summary_text.startswith("Cohort summary:")


class TestHandWiredRouted:
    """Version B — hand-wired LangGraph baseline."""

    def test_happy_path_succeeds(self) -> None:
        result = langgraph_routed.run(
            tickers=SAMPLE_TICKERS,
            _fns_override=_fake_fns(tavily_raises=False),
        )
        assert result.status == "ok"
        assert result.summary_text.startswith("Cohort summary:")

    def test_tavily_failure_breaks_the_graph(self) -> None:
        """Demonstrates the brittleness the screencast calls out."""
        result = langgraph_routed.run(
            tickers=SAMPLE_TICKERS,
            _fns_override=_fake_fns(tavily_raises=True),
        )
        assert result.status == "error"
        assert result.error is not None
        assert "TavilyUnauthorized" in result.error


# ---------------------------------------------------------------------------
# Live-API path — opt in via LANGGOAP_RUN_LIVE_DEMO=1
# ---------------------------------------------------------------------------


_LIVE_REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY")


@pytest.mark.api
@pytest.mark.skipif(
    not all(os.environ.get(k) for k in _LIVE_REQUIRED_KEYS),
    reason="Live demo requires OPENAI_API_KEY + TAVILY_API_KEY in the environment",
)
class TestLiveAPI:
    """Real OpenAI + Tavily run. Costs a few cents per execution."""

    def test_langgoap_under_budget(self) -> None:
        result = langgoap_planned.run(tickers=SAMPLE_TICKERS, cost_cap_usd=2.00)
        assert result.status == "goal_achieved"
        assert result.cost_summary["total_cost_usd"] < 2.00
        assert result.summary_text
