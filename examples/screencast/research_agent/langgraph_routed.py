"""Version B — hand-wired LangGraph ``StateGraph``.

The pattern a senior LangGraph user reaches for after ``create_react_agent``
proves too unpredictable: explicit per-ticker state machine with a
dispatcher node, deterministic tool ordering, no LLM-driven control
flow. Cheaper and tighter than version A on the happy path.

The intentional weakness: ``tavily_search`` is the only search edge.
There is no ``duckduckgo`` fallback wired in, because adding one would
require new edges, a new router branch, and bookkeeping for the
fallback verdict. Version C (LangGOAP) gets the fallback for free.

Run::

    uv run python -m examples.screencast.research_agent.langgraph_routed
"""

from __future__ import annotations

import argparse
import time
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from examples.screencast.research_agent.shared import (
    TICKERS,
    CostMeter,
    RunResult,
    build_research_tools,
)


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    return [*left, *right]


class GraphState(TypedDict, total=False):
    """Per-cohort state. Closure-captured ``meter`` keeps cost off-state."""

    tickers_remaining: list[str]
    current_ticker: str | None
    current_snippet: str | None
    current_earnings: dict[str, Any] | None
    findings: Annotated[list[dict[str, Any]], _append]
    summary: str
    error: str


def run(
    *,
    tickers: list[str] | None = None,
    model: str = "gpt-4o-mini",
    _fns_override: dict[str, Any] | None = None,
) -> RunResult:
    """Run the hand-wired LangGraph baseline. Returns a ``RunResult``.

    ``_fns_override`` is a test-only seam (see :mod:`langgoap_planned`).
    """
    tickers = tickers or list(TICKERS)
    world_state: dict[str, Any] = {}
    meter = CostMeter(world_state=world_state)
    if _fns_override is not None:
        fns = _fns_override
    else:
        llm = ChatOpenAI(model=model, temperature=0, callbacks=[meter.llm_callback])
        fns = build_research_tools(llm=llm, meter=meter).callables

    def search(state: GraphState) -> dict[str, Any]:
        remaining = list(state.get("tickers_remaining", []))
        ticker = remaining.pop(0)
        snippet = fns["tavily_search"](ticker)
        return {
            "tickers_remaining": remaining,
            "current_ticker": ticker,
            "current_snippet": snippet,
        }

    def extract(state: GraphState) -> dict[str, Any]:
        earnings = fns["extract_earnings"](
            state["current_ticker"], state["current_snippet"]
        )
        return {"current_earnings": earnings}

    def compare(state: GraphState) -> dict[str, Any]:
        verdict = fns["compare_to_consensus"](
            state["current_ticker"], state.get("current_earnings") or {}
        )
        return {
            "findings": [verdict],
            "current_ticker": None,
            "current_snippet": None,
            "current_earnings": None,
        }

    def synthesize(state: GraphState) -> dict[str, Any]:
        text = fns["synthesize_summary"](state.get("findings", []))
        return {"summary": text}

    def route(state: GraphState) -> str:
        if state.get("tickers_remaining"):
            return "search"
        if not state.get("summary"):
            return "synthesize"
        return END

    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("search", search)
    builder.add_node("extract", extract)
    builder.add_node("compare", compare)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "search")
    builder.add_edge("search", "extract")
    builder.add_edge("extract", "compare")
    builder.add_conditional_edges(
        "compare", route, {"search": "search", "synthesize": "synthesize", END: END}
    )
    builder.add_edge("synthesize", END)
    graph = builder.compile()

    t0 = time.monotonic()
    error: str | None = None
    status = "ok"
    summary_text = ""
    try:
        result = graph.invoke(
            {"tickers_remaining": list(tickers), "findings": []},
            config={"recursion_limit": 10 * len(tickers) + 20},
        )
        summary_text = str(result.get("summary") or "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    elapsed = time.monotonic() - t0

    return RunResult(
        name="langgraph_routed",
        status=status,
        cost_summary=meter.snapshot(),
        summary_text=summary_text,
        elapsed_s=elapsed,
        error=error,
    )


def main() -> None:
    from examples.screencast.research_agent.shared.cost_meter import reveal_cost_live

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=int, default=len(TICKERS))
    parser.add_argument(
        "--reveal-cost-live",
        action="store_true",
        help="Print a one-line meter update on every LLM / Tavily / DDG event (stderr).",
    )
    args = parser.parse_args()
    if args.reveal_cost_live:
        with reveal_cost_live():
            result = run(tickers=TICKERS[: args.tickers])
    else:
        result = run(tickers=TICKERS[: args.tickers])
    _print_report(result)


def _print_report(result: RunResult) -> None:
    bar = "=" * 64
    print(bar)
    print(f"  {result.name} — status: {result.status}")
    print(bar)
    cs = result.cost_summary
    print(f"  cost:        ${cs['total_cost_usd']:.4f}")
    print(f"  tokens:      {cs['total_tokens']:,}")
    print(f"  LLM calls:   {cs['llm_call_count']}")
    print(f"  Tavily:      {cs['tavily_call_count']}")
    print(f"  DuckDuckGo:  {cs['ddg_call_count']}")
    print(f"  wall-clock:  {result.elapsed_s:.1f}s")
    if result.error:
        print(f"  error:       {result.error}")
    if result.summary_text:
        print("\n  Cohort summary:")
        print(f"    {result.summary_text[:400]}")


if __name__ == "__main__":
    main()
