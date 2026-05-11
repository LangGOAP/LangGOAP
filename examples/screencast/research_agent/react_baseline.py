"""Version A — ``create_react_agent`` baseline.

The canonical LangGraph idiom: hand the LLM all 5 tools and a system
prompt; let it decide everything. This is the comparison floor for
the screencast.

Run::

    uv run python -m examples.screencast.research_agent.react_baseline

Real APIs only — the mocked integration test exercises the GOAP
versions (`langgoap_planned`, `disrupted`) which are the framework's
own responsibility. The baselines are only meaningful when the LLM
is the real decision-maker.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from examples.screencast.research_agent.shared import (
    TICKERS,
    CostMeter,
    RunResult,
    build_research_tools,
)

SYSTEM_PROMPT = (
    "You are a financial research agent. For each ticker provided, you "
    "must: (1) search the web (prefer tavily_search; fall back to "
    "duckduckgo_search if tavily errors), (2) call extract_earnings on "
    "the snippet, (3) call compare_to_consensus with the extracted "
    "earnings. After all tickers are processed, call "
    "synthesize_summary once with the full JSON list of comparisons. "
    "Do not skip steps. Do not return prose; return only tool calls "
    "until synthesize_summary has been called."
)


def run(
    *,
    tickers: list[str] | None = None,
    recursion_limit: int = 80,
    model: str = "gpt-4o-mini",
) -> RunResult:
    """Run the react baseline. Returns a ``RunResult``."""
    tickers = tickers or list(TICKERS)
    world_state: dict[str, Any] = {}
    meter = CostMeter(world_state=world_state)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        callbacks=[meter.llm_callback],
    )
    tools = build_research_tools(llm=llm, meter=meter)
    agent = create_react_agent(llm, tools.langchain_tools)

    user_prompt = (
        f"Research these tickers and produce one cohort summary: "
        f"{', '.join(tickers)}."
    )
    t0 = time.monotonic()
    final_summary = ""
    error: str | None = None
    status = "ok"
    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ],
            },
            config={"recursion_limit": recursion_limit},
        )
        final_summary = _extract_summary(result)
    except Exception as exc:  # pragma: no cover - exercised by disrupted
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    elapsed = time.monotonic() - t0

    return RunResult(
        name="react_baseline",
        status=status,
        cost_summary=meter.snapshot(),
        summary_text=final_summary,
        elapsed_s=elapsed,
        error=error,
    )


def _extract_summary(result: dict[str, Any]) -> str:
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
    return ""


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
    print()
    if result.summary_text:
        print("  Cohort summary:")
        print(f"    {result.summary_text[:400]}")


if __name__ == "__main__":
    main()
