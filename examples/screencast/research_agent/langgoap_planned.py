"""Version C — LangGOAP planned, cost-bounded research agent.

Same 5 tools, declared as :class:`ActionSpec`s with per-action USD
cost estimates and a hard ``ConstraintSpec(cost_usd <= 2.00)``
budget. The planner picks Tavily on the happy path because it has the
lower A* cost; when Tavily is blacklisted (rate-limit / 401) the
planner replans through DuckDuckGo without any new edges.

Run::

    uv run python -m examples.screencast.research_agent.langgoap_planned
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from langchain_openai import ChatOpenAI

from examples.screencast.research_agent.shared import (
    TICKERS,
    CostMeter,
    RunResult,
    build_research_tools,
)
from langgoap import (
    ActionSpec,
    ConstraintSpec,
    GoalPolicy,
    GoalSpec,
    GoapGraph,
)
from langgoap.types import ReplanStrategy

# Per-action USD cost estimates (calibrated from one real run on
# gpt-4o-mini + Tavily). These flow into the CSP layer through
# ``resources={"cost_usd": ...}``; they are *estimates* the planner
# uses to feasibility-check a plan against the $2.00 hard cap.
COST_TAVILY_ALL = 0.085
COST_DDG_ALL = 0.0
COST_EXTRACT_ALL = 0.030
COST_COMPARE_ALL = 0.018
COST_SYNTHESIZE = 0.012


def build_actions(
    tickers: list[str],
    fns: dict[str, Any],
) -> list[ActionSpec]:
    """Build the 5 planner actions. Closure-captures ``fns`` for execution."""

    def _search_all_tavily(world_state: dict[str, Any]) -> dict[str, Any]:
        snippets = {t: fns["tavily_search"](t) for t in tickers}
        return {"snippets": snippets, "snippets_ready": True, "search_source": "tavily"}

    def _search_all_ddg(world_state: dict[str, Any]) -> dict[str, Any]:
        snippets = {t: fns["duckduckgo_search"](t) for t in tickers}
        return {"snippets": snippets, "snippets_ready": True, "search_source": "ddg"}

    def _extract_all(world_state: dict[str, Any]) -> dict[str, Any]:
        snippets = world_state["snippets"]
        earnings = {t: fns["extract_earnings"](t, snippets[t]) for t in tickers}
        return {"earnings": earnings, "earnings_ready": True}

    def _compare_all(world_state: dict[str, Any]) -> dict[str, Any]:
        earnings = world_state["earnings"]
        findings = [fns["compare_to_consensus"](t, earnings[t]) for t in tickers]
        return {"findings": findings, "consensus_compared": True}

    def _synthesize(world_state: dict[str, Any]) -> dict[str, Any]:
        text = fns["synthesize_summary"](world_state.get("findings", []))
        return {"summary": text, "summary_ready": True}

    return [
        ActionSpec(
            name="search_all_via_tavily",
            preconditions={},
            effects={"snippets_ready": True},
            cost=1.0,
            resources={"cost_usd": COST_TAVILY_ALL},
            execute=_search_all_tavily,
        ),
        ActionSpec(
            name="search_all_via_ddg",
            preconditions={},
            effects={"snippets_ready": True},
            # Higher A* cost so Tavily wins on the happy path. When the
            # observer blacklists Tavily after a failure, DDG becomes
            # the only feasible search and is selected by the replan.
            cost=3.0,
            resources={"cost_usd": COST_DDG_ALL},
            execute=_search_all_ddg,
        ),
        ActionSpec(
            name="extract_all_earnings",
            preconditions={"snippets_ready": True},
            effects={"earnings_ready": True},
            cost=1.0,
            resources={"cost_usd": COST_EXTRACT_ALL},
            execute=_extract_all,
        ),
        ActionSpec(
            name="compare_all_to_consensus",
            preconditions={"earnings_ready": True},
            effects={"consensus_compared": True},
            cost=1.0,
            resources={"cost_usd": COST_COMPARE_ALL},
            execute=_compare_all,
        ),
        ActionSpec(
            name="synthesize_summary",
            preconditions={"consensus_compared": True},
            effects={"summary_ready": True},
            cost=1.0,
            resources={"cost_usd": COST_SYNTHESIZE},
            execute=_synthesize,
        ),
    ]


def run(
    *,
    tickers: list[str] | None = None,
    model: str = "gpt-4o-mini",
    cost_cap_usd: float = 2.00,
    _fns_override: dict[str, Any] | None = None,
) -> RunResult:
    """Run the LangGOAP planned agent. Returns a ``RunResult``.

    ``_fns_override`` is a test-only seam: when provided, the planner
    actions execute the supplied callables instead of wiring real
    OpenAI + Tavily clients. Production callers should leave it
    ``None``.
    """
    tickers = tickers or list(TICKERS)
    world_state: dict[str, Any] = {}
    meter = CostMeter(world_state=world_state)
    if _fns_override is not None:
        fns = _fns_override
    else:
        llm = ChatOpenAI(model=model, temperature=0, callbacks=[meter.llm_callback])
        fns = build_research_tools(llm=llm, meter=meter).callables
    actions = build_actions(tickers, fns)

    goal = GoalSpec(
        conditions={"summary_ready": True},
        constraints=(ConstraintSpec(key="cost_usd", max=cost_cap_usd, level="hard"),),
        policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION, max_replans=3),
    )

    t0 = time.monotonic()
    error: str | None = None
    status = "ok"
    summary_text = ""
    path_taken: list[str] = []
    replans = 0
    try:
        result = GoapGraph(actions=actions).invoke(goal=goal, world_state=world_state)
        status = str(result.get("status", "ok"))
        summary_text = str(result["world_state"].get("summary") or "")
        replans = int(result.get("replan_count", 0))
        path_taken = [
            h.action_name for h in result.get("execution_history", []) if h.success
        ]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    elapsed = time.monotonic() - t0

    return RunResult(
        name="langgoap_planned",
        status=status,
        cost_summary=meter.snapshot(),
        summary_text=summary_text,
        elapsed_s=elapsed,
        error=error,
        replans=replans,
        path_taken=path_taken,
    )


def main() -> None:
    from examples.screencast.research_agent.shared.cost_meter import reveal_cost_live

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=int, default=len(TICKERS))
    parser.add_argument("--cost-cap", type=float, default=2.00)
    parser.add_argument(
        "--reveal-cost-live",
        action="store_true",
        help="Print a one-line meter update on every LLM / Tavily / DDG event (stderr).",
    )
    args = parser.parse_args()
    if args.reveal_cost_live:
        with reveal_cost_live():
            result = run(tickers=TICKERS[: args.tickers], cost_cap_usd=args.cost_cap)
    else:
        result = run(tickers=TICKERS[: args.tickers], cost_cap_usd=args.cost_cap)
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
    print(f"  replans:     {result.replans}")
    if result.path_taken:
        print(f"  path:        {' → '.join(result.path_taken)}")
    if result.error:
        print(f"  error:       {result.error}")
    if result.summary_text:
        print("\n  Cohort summary:")
        print(f"    {result.summary_text[:400]}")


if __name__ == "__main__":
    main()
