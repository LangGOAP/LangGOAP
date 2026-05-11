"""The screencast climax — revoke Tavily, run all three versions.

The premise: Tavily rotated its API keys overnight. Three agents wake
up this morning, all with the same brief, and only one finishes the
cohort summary under budget.

* React baseline (Version A): keeps choosing ``tavily_search``,
  retries, eventually exhausts its recursion budget or returns
  partial output — with the LLM-decision tokens already billed.
* Hand-wired LangGraph (Version B): errors out the first time
  ``tavily_search`` raises — there is no fallback edge.
* LangGOAP (Version C): observer blacklists ``tavily_search`` on
  first failure, planner replans through ``duckduckgo_search``, and
  finishes the cohort under the $2.00 cap.

Run::

    uv run python -m examples.screencast.research_agent.disrupted

Real Tavily / OpenAI keys must be set; the script copies and then
unsets ``TAVILY_API_KEY`` before each run so the failure is real
rather than mocked.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from typing import Iterator

from examples.screencast.research_agent import (
    langgoap_planned,
    langgraph_routed,
    react_baseline,
)
from examples.screencast.research_agent.shared import TICKERS, RunResult
from examples.screencast.research_agent.shared.cost_meter import reveal_cost_live


@contextmanager
def tavily_revoked() -> Iterator[None]:
    """Temporarily clear ``TAVILY_API_KEY`` so live calls 401."""
    saved = os.environ.pop("TAVILY_API_KEY", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["TAVILY_API_KEY"] = saved


def run(*, tickers: list[str] | None = None) -> list[RunResult]:
    """Run all three agents under a revoked Tavily key. Returns results."""
    tickers = tickers or list(TICKERS)
    results: list[RunResult] = []

    print("\n" + "=" * 64)
    print("  DISRUPTION: TAVILY_API_KEY revoked")
    print("=" * 64)

    with tavily_revoked():
        print("\n--- Version A: react_baseline ---")
        results.append(react_baseline.run(tickers=tickers))
        _summarize(results[-1])

        print("\n--- Version B: langgraph_routed ---")
        results.append(langgraph_routed.run(tickers=tickers))
        _summarize(results[-1])

        print("\n--- Version C: langgoap_planned ---")
        results.append(langgoap_planned.run(tickers=tickers))
        _summarize(results[-1])

    _print_comparison(results)
    return results


def _summarize(result: RunResult) -> None:
    cs = result.cost_summary
    print(
        f"  status={result.status:<14} cost=${cs['total_cost_usd']:.4f}  "
        f"tokens={cs['total_tokens']:,}  "
        f"tavily={cs['tavily_call_count']}  ddg={cs['ddg_call_count']}  "
        f"replans={result.replans}  wall={result.elapsed_s:.1f}s"
    )
    if result.error:
        print(f"  error: {result.error}")


def _print_comparison(results: list[RunResult]) -> None:
    print("\n" + "=" * 64)
    print("  Side-by-side")
    print("=" * 64)
    header = f"  {'version':<22} {'status':<14} {'cost':>8} {'replans':>8}"
    print(header)
    print(f"  {'-'*22} {'-'*14} {'-'*8} {'-'*8}")
    for r in results:
        cost = r.cost_summary["total_cost_usd"]
        print(f"  {r.name:<22} {r.status:<14} {cost:>7.4f}$ " f"{r.replans:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", type=int, default=len(TICKERS))
    parser.add_argument(
        "--reveal-cost-live",
        action="store_true",
        help=(
            "Print a one-line meter update on every LLM / Tavily / DDG event "
            "(to stderr). Useful for live screen-recording: viewers see the "
            "cost climb in real time."
        ),
    )
    args = parser.parse_args()
    if args.reveal_cost_live:
        with reveal_cost_live():
            run(tickers=TICKERS[: args.tickers])
    else:
        run(tickers=TICKERS[: args.tickers])


if __name__ == "__main__":
    main()
