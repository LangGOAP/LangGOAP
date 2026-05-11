# Research agent screencast — narrative

## The hook

Every LangChain developer has gotten the bill. You shipped a tool-calling agent on Friday, it ran fine in dev, and on Monday your OpenAI invoice was up 5x because the agent looped on a flaky search and you only noticed after the credit-card alert. **LangGOAP puts the budget in the goal, not in retry logic.**

## The brief

> "For these 10 tickers, find the most recent quarterly earnings, flag whether EPS beat consensus, and write a one-paragraph cohort summary. Stay under $2."

10 real tickers: `AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, JPM, WMT, NFLX`. Five real tools: `tavily_search`, `duckduckgo_search`, `extract_earnings`, `compare_to_consensus`, `synthesize_summary`. Real OpenAI `gpt-4o-mini`, real Tavily API.

## The cast

| Version | Shape | What the dev wrote |
|---|---|---|
| **A. React baseline** | `langgraph.prebuilt.create_react_agent(llm, tools)` | One line. LLM decides everything. |
| **B. Hand-wired graph** | Explicit `StateGraph` with dispatcher + per-tool nodes | ~80 lines. Senior-dev pattern after react proved unpredictable. |
| **C. LangGOAP planned** | 5 `ActionSpec`s + `ConstraintSpec(cost_usd <= 2.00)` | ~60 lines, including the cost numbers. Planner picks the path. |

## Act 1 — the happy path

All three finish. The dollar totals land within a rounding error of each other — **react $0.0846, hand-wired $0.0810, LangGOAP $0.0809** — because Tavily call costs (10 searches × paid tier) dominate the bill and all three make the same 10 searches.

The token meter tells the quieter, more interesting story: **react burns 21,281 LLM tokens to do the same work the other two finish in ~4,200** — 5× more model chatter, because react re-derives the plan on every turn. On `gpt-4o-mini` that's noise; on `gpt-4` or `claude-opus` it isn't.

The viewer is unconvinced. *"Why bother with a planner if my hand-wired graph costs the same?"*

## Act 2 — the disruption

We pop `TAVILY_API_KEY` from the environment and rerun. *No code changes. No retry logic.*

- **A. React** picks `tavily_search` for ticker one, the tool raises `TavilyUnauthorized`, and because nobody passed `handle_tool_error=True` the exception propagates out of `agent.invoke()`. **One LLM call, 511 tokens, $0.0002, no summary, 3.3s.** This is the Friday-default failure mode.
- **B. Hand-wired** errors out *before* the first LLM call. The `search` node is the first edge from `START`, and it calls `tavily_search` directly. **0 tokens, $0.0000, no summary, 0.0s.** The graph has no edge from "tavily failed" to "try DDG" because that would have meant new nodes, new conditional edges, new state fields. The dev decided that was overkill on Friday. It is no longer overkill.
- **C. LangGOAP** sees the same exception. The observer node blacklists `search_all_via_tavily`. The next planning cycle finds exactly one feasible plan: `search_all_via_ddg → extract_all_earnings → compare_all_to_consensus → synthesize_summary`. DuckDuckGo costs A* three units to Tavily's one — it was the second choice on the happy path; with Tavily blacklisted it becomes the only choice. **The summary ships. 1 replan, 10 DDG calls, 4,074 tokens, $0.0009, 39.8s.** Cheaper than its own happy-path cost because DDG is free; the only spend is LLM tokens for extract / compare / synthesize.

## Measured numbers

Live `gpt-4o-mini` + Tavily run, 10-ticker cohort. These are the actual figures from the dry-run you'd record; they will vary by ±10% across runs because the LLM is non-deterministic and Tavily call counts fluctuate with retries.

**Happy path (all keys present):**

| Version | Status | Cost | Tokens | LLM calls | Tavily | Wall |
|---|---|---:|---:|---:|---:|---:|
| A. react_baseline | `ok` | $0.0846 | 21,281 | 26 | 10 | 57.2s |
| B. langgraph_routed | `ok` | $0.0810 | 4,257 | 21 | 10 | 41.1s |
| C. langgoap_planned | `goal_achieved` | $0.0809 | 4,211 | 21 | 10 | 39.2s |

**Disrupted (`TAVILY_API_KEY` revoked):**

| Version | Status | Cost | Tokens | DDG | Replans | Wall |
|---|---|---:|---:|---:|---:|---:|
| A. react_baseline | `error` | $0.0002 | 511 | 0 | 0 | 3.3s |
| B. langgraph_routed | `error` | $0.0000 | 0 | 0 | 0 | 0.0s |
| C. langgoap_planned | `goal_achieved` | $0.0009 | 4,074 | 10 | 1 | 39.8s |

## The takeaway

The dev didn't write recovery code. The recovery emerged because the dev declared *what was possible* (5 actions, their pre- and postconditions, their estimated cost), not *what to do when* (edge per failure mode).

Same five tools. Same brief. Three agents, one survives the disruption.

## What this screencast deliberately does not claim

- LangGOAP is not faster than react on the happy path. It runs at LLM-tool latency. The A* planning overhead is sub-millisecond.
- LangGOAP does not eliminate the need for prompts. `extract_earnings` and `synthesize_summary` are still LLM-backed.
- LangGOAP does not write recovery edges automatically. It uses the actions you declared. If you only declare Tavily, it cannot replan through DDG — *you have to give it the option*.

The win is: when you give the planner options, it uses them under failure. The hand-wired graph cannot, without new edges.

## Next viewer step

The notebook in this directory (`screencast.ipynb`) reproduces the bar chart and the disruption. The [`examples/tutorials/cost_bounded_research_agent.ipynb`](../../tutorials/cost_bounded_research_agent.ipynb) tutorial then shows the full termination-policy and wall-clock-cap surface that this screencast only sketches.
