# Research agent screencast — flagship demo

> The companion to the LangGOAP YouTube intro. One brief, four agent shapes, real OpenAI + Tavily, real dollars.

This is the **flagship** example for new viewers. It uses live LLM and search APIs to put concrete USD costs on each agent shape, so the planner's advantage is visible in numbers rather than abstractions.

For deeper case studies on different verticals (incident response, supply chain, travel) see the [parent screencast catalog](../README.md).

---

## The brief

> *"For these 10 tickers, find the most recent quarterly earnings, flag whether EPS beat consensus, and write a one-paragraph cohort summary. Stay under $2."*

| Ticker pool | Tools | Model |
|---|---|---|
| `AAPL, MSFT, GOOG, AMZN, META, NVDA, TSLA, JPM, WMT, NFLX` | `tavily_search`, `duckduckgo_search`, `extract_earnings`, `compare_to_consensus`, `synthesize_summary` | `gpt-4o-mini` |

## The four versions

| File | Shape | Lines | What it shows |
|---|---|---:|---|
| [`react_baseline.py`](react_baseline.py) | `langgraph.prebuilt.create_react_agent` | ~150 | LLM decides everything; canonical comparison floor. |
| [`langgraph_routed.py`](langgraph_routed.py) | Hand-wired `StateGraph` with dispatcher node | ~170 | What a senior LangGraph dev writes after react gets unpredictable. |
| [`langgoap_planned.py`](langgoap_planned.py) | `GoapGraph` + `ActionSpec`s + `ConstraintSpec(cost_usd <= 2.00)` | ~205 | The planner is cost-aware and self-healing. |
| [`disrupted.py`](disrupted.py) | All three, run under a revoked `TAVILY_API_KEY` | ~110 | The climax: which one survives? |

The full story is in [`narrative.md`](narrative.md). The recording shot-list with timecodes is in [`script.md`](script.md).

---

## Reproduce

### 1. Mocked (no API keys, ~0.3s)

```bash
uv run pytest tests/integration/test_screencast_research_agent.py -v
```

Four mocked-tool tests exercise the GOAP happy path, the GOAP → DuckDuckGo fallback under Tavily failure, the hand-wired happy path, and the hand-wired graph erroring out on Tavily failure. The live test is gated and stays skipped.

### 2. Live, single version

```bash
# Either export the keys or drop them in a project-root `.env`:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...
# The shared package calls `load_dotenv()` automatically on import.

uv run python -m examples.screencast.research_agent.langgoap_planned
```

Replace the module name with `react_baseline` or `langgraph_routed` to compare. Each script accepts `--tickers N` to shrink the cohort for faster recordings, and `--reveal-cost-live` to print a one-line meter update on every LLM / Tavily / DDG event (stderr) — designed for live screen-recording so viewers see the cost climb in real time.

### 3. Live, four-way head-to-head with disruption

```bash
uv run python -m examples.screencast.research_agent.disrupted --reveal-cost-live
```

This runs **all three** agents back-to-back under a revoked Tavily key (the script unsets and restores it in a context manager). Expected output: React baseline raises `TavilyUnauthorized` on its first tool call, hand-wired errors out at the `search` node before reaching any LLM, LangGOAP blacklists Tavily, replans through DuckDuckGo, and finishes the cohort summary.

### 4. Live, full real-API integration test

```bash
LANGGOAP_RUN_LIVE_DEMO=1 OPENAI_API_KEY=... TAVILY_API_KEY=... \
  uv run pytest tests/integration/test_screencast_research_agent.py::TestLiveAPI -v
```

Costs a few cents per execution. Asserts the GOAP run finishes under the $2.00 cap with a non-empty summary.

### 5. Notebook

[`screencast.ipynb`](screencast.ipynb) reproduces the comparison and renders the matplotlib bar chart. By default it runs with mocked tools (no API keys); flip `LIVE = True` in the second cell to switch to real APIs.

---

## Reference numbers

From a clean live run on `gpt-4o-mini` + Tavily, 10-ticker cohort. Numbers vary ±10% across runs.

**Happy path:**

| Version | Status | Cost | Tokens | LLM calls | Tavily | Wall |
|---|---|---:|---:|---:|---:|---:|
| react_baseline | `ok` | $0.0846 | 21,281 | 26 | 10 | 57.2s |
| langgraph_routed | `ok` | $0.0810 | 4,257 | 21 | 10 | 41.1s |
| langgoap_planned | `goal_achieved` | $0.0809 | 4,211 | 21 | 10 | 39.2s |

**Disrupted (`TAVILY_API_KEY` revoked):**

| Version | Status | Cost | Tokens | DDG | Replans | Wall |
|---|---|---:|---:|---:|---:|---:|
| react_baseline | `error` | $0.0002 | 511 | 0 | 0 | 3.3s |
| langgraph_routed | `error` | $0.0000 | 0 | 0 | 0 | 0.0s |
| langgoap_planned | `goal_achieved` | $0.0009 | 4,074 | 10 | 1 | 39.8s |

---

## What's intentionally not here

- **No retry logic**. The hand-wired graph deliberately does not handle Tavily failure. Adding it would have meant new edges, new nodes, new state — which is precisely the technical debt the planner avoids.
- **No prompt-engineering hacks**. The system prompt for the React baseline is plain and short. The win comes from declaring options to the planner, not from coaxing a better LLM call order.
- **No mocked LLM in the live runs**. The numbers on screen are what OpenAI and Tavily actually charged.
