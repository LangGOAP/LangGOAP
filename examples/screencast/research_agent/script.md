# Research agent screencast — video script (≈7 min)

Target length **6–7 min**. Tab and font sizes assume a 1920×1080 recording with the editor at 14pt and the terminal at 16pt.

---

## 00:00 — 00:25 · Cold open (25s)

**On screen**: split view: left half an OpenAI usage dashboard with a $42.18 line item flashing red; right half a Slack DM saying "did you push something to prod?"

**VO**:
> "Last month my research agent cost me forty-two dollars and got the answer wrong. This screencast is how I fixed both numbers."

Cut to title card: **"LangGOAP — putting the budget in the goal"**

---

## 00:25 — 01:10 · The brief (45s)

**On screen**: editor showing `examples/screencast/research_agent/shared/tickers.py`. Cursor on the `TICKERS` list.

**VO**:
> "One brief, ten tickers: AAPL, MSFT, GOOG, ten in total. For each one, search the web for the most recent quarterly earnings, flag whether EPS beat consensus, and write a one-paragraph cohort summary. Hard cap: two dollars. Same task, same five tools, three different agent shapes — plus a fourth run where things go wrong."

Brief shot of `shared/tools.py` showing the five `@tool` decorators (highlight names only, don't read code).

---

## 01:10 — 02:00 · Version A: react baseline (50s)

**On screen**: editor on `react_baseline.py`. Highlight the `create_react_agent(llm, tools)` line.

**VO**:
> "Version A is the LangGraph canonical idiom — hand the LLM all five tools, write a system prompt, let it decide. One line of agent code. We add a cost meter so we can see the bill."

Cut to terminal. Run with `--reveal-cost-live` so each LLM / Tavily event prints one line to stderr in real time:

```bash
uv run python -m examples.screencast.research_agent.react_baseline --reveal-cost-live
```

Let it run. The `· t=Xs cost=$Y tok=Z` lines stream as the cost climbs. Final: `cost: $0.0846, tokens: 21,281, LLM calls: 26, Tavily: 10, wall: 57.2s` (varies ±10%).

**VO**:
> "Eight cents. The LLM made the right choices and Tavily did the heavy lifting on the bill. But look at the tokens — almost twenty thousand. The model talked to itself before every single tool call, re-deciding the plan."

---

## 02:00 — 03:00 · Version B: the hand-wired fix (60s)

**On screen**: editor on `langgraph_routed.py`. Slowly scroll the `StateGraph` builder, highlighting the explicit `search → extract → compare → route` nodes.

**VO**:
> "Version B is what most senior LangGraph devs reach for next: write the graph yourself. Explicit nodes, explicit order, no LLM-driven control flow. Tighter, cheaper."

Cut to terminal:

```bash
uv run python -m examples.screencast.research_agent.langgraph_routed --reveal-cost-live
```

Show final: `cost: $0.0810, tokens: 4,257, LLM calls: 21`.

**VO**:
> "Same cost — Tavily dominates the bill — but a quarter the tokens. We took the LLM out of the dispatcher and the model only got called for extract, compare, and the final summary. Tighter code, fewer turns. The viewer might stop here — *good enough*."

**Beat. Lean in.**

> "...until Monday."

---

## 03:00 — 04:15 · The disruption — act 1 (75s)

**On screen**: terminal — one shot, full disruption.

```bash
uv run python -m examples.screencast.research_agent.disrupted --reveal-cost-live
```

The script pops `TAVILY_API_KEY` itself and runs all three versions back-to-back. First two segments are the failures:

```
--- Version A: react_baseline ---
  status=error  cost=$0.0002  tokens=511  tavily=0  ddg=0  wall=3.3s
  error: TavilyUnauthorized: TAVILY_API_KEY missing or revoked

--- Version B: langgraph_routed ---
  status=error  cost=$0.0000  tokens=0  tavily=0  ddg=0  wall=0.0s
  error: TavilyUnauthorized: TAVILY_API_KEY missing or revoked
```

**VO** (over Version A):
> "Version A picks Tavily, the tool raises, the exception bubbles out of `agent.invoke()`. Nobody passed `handle_tool_error=True` on Friday, so this is just a crash. Two hundredths of a cent — for the one LLM call before the exception."

**VO** (over Version B):
> "Version B doesn't even reach the LLM. The first node calls `tavily_search` directly. Zero tokens, zero dollars, zero output. We never wrote an edge for 'Tavily failed' because that would have meant new nodes, a new branch, new state. On Friday it felt like overkill. It isn't anymore."

---

## 04:15 — 05:30 · Version C: the LangGOAP fix (75s)

**On screen**: editor on `langgoap_planned.py`. Highlight three things in this order:

1. The five `ActionSpec(... resources={"cost_usd": ...})` blocks
2. The `cost=3.0` annotation on `search_all_via_ddg`
3. The `GoalSpec(... constraints=(ConstraintSpec(key="cost_usd", max=2.0, level="hard"),))` block

**VO** (over the highlights):
> "Version C declares the same tools as `ActionSpec`s. Each one has a per-action USD estimate. Tavily costs A* one unit, DuckDuckGo costs three — Tavily wins on the happy path. The two-dollar cap lives on the goal, not in a callback."

Cut back to the still-running `disrupted` terminal — Version C section:

```
--- Version C: langgoap_planned ---
Action 'search_all_via_tavily' failed: TAVILY_API_KEY missing or revoked
  status=goal_achieved  cost=$0.0009  tokens=4,074  tavily=0  ddg=10  replans=1  wall=39.8s
```

**VO**:
> "Same exception. The observer node sees the failure, blacklists Tavily, the planner replans. There is exactly one feasible plan left: through DuckDuckGo. It was second choice on the happy path; now it's the only choice. Less than a tenth of a cent. Summary written. One replan. No new edges, no retry logic, no try-except in the dispatcher."

---

## 05:30 — 06:30 · The side-by-side (60s)

**On screen**: the notebook (`screencast.ipynb`) with the matplotlib bar chart rendered. Three grey bars (happy path), two red bars (disrupted), pointing at the difference.

**VO**:
> "Three bars on top — happy path. Eight cents, eight cents, eight cents. *This is where most agent comparisons stop.* Three bars on the bottom — same agents under one revoked key. Version A's bar is a sliver — it crashed at four-tenths of a cent. Version B doesn't have a bar at all — zero dollars, zero output. Version C — the planner — came in at less than a tenth of a cent, cheaper than its own happy-path cost, because DuckDuckGo is free."

Pause on the chart. Let it land.

**VO**:
> "The win isn't fewer dollars. The win is *the dollars you didn't spend on retry logic you didn't write.*"

---

## 06:30 — 07:00 · Outro (30s)

**On screen**: GitHub repo `LangGOAP/LangGOAP`, the README. Zoom on the "Quickstart" section.

**VO**:
> "Repo is in the description. The mocked integration test for this entire screencast is committed and runs in CI — nothing in this video is staged. Star it, kick the tires, let me know what scenario you'd like to see next. I'm the maintainer; I read every issue."

End card.

---

## Production notes

- **Real APIs throughout**. Do not stage the costs. The number on screen is what the run actually charged.
- **Don't speed up the terminal**. The LLM-thrash on the react baseline is visually compelling; let the viewer feel it.
- **Color**: grey for happy-path runs, red for disrupted runs. Stick to it across the chart and the terminal screenshots in cuts.
- **B-roll**: the cost meter ticking up during react_baseline. Worth a separate take with the terminal full-screen, no editor.
- **Audio**: keep the screencast quiet on the technical sections; light cinematic sting on the "...until Monday" beat.
- **Captions**: hard-burn the dollar amounts on each terminal cut. They are the proof.
