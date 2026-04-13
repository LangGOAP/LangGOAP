# Script: "Get to the Wedding"

**Duration:** ~7 minutes

---

## 0:00–0:30 — Hook

"Your best friend's wedding. Ceremony at 3pm Saturday. You're in a different
city. You booked a direct flight. Then the airline cancels it. What does your
AI agent do?

Most agentic workflows? They crash. Or they follow a hardcoded fallback that
the developer anticipated months ago. Let me show you a better way."

---

## 0:30–1:30 — The "Before" (Hardcoded LangGraph)

Open `before.py`. Walk through:

- **8 node functions** — the actual business logic (search flights, book hotel, etc.)
- **4 routing functions** — `route_after_flight_search`, `route_after_connecting`,
  `route_after_ground_search`, `route_after_train`
- **4 conditional edges** — hand-wired branches for every failure mode

"See all these routing functions? Every one is a developer decision frozen in
code. 'If flights fail, try ground transport.' 'If train fails, try car.' You
have to anticipate every failure mode at design time."

Run happy path: `uv run python examples/screencast/travel/before.py`

"Works great. Direct flight booked, hotel reserved, arrival confirmed."

---

## 1:30–2:30 — Show the Fallback Pain

Run with disruptions: flights cancelled, then flights + train.

"It handles the disruptions — because we hand-coded every branch. But look at
the cost: 4 routing functions, 4 conditional edges. And that's with just 3
transport modes. What if we add bus? Rideshare? Charter? Each one means more
routing code, more type hints, more edges."

---

## 2:30–3:00 — The Transition

"What if instead of coding every possible path, you just described what's
possible — and let the system figure out which path to take? And when that
path breaks, it figures out a new one. Automatically."

---

## 3:00–5:30 — Build the "After" (Live Coding)

Open `after.py`. Build up incrementally:

1. "Same business logic functions. `search_flights`, `book_hotel` — identical
   purpose."

2. "But instead of wiring them into a graph, I declare them as actions with
   preconditions, effects, and costs."

   Point to: `preconditions={"has_destination": True}`,
   `effects={"flight_options_found": True}`, `cost=1.0`

3. "Notice the costs: direct flight is 2.0, connecting is 3.0, train is 2.0,
   rental car is 3.0. The planner will prefer cheaper options."

4. "And the invocation — one line:"
   ```python
   result = GoapGraph(actions=travel_actions).invoke(
       goal=GoalSpec(conditions={"at_venue": True, "on_time": True}),
       world_state={"has_destination": True},
   )
   ```

5. "I declared the WHAT — 'be at the venue, on time.' The planner found the
   HOW — search flights, book direct, book hotel, confirm arrival."

Run: `uv run python examples/screencast/travel/after.py`

---

## 5:30–6:30 — Break It

Open `after_disrupted.py`.

"Same actions. But now the execute functions check world_state flags and raise
exceptions when disrupted. Watch what happens."

Run direct flight cancelled:
"Direct flight fails → planner blacklists it → replans → picks connecting
flight. Same goal achieved. Zero routing code changed."

---

## 6:30–7:30 — Escalate

Run all flights grounded + train sold out:

"Now ALL flights are grounded AND the train is sold out. The planner has to
find a completely different path: ground transport → rental car."

"Look at the execution trace: search_flights failed, book_train failed,
planner adapted each time. It found rental car as the last option that
still reaches the goal."

---

## 7:30–8:00 — Close

"Zero routing functions. Zero conditional edges. The planner adapts to
disruptions I never explicitly coded for. The business logic stays clean —
it's just 'what can I do' and 'what do I want.' The planner figures out the
path.

`pip install langgoap`. Link in the description."
