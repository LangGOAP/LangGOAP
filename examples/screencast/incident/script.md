# Script: "Service Down, Clock Ticking"

**Duration:** ~7 minutes

---

## 0:00–0:30 — Hook

"2am. PagerDuty fires. Your payment API is returning 500s. SLA clock is
running — every minute costs money. You need to fix it NOW.

Most incident runbooks are rigid decision trees. 'Step 1: restart. Step 2:
rollback.' What happens when Step 1 doesn't work and Step 2 is blocked?
Let me show you how GOAP handles incident response."

---

## 0:30–1:30 — The "Before" (Hardcoded LangGraph)

Open `before.py`. Walk through:

- **6 node functions** — restart, rollback, analyze logs, hotfix, failover, notify
- **3 routing functions** — route after restart, after rollback, after hotfix
- **3 conditional edges** — each escalation step is explicit

"This IS a runbook encoded as a graph. 'If restart fails → rollback. If
rollback fails → analyze logs → hotfix. If hotfix fails → failover.' Every
escalation is a developer decision frozen in code."

Run happy path: restart works, stakeholders notified.

---

## 1:30–2:30 — Show Escalation Complexity

Run with restart failing, then restart + rollback failing.

"It handles it — because the escalation ladder is hardcoded. But what about a
new recovery strategy? Canary rollback, circuit breaker, horizontal scaling?
Each one means more routing functions, more conditional edges."

---

## 2:30–3:00 — The Transition

"What if the recovery options were just a bag of actions with costs? The
cheapest one gets tried first. If it fails, the next cheapest. The escalation
ladder emerges from the cost structure — you don't hard-wire it."

---

## 3:00–5:30 — Build the "After"

Open `after.py`. Highlight:

1. "Six actions, all with `incident_detected` as precondition, all producing
   `service_healthy`. The planner can pick ANY of them."
2. "Costs encode priority: restart(1) < rollback(2) = scale(2) < hotfix(3)
   < failover(4). The planner tries the cheapest first."
3. "The hotfix path requires `root_cause_hypothesized` — so the planner chains
   `analyze_error_logs → apply_hotfix`. That's a 2-step path (cost 4), more
   expensive than restart(1) but cheaper than failover(4)."
4. "One-line invocation. Two goal conditions: service healthy AND stakeholders
   notified."

Run: restart + notify (cheapest path, total cost 2).

---

## 5:30–6:30 — Break It

Open `after_disrupted.py`. Run with restart failing.

"Restart raises an exception — memory leak, not a transient crash. Planner
blacklists restart, replans. Next cheapest: rollback(2). Rollback works,
stakeholders notified."

---

## 6:30–7:30 — Escalate

Run with restart + rollback + scaling all failing.

"Three recovery strategies failed. Planner has tried cost-1, cost-2 options.
Now it routes through analyze_logs(1) → apply_hotfix(3) — a 2-step path
the developer never explicitly coded as a fallback. The planner discovered it
from the action preconditions and effects."

"The hotfix action sees the root cause from the log analysis and applies a
targeted fix. That's where the LLM reasoning shines — analyzing logs,
generating the fix, verifying the health check."

---

## 7:30–8:00 — Close

"The escalation ladder isn't hardcoded — it emerges from action costs. Adding
a new recovery strategy? Add an ActionSpec with the right cost. The planner
slots it into the right place in the escalation order automatically.

`pip install langgoap`."
