# Screencast: "Service Down, Clock Ticking" (Incident Response)

Production API returning 500s. SLA clock running. Restart doesn't fix it.
Rollback is blocked. How does the system recover?

## The Contrast

**before.py** — Hardcoded LangGraph with 3 routing functions and 3 conditional
edges. Every escalation step (restart → rollback → analyze → hotfix → failover)
is a hand-wired branch.

**after.py** — GOAP version. Same recovery logic. Zero routing code. Costs
encode escalation priority: restart(1) < rollback(2) < hotfix(3) < failover(4).

**after_disrupted.py** — Restart fails? Planner escalates to rollback. Rollback
blocked? Planner routes through log analysis → hotfix. No routing code changed.

## Run

```bash
cd langgoap
uv run python examples/screencast/incident/before.py
uv run python examples/screencast/incident/after.py
uv run python examples/screencast/incident/after_disrupted.py
```

## Test

```bash
uv run pytest tests/integration/test_screencast_incident.py -vv
```
