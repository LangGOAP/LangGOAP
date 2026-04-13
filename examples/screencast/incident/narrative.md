# Service Down, Clock Ticking

Your payment API is returning 500s. The SLA clock is running. Every minute
of downtime costs money and erodes customer trust. You need the service
healthy and stakeholders notified.

There are several ways to recover: restart the service, roll back to the
last good deployment, scale horizontally, analyze the logs and apply a
targeted hotfix, or fail over to a backup region. Each has a different cost,
a different speed, and a different chance of actually fixing the problem. The
right choice depends on what's actually wrong — and you often don't know
that until a cheaper option has already failed.

## The traditional approach

In a standard LangGraph workflow, the escalation ladder is hardcoded. Try
restart. If restart fails, route to rollback. If rollback fails, route to
log analysis, then hotfix. If hotfix fails, route to failover. Each step is
an explicit conditional edge.

This is a runbook encoded as a graph. It works when the escalation sequence
matches reality. But real incidents don't always follow the script. A
restart might mask the problem temporarily. A rollback might be blocked by
an irreversible database migration. The rigid ladder can't skip steps or
discover paths the developer didn't wire.

```
before.py — 3 routing functions, 3 conditional edges
```

## The GOAP approach

With GOAP, you declare every recovery action as an independent option. All
of them share the same precondition (`incident_detected: True`) and the same
effect (`service_healthy: True`). The difference is cost: restart at 1,
rollback at 2, horizontal scaling at 2, hotfix at 3 (but requires log
analysis first), failover at 4.

The planner picks the cheapest. On a clean run, that's restart (cost 1)
plus notify stakeholders (cost 1). Total cost 2. The escalation ladder
isn't wired — it emerges from the cost structure.

```
after.py — 0 routing functions, 0 conditional edges
```

## When things go wrong

The restart doesn't hold. The service crashes again within 30 seconds —
it's an OOM from a memory leak, not a transient crash. The observer sees
the failure, blacklists restart, and the planner re-searches. Next cheapest
single-step option: rollback (cost 2) or horizontal scaling (cost 2).
Service recovered, stakeholders notified.

Then the rollback is blocked. An irreversible database migration in the
latest deploy means there's nothing to roll back to. Now restart and
rollback are both blacklisted. The planner evaluates the remaining options:
horizontal scaling (cost 2), the analyze-then-hotfix path (cost 1 + 3 = 4),
or failover (cost 4). It picks scaling if available, otherwise the
hotfix path.

In the worst case — restart, rollback, and scaling all fail — the planner
routes through `analyze_error_logs` then `apply_hotfix`, or falls through
to `failover_to_backup`. These are paths the developer never explicitly
wired as "third fallback" or "fourth fallback." The planner discovered them
from the action preconditions and costs.

```
after_disrupted.py — same actions, three cascading failures, all recovered
```

## What this demonstrates

Incident response is fundamentally about trying things in order of
cost and risk, falling back when they don't work. The traditional approach
hardcodes that ordering as a graph. GOAP derives it from action costs.

The key difference shows up when you add a new recovery strategy — say, a
canary rollback or a circuit breaker. In the hardcoded version, you edit
routing functions, add conditional edges, figure out where the new option
fits in the escalation ladder. In the GOAP version, you add an `ActionSpec`
with the right cost. The planner slots it in automatically.

The hotfix path is particularly interesting. It requires two actions —
analyze logs, then apply the fix. The planner chains these because
`apply_hotfix` needs `root_cause_hypothesized`, which only `analyze_error_logs`
produces. That two-step recovery path was never explicitly wired. It fell out
of the precondition/effect declarations. The planner is doing the routing
that a developer would otherwise have to think through and hardcode.
