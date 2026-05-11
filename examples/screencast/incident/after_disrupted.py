"""Incident response: GOAP handles cascading recovery failures automatically.

Same actions as after.py. Disruptions injected via world_state flags. The
planner escalates through recovery strategies by cost — restart → rollback
→ analyze+hotfix → failover — without any routing code.

Run: uv run python examples/screencast/incident/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalPolicy, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Execute functions — disruption-aware
# ---------------------------------------------------------------------------


def restart_service(ws: dict[str, Any]) -> dict[str, Any]:
    """Restart — fails when the issue is a memory leak, not a transient crash."""
    print("  [restart_service] Attempting service restart...")
    if ws.get("restart_ineffective"):
        raise RuntimeError("Service crashed again within 30s — OOM from memory leak")
    return {
        "service_healthy": True,
        "recovery_method": "restart",
    }


def rollback_deployment(ws: dict[str, Any]) -> dict[str, Any]:
    """Rollback — fails when DB migration is irreversible."""
    print("  [rollback_deployment] Rolling back to last good deploy...")
    if ws.get("rollback_blocked"):
        raise RuntimeError("Rollback blocked — irreversible DB migration in v2.4.0")
    return {
        "service_healthy": True,
        "recovery_method": "rollback",
    }


def analyze_error_logs(ws: dict[str, Any]) -> dict[str, Any]:
    """Analyze logs to identify root cause."""
    print("  [analyze_error_logs] Analyzing error patterns in logs...")
    return {
        "root_cause_hypothesized": True,
        "root_cause": "Memory leak in RequestHandler — unbounded cache growth",
    }


def apply_hotfix(ws: dict[str, Any]) -> dict[str, Any]:
    """Apply targeted fix based on root cause."""
    cause = ws.get("root_cause", "unknown")
    print(f"  [apply_hotfix] Generating fix for: {cause}...")
    return {
        "service_healthy": True,
        "recovery_method": "hotfix",
    }


def scale_horizontally(ws: dict[str, Any]) -> dict[str, Any]:
    """Scale horizontally to absorb load."""
    print("  [scale_horizontally] Adding replicas...")
    if ws.get("scaling_blocked"):
        raise RuntimeError("Auto-scaling blocked — resource quota exceeded")
    return {
        "service_healthy": True,
        "recovery_method": "horizontal_scale",
    }


def failover_to_backup(ws: dict[str, Any]) -> dict[str, Any]:
    """Failover to backup region — expensive but reliable."""
    print("  [failover_to_backup] Triggering failover to us-west-2...")
    return {
        "service_healthy": True,
        "recovery_method": "failover",
    }


def notify_stakeholders(ws: dict[str, Any]) -> dict[str, Any]:
    """Draft and send incident summary."""
    method = ws.get("recovery_method", "unknown")
    print(f"  [notify_stakeholders] Sending summary (resolved via {method})...")
    return {"stakeholders_notified": True}


# ---------------------------------------------------------------------------
# Action declarations
# ---------------------------------------------------------------------------

incident_actions = [
    ActionSpec(
        name="restart_service",
        preconditions={"incident_detected": True},
        effects={"service_healthy": True},
        cost=1.0,
        execute=restart_service,
    ),
    ActionSpec(
        name="rollback_deployment",
        preconditions={"incident_detected": True},
        effects={"service_healthy": True},
        cost=2.0,
        execute=rollback_deployment,
    ),
    ActionSpec(
        name="analyze_error_logs",
        preconditions={"incident_detected": True},
        effects={"root_cause_hypothesized": True},
        cost=1.0,
        execute=analyze_error_logs,
    ),
    ActionSpec(
        name="apply_hotfix",
        preconditions={"root_cause_hypothesized": True},
        effects={"service_healthy": True},
        cost=3.0,
        execute=apply_hotfix,
    ),
    ActionSpec(
        name="scale_horizontally",
        preconditions={"incident_detected": True},
        effects={"service_healthy": True},
        cost=2.0,
        execute=scale_horizontally,
    ),
    ActionSpec(
        name="failover_to_backup",
        preconditions={"incident_detected": True},
        effects={"service_healthy": True},
        cost=4.0,
        execute=failover_to_backup,
    ),
    ActionSpec(
        name="notify_stakeholders",
        preconditions={"service_healthy": True},
        effects={"stakeholders_notified": True},
        cost=1.0,
        execute=notify_stakeholders,
    ),
]


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------


def run(label: str, world_state: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  AFTER (GOAP) — {label}")
    print(f"{'='*60}")

    result = GoapGraph(actions=incident_actions).invoke(
        goal=GoalSpec(
            conditions={"service_healthy": True, "stakeholders_notified": True},
            policy=GoalPolicy(replan_strategy=ReplanStrategy.ON_DEVIATION),
        ),
        world_state=world_state,
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Recovery: {result['world_state'].get('recovery_method')}")
    print(f"  Replans: {result.get('replan_count', 0)}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Path taken: {' → '.join(plan_path)}")

    failures = [h for h in result["execution_history"] if not h.success]
    if failures:
        print(f"  Failures recovered from:")
        for f in failures:
            print(f"    ✗ {f.action_name}: {f.error}")


if __name__ == "__main__":
    base_ws = {
        "incident_detected": True,
        "service_name": "payment-api",
        "error_rate": 0.45,
    }

    # Disruption 1: Restart doesn't fix it (memory leak, not transient)
    #   → planner escalates to rollback
    run(
        "Restart fails (memory leak)",
        {**base_ws, "restart_ineffective": True},
    )

    # Disruption 2: Restart + rollback both fail
    #   → planner routes through analyze_logs → apply_hotfix
    run(
        "Restart + rollback fail (irreversible migration)",
        {**base_ws, "restart_ineffective": True, "rollback_blocked": True},
    )

    # Disruption 3: Restart + rollback + scaling all fail
    #   → planner falls through to analyze → hotfix (or failover)
    run(
        "Restart + rollback + scaling all fail",
        {
            **base_ws,
            "restart_ineffective": True,
            "rollback_blocked": True,
            "scaling_blocked": True,
        },
    )
