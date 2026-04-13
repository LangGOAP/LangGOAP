"""Incident response: GOAP version — declare recovery actions, planner routes.

Same recovery logic. Zero routing code. The planner picks the cheapest
recovery path. Escalation happens automatically when cheaper options fail.

Run: uv run python examples/screencast/incident/after.py
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, GoapGraph

# ---------------------------------------------------------------------------
# Execute functions
# ---------------------------------------------------------------------------


def restart_service(ws: dict[str, Any]) -> dict[str, Any]:
    """Restart the service — cheapest recovery option."""
    print("  [restart_service] Attempting service restart...")
    return {
        "service_healthy": True,
        "recovery_method": "restart",
    }


def rollback_deployment(ws: dict[str, Any]) -> dict[str, Any]:
    """Roll back to last known good deployment."""
    print("  [rollback_deployment] Rolling back to last good deploy...")
    return {
        "service_healthy": True,
        "recovery_method": "rollback",
    }


def analyze_error_logs(ws: dict[str, Any]) -> dict[str, Any]:
    """Analyze logs to identify root cause."""
    print("  [analyze_error_logs] Analyzing error patterns...")
    return {
        "root_cause_hypothesized": True,
        "root_cause": "Memory leak in RequestHandler — unbounded cache",
    }


def apply_hotfix(ws: dict[str, Any]) -> dict[str, Any]:
    """Generate and apply a fix based on root cause analysis."""
    cause = ws.get("root_cause", "unknown")
    print(f"  [apply_hotfix] Applying fix for: {cause}...")
    return {
        "service_healthy": True,
        "recovery_method": "hotfix",
    }


def scale_horizontally(ws: dict[str, Any]) -> dict[str, Any]:
    """Add capacity to absorb the load spike."""
    print("  [scale_horizontally] Scaling to 3x replicas...")
    return {
        "service_healthy": True,
        "recovery_method": "horizontal_scale",
    }


def failover_to_backup(ws: dict[str, Any]) -> dict[str, Any]:
    """Trigger DNS failover to backup region."""
    print("  [failover_to_backup] Failing over to us-west-2...")
    return {
        "service_healthy": True,
        "recovery_method": "failover",
    }


def notify_stakeholders(ws: dict[str, Any]) -> dict[str, Any]:
    """Draft incident summary and notify stakeholders."""
    method = ws.get("recovery_method", "unknown")
    print(f"  [notify_stakeholders] Sending summary (resolved via {method})...")
    return {"stakeholders_notified": True}


# ---------------------------------------------------------------------------
# Action declarations — costs encode escalation priority
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
# Demo
# ---------------------------------------------------------------------------


def run() -> None:
    print(f"\n{'='*60}")
    print("  AFTER (GOAP) — Happy Path")
    print(f"{'='*60}")

    result = GoapGraph(actions=incident_actions).invoke(
        goal=GoalSpec(
            conditions={"service_healthy": True, "stakeholders_notified": True}
        ),
        world_state={
            "incident_detected": True,
            "service_name": "payment-api",
            "error_rate": 0.45,
        },
    )

    print(f"\n  Status: {result['status']}")
    print(f"  Recovery: {result['world_state'].get('recovery_method')}")

    plan_path = [h.action_name for h in result["execution_history"] if h.success]
    print(f"  Plan: {' → '.join(plan_path)}")
    print(f"\n  ROUTING FUNCTIONS: 0")
    print(f"  CONDITIONAL EDGES: 0")
    print(f"  Costs encode priority: restart(1) < rollback(2) < hotfix(3) < failover(4)")


if __name__ == "__main__":
    run()
