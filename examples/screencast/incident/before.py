"""Incident response: Hardcoded LangGraph workflow with manual fallback edges.

This version handles service recovery — but every escalation path is a
hand-wired conditional edge. Adding a new recovery strategy (e.g., canary
rollback, circuit breaker) means editing routing functions and graph structure.

Run: uv run python examples/screencast/incident/before.py
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class IncidentState(TypedDict, total=False):
    service_name: str
    status: str
    root_cause: str | None
    recovery_method: str | None
    disruptions: dict[str, bool]
    log: list[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def restart_service(state: IncidentState) -> dict[str, Any]:
    print("  [restart_service] Attempting service restart...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("restart_fails"):
        return {
            "status": "still_down",
            "log": ["Restart failed — service crashed again within 30s (OOM)"],
        }
    return {
        "status": "service_healthy",
        "recovery_method": "restart",
        "log": ["Service restarted successfully — health check passing"],
    }


def rollback_deployment(state: IncidentState) -> dict[str, Any]:
    print("  [rollback_deployment] Rolling back to last good deployment...")
    disruptions = state.get("disruptions", {})
    if disruptions.get("rollback_fails"):
        return {
            "status": "still_down",
            "log": ["Rollback failed — irreversible DB migration in latest deploy"],
        }
    return {
        "status": "service_healthy",
        "recovery_method": "rollback",
        "log": ["Rolled back to v2.3.1 — health check passing"],
    }


def analyze_error_logs(state: IncidentState) -> dict[str, Any]:
    print("  [analyze_error_logs] Analyzing error logs...")
    return {
        "status": "root_cause_found",
        "root_cause": "Memory leak in request handler — unbounded cache growth",
        "log": ["Root cause: memory leak in RequestHandler.cache — no eviction policy"],
    }


def apply_hotfix(state: IncidentState) -> dict[str, Any]:
    cause = state.get("root_cause", "unknown")
    print(f"  [apply_hotfix] Applying fix for: {cause}...")
    return {
        "status": "service_healthy",
        "recovery_method": "hotfix",
        "log": [f"Hotfix applied: added LRU eviction to cache (cause: {cause})"],
    }


def failover_to_backup(state: IncidentState) -> dict[str, Any]:
    print("  [failover_to_backup] Triggering failover to backup region...")
    return {
        "status": "service_healthy",
        "recovery_method": "failover",
        "log": ["Failed over to us-west-2 — DNS propagation complete"],
    }


def notify_stakeholders(state: IncidentState) -> dict[str, Any]:
    method = state.get("recovery_method", "unknown")
    print(f"  [notify_stakeholders] Sending incident summary (resolved via {method})...")
    return {
        "status": "resolved",
        "log": [f"Stakeholders notified — resolved via {method}"],
    }


# ---------------------------------------------------------------------------
# Routing functions — every escalation step is explicit
# ---------------------------------------------------------------------------


def route_after_restart(
    state: IncidentState,
) -> Literal["notify_stakeholders", "rollback_deployment"]:
    if state.get("status") == "service_healthy":
        return "notify_stakeholders"
    return "rollback_deployment"


def route_after_rollback(
    state: IncidentState,
) -> Literal["notify_stakeholders", "analyze_error_logs"]:
    if state.get("status") == "service_healthy":
        return "notify_stakeholders"
    return "analyze_error_logs"


def route_after_hotfix(
    state: IncidentState,
) -> Literal["notify_stakeholders", "failover_to_backup"]:
    if state.get("status") == "service_healthy":
        return "notify_stakeholders"
    return "failover_to_backup"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_incident_graph() -> StateGraph:
    builder = StateGraph(IncidentState)

    builder.add_node("restart_service", restart_service)
    builder.add_node("rollback_deployment", rollback_deployment)
    builder.add_node("analyze_error_logs", analyze_error_logs)
    builder.add_node("apply_hotfix", apply_hotfix)
    builder.add_node("failover_to_backup", failover_to_backup)
    builder.add_node("notify_stakeholders", notify_stakeholders)

    builder.add_edge(START, "restart_service")
    builder.add_conditional_edges("restart_service", route_after_restart)
    builder.add_conditional_edges("rollback_deployment", route_after_rollback)
    builder.add_edge("analyze_error_logs", "apply_hotfix")
    builder.add_conditional_edges("apply_hotfix", route_after_hotfix)
    builder.add_edge("failover_to_backup", "notify_stakeholders")
    builder.add_edge("notify_stakeholders", END)

    return builder


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------


def run(disruptions: dict[str, bool] | None = None) -> None:
    label = "HAPPY PATH" if not disruptions else f"DISRUPTIONS: {disruptions}"
    print(f"\n{'='*60}")
    print(f"  BEFORE (Hardcoded LangGraph) — {label}")
    print(f"{'='*60}")

    graph = build_incident_graph().compile()
    result = graph.invoke(
        {
            "service_name": "payment-api",
            "status": "incident_detected",
            "disruptions": disruptions or {},
            "log": [],
        }
    )

    print(f"\n  Result: {result['status']}")
    print(f"  Recovery: {result.get('recovery_method', 'none')}")
    for entry in result.get("log", []):
        print(f"    • {entry}")


if __name__ == "__main__":
    # Happy path — restart works
    run()

    # Disruption 1: Restart fails → rollback
    run({"restart_fails": True})

    # Disruption 2: Restart + rollback fail → analyze → hotfix
    run({"restart_fails": True, "rollback_fails": True})

    print(f"\n{'='*60}")
    print("  ROUTING FUNCTIONS: 3 (route_after_restart,")
    print("    route_after_rollback, route_after_hotfix)")
    print("  CONDITIONAL EDGES: 3")
    print("  To add a new recovery strategy (canary, circuit breaker)?")
    print("  → Edit routing functions, add edges, update type hints")
    print(f"{'='*60}")
