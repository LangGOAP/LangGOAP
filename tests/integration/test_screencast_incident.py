"""Integration tests for the "Service Down, Clock Ticking" screencast scenario.

Verifies that the GOAP planner picks the cheapest recovery strategy and
escalates through alternatives when cheaper options fail, with costs
encoding the escalation priority.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy

# ---------------------------------------------------------------------------
# Shared execute functions
# ---------------------------------------------------------------------------


def restart_service(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("restart_ineffective"):
        raise RuntimeError("OOM — memory leak, not a transient crash")
    return {"service_healthy": True, "recovery_method": "restart"}


def rollback_deployment(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("rollback_blocked"):
        raise RuntimeError("Irreversible DB migration blocks rollback")
    return {"service_healthy": True, "recovery_method": "rollback"}


def analyze_error_logs(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_cause_hypothesized": True,
        "root_cause": "Memory leak in RequestHandler",
    }


def apply_hotfix(ws: dict[str, Any]) -> dict[str, Any]:
    return {"service_healthy": True, "recovery_method": "hotfix"}


def scale_horizontally(ws: dict[str, Any]) -> dict[str, Any]:
    if ws.get("scaling_blocked"):
        raise RuntimeError("Resource quota exceeded")
    return {"service_healthy": True, "recovery_method": "horizontal_scale"}


def failover_to_backup(ws: dict[str, Any]) -> dict[str, Any]:
    return {"service_healthy": True, "recovery_method": "failover"}


def notify_stakeholders(ws: dict[str, Any]) -> dict[str, Any]:
    return {"stakeholders_notified": True}


def _incident_actions() -> list[ActionSpec]:
    return [
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


_GOAL = GoalSpec(
    conditions={"service_healthy": True, "stakeholders_notified": True},
    replan_strategy=ReplanStrategy.ON_DEVIATION,
)

_BASE_WS: dict[str, Any] = {
    "incident_detected": True,
    "service_name": "payment-api",
}


class TestIncidentHappyPath:
    """Planner picks the cheapest recovery strategy when nothing fails."""

    def test_prefers_restart_as_cheapest(self) -> None:
        """Restart (cost=1) + notify (cost=1) = total 2, cheapest path."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["service_healthy"] is True
        assert result["world_state"]["stakeholders_notified"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert "restart_service" in successful
        assert "notify_stakeholders" in successful
        # Should not use expensive alternatives
        assert "rollback_deployment" not in successful
        assert "failover_to_backup" not in successful

    def test_recovery_method_in_world_state(self) -> None:
        """The recovery method used is recorded in world state."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL, world_state=_BASE_WS
        )
        assert result["world_state"]["recovery_method"] == "restart"


class TestIncidentEscalation:
    """Planner escalates through recovery strategies as cheaper ones fail."""

    def test_restart_fails_escalates_to_rollback(self) -> None:
        """Restart fails → planner escalates to rollback (next cheapest)."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "restart_ineffective": True},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # Should have escalated past restart
        assert "restart_service" not in successful
        # Rollback or scale (both cost=2) should be used
        assert "rollback_deployment" in successful or "scale_horizontally" in successful

    def test_restart_and_rollback_fail_uses_hotfix_path(self) -> None:
        """Restart + rollback fail → planner routes through analyze → hotfix."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL,
            world_state={
                **_BASE_WS,
                "restart_ineffective": True,
                "rollback_blocked": True,
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # With restart and rollback failed, should use scale, hotfix path, or failover
        assert (
            "scale_horizontally" in successful
            or ("analyze_error_logs" in successful and "apply_hotfix" in successful)
            or "failover_to_backup" in successful
        )

    def test_multiple_failures_still_reaches_goal(self) -> None:
        """Restart + rollback + scaling fail → analyze → hotfix or failover."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL,
            world_state={
                **_BASE_WS,
                "restart_ineffective": True,
                "rollback_blocked": True,
                "scaling_blocked": True,
            },
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # Must have found an alternative path
        assert (
            "analyze_error_logs" in successful and "apply_hotfix" in successful
        ) or "failover_to_backup" in successful

    def test_failure_details_tracked(self) -> None:
        """Failed recovery attempts appear in execution history."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "restart_ineffective": True},
        )

        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1
        assert any(
            "OOM" in (f.error or "") or "memory leak" in (f.error or "")
            for f in failures
        )

    def test_blacklisted_actions_reflect_failures(self) -> None:
        """Failed actions are blacklisted so the planner skips them on replan."""
        result = GoapGraph(actions=_incident_actions()).invoke(
            goal=_GOAL,
            world_state={**_BASE_WS, "restart_ineffective": True},
        )

        assert "restart_service" in result.get("blacklisted_actions", [])
