"""Integration tests for the weather-disaster-management screencast.

The upstream notebook (NirDiamant/GenAI_Agents,
``Weather_Disaster_Management_AI_AGENT.ipynb``) fetches weather, names a
disaster, grades its severity, routes to one department with ``route_response``
(whose first branch carries an unreachable ``"send_email_alert"`` literal where
a second target was evidently intended), asks a human for low/medium alerts,
and sends an email whose failure is swallowed.  ``after.py`` keeps the
behaviour as eight actions: a sensor grades once, preconditions pick the
department, ``approve_alert`` carries the human gate, the e-mail action
retries through ``ActionQos`` and falls back to SMS, and a denied approval is
handled by a stuck handler that relaxes the goal to "log the decision".
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.screencast.weather_alert import before
from examples.screencast.weather_alert.after import (
    SCENARIOS,
    FakeSmtp,
    build_actions,
    build_goal,
    build_graph,
    initial_world_state,
)
from langgoap import GoapGraph


def _config(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


def _history(graph: GoapGraph, result: dict) -> list[str]:
    return [h.action_name for h in graph.rehydrate(result)["execution_history"]]


def _interrupt(result: dict) -> dict:
    interrupts = result.get("__interrupt__", [])
    assert interrupts, f"expected an interrupt, got keys {sorted(result)}"
    return interrupts[0].value


def _run(
    scenario: str,
    *,
    smtp: FakeSmtp | None = None,
    resume: Any = None,
    thread: str = "t",
) -> tuple[GoapGraph, dict[str, Any]]:
    graph = build_graph(smtp=smtp)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = _config(thread)
    result = compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(SCENARIOS[scenario])),
        config=config,
    )
    if "__interrupt__" in result and resume is not None:
        result = compiled.invoke(Command(resume=resume), config=config)
    return graph, result


class TestSeverityPicksTheDepartment:
    def test_hurricane_is_severe_so_no_human_is_asked(self) -> None:
        graph, result = _run("hurricane")
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == ["emergency_response", "send_email_alert"]
        ws = result["world_state"]
        assert (ws["disaster_type"], ws["severity"]) == ("Hurricane", "Critical")
        assert ws["approved"] is True  # the sensor's auto-approval policy
        assert ws["department_notified"] is True and ws["alert_sent"] is True

    def test_flood_goes_to_public_works_and_waits_for_a_human(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("flood")
        paused = compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(SCENARIOS["flood"])),
            config=config,
        )
        gate = _interrupt(paused)
        assert gate["type"] == "goap_action_approval"
        assert gate["action"] == "approve_alert"
        assert gate["world_state"]["department_notified"] is True, "plan drafted first"
        assert "Public works" in gate["world_state"]["response_plan"]

        result = compiled.invoke(Command(resume=True), config=config)
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == [
            "public_works_response",
            "approve_alert",
            "send_email_alert",
        ]

    def test_heatwave_goes_to_civil_defense(self) -> None:
        graph, result = _run("heatwave", resume=True)
        assert _history(graph, result)[0] == "civil_defense_response"
        assert result["world_state"]["alert_sent"] is True

    def test_calm_weather_sends_nothing(self) -> None:
        graph, result = _run("calm")
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == ["log_all_clear"]
        assert "alert_sent" not in result["world_state"]


class TestHumanDenial:
    def test_denied_alert_is_logged_not_sent(self) -> None:
        graph, result = _run(
            "flood", resume={"approved": False, "reason": "river gauge is stable"}
        )
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert result["blacklisted_actions"] == ["approve_alert"]
        assert _history(graph, result) == [
            "public_works_response",
            "approve_alert",
            "withhold_alert",
        ]
        typed = graph.rehydrate(result)
        assert dict(typed["goal"].conditions) == {"decision_logged": True}
        failed = [h for h in typed["execution_history"] if not h.success]
        assert [h.action_name for h in failed] == ["approve_alert"]
        assert "river gauge is stable" in str(failed[0].error)
        ws = result["world_state"]
        assert "alert_sent" not in ws
        assert "river gauge is stable" in ws["decision_log"]


class TestDelivery:
    def test_flaky_smtp_is_retried_inside_the_action(self) -> None:
        smtp = FakeSmtp(fail_first=1)
        graph, result = _run("hurricane", smtp=smtp)
        assert smtp.attempts == 2
        assert _history(graph, result) == ["emergency_response", "send_email_alert"]
        assert result.get("replan_count", 0) == 0
        assert result["world_state"]["alert_channel"] == "email"

    def test_dead_smtp_falls_back_to_sms(self) -> None:
        smtp = FakeSmtp(down=True)
        graph, result = _run("hurricane", smtp=smtp)
        assert smtp.attempts == 1, "a hard failure is not retried"
        assert result["status"] == "goal_achieved"
        assert result["blacklisted_actions"] == ["send_email_alert"]
        assert _history(graph, result) == [
            "emergency_response",
            "send_email_alert",
            "send_sms_alert",
        ]
        assert result["world_state"]["alert_channel"] == "sms"
        assert result["world_state"]["alert_sent"] is True


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_ainvoke_runs_the_same_gate(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("async")
        paused = await compiled.ainvoke(
            graph.initial_state(build_goal(), initial_world_state(SCENARIOS["flood"])),
            config=config,
        )
        assert _interrupt(paused)["action"] == "approve_alert"
        result = await compiled.ainvoke(Command(resume=True), config=config)
        assert result["status"] == "goal_achieved"
        assert _history(graph, result)[-1] == "send_email_alert"


class TestActionLibrary:
    def test_eight_actions_and_one_gate(self) -> None:
        actions = build_actions()
        assert {a.name for a in actions} == {
            "emergency_response",
            "public_works_response",
            "civil_defense_response",
            "approve_alert",
            "send_email_alert",
            "send_sms_alert",
            "withhold_alert",
            "log_all_clear",
        }
        assert [a.name for a in actions if a.require_human_approval] == [
            "approve_alert"
        ]
        email = next(a for a in actions if a.name == "send_email_alert")
        assert email.qos is not None and email.qos.max_attempts == 3


class TestBeforeIsFaithful:
    """The condensed hand-wired version keeps the upstream behaviour."""

    def test_route_response_keeps_the_unreachable_literal(self) -> None:
        source = inspect.getsource(before.route_response)
        assert 'return "emergency_response"\n        "send_email_alert"' in source

    def test_severe_alert_is_auto_approved_and_sent(self) -> None:
        state = before.run("hurricane")
        assert state["human_asked"] is False
        assert state["alerts"], "an alert was sent"

    def test_rejected_alert_is_not_sent(self) -> None:
        state = before.run("flood", approve=False)
        assert state["human_asked"] is True
        assert state["alerts"] == []
        assert "not approved" in state["messages"][-1].content

    def test_failed_severity_assessment_is_routed_and_auto_approved(self) -> None:
        """Upstream: anything that is not 'low'/'medium' is auto-approved, so a
        hurricane whose severity call failed goes to civil defense and is
        e-mailed with no human in the loop."""
        state = before.run("hurricane", severity="Assessment Failed")
        assert state["department"] == "civil_defense_response"
        assert state["human_asked"] is False
        assert state["alerts"] == ["Email alert sent"]

    def test_smtp_failure_is_swallowed(self) -> None:
        state = before.run("hurricane", smtp_down=True)
        assert state["alerts"] == []
        assert "Failed to send email alert" in state["messages"][-1].content
