"""AFTER — the weather emergency workflow as eight GOAP actions.

The upstream notebook routes with two hand-written routers: ``route_response``
picks a department from ``severity`` and ``disaster_type`` strings (its first
branch carries an unreachable ``"send_email_alert"`` literal), and
``verify_approval_router`` reads a flag set by a node that blocks on
``input()`` for low/medium alerts and auto-approves the rest.  A failed e-mail
is swallowed; a failed severity assessment is routed like a mild event.

Here a sensor grades the forecast once and writes facts.  The three
departments are actions whose preconditions are those facts; the goal is
``forecast_handled``.  ``approve_alert`` carries the human gate and the
auto-approval policy is one line in the sensor.  ``send_email_alert`` retries
connection errors through ``ActionQos``; when the relay is down for good the
action is blacklisted and ``send_sms_alert`` is the next cheapest route.  A
denied approval leaves no path to the goal, so a stuck handler relaxes it to
"log the decision".  Everything is hermetic; no API, no LLM, no SMTP.

    uv run python examples/screencast/weather_alert/after.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgoap import (
    ActionQos,
    ActionSpec,
    FunctionalSensor,
    FunctionalStuckHandler,
    GoalSpec,
    GoapGraph,
    StuckHandlerResult,
    StuckHandlingResultCode,
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "hurricane": {
        "city": "Port Mayfield",
        "weather": "heavy intensity rain",
        "wind_speed": 38.0,
        "temperature": 27.0,
        "humidity": 95,
        "pressure": 960,
    },
    "flood": {
        "city": "Riverton",
        "weather": "moderate rain",
        "wind_speed": 9.0,
        "temperature": 18.0,
        "humidity": 90,
        "pressure": 1002,
    },
    "heatwave": {
        "city": "Dry Gulch",
        "weather": "clear sky",
        "wind_speed": 3.0,
        "temperature": 41.0,
        "humidity": 12,
        "pressure": 1010,
    },
    "calm": {
        "city": "Fairweather",
        "weather": "few clouds",
        "wind_speed": 4.0,
        "temperature": 22.0,
        "humidity": 50,
        "pressure": 1015,
    },
}


def grade(weather: dict[str, Any]) -> tuple[str, str]:
    """Rule-based stand-in for the two LLM calls (disaster type, severity)."""
    if weather["wind_speed"] >= 33:
        return "Hurricane", "Critical"
    if weather["pressure"] < 1005 and "rain" in weather["weather"]:
        return "Flood", "Medium"
    if weather["temperature"] >= 38:
        return "Heatwave", "Medium"
    return "No Immediate Threat", "Low"


# --- the sensor: assess once ----------------------------------------------------


def assess(ws: dict[str, Any]) -> dict[str, Any]:
    """Grade the forecast and write the facts the actions' preconditions read.

    Runs before every planning pass; it writes only the first time so a
    re-plan never clobbers what the actions have established since.
    """
    if "disaster_type" in ws:
        return {}
    disaster_type, severity = grade(ws["weather"])
    severe = severity in ("Critical", "High")
    print(f"  [assess] {ws['weather']['city']}: {disaster_type} / {severity}")
    facts: dict[str, Any] = {
        "disaster_type": disaster_type,
        "severity": severity,
        "threat": disaster_type != "No Immediate Threat",
        "severe": severe,
        "flood_or_storm": any(w in disaster_type.lower() for w in ("flood", "storm")),
    }
    if severe:
        facts["approved"] = True  # policy: critical/high alerts go out without a human
    return facts


# --- fakes for the two side effects -------------------------------------------


@dataclass
class FakeSmtp:
    """Stand-in for the SMTP relay.  ``fail_first`` connection errors, or ``down`` for good."""

    fail_first: int = 0
    down: bool = False
    attempts: int = 0

    def send(self, subject: str) -> None:
        self.attempts += 1
        if self.down:
            raise RuntimeError("SMTP relay rejected the message (550)")
        if self.attempts <= self.fail_first:
            raise ConnectionError(f"SMTP connection reset (attempt {self.attempts})")
        print(f"  [smtp] sent {subject!r} on attempt {self.attempts}")


# --- the actions --------------------------------------------------------------


def _department(name: str, focus: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        plan = f"{name} plan for a {ws['disaster_type']} ({ws['severity']}) in {ws['weather']['city']}: {focus}."
        print(f"  [{name.lower().replace(' ', '_')}_response] {plan}")
        return {"response_plan": plan, "plan_ready": True, "department_notified": True}

    return execute


def approve_alert(ws: dict[str, Any]) -> dict[str, Any]:
    print("  [approve_alert] operator approved")
    return {"approved": True}


def _send_email(smtp: FakeSmtp) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        subject = f"Weather Alert: {ws['severity']} {ws['disaster_type']} in {ws['weather']['city']}"
        smtp.send(subject)
        return {"alert_sent": True, "alert_channel": "email", "forecast_handled": True}

    return execute


def send_sms_alert(ws: dict[str, Any]) -> dict[str, Any]:
    print(
        f"  [send_sms_alert] SMS: {ws['severity']} {ws['disaster_type']} in {ws['weather']['city']}"
    )
    return {"alert_sent": True, "alert_channel": "sms", "forecast_handled": True}


def withhold_alert(ws: dict[str, Any]) -> dict[str, Any]:
    reason = ws.get("withhold_reason", "operator declined")
    line = f"Alert for {ws['weather']['city']} withheld: {reason}"
    print(f"  [withhold_alert] {line}")
    return {"decision_log": line, "decision_logged": True}


def log_all_clear(ws: dict[str, Any]) -> dict[str, Any]:
    print(
        f"  [log_all_clear] {ws['weather']['city']}: no immediate threat, nothing sent"
    )
    return {"decision_log": "all clear", "forecast_handled": True}


def build_actions(smtp: FakeSmtp | None = None) -> list[ActionSpec]:
    smtp = smtp or FakeSmtp()
    return [
        ActionSpec(
            name="emergency_response",
            preconditions={"threat": True, "severe": True},
            effects={"plan_ready": True, "department_notified": True},
            cost=1.0,
            execute=_department(
                "Emergency", "immediate actions, shelters, evacuation routes"
            ),
        ),
        ActionSpec(
            name="public_works_response",
            preconditions={"threat": True, "severe": False, "flood_or_storm": True},
            effects={"plan_ready": True, "department_notified": True},
            cost=1.0,
            execute=_department("Public works", "drainage, road closures, sandbags"),
        ),
        ActionSpec(
            name="civil_defense_response",
            preconditions={"threat": True, "severe": False, "flood_or_storm": False},
            effects={"plan_ready": True, "department_notified": True},
            cost=1.0,
            execute=_department(
                "Civil defense", "public safety advisories, cooling centres"
            ),
        ),
        ActionSpec(
            name="approve_alert",
            preconditions={"plan_ready": True, "severe": False},
            effects={"approved": True},
            cost=1.0,
            execute=approve_alert,
            require_human_approval=True,
        ),
        ActionSpec(
            name="send_email_alert",
            preconditions={"plan_ready": True, "approved": True},
            effects={"alert_sent": True, "forecast_handled": True},
            cost=1.0,
            execute=_send_email(smtp),
            qos=ActionQos(
                max_attempts=3,
                backoff_initial_ms=1.0,
                jitter=0.0,
                retryable_exceptions=(ConnectionError,),
            ),
        ),
        ActionSpec(
            name="send_sms_alert",
            preconditions={"plan_ready": True, "approved": True},
            effects={"alert_sent": True, "forecast_handled": True},
            cost=3.0,  # dearer than e-mail, so only when e-mail is off the table
            execute=send_sms_alert,
        ),
        ActionSpec(
            name="withhold_alert",
            preconditions={"plan_ready": True},
            effects={"decision_logged": True},
            cost=1.0,
            execute=withhold_alert,
        ),
        ActionSpec(
            name="log_all_clear",
            preconditions={"threat": False},
            effects={"forecast_handled": True},
            cost=1.0,
            execute=log_all_clear,
        ),
    ]


def operator_withheld(state: dict[str, Any], reason: Any) -> StuckHandlerResult:
    """When the operator denied the alert there is no path to ``forecast_handled``.

    Relax the goal to "log the decision" and record the operator's reason.
    """
    if "approve_alert" not in state.get("blacklisted_actions", []):
        return StuckHandlerResult(
            code=StuckHandlingResultCode.NO_RESOLUTION, message="not an operator denial"
        )
    denial = next(
        (
            h
            for h in reversed(state.get("execution_history", []))
            if _name(h) == "approve_alert" and not _ok(h)
        ),
        None,
    )
    why = (
        str(_error(denial)).split("human_approval_denied: ")[-1] if denial else "denied"
    )
    return StuckHandlerResult(
        code=StuckHandlingResultCode.REPLAN,
        message=f"operator withheld the alert ({why}); log the decision instead",
        state_updates={"withhold_reason": why},
        new_goal=GoalSpec(conditions={"decision_logged": True}),
    )


def _name(h: Any) -> str:
    return h["action_name"] if isinstance(h, dict) else h.action_name


def _ok(h: Any) -> bool:
    return bool(h["success"] if isinstance(h, dict) else h.success)


def _error(h: Any) -> Any:
    return h["error"] if isinstance(h, dict) else h.error


def build_goal() -> GoalSpec:
    return GoalSpec(conditions={"forecast_handled": True})


def initial_world_state(weather: dict[str, Any]) -> dict[str, Any]:
    return {"weather": dict(weather)}


def build_graph(smtp: FakeSmtp | None = None) -> GoapGraph:
    return GoapGraph(
        build_actions(smtp),
        sensors=[FunctionalSensor("assess", assess)],
        stuck_handlers=[FunctionalStuckHandler("operator_withheld", operator_withheld)],
    )


if __name__ == "__main__":
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    for label, weather in SCENARIOS.items():
        print("=" * 60)
        print(f"  AFTER (GOAP) — {label.upper()}: {weather['city']}")
        print("=" * 60)
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": label}}
        result = compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(weather)),
            config=config,
        )
        if "__interrupt__" in result:
            ask = result["__interrupt__"][0].value
            print(f"  [human] {ask['type']} for {ask['action']} -> approve")
            result = compiled.invoke(Command(resume=True), config=config)
        print(
            f"  status: {result['status']}  steps: {[h['action_name'] for h in result['execution_history']]}"
        )
        print()
