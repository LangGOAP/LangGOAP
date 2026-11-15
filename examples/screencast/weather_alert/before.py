"""BEFORE — a condensed, hermetic version of the GenAI_Agents weather-disaster graph.

Upstream: ``NirDiamant/GenAI_Agents`` →
``all_agents_tutorials/Weather_Disaster_Management_AI_AGENT.ipynb`` (Part 1,
"Real-Data Agent Graph").  The notebook fetches OpenWeatherMap data, asks a
local LLM to name a disaster type and grade its severity, logs, routes to one
department with ``route_response``, asks a human (blocking ``input()``) for
low/medium alerts and auto-approves the rest, then sends an e-mail or runs
``handle_no_approval``.

This file keeps the nodes, the two routers (``route_response`` verbatim,
including the unreachable ``"send_email_alert"`` literal in its first branch)
and the error handling (an SMTP failure is appended to ``messages`` and the
graph ends normally; a failed severity assessment is routed like a mild
event).  The weather API, the LLM, ``input()`` and SMTP are replaced by
fixtures and flags so it runs without keys.

    uv run python examples/screencast/weather_alert/before.py
"""

from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from examples.screencast.weather_alert.after import SCENARIOS, grade


class WeatherState(TypedDict, total=False):
    city: str
    weather_data: dict[str, Any]
    disaster_type: str
    severity: str
    response: str
    messages: list[BaseMessage]
    alerts: list[str]
    human_approved: bool
    # test hooks (upstream: live API, LLM, input(), SMTP)
    severity_override: str
    operator_says: str
    smtp_down: bool
    human_asked: bool
    department: str


def get_weather_data(state: WeatherState) -> WeatherState:
    weather = SCENARIOS[state["city"]]
    return {
        **state,
        "weather_data": dict(weather),
        "messages": state["messages"]
        + [
            SystemMessage(
                content=f"Weather data fetched successfully for {weather['city']}"
            )
        ],
    }


def analyze_disaster_type(state: WeatherState) -> WeatherState:
    disaster_type, _ = grade(state["weather_data"])
    return {
        **state,
        "disaster_type": disaster_type,
        "messages": state["messages"]
        + [SystemMessage(content=f"Disaster type identified: {disaster_type}")],
    }


def assess_severity(state: WeatherState) -> WeatherState:
    _, severity = grade(state["weather_data"])
    severity = (
        state.get("severity_override") or severity
    )  # upstream: "Assessment Failed" on an LLM error
    return {
        **state,
        "severity": severity,
        "messages": state["messages"]
        + [SystemMessage(content=f"Severity assessed as: {severity}")],
    }


def data_logging(state: WeatherState) -> WeatherState:
    print(
        f"  [data_logging] {state['weather_data']['city']}: {state['disaster_type']} / {state['severity']}"
    )
    return {
        **state,
        "messages": state["messages"] + [SystemMessage(content="Data logged")],
    }


def _plan(kind: str, focus: str) -> Callable[[WeatherState], WeatherState]:
    def node(state: WeatherState) -> WeatherState:
        response = f"{kind} plan for a {state['disaster_type']} ({state['severity']}) in {state['weather_data']['city']}: {focus}."
        print(f"  [{kind.lower().replace(' ', '_')}_response] {response}")
        return {
            **state,
            "response": response,
            "department": f"{kind.lower().replace(' ', '_')}_response",
            "messages": state["messages"]
            + [AIMessage(content=f"{kind} response plan generated")],
        }

    return node


emergency_response = _plan(
    "Emergency", "immediate actions, shelters, evacuation routes"
)
civil_defense_response = _plan(
    "Civil defense", "public safety advisories, cooling centres"
)
public_works_response = _plan("Public works", "drainage, road closures, sandbags")


def get_human_verification(state: WeatherState) -> WeatherState:
    """Upstream blocks on ``input()`` for low/medium and auto-approves the rest."""
    severity = state["severity"].strip().lower()
    if severity in ["low", "medium"]:
        approved = state.get("operator_says", "y") == "y"
        print(f"  [get_human_verification] operator says {'y' if approved else 'n'}")
        return {
            **state,
            "human_approved": approved,
            "human_asked": True,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"Human verification: {'Approved' if approved else 'Rejected'}"
                )
            ],
        }
    print(f"  [get_human_verification] auto-approved {severity} severity alert")
    return {
        **state,
        "human_approved": True,
        "messages": state["messages"]
        + [SystemMessage(content=f"Auto-approved {severity} severity alert")],
    }


def send_email_alert(state: WeatherState) -> WeatherState:
    """Upstream wraps ``smtplib`` in ``try/except`` and appends the error to ``messages``."""
    try:
        if state.get("smtp_down"):
            raise ConnectionRefusedError("[Errno 61] Connection refused")
        print(f"  [send_email_alert] e-mail sent for {state['weather_data']['city']}")
        return {
            **state,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"Successfully sent weather alert email for {state['weather_data']['city']}"
                )
            ],
            "alerts": state["alerts"] + ["Email alert sent"],
        }
    except Exception as e:
        print(f"  [send_email_alert] failed: {e} (graph continues)")
        return {
            **state,
            "messages": state["messages"]
            + [SystemMessage(content=f"Failed to send email alert: {str(e)}")],
        }


def handle_no_approval(state: WeatherState) -> WeatherState:
    print(
        "  [handle_no_approval] Verification was not approved by human, Email not sent"
    )
    message = (
        f"Alert not sent for {state['weather_data']['city']} - "
        f"Weather severity level '{state['severity']}' was deemed non-critical "
        f"by human operator and verification was not approved."
    )
    return {**state, "messages": state["messages"] + [SystemMessage(content=message)]}


# --- routers, as upstream wrote them -------------------------------------------


def route_response(
    state: WeatherState,
) -> Literal[
    "emergency_response",
    "send_email_alert",
    "civil_defense_response",
    "public_works_response",
]:
    """Route to appropriate department based on disaster type and severity"""
    disaster = state["disaster_type"].strip().lower()
    severity = state["severity"].strip().lower()

    if severity in ["critical", "high"]:
        return "emergency_response"
        "send_email_alert"
    elif "flood" in disaster or "storm" in disaster:
        return "public_works_response"
    else:
        return "civil_defense_response"


def verify_approval_router(
    state: WeatherState,
) -> Literal["send_email_alert", "handle_no_approval"]:
    """Route based on human approval decision"""
    return "send_email_alert" if state["human_approved"] else "handle_no_approval"


def build_graph() -> StateGraph:
    workflow = StateGraph(WeatherState)
    workflow.add_node("get_weather", get_weather_data)
    workflow.add_node("analyze_disaster", analyze_disaster_type)
    workflow.add_node("assess_severity", assess_severity)
    workflow.add_node("data_logging", data_logging)
    workflow.add_node("emergency_response", emergency_response)
    workflow.add_node("civil_defense_response", civil_defense_response)
    workflow.add_node("public_works_response", public_works_response)
    workflow.add_node("get_human_verification", get_human_verification)
    workflow.add_node("send_email_alert", send_email_alert)
    workflow.add_node("handle_no_approval", handle_no_approval)
    workflow.add_edge(START, "get_weather")
    workflow.add_edge("get_weather", "analyze_disaster")
    workflow.add_edge("analyze_disaster", "assess_severity")
    workflow.add_edge("assess_severity", "data_logging")
    workflow.add_conditional_edges("data_logging", route_response)
    workflow.add_edge("civil_defense_response", "get_human_verification")
    workflow.add_edge("public_works_response", "get_human_verification")
    workflow.add_conditional_edges("get_human_verification", verify_approval_router)
    workflow.add_edge("emergency_response", "send_email_alert")
    workflow.add_edge("send_email_alert", END)
    workflow.add_edge("handle_no_approval", END)
    return workflow


def run(
    scenario: str,
    *,
    approve: bool = True,
    severity: str | None = None,
    smtp_down: bool = False,
) -> dict[str, Any]:
    compiled = build_graph().compile()
    initial: WeatherState = {
        "city": scenario,
        "weather_data": {},
        "disaster_type": "",
        "severity": "",
        "response": "",
        "messages": [
            HumanMessage(content=f"Check the weather in {SCENARIOS[scenario]['city']}")
        ],
        "alerts": [],
        "human_approved": False,
        "operator_says": "y" if approve else "n",
        "smtp_down": smtp_down,
        "human_asked": False,
    }
    if severity is not None:
        initial["severity_override"] = severity
    return dict(compiled.invoke(initial))


if __name__ == "__main__":
    for label in SCENARIOS:
        print("=" * 60)
        print(
            f"  BEFORE (hand-wired LangGraph) — {label.upper()}: {SCENARIOS[label]['city']}"
        )
        print("=" * 60)
        final = run(label)
        print(f"  alerts={final['alerts']} human_asked={final['human_asked']}")
        print()
    print("=" * 60)
    print("  BEFORE — operator rejects a flood alert")
    print("=" * 60)
    final = run("flood", approve=False)
    print(f"  alerts={final['alerts']}  last message: {final['messages'][-1].content}")
    print()
    print("=" * 60)
    print("  BEFORE — the severity LLM call fails during a hurricane")
    print("=" * 60)
    final = run("hurricane", severity="Assessment Failed")
    print(f"  routed to {final['department']}; human_asked={final['human_asked']}")
    print()
    print("=" * 60)
    print("  BEFORE — SMTP is down during a hurricane")
    print("=" * 60)
    final = run("hurricane", smtp_down=True)
    print(f"  alerts={final['alerts']}  last message: {final['messages'][-1].content}")
