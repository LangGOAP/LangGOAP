"""AFTER, DISRUPTED — the operator says no, the relay flaps, the relay dies.

Upstream handles the first with a dedicated node and router, swallows the
other two in a ``try/except`` and ends the graph as if the alert had gone out.
Here each is an ordinary action failure the observer and planner deal with.

    uv run python examples/screencast/weather_alert/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.screencast.weather_alert.after import (
    SCENARIOS,
    FakeSmtp,
    build_goal,
    build_graph,
    initial_world_state,
)


def _report(result: dict[str, Any]) -> None:
    goal = result["goal"]
    conditions = goal["conditions"] if isinstance(goal, dict) else dict(goal.conditions)
    print(
        f"  status: {result['status']}  goal: {conditions}  replans: {result.get('replan_count', 0)}"
        f"  blacklisted: {result.get('blacklisted_actions', [])}"
    )
    for h in result["execution_history"]:
        mark = "ok  " if h["success"] else "FAIL"
        print(
            f"    {mark} {h['action_name']}"
            + ("" if h["success"] else f" — {h['error']}")
        )
    ws = result["world_state"]
    print(
        f"  alert_sent={ws.get('alert_sent', False)} channel={ws.get('alert_channel')} decision_log={ws.get('decision_log')!r}"
    )


def denied_alert() -> dict[str, Any]:
    print("=" * 60)
    print("  DISRUPTION 1 — the operator denies the flood alert")
    print("=" * 60)
    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "denied"}}
    compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(SCENARIOS["flood"])),
        config=config,
    )
    print("  [human] approval for approve_alert -> DENY ('river gauge is stable')")
    result = compiled.invoke(
        Command(resume={"approved": False, "reason": "river gauge is stable"}),
        config=config,
    )
    _report(result)
    print()
    return result


def flaky_smtp() -> Any:
    print("=" * 60)
    print("  DISRUPTION 2 — the SMTP connection drops once (ActionQos retries)")
    print("=" * 60)
    smtp = FakeSmtp(fail_first=1)
    graph = build_graph(smtp=smtp)
    result = graph.invoke(
        goal=build_goal(), world_state=initial_world_state(SCENARIOS["hurricane"])
    )
    print(f"  smtp attempts: {smtp.attempts}")
    _report(_json(result))
    print()
    return result


def dead_smtp() -> Any:
    print("=" * 60)
    print("  DISRUPTION 3 — the relay rejects the message outright (SMS fallback)")
    print("=" * 60)
    smtp = FakeSmtp(down=True)
    graph = build_graph(smtp=smtp)
    result = graph.invoke(
        goal=build_goal(), world_state=initial_world_state(SCENARIOS["hurricane"])
    )
    print(f"  smtp attempts: {smtp.attempts}")
    _report(_json(result))
    print()
    return result


def _json(result: Any) -> dict[str, Any]:
    """``GoapGraph.invoke`` returns typed objects; ``_report`` reads the JSON shape."""
    out = dict(result)
    out["execution_history"] = [
        {"action_name": h.action_name, "success": h.success, "error": h.error}
        for h in result["execution_history"]
    ]
    return out


if __name__ == "__main__":
    denied_alert()
    flaky_smtp()
    dead_smtp()
