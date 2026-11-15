"""AFTER, DISRUPTED — the human says no, fills the form wrong, or the process dies.

Three things the upstream graph handles with ``raise ValueError`` or not at
all.  Here each is an ordinary action failure or an ordinary checkpoint resume.

    uv run python examples/screencast/email_assistant/after_disrupted.py
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.screencast.email_assistant.after import (
    EMAILS,
    build_goal,
    build_graph,
    initial_world_state,
)


def _report(result: dict[str, Any]) -> None:
    print(
        f"  status: {result['status']}  replans: {result.get('replan_count', 0)}  blacklisted: {result.get('blacklisted_actions', [])}"
    )
    for h in result["execution_history"]:
        mark = "ok  " if h["success"] else "FAIL"
        print(
            f"    {mark} {h['action_name']}"
            + ("" if h["success"] else f" — {h['error']}")
        )


def denied_send() -> dict[str, Any]:
    print("=" * 60)
    print("  DISRUPTION 1 — the operator denies the send")
    print("=" * 60)
    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "denied"}}
    compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(EMAILS["question"])),
        config=config,
    )
    print("  [human] approval for send_email -> DENY ('tone is off')")
    result = compiled.invoke(
        Command(resume={"approved": False, "reason": "tone is off"}), config=config
    )
    _report(result)
    print()
    return result


def invalid_form() -> dict[str, Any]:
    print("=" * 60)
    print("  DISRUPTION 2 — the invite form fails validation")
    print("=" * 60)
    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "bad-form"}}
    compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(EMAILS["meeting"])),
        config=config,
    )
    print(
        "  [human] form for send_cal_invite -> {'title': 'no times'}  (start/end missing)"
    )
    result = compiled.invoke(Command(resume={"title": "no times"}), config=config)
    _report(result)
    print()
    return result


def crash_and_resume() -> dict[str, Any]:
    print("=" * 60)
    print("  DISRUPTION 3 — the process dies while the human is deciding")
    print("=" * 60)
    checkpointer = MemorySaver()  # stands in for Postgres / Redis / SQLite
    config: RunnableConfig = {"configurable": {"thread_id": "crash"}}
    first = build_graph()
    paused = first.compile(checkpointer=checkpointer).invoke(
        first.initial_state(build_goal(), initial_world_state(EMAILS["question"])),
        config=config,
    )
    print(
        f"  paused at: {paused['__interrupt__'][0].value['action']}  -- process exits here"
    )
    del first
    print("  new process, new graph object, same checkpointer and thread")
    second = build_graph()
    result = second.compile(checkpointer=checkpointer).invoke(
        Command(resume={"approved": True}), config=config
    )
    _report(result)
    print()
    return result


if __name__ == "__main__":
    denied_send()
    invalid_form()
    crash_and_resume()
