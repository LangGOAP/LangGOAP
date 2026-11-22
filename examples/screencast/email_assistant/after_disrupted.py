"""AFTER, DISRUPTED — the human says no, fills the form wrong, or the process dies.

Three things the upstream graph handles with ``raise ValueError`` or not at
all.  Here a denial is an ordinary action failure, a bad form is asked
again with the errors, and a crash is an ordinary checkpoint resume.

    uv run python examples/screencast/email_assistant/after_disrupted.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from examples.screencast.email_assistant.after import (
    EMAILS,
    build_graph,
    initial_world_state,
)
from langgoap import GoapResult


def _report(result: GoapResult) -> None:
    print(
        f"  status: {result.status}  replans: {result['replan_count']}  blacklisted: {result['blacklisted_actions']}"
    )
    for h in result["execution_history"]:
        mark = "ok  " if h.success else "FAIL"
        print(f"    {mark} {h.action_name}" + ("" if h.success else f" — {h.error}"))


def denied_send() -> GoapResult:
    print("=" * 60)
    print("  DISRUPTION 1 — the operator denies the send")
    print("=" * 60)
    graph = build_graph()
    graph.start(initial_world_state(EMAILS["question"]), thread_id="denied")
    print("  [human] approval for send_email -> DENY ('tone is off')")
    result = graph.resume(
        {"approved": False, "reason": "tone is off"}, thread_id="denied"
    )
    _report(result)
    print()
    return result


def invalid_form() -> GoapResult:
    print("=" * 60)
    print("  DISRUPTION 2 — the invite form fails validation, then is declined")
    print("=" * 60)
    graph = build_graph()
    graph.start(initial_world_state(EMAILS["meeting"]), thread_id="bad-form")
    print(
        "  [human] form for send_cal_invite -> {'title': 'no times'}  (start/end missing)"
    )
    ask = graph.resume({"title": "no times"}, thread_id="bad-form").interrupt
    print(
        f"  asked again: attempt {ask['attempt']}, rejected {ask['rejected_payload']}"
    )
    for err in ask["validation_errors"]:
        print(f"    {'.'.join(map(str, err['loc']))}: {err['msg']}")
    print("  [human] form for send_cal_invite -> DECLINE ('not this week')")
    result = graph.resume(
        {"approved": False, "reason": "not this week"}, thread_id="bad-form"
    )
    _report(result)
    print()
    return result


def crash_and_resume() -> GoapResult:
    print("=" * 60)
    print("  DISRUPTION 3 — the process dies while the human is deciding")
    print("=" * 60)
    checkpointer = MemorySaver()  # stands in for Postgres / Redis / SQLite
    first = build_graph(checkpointer)
    paused = first.start(initial_world_state(EMAILS["question"]), thread_id="crash")
    print(f"  paused at: {paused.interrupt['action']}  -- process exits here")
    del first
    print("  new process, new graph object, same checkpointer and thread")
    second = build_graph(checkpointer)
    result = second.resume({"approved": True}, thread_id="crash")
    _report(result)
    print()
    return result


if __name__ == "__main__":
    denied_send()
    invalid_form()
    crash_and_resume()
