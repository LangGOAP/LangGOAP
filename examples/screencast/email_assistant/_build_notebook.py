"""Generate ``email_assistant_screencast.ipynb`` (run from the repo root).

    uv run python examples/screencast/email_assistant/_build_notebook.py
    uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
        examples/screencast/email_assistant/email_assistant_screencast.ipynb

Markdown cells carry the narration; code cells import the three scripts so
the notebook and the scripts cannot drift apart. Everything is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "email_assistant_screencast.ipynb"
S = nbf.v4.new_markdown_cell
C = nbf.v4.new_code_cell

cells: list = []

cells.append(S("""# Executive AI assistant — before / after / after-disrupted

Notebook version of the three-script screencast in
[`examples/screencast/email_assistant/`](https://github.com/LangGOAP/LangGOAP/tree/main/examples/screencast/email_assistant).

**The task.** An email arrives. Ignore it, notify the owner, answer it, or schedule the meeting it asks
for. A human approves anything that leaves the mailbox. The human may say no.

**The original.** [`langchain-ai/executive-ai-assistant`](https://github.com/langchain-ai/executive-ai-assistant)
(`eaia/main/graph.py`: 190 lines, 13 nodes) triages with an LLM, has a second LLM pick a tool, sends the
choice to a human through Agent Inbox, and routes on the *shape of the last message*. Three conditional
routers each end in `raise ValueError`; four interrupt handlers (395 lines) each re-implement the
accept / edit / ignore / respond branches; a `bad_tool_name` node strips a stray `:` from tool names the
model sometimes emits.

| | Hand-wired (`before.py`) | GOAP (`after.py`) |
|---|---|---|
| Which path | 3 routers, 11 `if` branches, 4 `raise ValueError` | triage facts as preconditions |
| Human gate | 4 handlers × (accept / edit / ignore / respond) | `require_human_approval=True` or a Pydantic model |
| Human says no | an `Ignore` tool call is synthesised so a router can see it | action failure, blacklist, re-plan |
| Human fills the form wrong | `raise ValueError(f\"Unexpected response: ...\")` | asked again, errors in the payload |
| Bad model output | `bad_tool_name` patches the tool name and loops | no tool names to patch |
"""))

cells.append(S("""## 1. The hand-wired version

`before.py` keeps the upstream control flow: the same nodes, the same three routers with their
`ValueError` fall-throughs, one Agent Inbox round trip shared by the four handlers, the `bad_tool_name`
patch loop. The LLM is a scripted drafter; Gmail is a pair of flags. `run()` answers each interrupt
from a scripted list of inbox responses."""))

cells.append(C("""from examples.screencast.email_assistant import before
from examples.screencast.email_assistant.after import EMAILS

print(before.build_graph().compile().get_graph().draw_mermaid())"""))

cells.append(
    C(
        """for label, answers in {
    "newsletter": [],
    "urgent": [{"type": "accept"}],
    "question": [{"type": "accept"}],
    "meeting": [{"type": "accept"}, {"type": "accept"}],
}.items():
    print(f"--- {label}: {EMAILS[label]['subject']} ---")
    final = before.run(EMAILS[label], answers)
    print(f"  sent={final.get('sent', False)} read={final.get('read', False)} notified={final.get('notified', False)}\\n")"""
    )
)

cells.append(
    S(
        """Two upstream failure modes, reproduced. First the model emits `ResponseEmailDraft:` and the
`bad_tool_name` node patches it and loops back to the drafter. Then the inbox returns a response type
the handler does not know and the graph raises."""
    )
)

cells.append(
    C(
        """final = before.run(EMAILS["question"], [{"type": "accept"}], mangle_tool_name=True)
print(f"  bad_tool_names={final['bad_tool_names']} sent={final['sent']}\\n")

try:
    before.run(EMAILS["question"], [{"type": "shrug"}])
except ValueError as exc:
    print(f"  ValueError: {exc}")"""
    )
)

cells.append(S("""## 2. The GOAP version: triage facts are preconditions

A `FunctionalSensor` classifies the email once and writes four facts. Seven actions carry the paths as
preconditions; the goal is `email_handled`. The two human gates are declared on the actions:
`send_email` with `require_human_approval=True`, `send_cal_invite` with a Pydantic `CalendarInvite`
form whose validated answer lands in `world_state["invite"]`."""))

cells.append(
    C(
        """from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.screencast.email_assistant.after import (
    CalendarInvite, build_actions, build_goal, build_graph, initial_world_state,
)

for a in build_actions():
    gate = a.require_human_approval
    gate_s = "" if not gate else ("  [approve/deny]" if gate is True else f"  [form: {gate.__name__}]")
    print(f"{a.name:18s} cost={a.cost:<4} pre={dict(a.preconditions)}  eff={dict(a.effects)}{gate_s}")"""
    )
)

cells.append(
    S(
        """The same four emails. The newsletter is only marked as read; the urgent one pings the owner; the
question is drafted, approved and sent; the meeting request finds a slot, asks the operator to fill in
the invite form, then drafts a reply that mentions the invite, asks for approval, and sends."""
    )
)

cells.append(C("""ANSWERS = {
    "send_cal_invite": {
        "title": "Q3 roadmap review",
        "start": "2026-11-18T15:00:00",
        "end": "2026-11-18T15:30:00",
        "attendees": ["dana@example.com"],
    },
    "send_email": {"approved": True},
}


def handle(label: str, answers: dict = ANSWERS) -> dict:
    graph = build_graph()
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": label}}
    print(f"--- {label}: {EMAILS[label]['subject']} ---")
    result = compiled.invoke(
        graph.initial_state(build_goal(), initial_world_state(EMAILS[label])), config=config
    )
    while "__interrupt__" in result:
        ask = result["__interrupt__"][0].value
        print(f"  [human] {ask['type']} for {ask['action']} -> {answers[ask['action']]}")
        result = compiled.invoke(Command(resume=answers[ask["action"]]), config=config)
    print(f"  status: {result['status']}  steps: {[h['action_name'] for h in result['execution_history']]}\\n")
    return result


for label in EMAILS:
    handle(label)"""))

cells.append(
    S(
        """What the operator sees at each gate: the approval payload carries the action, its preconditions,
effects and the current world state; the form payload carries the model's JSON schema."""
    )
)

cells.append(C("""import json

graph = build_graph()
compiled = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "peek"}}
paused = compiled.invoke(
    graph.initial_state(build_goal(), initial_world_state(EMAILS["meeting"])), config=config
)
form = paused["__interrupt__"][0].value
print("type      :", form["type"])
print("action    :", form["action"])
print("model     :", form["model_name"])
print("required  :", form["form_schema"]["required"])
print("properties:", list(form["form_schema"]["properties"]))
paused = compiled.invoke(Command(resume=ANSWERS["send_cal_invite"]), config=config)
approval = paused["__interrupt__"][0].value
print()
print("type      :", approval["type"])
print("action    :", approval["action"])
print("effects   :", approval["effects"])
print("draft     :", approval["world_state"]["draft"])"""))

cells.append(
    S("""## 3. Disruption: the human says no, fills the form wrong, or the process dies

Upstream, an unexpected inbox response is a `ValueError`. Here a denial is an action failure with the
operator's reason attached. The observer blacklists `send_email`, the planner re-plans from the current
world state, and `escalate_to_user` (cost 5, dearer than the whole reply path) is now the cheapest way to
`follow_up_done`.

An invite form that fails Pydantic validation is a different thing: an input error, not a decision. The
operator is asked again, and the new interrupt payload carries the attempt number, the rejected payload
and the validation errors. Declining the form (`{"approved": False, "reason": ...}`) is a decision and
takes the same path as the denied send.""")
)

cells.append(C("""import logging, sys

logging.basicConfig(stream=sys.stdout, format="  log  %(message)s", level=logging.WARNING, force=True)

from examples.screencast.email_assistant import after_disrupted

denied = after_disrupted.denied_send()
bad_form = after_disrupted.invalid_form()"""))

cells.append(
    S(
        """The run is paused at `interrupt()` with the state in the checkpointer. A new process builds a new
`GoapGraph`, compiles it against the same checkpointer, and resumes the same thread."""
    )
)

cells.append(C("""resumed = after_disrupted.crash_and_resume()"""))

cells.append(
    S(
        """## 4. What replaced what

| Hand-wired piece | GOAP replacement |
|---|---|
| `route_after_triage` (4 branches + `ValueError`) | the sensor's facts are preconditions on `notify_user`, `draft_reply`, `find_meeting_time` |
| `take_action` (5 branches + `bad_tool_name`) | no tool names: the planner picks actions by effects |
| `enter_after_human` (5 branches + 2 `ValueError`) | the gate either lets the action run or fails it |
| 4 Agent Inbox handlers (accept / edit / ignore / respond) | `require_human_approval=True` and a Pydantic model |
| `Ignore` tool call synthesised on denial | blacklist + re-plan to `escalate_to_user` |
| `raise ValueError` on an unexpected inbox response | a bad form is asked again with the errors; a declined one is blacklisted |

The integration test
[`tests/integration/test_screencast_email_assistant.py`](https://github.com/LangGOAP/LangGOAP/blob/main/tests/integration/test_screencast_email_assistant.py)
pins the four paths, both gate payloads, the denial re-plan, the invalid-form re-ask, the declined form,
the crash-resume and the async path, and keeps the hand-wired version faithful to upstream (`ValueError` included)."""
    )
)


def main() -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT.relative_to(HERE.parents[2])}")


if __name__ == "__main__":
    main()
