"""BEFORE — a condensed, hermetic version of ``executive-ai-assistant/eaia/main/graph.py``.

The upstream graph (190 lines, 13 nodes) triages an email, lets an LLM pick
a tool (``ResponseEmailDraft``, ``MeetingAssistant``, ``SendCalendarInvite``,
``Question``, ``Ignore``), sends the choice to a human through Agent Inbox,
and routes on the *shape of the last message*.  Three conditional routers
(``route_after_triage``, ``take_action``, ``enter_after_human``) each end in
``raise ValueError``; four interrupt handlers (395 lines in ``human_inbox.py``)
each re-implement the accept / edit / ignore / respond branches; and a
``bad_tool_name`` node exists to strip a stray ``:`` from tool names the
model sometimes emits.

This file keeps that control flow and swaps the LLM for a scripted drafter
and Gmail for flags, so it runs without keys.  ``run()`` scripts the human's
Agent Inbox answers.

    uv run python examples/screencast/email_assistant/before.py
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

FREE_SLOTS = ["2026-11-18T15:00:00", "2026-11-19T10:00:00"]


class State(TypedDict, total=False):
    email: dict[str, str]
    triage: str
    messages: Annotated[list[BaseMessage], add_messages]
    mangle_tool_name: bool
    bad_tool_names: int
    sent: bool
    read: bool
    notified: bool
    invite: dict[str, Any]


# --- nodes ---------------------------------------------------------------------


def triage_input(state: State) -> dict[str, Any]:
    text = f"{state['email']['subject']} {state['email']['body']}".lower()
    if "unsubscribe" in text or "digest" in text:
        triage = "no"
    elif "urgent" in text:
        triage = "notify"
    elif "meet" in text:
        triage = "email"
    else:
        triage = "email"
    print(f"  [triage_input] -> {triage}")
    return {"triage": triage}


def _last_tool_call(state: State) -> ToolCall:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage), "the last message must be the model's tool call"
    return last.tool_calls[0]


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}]
    )


def draft_response(state: State) -> dict[str, Any]:
    """Scripted stand-in for the LLM: pick the next tool from what has happened."""
    seen = {m.name for m in state.get("messages", []) if isinstance(m, ToolMessage)}
    text = f"{state['email']['subject']} {state['email']['body']}".lower()
    if "meet" in text and "MeetingAssistant" not in seen:
        call = _tool_call("MeetingAssistant", {"request": "30 minutes next week"})
    elif "meet" in text and "SendCalendarInvite" not in seen:
        call = _tool_call(
            "SendCalendarInvite",
            {
                "title": "Q3 roadmap review",
                "start_time": FREE_SLOTS[0],
                "emails": [state["email"]["from"]],
            },
        )
    else:
        content = f"Hi, {'the invite is on its way' if 'meet' in text else 'the limit is 600 requests per minute'}."
        call = _tool_call(
            "ResponseEmailDraft", {"content": content, "new_recipients": []}
        )
    if state.get("mangle_tool_name") and not state.get("bad_tool_names"):
        call.tool_calls[0]["name"] += ":"  # the quirk bad_tool_name exists for
    print(f"  [draft_response] tool call {call.tool_calls[0]['name']}")
    return {"messages": [call]}


def bad_tool_name(state: State) -> dict[str, Any]:
    call = _last_tool_call(state)
    message = f"Could not find tool with name `{call['name']}`. Make sure you are calling one of the allowed tools!"
    print(f"  [bad_tool_name] {message}")
    call["name"] = call["name"].replace(":", "")
    return {
        "messages": [
            ToolMessage(content=message, tool_call_id=call["id"], name="bad_tool_name")
        ],
        "bad_tool_names": state.get("bad_tool_names", 0) + 1,
    }


def find_meeting_time(state: State) -> dict[str, Any]:
    call = _last_tool_call(state)
    print(f"  [find_meeting_time] {FREE_SLOTS}")
    return {
        "messages": [
            ToolMessage(
                content=str(FREE_SLOTS),
                tool_call_id=call["id"],
                name="MeetingAssistant",
            )
        ]
    }


def _ask_human(state: State, kind: str) -> dict[str, Any]:
    """One Agent Inbox round trip (upstream has four copies of this function)."""
    call = _last_tool_call(state)
    request = {
        "action_request": {"action": call["name"], "args": call["args"]},
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": f"{kind} for {state['email']['subject']!r}",
    }
    response = interrupt([request])[0]
    print(f"  [{kind}] inbox response: {response['type']}")
    if response["type"] == "response":
        return {
            "messages": [
                ToolMessage(
                    content=str(response["args"]),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            ]
        }
    if response["type"] == "ignore":
        return {"messages": [_tool_call("Ignore", {"ignore": True})]}
    if response["type"] == "edit":
        return {"messages": [_tool_call(call["name"], response["args"]["args"])]}
    if response["type"] == "accept":
        return {}
    raise ValueError(f"Unexpected response: {response}")


def send_email_draft(state: State) -> dict[str, Any]:
    return _ask_human(state, "send_email_draft")


def send_cal_invite(state: State) -> dict[str, Any]:
    return _ask_human(state, "send_cal_invite")


def notify(state: State) -> dict[str, Any]:
    request = {
        "action_request": {"action": "Notify", "args": {}},
        "description": state["email"]["subject"],
    }
    response = interrupt([request])[0]
    print(f"  [notify] inbox response: {response['type']}")
    if response["type"] in ("ignore", "accept"):
        return {"notified": True}
    if response["type"] == "response":
        return {
            "notified": True,
            "messages": [HumanMessage(content=str(response["args"]))],
        }
    raise ValueError(f"Unexpected response: {response}")


def human_node(state: State) -> None:
    pass


def send_email_node(state: State) -> dict[str, Any]:
    call = _last_tool_call(state)
    print(
        f"  [send_email_node] -> {state['email']['from']}: {call['args']['content']!r}"
    )
    return {"sent": True}


def send_cal_invite_node(state: State) -> dict[str, Any]:
    call = _last_tool_call(state)
    print(f"  [send_cal_invite_node] {call['args']}")
    return {
        "invite": dict(call["args"]),
        "messages": [
            ToolMessage(
                content="Sent calendar invite!",
                tool_call_id=call["id"],
                name="SendCalendarInvite",
            )
        ],
    }


def mark_as_read_node(state: State) -> dict[str, Any]:
    print(f"  [mark_as_read_node] {state['email']['id']}")
    return {"read": True}


# --- routers (kept as upstream wrote them, ValueError fall-throughs included) --


def route_after_triage(
    state: State,
) -> Literal["draft_response", "mark_as_read_node", "notify"]:
    if state["triage"] == "email":
        return "draft_response"
    elif state["triage"] == "no":
        return "mark_as_read_node"
    elif state["triage"] == "notify":
        return "notify"
    elif state["triage"] == "question":
        return "draft_response"
    else:
        raise ValueError


def take_action(
    state: State,
) -> Literal[
    "send_email_draft",
    "mark_as_read_node",
    "find_meeting_time",
    "send_cal_invite",
    "bad_tool_name",
]:
    prediction = state["messages"][-1]
    assert isinstance(prediction, AIMessage)
    if len(prediction.tool_calls) != 1:
        raise ValueError
    name = prediction.tool_calls[0]["name"]
    if name == "ResponseEmailDraft":
        return "send_email_draft"
    elif name == "Ignore":
        return "mark_as_read_node"
    elif name == "MeetingAssistant":
        return "find_meeting_time"
    elif name == "SendCalendarInvite":
        return "send_cal_invite"
    else:
        return "bad_tool_name"


def enter_after_human(
    state: State,
) -> Literal[
    "mark_as_read_node", "draft_response", "send_email_node", "send_cal_invite_node"
]:
    messages = state.get("messages") or []
    if len(messages) == 0:
        if state["triage"] == "notify":
            return "mark_as_read_node"
        raise ValueError
    if isinstance(messages[-1], (ToolMessage, HumanMessage)):
        return "draft_response"
    execute = _last_tool_call(state)
    if execute["name"] == "ResponseEmailDraft":
        return "send_email_node"
    elif execute["name"] == "SendCalendarInvite":
        return "send_cal_invite_node"
    elif execute["name"] == "Ignore":
        return "mark_as_read_node"
    else:
        raise ValueError


def build_graph() -> StateGraph:
    b = StateGraph(State)
    for node in (
        triage_input,
        draft_response,
        bad_tool_name,
        find_meeting_time,
        send_email_draft,
        send_cal_invite,
        notify,
        human_node,
        send_email_node,
        send_cal_invite_node,
        mark_as_read_node,
    ):
        b.add_node(node)
    b.add_edge(START, "triage_input")
    b.add_conditional_edges("triage_input", route_after_triage)
    b.add_conditional_edges("draft_response", take_action)
    b.add_edge("bad_tool_name", "draft_response")
    b.add_edge("find_meeting_time", "draft_response")
    b.add_edge("send_email_draft", "human_node")
    b.add_edge("send_cal_invite", "human_node")
    b.add_edge("notify", "human_node")
    b.add_conditional_edges("human_node", enter_after_human)
    b.add_edge("send_email_node", "mark_as_read_node")
    b.add_edge("send_cal_invite_node", "draft_response")
    b.add_edge("mark_as_read_node", END)
    return b


def run(
    email: dict[str, str],
    responses: list[dict[str, Any]],
    mangle_tool_name: bool = False,
) -> dict[str, Any]:
    """Run one email through the graph, answering each Agent Inbox interrupt from ``responses``."""
    compiled = build_graph().compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": email["id"]}}
    queue = list(responses)
    result = compiled.invoke(
        {"email": email, "mangle_tool_name": mangle_tool_name, "bad_tool_names": 0},
        config=config,
    )
    while "__interrupt__" in result:
        if not queue:
            raise RuntimeError("the graph asked the human more times than scripted")
        result = compiled.invoke(Command(resume=[queue.pop(0)]), config=config)
    return dict(result)


if __name__ == "__main__":
    from examples.screencast.email_assistant.after import EMAILS

    scripts: dict[str, list[dict[str, Any]]] = {
        "newsletter": [],
        "urgent": [{"type": "accept"}],
        "question": [{"type": "accept"}],
        "meeting": [{"type": "accept"}, {"type": "accept"}],
    }
    for label, email in EMAILS.items():
        print("=" * 60)
        print(f"  BEFORE (hand-wired LangGraph) — {label.upper()}: {email['subject']}")
        print("=" * 60)
        final = run(email, scripts[label])
        print(
            f"  sent={final.get('sent', False)} read={final.get('read', False)} notified={final.get('notified', False)}"
        )
        print()
    print("=" * 60)
    print("  BEFORE — the model emits 'ResponseEmailDraft:' (bad_tool_name path)")
    print("=" * 60)
    final = run(EMAILS["question"], [{"type": "accept"}], mangle_tool_name=True)
    print(f"  bad_tool_names={final['bad_tool_names']} sent={final['sent']}")
    print()
    print("=" * 60)
    print("  BEFORE — an inbox response the handler does not expect")
    print("=" * 60)
    try:
        run(EMAILS["question"], [{"type": "shrug"}])
    except ValueError as exc:
        print(f"  ValueError: {exc}")
