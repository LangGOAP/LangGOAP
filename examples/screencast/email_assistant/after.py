"""AFTER — the executive assistant as seven GOAP actions.

The upstream graph has thirteen nodes and three conditional routers whose
fall-through branches ``raise ValueError``; four hand-written interrupt
handlers each re-implement the accept / edit / ignore / respond branches.
Here a sensor triages once, preconditions pick the path, and the two
human gates are declared on the actions:

* ``send_email``       — ``require_human_approval=True`` (approve / deny)
* ``send_cal_invite``  — ``require_human_approval=CalendarInvite`` (typed form)

A denied send, or a declined invite form, is an ordinary action failure:
the observer blacklists the action and the planner routes through
``escalate_to_user`` instead.  An invite form that fails validation is an
input error, not a decision: the operator is asked again with the errors.
Everything is hermetic; no mailbox, no LLM.

    uv run python examples/screencast/email_assistant/after.py
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from langgoap import ActionSpec, GoalSpec, GoapGraph

EMAILS: dict[str, dict[str, str]] = {
    "newsletter": {
        "id": "m-1",
        "from": "news@weekly.example.com",
        "subject": "Weekly digest: ten tips for faster builds",
        "body": "Here is this week's digest. Unsubscribe at any time.",
    },
    "urgent": {
        "id": "m-2",
        "from": "cfo@example.com",
        "subject": "URGENT: board deck numbers",
        "body": "I need the Q3 numbers before noon today.",
    },
    "question": {
        "id": "m-3",
        "from": "dana@example.com",
        "subject": "Question about the API rate limits",
        "body": "Hi, what is the per-minute limit on the public API?",
    },
    "meeting": {
        "id": "m-4",
        "from": "dana@example.com",
        "subject": "Can we meet next week?",
        "body": "Could we find 30 minutes to review the Q3 roadmap?",
    },
}

FREE_SLOTS = ["2026-11-18T15:00:00", "2026-11-19T10:00:00"]


class CalendarInvite(BaseModel):
    """The typed form the operator fills in before an invite goes out."""

    title: str = Field(min_length=1)
    start: str = Field(description="ISO-8601 start time")
    end: str = Field(description="ISO-8601 end time")
    attendees: list[str] = Field(default_factory=list)


# --- the sensor: triage once --------------------------------------------------


def triage(ws: dict[str, Any]) -> dict[str, Any]:
    """Classify the email into one of four categories (keyword stand-in for the LLM).

    Runs before every planning pass; it only writes facts the first time so a
    re-plan never clobbers what the actions have established since.
    """
    if "triage" in ws:
        return {}
    email = ws["email"]
    text = f"{email['subject']} {email['body']}".lower()
    if "unsubscribe" in text or "digest" in text:
        category = "ignore"
    elif "urgent" in text:
        category = "notify"
    elif "meet" in text:
        category = "meeting"
    else:
        category = "respond"
    print(f"  [triage] {email['subject']!r} -> {category}")
    return {
        "triage": category,
        "needs_reply": category in ("respond", "meeting"),
        "needs_meeting": category == "meeting",
        "needs_notify": category == "notify",
        # A plain question can be answered now; a meeting request is answered
        # after the invite goes out (send_cal_invite sets this to True).
        "reply_context_ready": category == "respond",
        # Nothing to do for a newsletter: the follow-up is already "done".
        "follow_up_done": category == "ignore",
    }


# --- the actions --------------------------------------------------------------


def find_meeting_time(ws: dict[str, Any]) -> dict[str, Any]:
    print(f"  [find_meeting_time] free slots: {FREE_SLOTS}")
    return {"free_slots": FREE_SLOTS, "free_slots_known": True}


def send_cal_invite(ws: dict[str, Any]) -> dict[str, Any]:
    invite = ws["invite"]  # the validated CalendarInvite, stored by the form gate
    print(
        f"  [send_cal_invite] {invite['title']!r} {invite['start']} -> {invite['attendees']}"
    )
    return {"invite_sent": True, "reply_context_ready": True}


def draft_reply(ws: dict[str, Any]) -> dict[str, Any]:
    email = ws["email"]
    sender = email["from"].split("@")[0].title()
    invite = ws.get("invite")
    if invite:
        draft = (
            f"Hi {sender}, I have sent an invite for {invite['title']!r} starting "
            f"{invite['start']}. Looking forward to it."
        )
    else:
        draft = f"Hi {sender}, the public API allows 600 requests per minute per key."
    print(f"  [draft_reply] {draft}")
    return {"draft": draft, "draft_ready": True}


def send_email(ws: dict[str, Any]) -> dict[str, Any]:
    print(f"  [send_email] -> {ws['email']['from']}")
    return {"reply_sent": True, "follow_up_done": True}


def escalate_to_user(ws: dict[str, Any]) -> dict[str, Any]:
    what = "draft left in Drafts" if ws.get("draft_ready") else "no draft yet"
    print(f"  [escalate_to_user] {what}; owner pinged to handle the thread by hand")
    return {"escalated": True, "follow_up_done": True}


def notify_user(ws: dict[str, Any]) -> dict[str, Any]:
    print(f"  [notify_user] ping: {ws['email']['subject']!r}")
    return {"user_notified": True, "follow_up_done": True}


def mark_as_read(ws: dict[str, Any]) -> dict[str, Any]:
    print(f"  [mark_as_read] {ws['email']['id']}")
    return {"read": True, "email_handled": True}


def build_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="find_meeting_time",
            preconditions={"needs_meeting": True},
            effects={"free_slots_known": True},
            cost=1.0,
            execute=find_meeting_time,
        ),
        ActionSpec(
            name="send_cal_invite",
            preconditions={"free_slots_known": True},
            effects={"invite_sent": True, "reply_context_ready": True},
            cost=1.0,
            execute=send_cal_invite,
            require_human_approval=CalendarInvite,
            human_input_key="invite",
        ),
        ActionSpec(
            name="draft_reply",
            preconditions={"needs_reply": True, "reply_context_ready": True},
            effects={"draft_ready": True},
            cost=1.0,
            execute=draft_reply,
        ),
        ActionSpec(
            name="send_email",
            preconditions={"draft_ready": True},
            effects={"reply_sent": True, "follow_up_done": True},
            cost=1.0,
            execute=send_email,
            require_human_approval=True,
        ),
        ActionSpec(
            name="escalate_to_user",
            preconditions={"needs_reply": True},
            effects={"escalated": True, "follow_up_done": True},
            cost=5.0,  # dearer than the whole reply path, so only when that path is closed
            execute=escalate_to_user,
        ),
        ActionSpec(
            name="notify_user",
            preconditions={"needs_notify": True},
            effects={"user_notified": True, "follow_up_done": True},
            cost=1.0,
            execute=notify_user,
        ),
        ActionSpec(
            name="mark_as_read",
            preconditions={"follow_up_done": True},
            effects={"email_handled": True},
            cost=1.0,
            execute=mark_as_read,
        ),
    ]


def build_goal() -> GoalSpec:
    return GoalSpec(conditions={"email_handled": True})


def initial_world_state(email: dict[str, str]) -> dict[str, Any]:
    return {"email": dict(email)}


def build_graph(checkpointer: Any = None) -> GoapGraph:
    return GoapGraph(
        build_actions(), goal=build_goal(), sensors=[triage], checkpointer=checkpointer
    )


if __name__ == "__main__":
    answers = {
        "send_cal_invite": {
            "title": "Q3 roadmap review",
            "start": FREE_SLOTS[0],
            "end": "2026-11-18T15:30:00",
            "attendees": ["dana@example.com"],
        },
        "send_email": {"approved": True},
    }
    for label, email in EMAILS.items():
        print("=" * 60)
        print(f"  AFTER (GOAP) — {label.upper()}: {email['subject']}")
        print("=" * 60)
        graph = build_graph()
        result = graph.start(initial_world_state(email), thread_id=label)
        while (ask := graph.waiting(label)) is not None:
            print(
                f"  [human] {ask['type']} for {ask['action']} -> {answers[ask['action']]}"
            )
            result = graph.resume(answers[ask["action"]], thread_id=label)
        print(
            f"  status: {result.status}  steps: {[h.action_name for h in result['execution_history']]}"
        )
        print()
