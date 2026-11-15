"""Integration tests for the executive-ai-assistant screencast.

The upstream `executive-ai-assistant <https://github.com/langchain-ai/executive-ai-assistant>`_
triages an email, drafts a reply, asks a human through four hand-written
interrupt handlers, and routes with three conditional routers whose
fall-through branches ``raise ValueError``.  ``after.py`` keeps the same
behaviour as a seven-action GOAP library: a sensor triages, preconditions pick
the path, ``require_human_approval`` gates the send, a Pydantic form gates the
calendar invite, and a denied send is re-planned around rather than raised.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from examples.screencast.email_assistant import before
from examples.screencast.email_assistant.after import (
    EMAILS,
    CalendarInvite,
    build_actions,
    build_goal,
    build_graph,
    initial_world_state,
)
from langgoap import GoapGraph

VALID_INVITE = {
    "title": "Q3 roadmap review",
    "start": "2026-11-18T15:00:00",
    "end": "2026-11-18T15:30:00",
    "attendees": ["dana@example.com"],
}


def _config(thread: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread}}


def _history(graph: GoapGraph, result: dict) -> list[str]:
    return [h.action_name for h in graph.rehydrate(result)["execution_history"]]


def _interrupt(result: dict) -> dict:
    interrupts = result.get("__interrupt__", [])
    assert interrupts, f"expected an interrupt, got keys {sorted(result)}"
    return interrupts[0].value


class TestTriagePicksThePath:
    """One sensor, one goal; the triage facts decide which actions run."""

    def test_newsletter_is_only_marked_as_read(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            graph.initial_state(
                build_goal(), initial_world_state(EMAILS["newsletter"])
            ),
            config=_config("newsletter"),
        )
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == ["mark_as_read"]

    def test_urgent_email_notifies_then_marks_read(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        result = compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["urgent"])),
            config=_config("urgent"),
        )
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == ["notify_user", "mark_as_read"]
        assert result["world_state"]["user_notified"] is True

    def test_question_is_drafted_approved_sent_and_marked_read(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("question")
        paused = compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["question"])),
            config=config,
        )
        payload = _interrupt(paused)
        assert payload["type"] == "goap_action_approval"
        assert payload["action"] == "send_email"
        assert "draft" in payload["world_state"]

        result = compiled.invoke(Command(resume={"approved": True}), config=config)
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == ["draft_reply", "send_email", "mark_as_read"]
        assert result["world_state"]["reply_sent"] is True

    def test_meeting_request_schedules_before_it_replies(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("meeting")
        paused = compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["meeting"])),
            config=config,
        )
        form = _interrupt(paused)
        assert form["type"] == "goap_action_form"
        assert form["action"] == "send_cal_invite"
        assert form["model_name"] == CalendarInvite.__name__
        assert set(form["form_schema"]["required"]) >= {"title", "start", "end"}
        assert form["world_state"]["free_slots"], "find_meeting_time ran first"

        paused = compiled.invoke(Command(resume=VALID_INVITE), config=config)
        approval = _interrupt(paused)
        assert approval["action"] == "send_email"
        assert "Q3 roadmap review" in approval["world_state"]["draft"]

        result = compiled.invoke(Command(resume=True), config=config)
        assert result["status"] == "goal_achieved"
        assert _history(graph, result) == [
            "find_meeting_time",
            "send_cal_invite",
            "draft_reply",
            "send_email",
            "mark_as_read",
        ]
        assert result["world_state"]["invite"] == VALID_INVITE


class TestHumanGates:
    def test_denied_send_is_replanned_to_an_escalation(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("denied")
        compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["question"])),
            config=config,
        )
        result = compiled.invoke(
            Command(resume={"approved": False, "reason": "tone is off"}), config=config
        )
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert "send_email" in result["blacklisted_actions"]
        assert result["replan_count"] == 1
        failures = [
            h for h in graph.rehydrate(result)["execution_history"] if not h.success
        ]
        assert [f.action_name for f in failures] == ["send_email"]
        assert "tone is off" in str(failures[0].error)
        assert _history(graph, result) == [
            "draft_reply",
            "send_email",
            "escalate_to_user",
            "mark_as_read",
        ]
        ws = result["world_state"]
        assert ws.get("reply_sent") is not True
        assert ws["escalated"] is True

    def test_invalid_invite_form_fails_the_action_and_escalates(self) -> None:
        """A form that fails validation is a deliberate failure, like a denial:
        the action is blacklisted and the planner falls back to escalation."""
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("bad-form")
        compiled.invoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["meeting"])),
            config=config,
        )
        result = compiled.invoke(Command(resume={"title": "no times"}), config=config)
        assert "__interrupt__" not in result
        assert result["status"] == "goal_achieved"
        assert result["blacklisted_actions"] == ["send_cal_invite"]
        history = graph.rehydrate(result)["execution_history"]
        failed = [h for h in history if not h.success]
        assert [h.action_name for h in failed] == ["send_cal_invite"]
        assert "human_form_invalid" in str(failed[0].error)
        assert [h.action_name for h in history] == [
            "find_meeting_time",
            "send_cal_invite",
            "escalate_to_user",
            "mark_as_read",
        ]
        assert "invite" not in result["world_state"]
        assert result["world_state"]["escalated"] is True


class TestCrashResume:
    def test_a_new_process_resumes_the_paused_thread(self) -> None:
        checkpointer = MemorySaver()
        config = _config("crash")
        first = build_graph()
        paused = first.compile(checkpointer=checkpointer).invoke(
            first.initial_state(build_goal(), initial_world_state(EMAILS["question"])),
            config=config,
        )
        assert _interrupt(paused)["action"] == "send_email"
        del first  # the process that asked is gone

        second = build_graph()
        result = second.compile(checkpointer=checkpointer).invoke(
            Command(resume={"approved": True}), config=config
        )
        assert result["status"] == "goal_achieved"
        assert _history(second, result) == ["draft_reply", "send_email", "mark_as_read"]


class TestAsyncParity:
    @pytest.mark.asyncio
    async def test_ainvoke_runs_the_same_gates(self) -> None:
        graph = build_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config("async")
        paused = await compiled.ainvoke(
            graph.initial_state(build_goal(), initial_world_state(EMAILS["meeting"])),
            config=config,
        )
        assert _interrupt(paused)["action"] == "send_cal_invite"
        paused = await compiled.ainvoke(Command(resume=VALID_INVITE), config=config)
        assert _interrupt(paused)["action"] == "send_email"
        result = await compiled.ainvoke(Command(resume=True), config=config)
        assert result["status"] == "goal_achieved"
        assert _history(graph, result)[-1] == "mark_as_read"


class TestActionLibrary:
    def test_seven_actions_cover_four_paths(self) -> None:
        names = {a.name for a in build_actions()}
        assert names == {
            "find_meeting_time",
            "send_cal_invite",
            "draft_reply",
            "send_email",
            "escalate_to_user",
            "notify_user",
            "mark_as_read",
        }
        gated = {a.name for a in build_actions() if a.require_human_approval}
        assert gated == {"send_email", "send_cal_invite"}


class TestBeforeIsFaithful:
    """The condensed hand-wired version keeps the upstream failure modes."""

    def test_accepted_draft_is_sent(self) -> None:
        state = before.run(EMAILS["question"], responses=[{"type": "accept"}])
        assert state["sent"] is True
        assert state["read"] is True

    def test_unexpected_inbox_response_raises(self) -> None:
        with pytest.raises(ValueError, match="Unexpected response"):
            before.run(EMAILS["question"], responses=[{"type": "shrug"}])

    def test_bad_tool_name_is_patched_and_retried(self) -> None:
        state = before.run(
            EMAILS["question"], responses=[{"type": "accept"}], mangle_tool_name=True
        )
        assert state["bad_tool_names"] == 1
        assert state["sent"] is True
