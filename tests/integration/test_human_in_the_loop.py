"""Integration tests for human-in-the-loop via require_human_approval.

Verifies that ActionSpec.require_human_approval gates execution behind
LangGraph's interrupt()/Command(resume=...) mechanism, including:

- Approval continues execution normally
- Denial blacklists the action and triggers replanning
- Denial is decisive even with max_retries > 0
- Async parity via ainvoke
"""

from __future__ import annotations

import sys
import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph


def _make_actions(
    *, approval_on: str = "step_b", max_retries: int = 0
) -> list[ActionSpec]:
    """Build a 3-action linear chain: step_a → step_b → step_c.

    ``step_b`` requires human approval by default.  An alternative
    action ``step_b_alt`` reaches the same effect without approval,
    so the planner can route around a denied ``step_b``.
    """
    step_a = ActionSpec(
        name="step_a",
        preconditions={},
        effects={"a_done": True},
        cost=1.0,
        execute=lambda s: {"a_done": True},
    )
    step_b = ActionSpec(
        name="step_b",
        preconditions={"a_done": True},
        effects={"b_done": True},
        cost=1.0,
        execute=lambda s: {"b_done": True},
        require_human_approval=approval_on == "step_b",
        max_retries=max_retries,
    )
    step_b_alt = ActionSpec(
        name="step_b_alt",
        preconditions={"a_done": True},
        effects={"b_done": True},
        cost=5.0,  # higher cost so planner prefers step_b
        execute=lambda s: {"b_done": True},
    )
    step_c = ActionSpec(
        name="step_c",
        preconditions={"b_done": True},
        effects={"c_done": True},
        cost=1.0,
        execute=lambda s: {"c_done": True},
    )
    return [step_a, step_b, step_b_alt, step_c]


def _make_goal() -> GoalSpec:
    return GoalSpec(conditions={"c_done": True})


def _thread_config(thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


class TestHumanApprovalSync:
    """Sync invoke tests for require_human_approval."""

    def test_approve_continues_execution(self) -> None:
        """Approving the interrupted action lets the plan complete."""
        actions = _make_actions()
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        # First invoke — executes step_a, then hits interrupt at step_b
        result = compiled.invoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        # The graph should have interrupted — check the interrupt metadata
        snapshot = compiled.get_state(config)
        assert snapshot.next == (
            "executor",
        ), f"Expected interrupt before executor, got {snapshot.next}"
        # Verify the interrupt payload includes the action name
        interrupts = [
            i for task in snapshot.tasks for i in getattr(task, "interrupts", [])
        ]
        assert len(interrupts) > 0, "Expected at least one interrupt"
        interrupt_val = interrupts[0].value
        assert interrupt_val["action"] == "step_b"
        assert interrupt_val["type"] == "goap_action_approval"

        # Resume with approval
        result = compiled.invoke(
            Command(resume={"approved": True}),
            config=config,
        )
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"

    def test_deny_triggers_replan_around_action(self) -> None:
        """Denying the action blacklists it and replans via step_b_alt."""
        actions = _make_actions()
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        # First invoke — hits interrupt at step_b
        compiled.invoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        # Deny
        result = compiled.invoke(
            Command(resume={"approved": False, "reason": "budget"}),
            config=config,
        )

        # Plan should have completed via step_b_alt
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"
        # step_b should be blacklisted
        assert "step_b" in result.get("blacklisted_actions", [])
        # Verify step_b_alt was used (check execution history)
        history = result.get("execution_history", [])
        action_names = [h.action_name for h in history]
        assert "step_b_alt" in action_names

    def test_deny_decisive_ignores_max_retries(self) -> None:
        """Denial blacklists immediately even with max_retries=3."""
        actions = _make_actions(max_retries=3)
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        # First invoke — hits interrupt at step_b
        compiled.invoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        # Deny — should blacklist despite max_retries=3
        result = compiled.invoke(
            Command(resume=False),
            config=config,
        )

        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"
        assert "step_b" in result.get("blacklisted_actions", [])

    def test_approve_with_true_resume(self) -> None:
        """Resuming with True is treated as approval."""
        actions = _make_actions()
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        compiled.invoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        result = compiled.invoke(
            Command(resume=True),
            config=config,
        )
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason=(
        "LangGraph interrupt() relies on get_config(), which cannot "
        "propagate RunnableConfig through asyncio tasks on Python 3.10. "
        "Upstream raises 'Python 3.11 or later required to use this in "
        "an async context'. Async HITL is therefore supported only on "
        "Python 3.11+."
    ),
)
class TestHumanApprovalAsync:
    """Async ainvoke tests for require_human_approval."""

    @pytest.mark.asyncio
    async def test_approve_async(self) -> None:
        """Async approval path mirrors the sync one."""
        actions = _make_actions()
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        await compiled.ainvoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        snapshot = await compiled.aget_state(config)
        assert snapshot.next == ("executor",)

        result = await compiled.ainvoke(
            Command(resume={"approved": True}),
            config=config,
        )
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"

    @pytest.mark.asyncio
    async def test_deny_async(self) -> None:
        """Async denial triggers replan via alternative action."""
        actions = _make_actions()
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        await compiled.ainvoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        result = await compiled.ainvoke(
            Command(resume={"approved": False}),
            config=config,
        )
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"
        assert "step_b" in result.get("blacklisted_actions", [])


class TestHumanApprovalNoApprovalNeeded:
    """Verify that actions without require_human_approval work unchanged."""

    def test_no_approval_no_interrupt(self) -> None:
        """Actions without require_human_approval execute without pause."""
        actions = _make_actions(approval_on="none")
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _thread_config()

        result = compiled.invoke(
            {"goal": _make_goal(), "world_state": {}},
            config=config,
        )

        # Plan should complete in a single invoke — no interrupt
        assert result["world_state"]["c_done"] is True
        assert result["status"] == "goal_achieved"
