"""Integration tests for error recovery / blacklisting through the full GOAP loop.

Tests exercise GoapGraph.invoke() and ainvoke() with actions that fail,
verifying that the blacklist mechanism switches to alternatives, respects
max_retries, and applies the blacklist fallback correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from tests.conftest import make_action as _action

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_fail_count: dict[str, int] = {}


def _make_flaky(
    name: str,
    *,
    pre: dict[str, Any] | None = None,
    eff: dict[str, Any],
    fail_times: int,
    cost: float = 1.0,
    max_retries: int = 0,
) -> ActionSpec:
    """Create an action that fails the first *fail_times* calls, then succeeds."""
    key = name

    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        _fail_count[key] = _fail_count.get(key, 0) + 1
        if _fail_count[key] <= fail_times:
            raise RuntimeError(f"{name} failed (attempt {_fail_count[key]})")
        return dict(eff)

    return ActionSpec(
        name=name,
        preconditions=pre or {},
        effects=eff,
        cost=cost,
        execute=execute,
        max_retries=max_retries,
    )


@pytest.fixture(autouse=True)
def _reset_fail_count() -> None:
    """Reset the global fail counter between tests."""
    _fail_count.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlacklistingGoapLoop:
    def test_blacklist_switches_to_alternative(self) -> None:
        """Action A fails → blacklisted → planner selects Action B → goal achieved."""
        action_a = _make_flaky(
            "action_a",
            eff={"x": True},
            fail_times=999,  # always fails
            cost=1.0,
        )
        action_b = _action(
            "action_b",
            eff={"x": True},
            cost=5.0,
            execute=lambda ws: {"x": True},
        )

        graph = GoapGraph(actions=[action_a, action_b])
        result = graph.invoke(
            goal=GoalSpec(conditions={"x": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["x"] is True
        assert result["replan_count"] >= 1

        # Verify action_a was blacklisted
        assert "action_a" in result.get("blacklisted_actions", [])

        # Verify action_b was the one that succeeded
        successes = [h for h in result["execution_history"] if h.success]
        assert any(h.action_name == "action_b" for h in successes)

    def test_max_retries_allows_retry_before_blacklist(self) -> None:
        """Action with max_retries=2 fails 3x → blacklisted → alternative used."""
        action_a = _make_flaky(
            "action_a",
            eff={"x": True},
            fail_times=3,  # fails 3 times total
            cost=1.0,
            max_retries=2,  # allowed 2 retries (3 total attempts before blacklist)
        )
        action_b = _action(
            "action_b",
            eff={"x": True},
            cost=5.0,
            execute=lambda ws: {"x": True},
        )

        graph = GoapGraph(actions=[action_a, action_b])
        result = graph.invoke(
            goal=GoalSpec(conditions={"x": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"

        # action_a should have been tried 3 times before blacklisting
        a_failures = [
            h
            for h in result["execution_history"]
            if h.action_name == "action_a" and not h.success
        ]
        assert len(a_failures) == 3

        # action_a should be blacklisted
        assert "action_a" in result.get("blacklisted_actions", [])

        # Replanning count: 3 failures → 3 replans, plus 1 to switch to action_b
        assert result["replan_count"] >= 3

    def test_blacklist_fallback_retries_only_action(self) -> None:
        """Only one action can achieve goal → blacklisted → fallback clears blacklist → retried."""
        # Action fails once, then succeeds on second attempt
        action = _make_flaky(
            "only_action",
            eff={"x": True},
            fail_times=1,
            cost=1.0,
        )

        graph = GoapGraph(actions=[action])
        result = graph.invoke(
            goal=GoalSpec(conditions={"x": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["x"] is True
        assert result["replan_count"] >= 1

        # Should have at least one failure and one success
        failures = [h for h in result["execution_history"] if not h.success]
        successes = [h for h in result["execution_history"] if h.success]
        assert len(failures) >= 1
        assert len(successes) >= 1

        # Blacklist fallback should have cleared the blacklist and failure counts
        assert result.get("blacklisted_actions", []) == []
        assert result.get("action_failure_counts", {}) == {}

    async def test_blacklist_with_async_actions(self) -> None:
        """Blacklist switching works with ainvoke() and async actions."""

        async def failing_async(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("async fail")

        async def working_async(ws: dict[str, Any]) -> dict[str, Any]:
            return {"x": True}

        action_a = ActionSpec(
            name="async_fail",
            effects={"x": True},
            aexecute=failing_async,
            cost=1.0,
        )
        action_b = ActionSpec(
            name="async_ok",
            effects={"x": True},
            aexecute=working_async,
            cost=5.0,
        )

        graph = GoapGraph(actions=[action_a, action_b])
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"x": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert "async_fail" in result.get("blacklisted_actions", [])

        successes = [h for h in result["execution_history"] if h.success]
        assert any(h.action_name == "async_ok" for h in successes)

    def test_multiple_actions_blacklisted(self) -> None:
        """Two actions fail → both blacklisted → third action used."""
        action_a = _make_flaky(
            "action_a",
            eff={"x": True},
            fail_times=999,
            cost=1.0,
        )
        action_b = _make_flaky(
            "action_b",
            eff={"x": True},
            fail_times=999,
            cost=2.0,
        )
        action_c = _action(
            "action_c",
            eff={"x": True},
            cost=10.0,
            execute=lambda ws: {"x": True},
        )

        graph = GoapGraph(actions=[action_a, action_b, action_c])
        result = graph.invoke(
            goal=GoalSpec(conditions={"x": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        blacklisted = result.get("blacklisted_actions", [])
        assert "action_a" in blacklisted
        assert "action_b" in blacklisted

        successes = [h for h in result["execution_history"] if h.success]
        assert any(h.action_name == "action_c" for h in successes)

    def test_blacklist_preserves_world_state(self) -> None:
        """World state and execution history correct through blacklist-triggered replanning."""
        call_count = 0

        def flaky_step1(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("step1 failed")
            return {"a": True}

        step1_main = ActionSpec(
            name="step1_main",
            effects={"a": True},
            cost=1.0,
            execute=flaky_step1,
        )
        step1_alt = _action(
            "step1_alt",
            eff={"a": True},
            cost=3.0,
            execute=lambda ws: {"a": True},
        )
        step2 = _action(
            "step2",
            pre={"a": True},
            eff={"b": True},
            execute=lambda ws: {"b": True},
        )

        graph = GoapGraph(actions=[step1_main, step1_alt, step2])
        result = graph.invoke(
            goal=GoalSpec(conditions={"b": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["a"] is True
        assert result["world_state"]["b"] is True

        # step1_main should have failed once
        failures = [
            h
            for h in result["execution_history"]
            if h.action_name == "step1_main" and not h.success
        ]
        assert len(failures) == 1

        # step1_alt should have been used after blacklisting step1_main
        successes = [h for h in result["execution_history"] if h.success]
        success_names = [h.action_name for h in successes]
        assert "step2" in success_names
        assert "step1_alt" in success_names
