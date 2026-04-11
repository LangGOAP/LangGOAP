"""Tests for async parity: ActionSpec.aexecute, goap_action async detection,
GoapAction.aexecute, GoapExecutor.acall, and GoapGraph.ainvoke.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgoap.actions import ActionSpec, GoapAction, goap_action
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.graph.nodes import GoapExecutor, async_execute_action
from langgoap.graph.state import GoapState
from langgoap.types import ReplanStrategy
from tests.conftest import make_action as _action
from tests.conftest import make_plan as _make_plan

# ---------------------------------------------------------------------------
# ActionSpec async detection
# ---------------------------------------------------------------------------


class TestAsyncActionSpec:
    def test_is_async_execute_false_for_sync(self) -> None:
        action = ActionSpec(name="sync", execute=lambda ws: {})
        assert action.is_async_execute() is False

    def test_is_async_execute_true_for_async_execute(self) -> None:
        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {}

        action = ActionSpec(name="async", execute=async_fn)
        assert action.is_async_execute() is True

    def test_is_async_execute_true_for_aexecute_field(self) -> None:
        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {}

        action = ActionSpec(name="async", execute=lambda ws: {}, aexecute=async_fn)
        assert action.is_async_execute() is True

    def test_is_async_execute_false_for_no_execute(self) -> None:
        action = ActionSpec(name="noop")
        assert action.is_async_execute() is False

    def test_aexecute_field_stored(self) -> None:
        async def afn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        action = ActionSpec(name="test", aexecute=afn)
        assert action.aexecute is afn


# ---------------------------------------------------------------------------
# goap_action decorator async detection
# ---------------------------------------------------------------------------


class TestAsyncGoapActionDecorator:
    def test_sync_function_no_aexecute(self) -> None:
        @goap_action(effects={"done": True})
        def sync_action(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert sync_action.aexecute is None
        assert sync_action.execute is not None

    def test_async_function_sets_aexecute_not_execute(self) -> None:
        """Async-decorated actions must have aexecute set and execute=None.

        Storing an async function in `execute` would cause the sync executor
        to call it, receive a coroutine, silently discard it, and apply
        declared effects instead — corrupting runtime-computed state.
        """

        @goap_action(effects={"done": True})
        async def async_action(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert async_action.aexecute is not None
        # execute must be None so the sync executor never sees the coroutine fn
        assert async_action.execute is None

    def test_async_function_via_sync_invoke_fails_with_clear_error(self) -> None:
        """Using an async @goap_action with sync invoke() fails cleanly.

        Regression test: previously the coroutine was silently leaked and
        declared effects were applied instead of the actual return value.

        The executor now raises a RuntimeError that is caught by the failure
        handler; the graph terminates with status='failed' and the error
        message appears in the execution history — never silently succeeds
        with wrong data.
        """

        @goap_action(effects={"done": True, "answer": "declared"})
        async def async_act(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True, "answer": "runtime_computed"}

        graph = GoapGraph(actions=[async_act])
        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}, max_replans=1),
            world_state={},
        )

        # Must NOT silently achieve the goal via declared-effects fallback
        assert result["status"] != "goal_achieved"
        # Every execution attempt records the actionable error
        errors = [h.error or "" for h in result["execution_history"] if not h.success]
        assert all(
            "async-only" in e or "async execute" in e for e in errors
        ), f"Expected async error messages, got: {errors}"
        # answer must NOT be the declared fallback — runtime data was not applied
        assert result["world_state"].get("answer") != "runtime_computed"

    def test_async_decorator_preserves_name(self) -> None:
        @goap_action(effects={"done": True}, name="custom_name")
        async def my_action(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert my_action.name == "custom_name"

    def test_async_decorator_default_name(self) -> None:
        @goap_action(effects={"done": True})
        async def my_async_action(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        assert my_async_action.name == "my_async_action"


# ---------------------------------------------------------------------------
# GoapAction base class async support
# ---------------------------------------------------------------------------


class TestAsyncGoapAction:
    async def test_default_aexecute_delegates_to_sync(self) -> None:
        """Default aexecute runs sync execute in executor."""

        class SyncAction(GoapAction):
            effects = {"done": True}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"done": True, "source": "sync"}

        action = SyncAction()
        result = await action.aexecute({"input": 1})
        assert result == {"done": True, "source": "sync"}

    async def test_override_aexecute(self) -> None:
        """Subclass can override aexecute for native async."""

        class AsyncAction(GoapAction):
            effects = {"done": True}

            async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
                await asyncio.sleep(0)  # simulate async work
                return {"done": True, "source": "async"}

        action = AsyncAction()
        result = await action.aexecute({"input": 1})
        assert result == {"done": True, "source": "async"}

    def test_to_spec_no_aexecute_when_not_overridden(self) -> None:
        """to_spec() does NOT set aexecute when subclass uses default."""

        class SyncOnlyAction(GoapAction):
            effects = {"done": True}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"done": True}

        spec = SyncOnlyAction().to_spec()
        assert spec.aexecute is None

    def test_to_spec_sets_aexecute_when_overridden(self) -> None:
        """to_spec() sets aexecute when subclass overrides it."""

        class CustomAsyncAction(GoapAction):
            effects = {"done": True}

            async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"done": True}

        spec = CustomAsyncAction().to_spec()
        assert spec.aexecute is not None
        assert spec.is_async_execute() is True


# ---------------------------------------------------------------------------
# _async_execute_action helper
# ---------------------------------------------------------------------------


class TestAsyncExecuteAction:
    async def test_prefers_aexecute(self) -> None:
        """aexecute is preferred over execute."""
        calls: list[str] = []

        async def afn(ws: dict[str, Any]) -> dict[str, Any]:
            calls.append("async")
            return {"done": True}

        def sfn(ws: dict[str, Any]) -> dict[str, Any]:
            calls.append("sync")
            return {"done": True}

        action = ActionSpec(name="test", execute=sfn, aexecute=afn)
        result = await async_execute_action(action, {})
        assert result == {"done": True}
        assert calls == ["async"]

    async def test_awaits_async_execute(self) -> None:
        """When execute is a coroutine function, it's awaited."""

        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"result": 42}

        action = ActionSpec(name="test", execute=async_fn)
        result = await async_execute_action(action, {})
        assert result == {"result": 42}

    async def test_wraps_sync_execute(self) -> None:
        """Sync execute is run in executor."""

        def sync_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"result": 99}

        action = ActionSpec(name="test", execute=sync_fn)
        result = await async_execute_action(action, {})
        assert result == {"result": 99}

    async def test_returns_none_when_no_execute(self) -> None:
        """No execute callable → returns None (effects applied by caller)."""
        action = ActionSpec(name="noop", effects={"done": True})
        result = await async_execute_action(action, {})
        assert result is None


# ---------------------------------------------------------------------------
# GoapExecutor.acall
# ---------------------------------------------------------------------------


class TestAsyncGoapExecutor:
    async def test_acall_with_async_action(self) -> None:
        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"a": True}

        action = ActionSpec(name="async_act", effects={"a": True}, aexecute=async_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = await executor.acall(state)

        assert result["world_state"]["a"] is True
        assert result["current_step"] == 1
        assert result["execution_history"][0].success is True

    async def test_acall_with_sync_action(self) -> None:
        """Sync actions work through acall via run_in_executor."""
        action = _action("sync_act", eff={"a": True})
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = await executor.acall(state)

        assert result["world_state"]["a"] is True
        assert result["current_step"] == 1

    async def test_acall_aexecute_preferred_over_execute(self) -> None:
        """When both execute and aexecute exist, aexecute wins."""
        calls: list[str] = []

        def sync_fn(ws: dict[str, Any]) -> dict[str, Any]:
            calls.append("sync")
            return {"a": True}

        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            calls.append("async")
            return {"a": True}

        action = ActionSpec(
            name="dual", effects={"a": True}, execute=sync_fn, aexecute=async_fn
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        await executor.acall(state)
        assert calls == ["async"]

    async def test_acall_async_failure_handling(self) -> None:
        async def failing_fn(ws: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("async boom")

        action = ActionSpec(name="fail", effects={"x": True}, aexecute=failing_fn)
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = await executor.acall(state)

        assert result["status"] == "action_failed"
        assert result["execution_history"][0].success is False
        assert "async boom" in (result["execution_history"][0].error or "")

    async def test_acall_no_plan_short_circuit(self) -> None:
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": None,
            "current_step": 0,
            "status": "no_plan",
        }
        result = await executor.acall(state)
        assert result == {}

    async def test_acall_no_action_to_execute(self) -> None:
        executor = GoapExecutor()
        state: GoapState = {
            "world_state": {},
            "plan": None,
            "current_step": 0,
        }
        result = await executor.acall(state)
        assert result["status"] == "error"

    async def test_acall_effect_validation(self) -> None:
        """acall calls effect_validator just like sync __call__."""

        def always_fail(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return False

        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True}

        action = ActionSpec(
            name="validated",
            effects={"done": True},
            aexecute=async_fn,
            effect_validator=always_fail,
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = await executor.acall(state)

        assert result["status"] == "action_failed"
        assert "validation failed" in (result["execution_history"][0].error or "")

    async def test_acall_validator_rejection_rolls_back_world_state(self) -> None:
        """Async path mirrors the sync rollback contract on validator rejection."""

        def always_fail(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return False

        async def async_fn(ws: dict[str, Any]) -> dict[str, Any]:
            return {"done": True, "side_effect": "leaked"}

        action = ActionSpec(
            name="validated",
            effects={"done": True},
            aexecute=async_fn,
            effect_validator=always_fail,
        )
        plan_obj = _make_plan(action)
        executor = GoapExecutor()

        state: GoapState = {
            "world_state": {"prior": "value"},
            "plan": plan_obj,
            "current_step": 0,
        }
        result = await executor.acall(state)

        assert result["world_state"] == {"prior": "value"}
        assert "done" not in result["world_state"]
        assert "side_effect" not in result["world_state"]
        history_entry = result["execution_history"][0]
        assert history_entry.state_after.get("done") is True
        assert history_entry.state_after.get("side_effect") == "leaked"
        assert history_entry.state_before == {"prior": "value"}


# ---------------------------------------------------------------------------
# GoapGraph.ainvoke
# ---------------------------------------------------------------------------


class TestAsyncGoapGraph:
    async def test_ainvoke_achieves_goal(self) -> None:
        """Basic ainvoke with sync actions achieves goal."""
        actions = [
            _action("step1", eff={"a": True}),
            _action("step2", pre={"a": True}, eff={"b": True}),
        ]
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"b": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["b"] is True

    async def test_ainvoke_with_async_actions(self) -> None:
        """ainvoke properly awaits async action callables."""

        async def async_gather(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"has_data": True}

        async def async_process(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"done": True}

        actions = [
            ActionSpec(
                name="gather", effects={"has_data": True}, aexecute=async_gather
            ),
            ActionSpec(
                name="process",
                preconditions={"has_data": True},
                effects={"done": True},
                aexecute=async_process,
            ),
        ]
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["done"] is True

    async def test_ainvoke_mixed_sync_async(self) -> None:
        """Mixed sync/async action chain works through ainvoke."""

        async def async_step(ws: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"b": True}

        actions = [
            _action("sync_step", eff={"a": True}),
            ActionSpec(
                name="async_step",
                preconditions={"a": True},
                effects={"b": True},
                aexecute=async_step,
            ),
        ]
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"b": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["a"] is True
        assert result["world_state"]["b"] is True

    async def test_ainvoke_replanning_with_async(self) -> None:
        """ainvoke handles replanning when async action fails."""
        fail_count = 0

        async def sometimes_fails(ws: dict[str, Any]) -> dict[str, Any]:
            nonlocal fail_count
            fail_count += 1
            if fail_count == 1:
                raise RuntimeError("transient async failure")
            return {"done": True}

        actions = [
            ActionSpec(
                name="flaky",
                effects={"done": True},
                aexecute=sometimes_fails,
            ),
        ]
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        failures = [h for h in result["execution_history"] if not h.success]
        assert len(failures) >= 1

    async def test_ainvoke_matches_invoke_for_sync(self) -> None:
        """ainvoke produces the same result as invoke for sync-only actions."""
        actions = [
            _action("step1", eff={"a": True}),
            _action("step2", pre={"a": True}, eff={"b": True}),
        ]
        graph = GoapGraph(actions=actions)
        goal = GoalSpec(conditions={"b": True})

        sync_result = graph.invoke(goal=goal, world_state={})
        async_result = await graph.ainvoke(goal=goal, world_state={})

        assert sync_result["status"] == async_result["status"]
        assert sync_result["world_state"] == async_result["world_state"]

    async def test_ainvoke_compiled_directly(self) -> None:
        """compiled.ainvoke() works directly (power-user API)."""
        actions = [_action("act", eff={"done": True})]
        compiled = GoapGraph(actions=actions).compile()

        result = await compiled.ainvoke(
            {
                "goal": GoalSpec(conditions={"done": True}),
                "world_state": {},
            }
        )

        assert result["status"] == "goal_achieved"

    async def test_ainvoke_goap_action_subclass_with_aexecute(self) -> None:
        """GoapAction subclass with aexecute override works via ainvoke."""

        class AsyncFetch(GoapAction):
            preconditions: dict[str, Any] = {}
            effects = {"has_data": True}

            async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
                await asyncio.sleep(0)
                return {"has_data": True, "data": "fetched"}

        class SyncProcess(GoapAction):
            preconditions = {"has_data": True}
            effects = {"done": True}

            def execute(self, state: dict[str, Any]) -> dict[str, Any]:
                return {"done": True}

        actions = [AsyncFetch().to_spec(), SyncProcess().to_spec()]
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["data"] == "fetched"
        assert result["world_state"]["done"] is True
