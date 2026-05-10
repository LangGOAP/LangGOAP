"""``ActionQos`` retry policy under ``ParallelGoapExecutor``.

The parallel executor runs independent wave actions concurrently; each
action's ``qos`` is honoured per-action.  A flaky action that succeeds
on its third attempt does not abort the wave — its sibling actions
still complete in parallel.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.graph.nodes import ParallelGoapExecutor
from langgoap.graph.state import GoapState
from langgoap.qos import ActionQos
from tests.conftest import make_plan as _make_plan


def _flaky_action(
    name: str, *, fail_first_n: int, qos: ActionQos
) -> tuple[ActionSpec, list[int]]:
    invocations: list[int] = [0]

    def execute(world_state: dict[str, Any]) -> dict[str, Any] | None:
        invocations[0] += 1
        if invocations[0] <= fail_first_n:
            raise TimeoutError(f"flake on attempt {invocations[0]}")
        return {f"{name}_done": True}

    action = ActionSpec(
        name=name,
        preconditions={},
        effects={f"{name}_done": True},
        cost=1.0,
        execute=execute,
        qos=qos,
    )
    return action, invocations


_FAST_QOS = ActionQos(
    max_attempts=3,
    idempotent=True,
    backoff_initial_ms=1.0,
    backoff_multiplier=1.0,
    jitter=0.0,
)


class TestParallelExecutorActionQos:
    def test_flaky_action_retried_within_wave(self) -> None:
        flaky, calls = _flaky_action("flaky", fail_first_n=2, qos=_FAST_QOS)
        sibling, sibling_calls = _flaky_action("sibling", fail_first_n=0, qos=_FAST_QOS)
        executor = ParallelGoapExecutor(actions=[flaky, sibling])
        state: GoapState = {
            "world_state": {},
            "plan": _make_plan(flaky, sibling),
            "current_step": 0,
        }

        result = executor(state)

        assert result.get("status") != "action_failed"
        assert calls[0] == 3, f"flaky should retry to success, got {calls[0]} calls"
        assert sibling_calls[0] == 1
        assert result["world_state"]["flaky_done"] is True
        assert result["world_state"]["sibling_done"] is True

    def test_attempts_exhausted_aborts_wave(self) -> None:
        always_fails, calls = _flaky_action(
            "always_fails", fail_first_n=99, qos=_FAST_QOS
        )
        executor = ParallelGoapExecutor(actions=[always_fails])
        state: GoapState = {
            "world_state": {},
            "plan": _make_plan(always_fails),
            "current_step": 0,
        }

        result = executor(state)

        assert result["status"] == "action_failed"
        assert calls[0] == 3, "must use up all attempts"

    @pytest.mark.asyncio
    async def test_async_flaky_action_retried_within_wave(self) -> None:
        invocations: list[int] = [0]

        async def aexecute(world_state: dict[str, Any]) -> dict[str, Any]:
            invocations[0] += 1
            if invocations[0] <= 2:
                raise TimeoutError(f"async flake {invocations[0]}")
            return {"a_done": True}

        flaky = ActionSpec(
            name="async_flaky",
            preconditions={},
            effects={"a_done": True},
            cost=1.0,
            aexecute=aexecute,
            execute=lambda ws: None,
            qos=_FAST_QOS,
        )

        async def aexecute_sib(ws: dict[str, Any]) -> dict[str, Any]:
            return {"b_done": True}

        sibling = ActionSpec(
            name="async_sib",
            preconditions={},
            effects={"b_done": True},
            cost=1.0,
            aexecute=aexecute_sib,
            execute=lambda ws: None,
            qos=ActionQos(max_attempts=1),
        )

        executor = ParallelGoapExecutor(actions=[flaky, sibling])
        state: GoapState = {
            "world_state": {},
            "plan": _make_plan(flaky, sibling),
            "current_step": 0,
        }

        result = await executor.acall(state)

        assert result.get("status") != "action_failed"
        assert invocations[0] == 3
        assert result["world_state"]["a_done"] is True
        assert result["world_state"]["b_done"] is True


class TestParallelExecutorRebind:
    def test_rebind_restores_qos_after_serializer_strip(self) -> None:
        """Simulate the post-checkpoint case where ActionSpec round-trips
        through the serializer with callable fields nullified.  Without
        ``actions=`` on the executor, qos retries would be lost; with it,
        ``_rebind`` restores the live ActionSpec by name."""
        live, calls = _flaky_action("live", fail_first_n=2, qos=_FAST_QOS)

        # The serialized variant has callables stripped (mirrors what
        # langgoap.serde does to ActionSpec.execute on the way through
        # msgpack).
        from types import MappingProxyType

        stripped = ActionSpec(
            name="live",
            preconditions=MappingProxyType({}),
            effects=MappingProxyType({"live_done": True}),
            cost=1.0,
            execute=None,  # serializer stripped this
            qos=None,  # and this
        )
        executor = ParallelGoapExecutor(actions=[live])
        state: GoapState = {
            "world_state": {},
            "plan": _make_plan(stripped),
            "current_step": 0,
        }

        result = executor(state)

        # Rebind picked up the LIVE action with its callable + qos, so
        # the retry actually fired.  Without rebind, the executor would
        # have called execute=None (no-op), found no error, applied
        # the declared effects, and reported success in 1 invocation.
        assert (
            calls[0] == 3
        ), f"_rebind must restore the live ActionSpec; got {calls[0]} calls"
        assert result["world_state"]["live_done"] is True
