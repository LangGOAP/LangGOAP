"""The async graph path is async all the way down (issue #53).

``GoapPlanner.acall`` prefers a strategy's ``aplan`` when it has one, and
stuck handlers can be coroutine-based (``AsyncStuckHandler.ahandle_stuck``)
so a handler that needs to await an LLM or a store works under ``ainvoke``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.planner.strategy import AStarStrategy
from langgoap.stuck import (
    AsyncStuckHandler,
    FunctionalStuckHandler,
    MulticastStuckHandler,
    StuckHandler,
    StuckHandlerResult,
    StuckHandlingResultCode,
)


class _AsyncAwareStrategy:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0
        self._inner = AStarStrategy()

    def plan(self, start: Any, goal: Any, actions: Any, **kwargs: Any) -> Any:
        self.sync_calls += 1
        return self._inner.plan(start, goal, actions, **kwargs)

    async def aplan(self, start: Any, goal: Any, actions: Any, **kwargs: Any) -> Any:
        self.async_calls += 1
        await asyncio.sleep(0)
        return self._inner.plan(start, goal, actions, **kwargs)


def _chain() -> list[ActionSpec]:
    return [
        ActionSpec(name="a", effects={"a": True}, execute=lambda ws: {"a": True}),
        ActionSpec(
            name="b",
            preconditions={"a": True},
            effects={"done": True},
            execute=lambda ws: {"done": True},
        ),
    ]


class TestStrategyAplan:
    @pytest.mark.asyncio
    async def test_ainvoke_uses_aplan(self) -> None:
        strategy = _AsyncAwareStrategy()
        result = await GoapGraph(_chain(), strategy=strategy).ainvoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert strategy.async_calls >= 1
        assert strategy.sync_calls == 0

    def test_invoke_uses_plan(self) -> None:
        strategy = _AsyncAwareStrategy()
        result = GoapGraph(_chain(), strategy=strategy).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert strategy.sync_calls >= 1
        assert strategy.async_calls == 0


class _AsyncRelaxer:
    """Relaxes an unreachable goal after awaiting something."""

    name = "relax_async"

    def __init__(self) -> None:
        self.calls = 0

    async def ahandle_stuck(self, state: Any, reason: Any) -> StuckHandlerResult:
        self.calls += 1
        await asyncio.sleep(0)
        return StuckHandlerResult(
            code=StuckHandlingResultCode.REPLAN,
            message="relaxed",
            new_goal=GoalSpec(conditions={"a": True}),
        )


def _sync_relaxer() -> FunctionalStuckHandler:
    return FunctionalStuckHandler(
        name="relax_sync",
        fn=lambda state, reason: StuckHandlerResult(
            code=StuckHandlingResultCode.REPLAN,
            message="relaxed",
            new_goal=GoalSpec(conditions={"a": True}),
        ),
    )


UNREACHABLE = GoalSpec(conditions={"never": True})


class TestAsyncStuckHandlers:
    def test_protocols(self) -> None:
        assert isinstance(_AsyncRelaxer(), AsyncStuckHandler)
        assert not isinstance(_AsyncRelaxer(), StuckHandler)
        assert isinstance(_sync_relaxer(), StuckHandler)

    @pytest.mark.asyncio
    async def test_async_handler_resolves_under_ainvoke(self) -> None:
        handler = _AsyncRelaxer()
        result = await GoapGraph(_chain(), stuck_handlers=[handler]).ainvoke(
            goal=UNREACHABLE, world_state={}
        )
        assert handler.calls == 1
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["a"] is True

    @pytest.mark.asyncio
    async def test_sync_handler_still_works_under_ainvoke(self) -> None:
        result = await GoapGraph(_chain(), stuck_handlers=[_sync_relaxer()]).ainvoke(
            goal=UNREACHABLE, world_state={}
        )
        assert result["status"] == "goal_achieved"

    def test_async_only_handler_under_invoke_is_skipped_not_crashed(self) -> None:
        handler = _AsyncRelaxer()
        result = GoapGraph(_chain(), stuck_handlers=[handler]).invoke(
            goal=UNREACHABLE, world_state={}
        )
        assert result["status"] == "no_plan"
        assert handler.calls == 0

    @pytest.mark.asyncio
    async def test_multicast_mixes_both_kinds(self) -> None:
        class _Declines:
            name = "declines"

            async def ahandle_stuck(
                self, state: Any, reason: Any
            ) -> StuckHandlerResult:
                return StuckHandlerResult.no_resolution(
                    handler_name=self.name, message="no"
                )

        multicast = MulticastStuckHandler([_Declines(), _sync_relaxer()])
        outcome = await multicast.ahandle_stuck({}, None)
        assert outcome.code is StuckHandlingResultCode.REPLAN
        assert outcome.message == "relaxed"
