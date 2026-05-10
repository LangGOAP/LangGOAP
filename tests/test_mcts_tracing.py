"""Tests for MCTS search-tree tracing via :class:`PlanningTracer`.

The MCTS strategy shares the same tracer hooks as A*
(:meth:`PlanningTracer.on_search_expand` /
:meth:`PlanningTracer.on_search_complete`) with a semantic remap so a
single downstream panel can consume both planners' event streams:

* ``g`` = node ``visits``
* ``h`` = node ``ucb1`` score at selection time
* ``f`` = node running-mean ``value`` (reward)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
)
from langgoap.state import PlanningState


class _RecordingTracer:
    def __init__(self) -> None:
        self.expands: list[dict[str, Any]] = []
        self.completes: list[dict[str, Any]] = []

    def on_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.expands.append(
            {
                "node_id": node_id,
                "parent_id": parent_id,
                "action_name": action_name,
                "g": g,
                "h": h,
                "f": f,
            }
        )

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.completes.append({"nodes_explored": nodes_explored, "found": found})

    async def aon_search_expand(
        self,
        node_id: int,
        state: Any,
        g: float,
        h: float,
        f: float,
        parent_id: int | None,
        action_name: str | None,
    ) -> None:
        self.on_search_expand(node_id, state, g, h, f, parent_id, action_name)

    async def aon_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.on_search_complete(nodes_explored, duration_ms, found)


def _toy_domain() -> tuple[PlanningState, GoalSpec, list[ActionSpec]]:
    start = PlanningState.from_dict({"at": "a", "done": False})
    goal = GoalSpec(conditions={"done": True})
    actions = [
        ActionSpec(
            name="go_b",
            preconditions={"at": "a"},
            effects={"at": "b"},
            cost=1.0,
        ),
        ActionSpec(
            name="finish",
            preconditions={"at": "b"},
            effects={"done": True},
            cost=1.0,
        ),
        ActionSpec(
            name="noop",
            preconditions={"at": "a"},
            effects={"at": "a"},
            cost=1.0,
        ),
    ]
    return start, goal, actions


def test_mcts_emits_search_expand_events() -> None:
    start, goal, actions = _toy_domain()
    tracer = _RecordingTracer()
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=20, wall_clock_ms=0.0, seed=7),
        tracing=MCTSTracingConfig(tracer=tracer, record_expansions=True),
    )
    plan = strategy.plan(start, goal, actions)
    assert plan is not None
    assert len(tracer.expands) > 0
    root_evt = tracer.expands[0]
    assert root_evt["parent_id"] is None
    assert root_evt["action_name"] in {a.name for a in actions}
    for e in tracer.expands:
        assert e["g"] >= 0  # visits
        assert -1.0 <= e["f"] <= 1.0  # mean value


def test_mcts_emits_single_search_complete() -> None:
    start, goal, actions = _toy_domain()
    tracer = _RecordingTracer()
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=1),
        tracing=MCTSTracingConfig(tracer=tracer, record_expansions=True),
    )
    strategy.plan(start, goal, actions)
    assert len(tracer.completes) == 1
    assert tracer.completes[0]["found"] is True


def test_mcts_silent_when_record_expansions_false() -> None:
    start, goal, actions = _toy_domain()
    tracer = _RecordingTracer()
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=1),
        tracing=MCTSTracingConfig(tracer=tracer),
    )
    strategy.plan(start, goal, actions)
    assert tracer.expands == []
    assert tracer.completes == []


def test_mcts_aplan_emits_async_events() -> None:
    start, goal, actions = _toy_domain()
    tracer = _RecordingTracer()
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=2),
        tracing=MCTSTracingConfig(tracer=tracer, record_expansions=True),
    )
    asyncio.run(strategy.aplan(start, goal, actions))
    assert len(tracer.expands) > 0
    assert len(tracer.completes) == 1


def test_mcts_aplan_does_not_mutate_strategy_tracer() -> None:
    start, goal, actions = _toy_domain()
    tracer = _RecordingTracer()
    strategy = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=3),
        tracing=MCTSTracingConfig(tracer=tracer, record_expansions=True),
    )
    asyncio.run(strategy.aplan(start, goal, actions))
    assert strategy.tracing.tracer is tracer


def test_mcts_concurrent_aplan_does_not_clobber_tracers() -> None:
    start, goal, actions = _toy_domain()
    tracer_a = _RecordingTracer()
    tracer_b = _RecordingTracer()
    strategy_a = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=4),
        tracing=MCTSTracingConfig(tracer=tracer_a, record_expansions=True),
    )
    strategy_b = MCTSStrategy(
        exploration=MCTSExploration(iterations=10, wall_clock_ms=0.0, seed=4),
        tracing=MCTSTracingConfig(tracer=tracer_b, record_expansions=True),
    )

    async def _run_both() -> None:
        await asyncio.gather(
            strategy_a.aplan(start, goal, actions),
            strategy_b.aplan(start, goal, actions),
        )

    asyncio.run(_run_both())
    assert strategy_a.tracing.tracer is tracer_a
    assert strategy_b.tracing.tracer is tracer_b
    assert len(tracer_a.expands) > 0
    assert len(tracer_b.expands) > 0
    assert len(tracer_a.completes) == 1
    assert len(tracer_b.completes) == 1
