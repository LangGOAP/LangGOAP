"""Shared drivers for the checkpointer integration tests.

Imported by ``test_checkpointing_memory.py`` (always runs) and
``test_checkpointing.py`` (Postgres / Redis, gated on Docker) so the
same three contracts are asserted against every backend (issue #43).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _actions() -> list[ActionSpec]:
    """A two-step linear plan: ``gather`` then ``process``.

    Small enough that every checkpoint boundary is easy to reason about
    in test assertions, and still exercises the full planner → executor
    → observer loop at least twice.
    """
    return [
        ActionSpec(
            name="gather",
            preconditions={},
            effects={"data": True},
            cost=1.0,
        ),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            cost=1.0,
        ),
    ]


def _goal() -> GoalSpec:
    return GoalSpec(conditions={"done": True})


def _config(thread_id: str | None = None) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id or str(uuid4())}}


# ---------------------------------------------------------------------------
# Contracts exercised against every backend
# ---------------------------------------------------------------------------


def _assert_run_is_goal_achieved(result: dict[str, Any]) -> None:
    assert result.get("status") == "goal_achieved"
    assert result.get("world_state", {}).get("data") is True
    assert result.get("world_state", {}).get("done") is True


def _run_mid_plan_resume_sync(checkpointer: Any) -> None:
    """Drive a synchronous mid-plan resume cycle against ``checkpointer``.

    Compiles the graph with ``interrupt_before=["executor"]`` so the
    planner produces the full plan but each executor invocation is
    gated behind a checkpoint.  The helper then drives three rounds
    of ``invoke(None, config)`` with the same thread_id:

    1. First invoke: planner plans; executor is paused before action 1.
    2. Second invoke: action 1 executes; world_state gains ``data=True``;
       executor is paused before action 2.
    3. Third invoke: action 2 executes; observer routes to END;
       final state reports ``goal_achieved``.

    At every pause boundary the helper asserts that ``get_state``
    returns the saved state from the checkpointer, not a stale copy.
    """
    graph = GoapGraph(_actions())
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["executor"],
    )

    config = _config()

    # Round 1 — plan, then pause before executing action 1.
    round1 = compiled.invoke(
        graph.initial_state(_goal(), {}),
        config=config,
    )
    # At this point the plan has been produced and current_step==0.
    # The interrupt fires before the executor node runs.
    assert round1.get("plan") is not None
    assert round1.get("current_step", 0) == 0
    assert round1.get("world_state", {}).get("data") is not True

    state_after_round1 = compiled.get_state(config)
    assert state_after_round1.next == ("executor",)

    # Round 2 — execute action 1, observer loops back, pause again.
    round2 = compiled.invoke(None, config=config)
    assert round2.get("world_state", {}).get("data") is True
    assert round2.get("world_state", {}).get("done") is not True
    assert round2.get("current_step", 0) == 1

    state_after_round2 = compiled.get_state(config)
    assert state_after_round2.next == ("executor",)

    # Round 3 — execute action 2, observer routes to END, plan complete.
    round3 = compiled.invoke(None, config=config)
    _assert_run_is_goal_achieved(round3)

    state_after_round3 = compiled.get_state(config)
    assert state_after_round3.next == ()


async def _run_mid_plan_resume_async(checkpointer: Any) -> None:
    """Async mirror of :func:`_run_mid_plan_resume_sync`."""
    graph = GoapGraph(_actions())
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["executor"],
    )

    config = _config()

    round1 = await compiled.ainvoke(
        graph.initial_state(_goal(), {}),
        config=config,
    )
    assert round1.get("plan") is not None
    assert round1.get("current_step", 0) == 0

    state_after_round1 = await compiled.aget_state(config)
    assert state_after_round1.next == ("executor",)

    round2 = await compiled.ainvoke(None, config=config)
    assert round2.get("world_state", {}).get("data") is True
    assert round2.get("current_step", 0) == 1

    round3 = await compiled.ainvoke(None, config=config)
    _assert_run_is_goal_achieved(round3)


def _assert_threads_isolated_sync(checkpointer: Any) -> None:
    """Two ``thread_id``s must not see each other's state."""
    graph = GoapGraph(_actions())
    compiled = graph.compile(checkpointer=checkpointer)

    config_a = _config("thread-a")
    config_b = _config("thread-b")

    # Thread A reaches the goal.
    result_a = compiled.invoke(
        graph.initial_state(_goal(), {}),
        config=config_a,
    )
    _assert_run_is_goal_achieved(result_a)

    # Thread B reaches a DIFFERENT terminal state (unreachable goal),
    # which rules out any accidental cross-thread reads: if the two
    # threads shared state, thread B would inherit ``done=True`` from
    # thread A and the assertion on ``no_plan`` would fail.
    result_b = compiled.invoke(
        graph.initial_state(GoalSpec(conditions={"impossible": True}), {}),
        config=config_b,
    )
    assert result_b.get("status") == "no_plan"

    # Final states read from the checkpointer confirm isolation.
    state_a = compiled.get_state(config_a)
    state_b = compiled.get_state(config_b)
    assert state_a.values.get("status") == "goal_achieved"
    assert state_b.values.get("status") == "no_plan"
    assert state_a.values.get("world_state", {}).get("done") is True
    assert state_b.values.get("world_state", {}).get("done") is not True


async def _assert_threads_isolated_async(checkpointer: Any) -> None:
    """Async mirror of :func:`_assert_threads_isolated_sync`."""
    graph = GoapGraph(_actions())
    compiled = graph.compile(checkpointer=checkpointer)

    config_a = _config("thread-a")
    config_b = _config("thread-b")

    result_a = await compiled.ainvoke(
        graph.initial_state(_goal(), {}),
        config=config_a,
    )
    _assert_run_is_goal_achieved(result_a)

    result_b = await compiled.ainvoke(
        graph.initial_state(GoalSpec(conditions={"impossible": True}), {}),
        config=config_b,
    )
    assert result_b.get("status") == "no_plan"

    state_a = await compiled.aget_state(config_a)
    state_b = await compiled.aget_state(config_b)
    assert state_a.values.get("status") == "goal_achieved"
    assert state_b.values.get("status") == "no_plan"


__all__ = [
    "_actions",
    "_goal",
    "_config",
    "_assert_run_is_goal_achieved",
    "_run_mid_plan_resume_sync",
    "_run_mid_plan_resume_async",
    "_assert_threads_isolated_sync",
    "_assert_threads_isolated_async",
]
