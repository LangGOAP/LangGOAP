"""Integration tests for ``GoapGraph`` under LangGraph checkpointers.

These tests prove that the checkpointer wiring exposed by
``GoapGraph.compile(checkpointer=...)`` actually works end-to-end
against the three checkpointer backends recommended for production:

* ``MemorySaver`` — LangGraph's reference in-memory saver.
* ``PostgresSaver`` / ``AsyncPostgresSaver`` — ``langgraph-checkpoint-postgres``
  against a real Postgres container provisioned via TestContainers.
* ``RedisSaver`` / ``AsyncRedisSaver`` — ``langgraph-checkpoint-redis``
  against a real Redis 8 container provisioned via TestContainers.
  ``redis:8`` ships RedisJSON + RediSearch as built-in modules, which
  the Redis checkpointer requires.

The tests verify three contracts that every checkpointer must honour:

1. **Persistence** — running the GOAP loop under a checkpointer stores
   a non-trivial state history that can be queried via ``get_state``.
2. **Mid-plan resume** — compiling with ``interrupt_before=["executor"]``
   pauses the graph between actions; calling ``invoke(None, config)``
   with the same ``thread_id`` resumes from the saved state, executes
   the remaining actions, and reaches ``goal_achieved``.  Intermediate
   world-state reflects the partial effects of each executed action.
3. **Thread isolation** — concurrent ``thread_id``s on the same
   compiled graph operate on independent state and never read each
   other's checkpoints.

These three invariants are what users actually rely on when they reach
for a checkpointer, so the test suite asserts them directly rather than
testing individual LangGraph primitives.

LangGoap's frozen dataclasses wrap dict fields in ``MappingProxyType``
for deep immutability.  LangGraph's stock ``JsonPlusSerializer`` cannot
encode ``MappingProxyType``, so :class:`GoapGraph` auto-installs
``LangGoapSerializer`` on every checkpointer that reaches ``compile()``.
These tests exercise that wiring — a regression in the serde install
would make every backend test fail with ``TypeError: Type is not
msgpack serializable: GoalSpec``.

Postgres and Redis tests are guarded by ``pytest.importorskip`` against
the corresponding optional extra, so the suite still runs on
environments without ``langgraph-checkpoint-postgres`` /
``langgraph-checkpoint-redis`` installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

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
        {"goal": _goal(), "world_state": {}},
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
        {"goal": _goal(), "world_state": {}},
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
        {"goal": _goal(), "world_state": {}},
        config=config_a,
    )
    _assert_run_is_goal_achieved(result_a)

    # Thread B reaches a DIFFERENT terminal state (unreachable goal),
    # which rules out any accidental cross-thread reads: if the two
    # threads shared state, thread B would inherit ``done=True`` from
    # thread A and the assertion on ``no_plan`` would fail.
    result_b = compiled.invoke(
        {"goal": GoalSpec(conditions={"impossible": True}), "world_state": {}},
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
        {"goal": _goal(), "world_state": {}},
        config=config_a,
    )
    _assert_run_is_goal_achieved(result_a)

    result_b = await compiled.ainvoke(
        {"goal": GoalSpec(conditions={"impossible": True}), "world_state": {}},
        config=config_b,
    )
    assert result_b.get("status") == "no_plan"

    state_a = await compiled.aget_state(config_a)
    state_b = await compiled.aget_state(config_b)
    assert state_a.values.get("status") == "goal_achieved"
    assert state_b.values.get("status") == "no_plan"


# ---------------------------------------------------------------------------
# MemorySaver
# ---------------------------------------------------------------------------


class TestMemorySaver:
    def test_memory_saver_resume_from_mid_plan(self) -> None:
        _run_mid_plan_resume_sync(MemorySaver())

    async def test_memory_saver_resume_from_mid_plan_async(self) -> None:
        await _run_mid_plan_resume_async(MemorySaver())

    def test_memory_saver_concurrent_threads_isolated(self) -> None:
        _assert_threads_isolated_sync(MemorySaver())


# ---------------------------------------------------------------------------
# PostgresSaver / AsyncPostgresSaver
# ---------------------------------------------------------------------------


pytest.importorskip(
    "langgraph.checkpoint.postgres",
    reason="Install langgoap[checkpoint-postgres] to run Postgres checkpointer tests",
)
pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers[postgres] required for Postgres checkpointer tests",
)


@contextmanager
def _postgres_url() -> Iterator[str]:
    """Spin up an ephemeral Postgres 16 container and yield its URL.

    Uses the ``psycopg`` (v3) driver that ``langgraph-checkpoint-postgres``
    depends on; the URL must use the ``postgresql://`` scheme rather
    than the testcontainers default ``postgresql+psycopg2://``.
    """
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer("postgres:16", driver=None)
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        user = container.username
        password = container.password
        dbname = container.dbname
        yield f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    finally:
        container.stop()


class TestPostgresSaver:
    def test_postgres_saver_resume_from_mid_plan(self) -> None:
        from langgraph.checkpoint.postgres import PostgresSaver

        with _postgres_url() as url:
            with PostgresSaver.from_conn_string(url) as saver:
                saver.setup()
                _run_mid_plan_resume_sync(saver)

    async def test_async_postgres_saver_resume_from_mid_plan(self) -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        with _postgres_url() as url:
            async with AsyncPostgresSaver.from_conn_string(url) as saver:
                await saver.setup()
                await _run_mid_plan_resume_async(saver)

    async def test_async_postgres_saver_concurrent_threads_isolated(self) -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        with _postgres_url() as url:
            async with AsyncPostgresSaver.from_conn_string(url) as saver:
                await saver.setup()
                await _assert_threads_isolated_async(saver)


# ---------------------------------------------------------------------------
# RedisSaver / AsyncRedisSaver
# ---------------------------------------------------------------------------


pytest.importorskip(
    "langgraph.checkpoint.redis",
    reason="Install langgoap[checkpoint-redis] to run Redis checkpointer tests",
)
pytest.importorskip(
    "testcontainers.redis",
    reason="testcontainers[redis] required for Redis checkpointer tests",
)


@contextmanager
def _redis_url() -> Iterator[str]:
    """Spin up an ephemeral ``redis:8`` container and yield its URL.

    ``redis:8`` ships RedisJSON + RediSearch as built-in modules, which
    the Redis checkpointer uses to index checkpoint metadata.  Older
    base images (``redis:7``) do not ship these modules and would
    require ``redis/redis-stack-server`` instead.  We stick to
    ``redis:8`` because it is the same image used by the upstream
    ``langgraph-redis`` test suite.
    """
    from testcontainers.redis import RedisContainer

    container = RedisContainer("redis:8")
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        container.stop()


class TestRedisSaver:
    def test_redis_saver_resume_from_mid_plan(self) -> None:
        from langgraph.checkpoint.redis import RedisSaver

        with _redis_url() as url:
            with RedisSaver.from_conn_string(url) as saver:
                saver.setup()
                _run_mid_plan_resume_sync(saver)

    async def test_async_redis_saver_resume_from_mid_plan(self) -> None:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        with _redis_url() as url:
            async with AsyncRedisSaver.from_conn_string(url) as saver:
                await saver.asetup()
                await _run_mid_plan_resume_async(saver)
