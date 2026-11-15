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

``GoapState`` is JSON-native, so every backend runs with its **stock**
serializer: LangGOAP installs nothing on the checkpointer.  The input is
built with ``GoapGraph.initial_state`` because LangGraph serialises the
raw input before the state reducers can convert a ``GoalSpec`` object.
A regression that puts a non-JSON value back into the state would make
every backend test fail with ``TypeError: Type is not msgpack
serializable: ...``.

Postgres and Redis tests are guarded by ``pytest.importorskip`` against
the corresponding optional extra, so the suite still runs on
environments without ``langgraph-checkpoint-postgres`` /
``langgraph-checkpoint-redis`` installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from tests.integration.checkpointing_support import (
    _assert_threads_isolated_async,
    _assert_threads_isolated_sync,
    _run_mid_plan_resume_async,
    _run_mid_plan_resume_sync,
)

# ---------------------------------------------------------------------------
# PostgresSaver / AsyncPostgresSaver
# ---------------------------------------------------------------------------


pytest.importorskip(
    "langgraph.checkpoint.postgres",
    reason="Install langgoap[checkpoint-postgres] to run Postgres checkpointer tests",
    exc_type=ImportError,
)
pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers[postgres] required for Postgres checkpointer tests",
    exc_type=ImportError,
)


# Containerized backends below need a reachable docker daemon. macOS GitHub
# runners ship the ``docker`` Python package (via testcontainers) but not
# the daemon, so probe at module load time and skip the rest of the module
# the same way ``importorskip`` would for a missing extra.
try:
    import docker as _docker_lib

    _docker_lib.from_env(timeout=2).ping()
except Exception as _exc:  # pragma: no cover - environmental
    pytest.skip(
        f"docker daemon not reachable ({_exc.__class__.__name__}); "
        "skipping containerized checkpointer tests",
        allow_module_level=True,
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
