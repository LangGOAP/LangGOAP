"""``MemorySaver`` checkpointing contracts.

Kept apart from the containerised Postgres / Redis module so a missing
Docker daemon (macOS and Windows runners, most laptops) never skips the
in-memory resume and thread-isolation tests (issue #43).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from tests.integration.checkpointing_support import (
    _assert_threads_isolated_sync,
    _run_mid_plan_resume_async,
    _run_mid_plan_resume_sync,
)


class TestMemorySaver:
    def test_memory_saver_resume_from_mid_plan(self) -> None:
        _run_mid_plan_resume_sync(MemorySaver())

    async def test_memory_saver_resume_from_mid_plan_async(self) -> None:
        await _run_mid_plan_resume_async(MemorySaver())

    def test_memory_saver_concurrent_threads_isolated(self) -> None:
        _assert_threads_isolated_sync(MemorySaver())
