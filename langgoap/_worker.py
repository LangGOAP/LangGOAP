"""Run synchronous callables from the async path without losing the run's context."""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any, Callable


async def run_in_worker(func: Callable[..., Any], *args: Any) -> Any:
    """Run a synchronous callable in the default executor, in the caller's context.

    ``loop.run_in_executor`` does not copy context variables, so LangGraph's
    run config (and with it :func:`~langgoap.tracing.current_thread_id`) would be lost in sync
    actions, sensors, guards and resolvers on the async path.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(None, context.run, func, *args)


__all__ = ["run_in_worker"]
