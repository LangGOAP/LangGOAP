"""Run context for tracer hooks, actions and sensors.

Internal module; :func:`current_thread_id` is re-exported from
``langgoap.tracing`` and the top-level package.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from langgraph.config import get_config


def current_thread_id() -> str | None:
    """The ``thread_id`` of the graph run this code is executing in, or ``None``.

    Tracer hooks, actions, sensors and guards run inside a LangGraph node, so
    the run's config is in scope: a tracer shared by many threads can attribute
    each event without the application setting state on it before every
    invocation (issue #96).

    Returns ``None`` outside a run, when the run has no ``thread_id``, and on
    the async path under Python 3.10, where LangGraph cannot propagate the
    run's context.
    """
    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = (config.get("configurable") or {}).get("thread_id")
    return None if thread_id is None else str(thread_id)


def supported_kwargs(
    hook: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Drop the keyword arguments *hook* does not accept.

    Hooks gain optional keyword arguments over time (``on_replan`` got
    ``duration_ms`` in 0.3.1).  A tracer written against the older signature
    must keep working, so callers pass new arguments by keyword and through
    this filter.
    """
    if not kwargs:
        return kwargs
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):  # builtins and some C callables
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in parameters}


__all__ = ["current_thread_id", "supported_kwargs"]
