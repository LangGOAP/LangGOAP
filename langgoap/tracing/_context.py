"""Run context for tracer hooks, actions and sensors.

Internal module; :func:`current_thread_id` is re-exported from
``langgoap.tracing`` and the top-level package.
"""

from __future__ import annotations

import functools
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


_KEYWORD_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


@functools.lru_cache(maxsize=1024)
def _keyword_names(func: Callable[..., Any]) -> frozenset[str] | None:
    """The parameter names *func* takes by keyword, or ``None`` for "any".

    ``None`` covers a ``**kwargs`` parameter and callables that cannot be
    inspected (builtins and some C callables).
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return frozenset(n for n, p in parameters.items() if p.kind in _KEYWORD_KINDS)


def accepted_keywords(hook: Callable[..., Any]) -> frozenset[str] | None:
    """The keyword names *hook* accepts, or ``None`` when it accepts any.

    Remembered per function: a bound method is a new object on every
    attribute access, so the lookup is keyed on the function it shares with
    every other instance.  Signature inspection costs tens of microseconds,
    which matters for ``on_search_expand`` (fired per A* expansion).
    """
    target = getattr(hook, "__func__", hook)
    try:
        return _keyword_names(target)
    except TypeError:  # an unhashable callable object: look, do not remember
        return _keyword_names.__wrapped__(target)


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
    accepted = accepted_keywords(hook)
    if accepted is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in accepted}


def hook_kwargs(hook: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """*kwargs* as *hook* accepts them, plus the run's ``thread_id`` (issue #104).

    ``thread_id`` is added only when the hook takes it and the caller did not
    pass one, so a hook that does not want it costs no config lookup.
    """
    accepted = accepted_keywords(hook)
    passed = (
        kwargs
        if accepted is None
        else {k: v for k, v in kwargs.items() if k in accepted}
    )
    if "thread_id" in passed or (accepted is not None and "thread_id" not in accepted):
        return passed
    return {**passed, "thread_id": current_thread_id()}


__all__ = ["accepted_keywords", "current_thread_id", "hook_kwargs", "supported_kwargs"]
