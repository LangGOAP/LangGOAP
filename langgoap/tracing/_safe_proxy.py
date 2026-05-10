"""Exception-safe wrapper around any :class:`PlanningTracer`.

Consolidates the three pre-existing "never raise" patterns (per-call
``_safe_tracer_call`` guards in the node modules,
:class:`MultiTracer`'s catch-and-log, and ad-hoc inline try/excepts)
into a single proxy installed once at node construction time.  Once
the proxy wraps the user-supplied tracer, call sites can invoke hooks
directly (``self._tracer.on_action_start(action, state)``) without
their own try/except scaffolding.

Both sync (``on_*``) and async (``aon_*``) hooks are guarded.  Errors
are logged at ``WARNING`` level via the ``langgoap.tracing`` logger;
they are never propagated.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from langgoap.tracing._protocol import PlanningTracer

logger = logging.getLogger("langgoap.tracing")


class SafeTracerProxy:
    """Wrap a :class:`PlanningTracer` so every hook call is exception-safe.

    ``__getattr__`` intercepts attribute access, returning an
    error-swallowing wrapper for callables and the raw value for
    everything else (so attributes like
    :attr:`~langgoap.reflexion.ReflexionTracer.reflections` keep
    working transparently).

    Usage::

        tracer = LangSmithTracer()
        safe = SafeTracerProxy(tracer)
        safe.on_plan_start(goal, state, "AStar")  # never raises

    The proxy is itself a valid :class:`PlanningTracer` because the
    Protocol is structural and the proxy responds to every hook name
    via ``__getattr__``.

    Idempotent wrapping: ``SafeTracerProxy(SafeTracerProxy(t))`` is
    indistinguishable from ``SafeTracerProxy(t)`` because the proxy
    flattens itself in :meth:`__init__`.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: PlanningTracer) -> None:
        # Flatten nested proxies so node constructors can wrap
        # unconditionally without paying for nested try/except chains
        # when callers already supplied a SafeTracerProxy.
        if isinstance(inner, SafeTracerProxy):
            inner = inner._inner
        object.__setattr__(self, "_inner", inner)

    @property
    def inner(self) -> PlanningTracer:
        """The wrapped tracer (read-only handle for tests / introspection)."""
        return self._inner

    def __getattr__(self, name: str) -> Any:
        # Note: __getattr__ is only called when normal attribute lookup
        # fails, so ``self._inner`` (set via object.__setattr__) is
        # found first and never re-enters this branch.
        attr = getattr(self._inner, name)
        if not callable(attr):
            # Properties / data attributes (e.g. ReflexionTracer.reflections)
            # are returned untouched.
            return attr
        return _wrap_callable(attr, name)


def _wrap_callable(attr: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Return a sync or async wrapper around ``attr`` that swallows exceptions."""
    if inspect.iscoroutinefunction(attr) or name.startswith("aon_"):

        async def _async_wrapper(*args: Any, **kwargs: Any) -> None:
            try:
                await attr(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Tracer raised during %s: %s", name, exc)

        return _async_wrapper

    def _sync_wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            attr(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tracer raised during %s: %s", name, exc)

    return _sync_wrapper


__all__ = ["SafeTracerProxy"]
