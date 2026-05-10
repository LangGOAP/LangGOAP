"""Unit tests for :class:`langgoap.tracing.SafeTracerProxy`.

The proxy consolidates the previous three "never raise" patterns
(per-call ``_safe_tracer_call`` guards, :class:`MultiTracer`
catch-and-log, ad-hoc inline try/excepts) into a single wrapper
installed once at node construction time.  These tests pin its
contract so future refactors do not regress the exception-safety
boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from langgoap.tracing import (
    LoggingTracer,
    NullTracer,
    SafeTracerProxy,
)


class _ExplodingTracer:
    """Test double that raises from every hook."""

    def __init__(self) -> None:
        self.sync_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.async_calls: list[tuple[str, tuple[Any, ...]]] = []

    def on_action_start(self, *args: Any) -> None:
        self.sync_calls.append(("on_action_start", args))
        raise RuntimeError("sync boom")

    async def aon_action_start(self, *args: Any) -> None:
        self.async_calls.append(("aon_action_start", args))
        raise RuntimeError("async boom")


class _ReflectionTracer:
    """Test double exposing a non-callable ``reflections`` attribute."""

    def __init__(self) -> None:
        self.reflections: list[str] = ["r1", "r2"]


class TestSyncHookSafety:
    def test_sync_hook_exception_is_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inner = _ExplodingTracer()
        proxy = SafeTracerProxy(inner)
        with caplog.at_level(logging.WARNING, logger="langgoap.tracing"):
            proxy.on_action_start("a", {})  # must not raise
        assert inner.sync_calls == [("on_action_start", ("a", {}))]
        assert any("on_action_start" in rec.message for rec in caplog.records)

    def test_successful_sync_hook_passes_through(self) -> None:
        # Use a real LoggingTracer; success path returns None and does not raise.
        proxy = SafeTracerProxy(LoggingTracer())
        proxy.on_plan_start("goal", {}, "AStar")  # smoke


class TestAsyncHookSafety:
    @pytest.mark.asyncio
    async def test_async_hook_exception_is_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inner = _ExplodingTracer()
        proxy = SafeTracerProxy(inner)
        with caplog.at_level(logging.WARNING, logger="langgoap.tracing"):
            await proxy.aon_action_start("a", {})  # must not raise
        assert inner.async_calls == [("aon_action_start", ("a", {}))]
        assert any("aon_action_start" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_successful_async_hook_passes_through(self) -> None:
        proxy = SafeTracerProxy(LoggingTracer())
        await proxy.aon_plan_start("goal", {}, "AStar")  # smoke


class TestNonCallableForwarding:
    def test_data_attributes_are_returned_unwrapped(self) -> None:
        """Non-callable attributes (e.g. ``ReflexionTracer.reflections``)
        must pass through untouched so downstream callers can read them
        normally."""
        inner = _ReflectionTracer()
        proxy = SafeTracerProxy(inner)
        assert proxy.reflections == ["r1", "r2"]
        # Mutating the underlying list is reflected through the proxy.
        inner.reflections.append("r3")
        assert proxy.reflections == ["r1", "r2", "r3"]


class TestIdempotentWrapping:
    def test_proxy_flattens_nested_proxies(self) -> None:
        """Wrapping a SafeTracerProxy in another SafeTracerProxy must
        not nest \u2014 the outer proxy unwraps the inner one so that
        node constructors can wrap unconditionally without paying
        for chained try/except overhead.
        """
        inner = LoggingTracer()
        once = SafeTracerProxy(inner)
        twice = SafeTracerProxy(once)
        assert twice.inner is inner
        assert twice.inner is once.inner


class TestProtocolCompatibility:
    def test_proxy_exposes_every_planning_tracer_hook(self) -> None:
        """The proxy must respond to every hook name on the
        :class:`~langgoap.tracing.PlanningTracer` Protocol so call sites
        can invoke hooks directly without hasattr guards.

        ``isinstance(proxy, PlanningTracer)`` does **not** reliably
        return ``True`` in Python 3.12+ because
        :func:`typing.runtime_checkable` does not consult
        ``__getattr__``-served attributes; we exercise the structural
        contract via :func:`hasattr` instead.
        """
        proxy = SafeTracerProxy(NullTracer())
        # Spot-check the full sync + async hook surface.
        sync_hooks = (
            "on_plan_start",
            "on_plan_complete",
            "on_plan_failed",
            "on_action_start",
            "on_action_complete",
            "on_replan",
            "on_goal_achieved",
            "on_sensor_complete",
            "on_search_expand",
            "on_search_dead_end",
            "on_search_complete",
        )
        async_hooks = tuple("a" + h for h in sync_hooks)
        for h in sync_hooks + async_hooks:
            assert hasattr(proxy, h), f"proxy missing hook {h!r}"
            assert callable(getattr(proxy, h)), f"proxy hook {h!r} is not callable"
