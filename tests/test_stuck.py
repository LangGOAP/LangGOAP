"""Unit tests for ``langgoap.stuck`` — Protocol, result, multicast.

API reference:
  research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
    com/embabel/agent/api/common/StuckHandler.kt
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

import pytest

from langgoap.stuck import (
    FunctionalStuckHandler,
    MulticastStuckHandler,
    StuckHandler,
    StuckHandlerResult,
    StuckHandlingResultCode,
)

# ---------------------------------------------------------------------------
# StuckHandlerResult
# ---------------------------------------------------------------------------


class TestStuckHandlerResult:
    def test_replan_factory_sets_code_and_fields(self) -> None:
        r = StuckHandlerResult.replan(
            handler_name="h",
            message="ok",
            state_updates={"a": 1},
        )
        assert r.code is StuckHandlingResultCode.REPLAN
        assert r.handler_name == "h"
        assert r.message == "ok"
        assert dict(r.state_updates) == {"a": 1}
        assert r.new_goal is None

    def test_no_resolution_factory(self) -> None:
        r = StuckHandlerResult.no_resolution(handler_name="h", message="cannot help")
        assert r.code is StuckHandlingResultCode.NO_RESOLUTION
        assert r.handler_name == "h"
        assert dict(r.state_updates) == {}

    def test_state_updates_default_is_empty_mapping(self) -> None:
        r = StuckHandlerResult.replan(handler_name="h", message="m")
        assert dict(r.state_updates) == {}

    def test_timestamp_is_set(self) -> None:
        r = StuckHandlerResult.replan(handler_name="h", message="m")
        assert isinstance(r.timestamp, datetime)
        assert r.timestamp.tzinfo is timezone.utc

    def test_state_updates_immutable_after_construction(self) -> None:
        src = {"a": 1}
        r = StuckHandlerResult.replan(
            handler_name="h",
            message="m",
            state_updates=src,
        )
        src["a"] = 999  # mutating the source must not affect the result
        assert r.state_updates["a"] == 1

    def test_dataclass_is_frozen(self) -> None:
        r = StuckHandlerResult.replan(handler_name="h", message="m")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.message = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_functional_stuck_handler_is_a_handler(self) -> None:
        h = FunctionalStuckHandler(
            name="x",
            fn=lambda s, r: StuckHandlerResult.no_resolution(
                handler_name="x", message="m"
            ),
        )
        assert isinstance(h, StuckHandler)

    def test_multicast_is_a_handler(self) -> None:
        h = MulticastStuckHandler([])
        assert isinstance(h, StuckHandler)


# ---------------------------------------------------------------------------
# FunctionalStuckHandler
# ---------------------------------------------------------------------------


class TestFunctionalStuckHandler:
    def test_invokes_function_with_state_and_reason(self) -> None:
        captured: dict[str, Any] = {}

        def fn(state: Any, reason: Any) -> StuckHandlerResult:
            captured["state"] = state
            captured["reason"] = reason
            return StuckHandlerResult.no_resolution(handler_name="f", message="m")

        h = FunctionalStuckHandler(name="f", fn=fn)
        h.handle_stuck(state={"world_state": {"k": True}}, reason={"x": 1})

        assert captured["state"] == {"world_state": {"k": True}}
        assert captured["reason"] == {"x": 1}


# ---------------------------------------------------------------------------
# MulticastStuckHandler unit-level
# ---------------------------------------------------------------------------


class TestMulticastUnit:
    def _no_op(self, name: str) -> StuckHandler:
        return FunctionalStuckHandler(
            name=name,
            fn=lambda s, r: StuckHandlerResult.no_resolution(
                handler_name=name, message=f"{name} cannot help"
            ),
        )

    def _replan(self, name: str) -> StuckHandler:
        return FunctionalStuckHandler(
            name=name,
            fn=lambda s, r: StuckHandlerResult.replan(
                handler_name=name,
                message=f"{name} resolved",
                state_updates={"fixed_by": name},
            ),
        )

    def test_empty_list_returns_no_resolution(self) -> None:
        result = MulticastStuckHandler([]).handle_stuck(state={}, reason=None)
        assert result.code is StuckHandlingResultCode.NO_RESOLUTION
        assert result.handler_name is None

    def test_first_replan_short_circuits(self) -> None:
        m = MulticastStuckHandler([self._replan("h1"), self._replan("h2")])
        result = m.handle_stuck(state={}, reason=None)
        assert result.handler_name == "h1"

    def test_aggregated_message_lists_every_handler(self) -> None:
        m = MulticastStuckHandler(
            [self._no_op("h1"), self._no_op("h2"), self._no_op("h3")]
        )
        result = m.handle_stuck(state={}, reason=None)
        assert "h1" in result.message
        assert "h2" in result.message
        assert "h3" in result.message

    def test_exception_treated_as_no_resolution_and_does_not_short_circuit(
        self,
    ) -> None:
        def boom(s: Any, r: Any) -> StuckHandlerResult:
            raise RuntimeError("intentional")

        m = MulticastStuckHandler(
            [
                FunctionalStuckHandler(name="bad", fn=boom),
                self._replan("good"),
            ]
        )
        result = m.handle_stuck(state={}, reason=None)
        assert result.code is StuckHandlingResultCode.REPLAN
        assert result.handler_name == "good"
