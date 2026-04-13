"""Tests for the three-valued condition logic module."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from langgoap.conditions import (
    AsyncConditionResolver,
    ConditionResolver,
    ConditionStatus,
    FunctionalConditionResolver,
    PromptConditionResolver,
    aresolve_conditions,
    resolve_conditions,
)

# ---------------------------------------------------------------------------
# ConditionStatus
# ---------------------------------------------------------------------------


class TestConditionStatus:
    def test_module_constants_are_condition_status(self) -> None:
        assert isinstance(ConditionStatus.TRUE, ConditionStatus)
        assert isinstance(ConditionStatus.FALSE, ConditionStatus)
        assert isinstance(ConditionStatus.UNKNOWN, ConditionStatus)

    def test_true_is_truthy(self) -> None:
        assert bool(ConditionStatus.TRUE) is True

    def test_false_is_falsy(self) -> None:
        assert bool(ConditionStatus.FALSE) is False

    def test_unknown_is_falsy(self) -> None:
        assert bool(ConditionStatus.UNKNOWN) is False

    def test_equality(self) -> None:
        assert ConditionStatus.TRUE == ConditionStatus.TRUE
        assert ConditionStatus.FALSE != ConditionStatus.TRUE
        assert ConditionStatus.UNKNOWN != ConditionStatus.FALSE

    def test_repr(self) -> None:
        assert "TRUE" in repr(ConditionStatus.TRUE)
        assert "FALSE" in repr(ConditionStatus.FALSE)

    def test_str_equality(self) -> None:
        # ConditionStatus subclasses str — interoperable with plain strings
        assert ConditionStatus.TRUE == "true"
        assert ConditionStatus.FALSE == "false"

    def test_json_serializable(self) -> None:
        import json

        data = {"status": ConditionStatus.TRUE}
        serialized = json.dumps(data)
        assert '"true"' in serialized


# ---------------------------------------------------------------------------
# FunctionalConditionResolver
# ---------------------------------------------------------------------------


def _make_resolver(return_value: Any) -> FunctionalConditionResolver:
    return FunctionalConditionResolver("test", lambda k, ws: return_value)


class TestFunctionalConditionResolver:
    def test_name(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        assert r.name == "test"

    def test_resolve_returns_true(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        assert r.resolve("x", {}) == ConditionStatus.TRUE

    def test_resolve_returns_false(self) -> None:
        r = _make_resolver(ConditionStatus.FALSE)
        assert r.resolve("x", {}) == ConditionStatus.FALSE

    def test_resolve_returns_unknown_on_none(self) -> None:
        r = _make_resolver(None)
        assert r.resolve("x", {}) == ConditionStatus.UNKNOWN

    def test_resolve_coerces_bool_true(self) -> None:
        r = _make_resolver(True)
        assert r.resolve("x", {}) == ConditionStatus.TRUE

    def test_resolve_coerces_bool_false(self) -> None:
        r = _make_resolver(False)
        assert r.resolve("x", {}) == ConditionStatus.FALSE

    def test_resolve_coerces_string_true(self) -> None:
        r = _make_resolver("true")
        assert r.resolve("x", {}) == ConditionStatus.TRUE

    def test_resolve_coerces_string_unknown(self) -> None:
        r = _make_resolver("maybe")
        assert r.resolve("x", {}) == ConditionStatus.UNKNOWN

    def test_resolve_receives_world_state(self) -> None:
        captured: list[dict[str, Any]] = []
        r = FunctionalConditionResolver(
            "ws_check",
            lambda k, ws: captured.append(ws) or ConditionStatus.TRUE,
        )
        r.resolve("k", {"foo": 42})
        assert captured[0] == {"foo": 42}

    @pytest.mark.asyncio
    async def test_aresolve_returns_true(self) -> None:
        r = _make_resolver(True)
        assert await r.aresolve("x", {}) == ConditionStatus.TRUE

    def test_resolve_raises_type_error_for_async_callable(self) -> None:
        """resolve() must raise TypeError instead of silently trying to run the
        coroutine in a thread pool.  The caller should use aresolve() instead."""

        async def async_fn(k: str, ws: dict[str, Any]) -> bool:
            return True

        r = FunctionalConditionResolver("async_check", async_fn)
        with pytest.raises(TypeError, match="aresolve"):
            r.resolve("x", {})

    @pytest.mark.asyncio
    async def test_aresolve_coerces_async_callable(self) -> None:
        async def async_fn(k: str, ws: dict[str, Any]) -> bool:
            return True

        r = FunctionalConditionResolver("async", async_fn)
        assert await r.aresolve("key", {}) == ConditionStatus.TRUE


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_functional_resolver_satisfies_condition_resolver_protocol(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        assert isinstance(r, ConditionResolver)

    def test_functional_resolver_satisfies_async_resolver_protocol(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        assert isinstance(r, AsyncConditionResolver)


# ---------------------------------------------------------------------------
# resolve_conditions helper
# ---------------------------------------------------------------------------


class TestResolveConditions:
    def test_fills_missing_key(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        result = resolve_conditions([r], ["x"], {})
        assert result["x"] is True

    def test_does_not_overwrite_existing_key(self) -> None:
        r = _make_resolver(ConditionStatus.FALSE)
        result = resolve_conditions([r], ["x"], {"x": True})
        assert result["x"] is True  # existing value preserved

    def test_overwrite_flag_re_evaluates(self) -> None:
        r = _make_resolver(ConditionStatus.FALSE)
        result = resolve_conditions([r], ["x"], {"x": True}, overwrite=True)
        assert result["x"] is False

    def test_unknown_leaves_key_absent(self) -> None:
        r = _make_resolver(ConditionStatus.UNKNOWN)
        result = resolve_conditions([r], ["missing"], {})
        assert "missing" not in result

    def test_does_not_mutate_input(self) -> None:
        ws: dict[str, Any] = {}
        r = _make_resolver(ConditionStatus.TRUE)
        resolve_conditions([r], ["x"], ws)
        assert ws == {}

    def test_first_non_unknown_wins(self) -> None:
        r_unknown = _make_resolver(ConditionStatus.UNKNOWN)
        r_true = _make_resolver(ConditionStatus.TRUE)
        result = resolve_conditions([r_unknown, r_true], ["x"], {})
        assert result["x"] is True

    def test_resolver_exception_is_caught(self) -> None:
        def bad_fn(k: str, ws: dict[str, Any]) -> ConditionStatus:
            raise RuntimeError("oops")

        r = FunctionalConditionResolver("bad", bad_fn)
        # Should not raise; key stays absent
        result = resolve_conditions([r], ["x"], {})
        assert "x" not in result

    def test_multiple_keys_resolved_independently(self) -> None:
        def by_key(k: str, ws: dict[str, Any]) -> ConditionStatus:
            return ConditionStatus.TRUE if k == "a" else ConditionStatus.FALSE

        r = FunctionalConditionResolver("multi", by_key)
        result = resolve_conditions([r], ["a", "b"], {})
        assert result["a"] is True
        assert result["b"] is False


# ---------------------------------------------------------------------------
# aresolve_conditions helper
# ---------------------------------------------------------------------------


class TestAresolveConditions:
    @pytest.mark.asyncio
    async def test_fills_missing_key_async(self) -> None:
        r = _make_resolver(ConditionStatus.TRUE)
        result = await aresolve_conditions([r], ["x"], {})
        assert result["x"] is True

    @pytest.mark.asyncio
    async def test_unknown_leaves_absent_async(self) -> None:
        r = _make_resolver(ConditionStatus.UNKNOWN)
        result = await aresolve_conditions([r], ["y"], {})
        assert "y" not in result

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_async(self) -> None:
        r = _make_resolver(ConditionStatus.FALSE)
        result = await aresolve_conditions([r], ["x"], {"x": True})
        assert result["x"] is True

    @pytest.mark.asyncio
    async def test_resolver_exception_caught_async(self) -> None:
        def boom(k: str, ws: dict[str, Any]) -> ConditionStatus:
            raise RuntimeError("boom")

        r = FunctionalConditionResolver("boom", boom)
        result = await aresolve_conditions([r], ["x"], {})
        assert "x" not in result

    @pytest.mark.asyncio
    async def test_async_native_resolver_used(self) -> None:
        """Uses aresolve when available (async-native resolver)."""

        class AsyncOnlyResolver:
            @property
            def name(self) -> str:
                return "async_only"

            async def aresolve(
                self, key: str, world_state: dict[str, Any]
            ) -> ConditionStatus:
                return ConditionStatus.TRUE

        result = await aresolve_conditions([AsyncOnlyResolver()], ["x"], {})
        assert result["x"] is True


# ---------------------------------------------------------------------------
# PromptConditionResolver (mocked LLM)
# ---------------------------------------------------------------------------


def _make_fake_llm(status: str) -> MagicMock:
    """Build a fake BaseChatModel that returns *status* via structured output."""

    class _FakeAnswer:
        def __init__(self) -> None:
            self.status = status
            self.reasoning = ""

    llm = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = _FakeAnswer()

    async def _async_invoke(*args: Any, **kwargs: Any) -> _FakeAnswer:
        return _FakeAnswer()

    chain.ainvoke = _async_invoke
    llm.with_structured_output.return_value = chain
    return llm


class TestPromptConditionResolver:
    def test_name(self) -> None:
        r = PromptConditionResolver(MagicMock())
        assert r.name == "PromptConditionResolver"

    def test_resolve_true(self) -> None:
        r = PromptConditionResolver(_make_fake_llm("true"))
        assert r.resolve("x", {}) == ConditionStatus.TRUE

    def test_resolve_false(self) -> None:
        r = PromptConditionResolver(_make_fake_llm("false"))
        assert r.resolve("x", {}) == ConditionStatus.FALSE

    def test_resolve_unknown(self) -> None:
        r = PromptConditionResolver(_make_fake_llm("unknown"))
        assert r.resolve("x", {}) == ConditionStatus.UNKNOWN

    def test_resolve_returns_unknown_on_exception(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("no LLM")
        r = PromptConditionResolver(llm)
        # Must not raise — degrade to UNKNOWN
        assert r.resolve("x", {}) == ConditionStatus.UNKNOWN

    def test_custom_prompt_template_is_used(self) -> None:
        llm = _make_fake_llm("true")
        r = PromptConditionResolver(
            llm, prompt_template="Custom: {key} in {world_state}"
        )
        assert r.resolve("my_key", {"a": 1}) == ConditionStatus.TRUE

    @pytest.mark.asyncio
    async def test_aresolve_true(self) -> None:
        r = PromptConditionResolver(_make_fake_llm("true"))
        assert await r.aresolve("x", {}) == ConditionStatus.TRUE

    @pytest.mark.asyncio
    async def test_aresolve_returns_unknown_on_exception(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.side_effect = RuntimeError("no LLM")
        r = PromptConditionResolver(llm)
        assert await r.aresolve("x", {}) == ConditionStatus.UNKNOWN


# ---------------------------------------------------------------------------
# GoapPlanner integration: resolvers fill world_state before planning
# ---------------------------------------------------------------------------


class TestGoapPlannerResolverIntegration:
    """Resolvers must be called before A* runs, filling in missing keys."""

    def test_resolver_fills_missing_precondition_before_planning(self) -> None:
        from langgoap.actions import ActionSpec
        from langgoap.goals import GoalSpec
        from langgoap.graph.nodes import GoapPlanner
        from langgoap.graph.state import GoapState

        action = ActionSpec(
            name="do_it",
            preconditions={"ready": True},
            effects={"done": True},
        )
        goal = GoalSpec(conditions={"done": True})

        # Without resolver, planning fails (ready=True missing)
        planner_no_resolver = GoapPlanner([action])
        state_no_resolve: GoapState = {"world_state": {}, "goal": goal}
        result_no = planner_no_resolver(state_no_resolve)
        assert result_no.get("status") == "no_plan"

        # With resolver, ready=True is injected → planning succeeds
        resolver = FunctionalConditionResolver(
            "inject_ready",
            lambda k, ws: (
                ConditionStatus.TRUE if k == "ready" else ConditionStatus.UNKNOWN
            ),
        )
        planner = GoapPlanner([action], resolvers=[resolver])
        state: GoapState = {"world_state": {}, "goal": goal}
        result = planner(state)
        assert (
            result.get("plan") is not None
        ), "Planner should have found a plan after resolver filled 'ready'"

    @pytest.mark.asyncio
    async def test_async_resolver_fills_missing_precondition(self) -> None:
        from langgoap.actions import ActionSpec
        from langgoap.goals import GoalSpec
        from langgoap.graph.nodes import GoapPlanner
        from langgoap.graph.state import GoapState

        action = ActionSpec(
            name="async_act",
            preconditions={"enabled": True},
            effects={"result": True},
        )
        goal = GoalSpec(conditions={"result": True})
        resolver = FunctionalConditionResolver(
            "inject_enabled",
            lambda k, ws: (
                ConditionStatus.TRUE if k == "enabled" else ConditionStatus.UNKNOWN
            ),
        )
        planner = GoapPlanner([action], resolvers=[resolver])
        state: GoapState = {"world_state": {}, "goal": goal}
        result = await planner.acall(state)
        assert result.get("plan") is not None
