"""Integration tests for ``langgoap.integrations.tools.goapify_tool``.

The adapter converts a LangChain ``BaseTool`` to a LangGoap
``ActionSpec`` with explicit pre/eff/cost/resources.  Layer B of
the three-layer low-code on-ramp (AD-2).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langgoap.actions import ActionSpec
from langgoap.integrations import goapify_tool


@tool
def double_number(x: int) -> int:
    """Return x doubled."""
    return x * 2


@tool
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


class TestGoapifyToolBasic:
    def test_wraps_langchain_tool_into_action_spec(self) -> None:
        spec = goapify_tool(
            double_number,
            preconditions={"has_input": True},
            effects={"result_ready": True},
            cost=2.0,
        )
        assert isinstance(spec, ActionSpec)
        assert spec.name == "double_number"
        assert dict(spec.preconditions) == {"has_input": True}
        assert dict(spec.effects) == {"result_ready": True}
        assert spec.cost == 2.0

    def test_empty_pre_and_eff_allowed(self) -> None:
        spec = goapify_tool(double_number)
        assert spec.name == "double_number"
        assert dict(spec.preconditions) == {}
        assert dict(spec.effects) == {}
        assert spec.cost == 1.0

    def test_forwards_resources_and_duration(self) -> None:
        spec = goapify_tool(
            greet,
            preconditions={"knows_name": True},
            effects={"greeted": True},
            resources={"tokens": 10.0},
            duration=timedelta(seconds=1),
        )
        assert spec.resources is not None
        assert dict(spec.resources) == {"tokens": 10.0}
        assert spec.duration == timedelta(seconds=1)

    def test_forwards_max_retries(self) -> None:
        spec = goapify_tool(double_number, max_retries=3)
        assert spec.max_retries == 3

    def test_forwards_effect_validator(self) -> None:
        def validator(pre: dict[str, Any], post: dict[str, Any]) -> bool:
            return post.get("result_ready") is True

        spec = goapify_tool(
            double_number,
            effects={"result_ready": True},
            effect_validator=validator,
        )
        assert spec.effect_validator is validator


class TestGoapifyToolExecute:
    def test_execute_calls_underlying_tool(self) -> None:
        spec = goapify_tool(
            double_number,
            preconditions={"has_input": True},
            effects={"result_ready": True},
        )
        assert spec.execute is not None
        # The execute wrapper should accept a state dict and call the tool.
        result = spec.execute({"has_input": True, "x": 7})
        # The wrapper returns the effects dict; the tool's return value is
        # recorded in metadata or ignored — actions produce state changes,
        # not return values.
        assert isinstance(result, dict)
        assert result.get("result_ready") is True

    def test_tool_call_without_args_schema_uses_state_passthrough(self) -> None:
        @tool
        def touch() -> str:
            """A no-arg tool."""
            return "ok"

        spec = goapify_tool(touch, effects={"touched": True})
        assert spec.execute is not None
        result = spec.execute({})
        assert result.get("touched") is True


class TestGoapifyToolResultKey:
    """result_key bridges the tool return value into world_state."""

    def test_default_no_result_key_returns_effects_only(self) -> None:
        spec = goapify_tool(greet, effects={"greeted": True})
        assert spec.execute is not None
        result = spec.execute({"name": "Alice"})
        assert set(result.keys()) == {"greeted"}

    def test_result_key_stored_alongside_effects(self) -> None:
        spec = goapify_tool(
            greet,
            effects={"greeted": True},
            result_key="greeting_text",
        )
        assert spec.execute is not None
        result = spec.execute({"name": "Bob"})
        assert result["greeted"] is True
        assert result["greeting_text"] == "Hello, Bob!"

    def test_result_key_absent_from_action_spec_effects(self) -> None:
        """result_key must not pollute ActionSpec.effects — planning is boolean."""
        spec = goapify_tool(
            greet,
            effects={"greeted": True},
            result_key="greeting_text",
        )
        assert "greeting_text" not in dict(spec.effects)

    def test_result_key_captures_numeric_output(self) -> None:
        spec = goapify_tool(
            double_number,
            effects={"doubled": True},
            result_key="doubled_value",
        )
        assert spec.execute is not None
        result = spec.execute({"x": 6})
        assert result["doubled"] is True
        assert result["doubled_value"] == 12

    def test_result_key_none_is_the_same_as_omitting_it(self) -> None:
        spec_none = goapify_tool(greet, effects={"greeted": True}, result_key=None)
        spec_omit = goapify_tool(greet, effects={"greeted": True})
        assert spec_none.execute is not None
        assert spec_omit.execute is not None
        r_none = spec_none.execute({"name": "X"})
        r_omit = spec_omit.execute({"name": "X"})
        assert r_none == r_omit


class TestGoapifyToolCollisionValidation:
    """Construction-time guard: result_key must not overlap with effects keys."""

    def test_colliding_result_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="collides"):
            goapify_tool(greet, effects={"greeted": True}, result_key="greeted")

    def test_non_colliding_result_key_does_not_raise(self) -> None:
        action = goapify_tool(
            greet, effects={"greeted": True}, result_key="greeting_text"
        )
        assert action.name == "greet"

    def test_result_key_none_never_raises(self) -> None:
        action = goapify_tool(greet, effects={"greeted": True}, result_key=None)
        assert action.name == "greet"


class TestGoapifyToolAexecuteIntegration:
    """aexecute must be set and produce output identical to the sync path."""

    def test_aexecute_present_by_default(self) -> None:
        action = goapify_tool(greet, effects={"greeted": True})
        assert action.aexecute is not None

    @pytest.mark.asyncio
    async def test_aexecute_returns_effects_only_without_result_key(self) -> None:
        action = goapify_tool(greet, effects={"greeted": True})
        assert action.aexecute is not None
        result = await action.aexecute({"name": "Alice"})
        assert result == {"greeted": True}

    @pytest.mark.asyncio
    async def test_aexecute_captures_result_key(self) -> None:
        action = goapify_tool(
            greet, effects={"greeted": True}, result_key="greeting_text"
        )
        assert action.aexecute is not None
        result = await action.aexecute({"name": "Bob"})
        assert result["greeted"] is True
        assert result["greeting_text"] == "Hello, Bob!"

    @pytest.mark.asyncio
    async def test_aexecute_output_matches_sync_execute(self) -> None:
        action = goapify_tool(
            double_number, effects={"doubled": True}, result_key="doubled_value"
        )
        assert action.execute is not None
        assert action.aexecute is not None
        sync_result = action.execute({"x": 7})
        async_result = await action.aexecute({"x": 7})
        assert sync_result == async_result


class TestGoapifyToolRejects:
    def test_rejects_non_base_tool(self) -> None:
        def not_a_tool(x: int) -> int:
            return x

        with pytest.raises(TypeError, match="BaseTool"):
            goapify_tool(not_a_tool)  # type: ignore[arg-type]

    def test_accepts_custom_basetool_subclass(self) -> None:
        class MyTool(BaseTool):
            name: str = "my_tool"
            description: str = "custom tool"

            def _run(self, **kwargs: Any) -> str:
                return "result"

        spec = goapify_tool(MyTool(), effects={"done": True})
        assert spec.name == "my_tool"
        assert dict(spec.effects) == {"done": True}
