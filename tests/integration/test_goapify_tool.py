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
    return f"hello {name}"


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
