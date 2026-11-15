"""``goapify_tool`` execution contracts (issues #37 and #38).

* A tool created with ``handle_tool_error=True`` returns the error text
  instead of raising.  Wrapped as an action that must still be an
  ``action_failed`` so the observer blacklists and replans; previously the
  declared effects were applied and the run ended ``goal_achieved``.
* Input filtering must work for a JSON-schema ``args_schema`` (a dict in
  langchain-core 1.x) and must forward Pydantic aliased fields under the
  alias the schema validates by.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from langgoap import GoalSpec, GoapGraph
from langgoap.integrations import goapify_tool


def _boom(x: int) -> str:
    raise ToolException("kaboom")


class TestHandledToolErrorsAreFailures:
    @pytest.mark.parametrize("handler", [True, "custom message"])
    def test_handled_tool_error_becomes_action_failed(self, handler: Any) -> None:
        tool = StructuredTool.from_function(
            func=_boom,
            name="boom",
            description="always fails",
            handle_tool_error=handler,
        )
        action = goapify_tool(tool, effects={"done": True}, result_key="out")
        result = GoapGraph([action]).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"x": 1}
        )
        assert result["status"] != "goal_achieved"
        failures = [h for h in result["execution_history"] if not h.success]
        assert failures and "kaboom" in (failures[0].error or "")
        assert "out" not in result["world_state"]

    def test_the_users_tool_object_is_not_mutated(self) -> None:
        tool = StructuredTool.from_function(
            func=_boom, name="boom", description="always fails", handle_tool_error=True
        )
        goapify_tool(tool, effects={"done": True})
        assert tool.handle_tool_error is True
        assert tool.invoke({"x": 1}) == "kaboom"

    def test_successful_tool_still_succeeds(self) -> None:
        tool = StructuredTool.from_function(
            func=lambda x: f"ok {x}",
            name="fine",
            description="works",
            handle_tool_error=True,
        )
        action = goapify_tool(tool, effects={"done": True}, result_key="out")
        result = GoapGraph([action]).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"x": 1}
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["out"] == "ok 1"


def _add(a: int, b: int) -> int:
    return a + b


class _GreetArgs(BaseModel):
    user_id: str = Field(alias="userId")


def _greet(user_id: str) -> str:
    return f"hi {user_id}"


class TestInputFiltering:
    def test_dict_args_schema_forwards_declared_keys(self) -> None:
        schema = {
            "type": "object",
            "title": "add",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        tool = StructuredTool.from_function(
            func=_add, name="add", description="add", args_schema=schema
        )
        action = goapify_tool(tool, effects={"summed": True}, result_key="total")
        assert action.execute is not None
        update = action.execute({"a": 2, "b": 3, "unrelated": "x"})
        assert update == {"summed": True, "total": 5}

    @pytest.mark.parametrize("key", ["user_id", "userId"])
    def test_aliased_field_is_accepted_by_name_or_alias(self, key: str) -> None:
        tool = StructuredTool.from_function(
            func=_greet, name="greet", description="g", args_schema=_GreetArgs
        )
        action = goapify_tool(tool, effects={"greeted": True}, result_key="greeting")
        assert action.execute is not None
        update = action.execute({key: "u1"})
        assert update == {"greeted": True, "greeting": "hi u1"}

    @pytest.mark.asyncio
    async def test_async_path_uses_the_same_filtering(self) -> None:
        tool = StructuredTool.from_function(
            func=_greet, name="greet", description="g", args_schema=_GreetArgs
        )
        action = goapify_tool(tool, effects={"greeted": True}, result_key="greeting")
        assert action.aexecute is not None
        update = await action.aexecute({"user_id": "u2"})
        assert update == {"greeted": True, "greeting": "hi u2"}


class TestToolDescriptionReachesTheCatalog:
    """Issue #56: the docstring written for the LLM must survive wrapping."""

    def _tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=_greet,
            name="greet",
            description="Greet a user by their id.",
            args_schema=_GreetArgs,
        )

    def test_metadata_carries_description_and_args(self) -> None:
        spec = goapify_tool(self._tool(), effects={"greeted": True})
        assert spec.metadata is not None
        assert spec.metadata["description"] == "Greet a user by their id."
        assert "user_id" in spec.metadata["args"]

    def test_user_metadata_merges_and_wins(self) -> None:
        spec = goapify_tool(
            self._tool(),
            effects={"greeted": True},
            metadata={"team": "ops", "description": "custom"},
        )
        assert spec.metadata is not None
        assert spec.metadata["description"] == "custom"
        assert spec.metadata["team"] == "ops"
        assert "args" in spec.metadata

    def test_catalog_shows_description_and_args(self) -> None:
        from langgoap.interpreter import build_action_catalog

        text = build_action_catalog(
            [goapify_tool(self._tool(), effects={"greeted": True})]
        )
        assert "Greet a user by their id." in text
        assert "user_id" in text
