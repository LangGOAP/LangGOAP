"""Integration tests for the DeepAgents integration module.

Tests ``create_goap_tool``, ``create_goap_subagent``, and
``format_goap_result`` using ``FakeStructuredModel`` so no real LLM
is needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from langgoap.actions import ActionSpec
from langgoap.interpreter import InterpretedGoal
from langgoap.testing import FakeStructuredModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cicd_actions() -> list[ActionSpec]:
    """CI/CD pipeline: install_deps → lint → tests → build → deploy."""
    return [
        ActionSpec(
            name="install_deps",
            preconditions={},
            effects={"deps_installed": True},
            cost=1.0,
        ),
        ActionSpec(
            name="lint",
            preconditions={"deps_installed": True},
            effects={"lint_passed": True},
            cost=1.0,
        ),
        ActionSpec(
            name="run_tests",
            preconditions={"deps_installed": True},
            effects={"tests_passed": True},
            cost=2.0,
        ),
        ActionSpec(
            name="build",
            preconditions={"lint_passed": True, "tests_passed": True},
            effects={"build_ready": True},
            cost=3.0,
        ),
        ActionSpec(
            name="deploy",
            preconditions={"build_ready": True},
            effects={"deployed": True},
            cost=1.0,
        ),
    ]


def _simple_actions() -> list[ActionSpec]:
    """Minimal two-step action chain for basic tests."""
    return [
        ActionSpec(
            name="gather",
            preconditions={},
            effects={"data": True},
            cost=1.0,
        ),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            cost=1.0,
        ),
    ]


def _fake_llm(conditions: dict[str, Any] | None = None) -> FakeStructuredModel:
    """Return a ``FakeStructuredModel`` that produces an ``InterpretedGoal``."""
    conds = conditions or {"done": True}
    return FakeStructuredModel(
        response=InterpretedGoal(
            conditions=conds,
            constraints=[],
            objectives=[],
            reasoning="Test goal interpretation.",
        )
    )


# ---------------------------------------------------------------------------
# TestCreateGoapTool
# ---------------------------------------------------------------------------


class TestCreateGoapTool:
    def test_returns_base_tool(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
        )
        assert isinstance(tool, BaseTool)

    def test_custom_name_and_description(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
            name="my_planner",
            description="Plans stuff.",
        )
        assert tool.name == "my_planner"
        assert tool.description == "Plans stuff."

    def test_default_name_and_description(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
        )
        assert tool.name == "goap_planner"
        assert "GOAP" in tool.description

    def test_invoke_runs_goap_loop(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
        )
        result = tool.invoke({"request": "Process the data."})
        assert isinstance(result, str)
        assert "goal_achieved" in result

    @pytest.mark.asyncio
    async def test_ainvoke_runs_goap_loop(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
        )
        result = await tool.ainvoke({"request": "Process the data."})
        assert isinstance(result, str)
        assert "goal_achieved" in result

    def test_with_custom_world_state(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
            world_state={"data": True},
        )
        result = tool.invoke({"request": "Process the data."})
        assert isinstance(result, str)
        assert "goal_achieved" in result

    def test_graph_kwargs_forwarded(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        # tracer=None is a valid graph kwarg; just verifying it doesn't error.
        tool = create_goap_tool(
            _simple_actions(),
            llm=_fake_llm(),
            graph_kwargs={"tracer": None},
        )
        assert isinstance(tool, BaseTool)


# ---------------------------------------------------------------------------
# TestCreateGoapSubagent
# ---------------------------------------------------------------------------


class TestCreateGoapSubagent:
    def test_returns_dict_with_required_keys(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _simple_actions(),
            llm=_fake_llm(),
        )
        assert isinstance(spec, dict)
        assert "name" in spec
        assert "description" in spec
        assert "runnable" in spec

    def test_custom_name_and_description(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _simple_actions(),
            llm=_fake_llm(),
            name="my_goap_agent",
            description="Plans things.",
        )
        assert spec["name"] == "my_goap_agent"
        assert spec["description"] == "Plans things."

    def test_runnable_accepts_messages_state(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _simple_actions(),
            llm=_fake_llm(),
        )
        runnable = spec["runnable"]
        result = runnable.invoke(
            {"messages": [HumanMessage(content="Process the data.")]}
        )
        assert "messages" in result
        assert len(result["messages"]) >= 1

    def test_final_message_contains_goap_result(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _simple_actions(),
            llm=_fake_llm(),
        )
        result = spec["runnable"].invoke(
            {"messages": [HumanMessage(content="Process the data.")]}
        )
        final_msg = result["messages"][-1]
        assert "goal_achieved" in final_msg.content

    @pytest.mark.asyncio
    async def test_runnable_ainvoke_messages_state(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _simple_actions(),
            llm=_fake_llm(),
        )
        result = await spec["runnable"].ainvoke(
            {"messages": [HumanMessage(content="Process the data.")]}
        )
        assert "messages" in result
        final_msg = result["messages"][-1]
        assert "goal_achieved" in final_msg.content


# ---------------------------------------------------------------------------
# TestFormatGoapResult
# ---------------------------------------------------------------------------


class TestFormatGoapResult:
    def test_format_success(self) -> None:
        from langgoap.integrations.deepagents import format_goap_result

        result = {
            "status": "goal_achieved",
            "world_state": {"data": True, "done": True},
        }
        text = format_goap_result(result)
        assert "goal_achieved" in text
        assert "done" in text

    def test_format_failure(self) -> None:
        from langgoap.integrations.deepagents import format_goap_result

        result = {
            "status": "no_plan",
            "world_state": {},
        }
        text = format_goap_result(result)
        assert "no_plan" in text

    def test_format_includes_plan_actions(self) -> None:
        from langgoap.integrations.deepagents import format_goap_result

        result = {
            "status": "goal_achieved",
            "world_state": {"done": True},
            "execution_history": [
                {"action": "gather", "success": True},
                {"action": "process", "success": True},
            ],
        }
        text = format_goap_result(result)
        assert "gather" in text
        assert "process" in text


# ---------------------------------------------------------------------------
# TestGoapToolCorrectOrdering
# ---------------------------------------------------------------------------


class TestGoapToolCorrectOrdering:
    """CI/CD pipeline scenario showing GOAP finds correct action ordering."""

    def test_cicd_pipeline_deploys_successfully(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _cicd_actions(),
            llm=_fake_llm(conditions={"deployed": True}),
        )
        result = tool.invoke({"request": "Deploy the application."})
        assert isinstance(result, str)
        assert "goal_achieved" in result

    @pytest.mark.asyncio
    async def test_cicd_pipeline_deploys_async(self) -> None:
        from langgoap.integrations.deepagents import create_goap_tool

        tool = create_goap_tool(
            _cicd_actions(),
            llm=_fake_llm(conditions={"deployed": True}),
        )
        result = await tool.ainvoke({"request": "Deploy the application."})
        assert isinstance(result, str)
        assert "goal_achieved" in result

    def test_cicd_subagent_deploys_successfully(self) -> None:
        from langgoap.integrations.deepagents import create_goap_subagent

        spec = create_goap_subagent(
            _cicd_actions(),
            llm=_fake_llm(conditions={"deployed": True}),
        )
        result = spec["runnable"].invoke(
            {"messages": [HumanMessage(content="Deploy the application.")]}
        )
        assert "messages" in result
        assert "goal_achieved" in result["messages"][-1].content
