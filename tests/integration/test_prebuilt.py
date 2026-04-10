"""Integration tests for ``langgoap.integrations.prebuilt.create_goap_agent``.

Layer A of the three-layer low-code on-ramp (AD-2).
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph

from langgoap.goals import GoalSpec
from langgoap.integrations import create_goap_agent


@tool
def gather_data() -> str:
    """Gather raw data."""
    return "raw data"


@tool
def process_data() -> str:
    """Process gathered data."""
    return "processed"


@tool
def write_report() -> str:
    """Write the final report."""
    return "report.pdf"


class TestCreateGoapAgentBasic:
    def test_returns_compiled_state_graph(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, process_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "process_data": {"data_processed": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "process_data": {"has_data": True},
                "write_report": {"data_processed": True},
            },
        )
        assert isinstance(agent, CompiledStateGraph)

    def test_end_to_end_execution(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, process_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "process_data": {"data_processed": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "process_data": {"has_data": True},
                "write_report": {"data_processed": True},
            },
        )
        result = agent.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"report_written": True}),
            }
        )
        assert result["status"] == "goal_achieved"
        # plan should chain the three tools
        plan = result["plan"]
        assert plan is not None
        assert plan.action_names == ["gather_data", "process_data", "write_report"]


class TestCreateGoapAgentErrors:
    def test_string_goal_without_llm_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="string goal but no llm"):
            create_goap_agent(
                tools=[gather_data],
                goal="write a report",
            )

    def test_warns_on_tools_without_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations"):
            create_goap_agent(
                tools=[gather_data, process_data],
                goal=GoalSpec(conditions={"done": True}),
            )
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("gather_data" in w.message for w in warnings)
        assert any("process_data" in w.message for w in warnings)

    def test_no_warning_when_all_tools_have_effects(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="langgoap.integrations"):
            create_goap_agent(
                tools=[gather_data],
                goal=GoalSpec(conditions={"has_data": True}),
                effects={"gather_data": {"has_data": True}},
            )
        warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "has no" in r.message
        ]
        assert warnings == []


class TestCreateGoapAgentNLGoal:
    def test_string_goal_with_llm_invokes_interpreter(self) -> None:
        from langgoap.interpreter import InterpretedGoal
        from tests.conftest import FakeStructuredModel

        fake = FakeStructuredModel(
            response=InterpretedGoal(
                conditions={"report_written": True},
                reasoning="user asked for a written report",
            )
        )
        agent = create_goap_agent(
            tools=[gather_data, write_report],
            goal="please write me a report",
            llm=fake,
            effects={
                "gather_data": {"has_data": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "write_report": {"has_data": True},
            },
        )
        # The resolved goal is stashed for diagnostics.
        resolved = getattr(agent, "_langgoap_goal")
        assert dict(resolved.conditions) == {"report_written": True}
        # And the agent still runs end-to-end.
        result = agent.invoke(
            {
                "world_state": {},
                "goal": resolved,
            }
        )
        assert result["status"] == "goal_achieved"


class TestCreateGoapAgentResources:
    def test_forwards_resources(self) -> None:
        agent = create_goap_agent(
            tools=[gather_data, write_report],
            goal=GoalSpec(conditions={"report_written": True}),
            effects={
                "gather_data": {"has_data": True},
                "write_report": {"report_written": True},
            },
            preconditions={
                "write_report": {"has_data": True},
            },
            resources={
                "gather_data": {"cost_usd": 0.01},
                "write_report": {"cost_usd": 0.05},
            },
        )
        result = agent.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"report_written": True}),
            }
        )
        assert result["status"] == "goal_achieved"
