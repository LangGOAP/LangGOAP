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
        # The resolved goal is exposed as a public attribute so callers
        # can pass it back to ``invoke({"goal": ...})``.
        resolved = agent.goap_goal
        assert dict(resolved.conditions) == {"report_written": True}
        # And the agent still runs end-to-end.
        result = agent.invoke(
            {
                "world_state": {},
                "goal": resolved,
            }
        )
        assert result["status"] == "goal_achieved"


class TestReadmeQuickstart:
    """Mirrors the exact Quickstart example shown in the project README.

    The only difference from the README snippet is that ``ChatOpenAI`` is
    replaced with :class:`FakeStructuredModel` so the test never hits the
    network.  Any divergence here means the README has drifted from the
    actual ``create_goap_agent`` API and must be corrected.
    """

    def test_readme_quickstart_runs_end_to_end(self) -> None:
        from langchain_core.tools import tool as _tool

        from langgoap.interpreter import InterpretedGoal
        from tests.conftest import FakeStructuredModel

        @_tool
        def research_topic(topic: str) -> str:
            """Produce a short research brief for a topic."""
            return f"Brief on {topic}"

        @_tool
        def write_article(brief: str) -> str:
            """Turn a research brief into an article draft."""
            return f"Article from: {brief}"

        @_tool
        def publish_article(draft: str) -> str:
            """Publish an article draft."""
            return f"Published: {draft}"

        fake_llm = FakeStructuredModel(
            response=InterpretedGoal(
                conditions={"published": True},
                reasoning="user asked to publish an article",
            )
        )

        agent = create_goap_agent(
            tools=[research_topic, write_article, publish_article],
            goal="Publish an article about GOAP for LangGraph",
            llm=fake_llm,
            preconditions={
                "write_article":   {"have_brief": True},
                "publish_article": {"have_draft": True},
            },
            effects={
                "research_topic":  {"have_brief": True},
                "write_article":   {"have_draft": True},
                "publish_article": {"published":  True},
            },
            costs={
                "research_topic":  1.0,
                "write_article":   3.0,
                "publish_article": 1.0,
            },
            # Wire each tool's return value into the next tool's input
            # under its required argument name.
            result_keys={
                "research_topic": "brief",
                "write_article":  "draft",
            },
        )

        result = agent.invoke(
            {
                "world_state": {"topic": "GOAP for LangGraph"},
                "goal": agent.goap_goal,
            }
        )

        assert result["status"] == "goal_achieved"
        assert result["plan"].action_names == [
            "research_topic",
            "write_article",
            "publish_article",
        ]
        # The cost-weighted total of the chosen plan: 1.0 + 3.0 + 1.0 = 5.0
        assert result["plan"].total_cost == pytest.approx(5.0)
        # The result_keys plumbing should have bridged each tool's output
        # into the next tool's input, leaving a coherent trail in world_state.
        ws = result["world_state"]
        assert ws["brief"] == "Brief on GOAP for LangGraph"
        assert ws["draft"] == "Article from: Brief on GOAP for LangGraph"


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
