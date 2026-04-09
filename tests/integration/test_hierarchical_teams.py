"""Integration tests for GOAPified Hierarchical Agent Teams.

The original LangGraph Hierarchical Agent Teams uses LLM-based supervisors
to route between team members (research team, writing team). In the GOAPified
version, agent capabilities become GOAP actions with formal preconditions
(what the agent needs) and effects (what it produces). The planner automatically
selects and orders agents without supervisor LLM calls.

Key advantages over the original:
- No supervisor LLM calls: routing is formal planning, not LLM prompting
- Automatic team selection: planner picks the right agents based on capabilities
- Cost-optimal ordering: A* finds the cheapest path through available agents
- Verifiable plans: the agent sequence is type-checked via preconditions/effects

Reference: research/repos/langgraph/docs/docs/tutorials/multi_agent/hierarchical_agent_teams.ipynb
"""

from __future__ import annotations

from typing import Any

import pytest
from tutorial_examples.hierarchical_teams import (
    doc_writer_agent,
    full_team_actions,
    note_taker_agent,
    search_agent,
    web_scraper_agent,
)

from langgoap import ActionSpec, GoalSpec, GoapGraph, ReplanStrategy


class TestHierarchicalTeamsGoapified:
    """GOAPified Hierarchical Agent Teams: A* replaces LLM supervisors."""

    def test_full_research_and_writing_pipeline(self) -> None:
        """Planner discovers: search → scrape → notes → write (optimal path)."""
        actions = full_team_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_document": True}),
            world_state={"has_topic": True, "topic": "AI agents"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["has_document"] is True
        assert "AI agents" in ws["document"]

        # Verify the sequence: research team first, then writing team
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == [
            "search_agent",
            "web_scraper_agent",
            "note_taker_agent",
            "doc_writer_agent",
        ]

    def test_research_only_goal(self) -> None:
        """When goal only requires research, writing team is not invoked."""
        actions = full_team_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_detailed_content": True}),
            world_state={"has_topic": True, "topic": "LLM planning"},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert successful == ["search_agent", "web_scraper_agent"]
        # No writing team agents should have run
        assert "doc_writer_agent" not in successful
        assert "note_taker_agent" not in successful

    def test_visualization_requires_research_not_writing(self) -> None:
        """Chart generation requires detailed_content but not outline/document."""
        actions = full_team_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_visualization": True}),
            world_state={"has_topic": True, "topic": "agent architectures"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["has_visualization"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # Should go through research → chart, NOT through outline → doc_writer
        assert "search_agent" in successful
        assert "web_scraper_agent" in successful
        assert "chart_generator_agent" in successful
        assert "doc_writer_agent" not in successful

    def test_multi_goal_document_and_visualization(self) -> None:
        """Achieving both document and visualization requires broader planning.

        Since LangGoap currently supports single goals, we combine them into
        one goal with both conditions. The planner must find a path that
        satisfies both.
        """
        actions = full_team_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(
                conditions={
                    "has_document": True,
                    "has_visualization": True,
                }
            ),
            world_state={"has_topic": True, "topic": "AI agents"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        assert ws["has_document"] is True
        assert ws["has_visualization"] is True

        successful = [h.action_name for h in result["execution_history"] if h.success]
        # All five agents should run
        assert "search_agent" in successful
        assert "web_scraper_agent" in successful
        assert "note_taker_agent" in successful
        assert "doc_writer_agent" in successful
        assert "chart_generator_agent" in successful

    def test_missing_capability_reports_no_plan(self) -> None:
        """When required capabilities are missing, planner reports no_plan."""
        # Only writing team, no research team
        actions = [
            ActionSpec(
                name="note_taker_agent",
                preconditions={"has_detailed_content": True},
                effects={"has_outline": True},
                execute=note_taker_agent,
            ),
            ActionSpec(
                name="doc_writer_agent",
                preconditions={"has_outline": True},
                effects={"has_document": True},
                execute=doc_writer_agent,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_document": True}),
            world_state={"has_topic": True},
        )

        # No path from has_topic to has_document without research team
        assert result["status"] == "no_plan"

    def test_agent_failure_triggers_replan(self) -> None:
        """When a team member fails, observer triggers replanning."""
        call_count = {"scraper": 0}

        def flaky_scraper(ws: dict[str, Any]) -> dict[str, Any]:
            call_count["scraper"] += 1
            if call_count["scraper"] == 1:
                raise RuntimeError("Page load timeout")
            return web_scraper_agent(ws)

        actions = [
            ActionSpec(
                name="search_agent",
                preconditions={"has_topic": True},
                effects={"has_search_results": True},
                execute=search_agent,
            ),
            ActionSpec(
                name="web_scraper_agent",
                preconditions={"has_search_results": True},
                effects={"has_detailed_content": True},
                execute=flaky_scraper,
            ),
            ActionSpec(
                name="note_taker_agent",
                preconditions={"has_detailed_content": True},
                effects={"has_outline": True},
                execute=note_taker_agent,
            ),
            ActionSpec(
                name="doc_writer_agent",
                preconditions={"has_outline": True},
                effects={"has_document": True},
                execute=doc_writer_agent,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_document": True}),
            world_state={"has_topic": True, "topic": "AI agents"},
        )

        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        assert call_count["scraper"] >= 2

    def test_cost_based_agent_selection(self) -> None:
        """When multiple agents can produce the same effect, planner picks cheapest."""

        def quick_search(ws: dict[str, Any]) -> dict[str, Any]:
            return {"has_search_results": True, "search_results": ["quick result"]}

        def deep_search(ws: dict[str, Any]) -> dict[str, Any]:
            return {
                "has_search_results": True,
                "search_results": ["deep result 1", "deep result 2", "deep result 3"],
            }

        actions = [
            ActionSpec(
                name="quick_search",
                preconditions={"has_topic": True},
                effects={"has_search_results": True},
                cost=1.0,
                execute=quick_search,
            ),
            ActionSpec(
                name="deep_search",
                preconditions={"has_topic": True},
                effects={"has_search_results": True},
                cost=5.0,
                execute=deep_search,
            ),
        ]

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_search_results": True}),
            world_state={"has_topic": True},
        )

        assert result["status"] == "goal_achieved"
        successful = [h.action_name for h in result["execution_history"] if h.success]
        # Planner picks the cheaper option
        assert successful == ["quick_search"]

    def test_rich_data_flows_between_teams(self) -> None:
        """Rich execution data (lists, strings) flows correctly between agents."""
        actions = full_team_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"has_document": True}),
            world_state={"has_topic": True, "topic": "AI agents"},
        )

        assert result["status"] == "goal_achieved"
        ws = result["world_state"]
        # Research team output available to writing team
        assert isinstance(ws["search_results"], list)
        assert isinstance(ws["detailed_content"], str)
        assert isinstance(ws["outline"], list)
        assert len(ws["outline"]) == 5
        assert isinstance(ws["document"], str)
