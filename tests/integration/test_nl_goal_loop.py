"""Integration tests for the NL → GoalSpec → GOAP execution loop.

Verifies end-to-end behavior: natural language request interpreted via
FakeStructuredModel, converted to GoalSpec, and executed through the
full GOAP graph.
"""

from __future__ import annotations

import pytest

from langgoap import (
    ActionSpec,
    CSPStatus,
    GoalInterpreter,
    GoapGraph,
    InterpretedConstraint,
    InterpretedGoal,
)
from tests.conftest import FakeStructuredModel, make_action

# ---------------------------------------------------------------------------
# Shared action setup
# ---------------------------------------------------------------------------


def _report_pipeline_actions() -> list[ActionSpec]:
    """3-action pipeline: fetch → clean → report (with resources)."""
    return [
        make_action(
            "fetch_data",
            eff={"data_fetched": True},
            resources={"cost_usd": 0.5, "api_calls": 1},
        ),
        make_action(
            "clean_data",
            pre={"data_fetched": True},
            eff={"data_clean": True},
            resources={"cost_usd": 0.1},
        ),
        make_action(
            "generate_report",
            pre={"data_clean": True},
            eff={"report_complete": True},
            resources={"cost_usd": 1.0, "tokens": 500},
        ),
    ]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestNLGoalLoop:
    def test_nl_report_pipeline_achieves_goal(self) -> None:
        """NL request → GoalSpec → full GOAP loop achieves goal."""
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="User wants a report generated from fetched data.",
        )
        llm = FakeStructuredModel(response=fake_response)
        interpreter = GoalInterpreter(llm=llm, actions=actions)
        goal = interpreter.interpret("Generate a report from the raw data")

        graph = GoapGraph(actions=actions)
        result = graph.invoke(goal=goal, world_state={})

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["report_complete"] is True
        assert result["world_state"]["data_fetched"] is True
        assert result["world_state"]["data_clean"] is True
        history_names = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert "fetch_data" in history_names
        assert "clean_data" in history_names
        assert "generate_report" in history_names

    def test_invoke_nl_convenience_achieves_goal(self) -> None:
        """GoapGraph.invoke_nl() one-liner achieves goal."""
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="Convenience test.",
        )
        llm = FakeStructuredModel(response=fake_response)
        graph = GoapGraph(actions=actions)

        result = graph.invoke_nl("Generate a report from the data", llm=llm)

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["report_complete"] is True

    def test_nl_with_tight_constraint_feasible(self) -> None:
        """NL budget constraint that is tight but feasible: CSP runs and passes (M4).

        Pipeline cost: $0.5 + $0.1 + $1.0 = $1.60.
        Budget cap: $2.00 — tight enough that CSP must actually evaluate it,
        but wide enough for the full plan to be accepted as FEASIBLE/OPTIMAL.
        """
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"report_complete": True},
            constraints=[InterpretedConstraint(key="cost_usd", max=2.0)],
            reasoning="User set a $2 budget.",
        )
        llm = FakeStructuredModel(response=fake_response)
        interpreter = GoalInterpreter(llm=llm, actions=actions)
        goal = interpreter.interpret("Generate a report under $2")

        assert len(goal.constraints) == 1
        assert goal.constraints[0].key == "cost_usd"
        assert goal.constraints[0].max == 2.0

        graph = GoapGraph(actions=actions)
        result = graph.invoke(goal=goal, world_state={})

        assert result["status"] == "goal_achieved"
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        assert plan_obj.metadata.csp.status in (
            CSPStatus.OPTIMAL,
            CSPStatus.FEASIBLE,
        )

    def test_nl_with_impossible_constraint_records_infeasible(self) -> None:
        """NL budget constraint tighter than plan cost: CSP records INFEASIBLE (M4).

        Pipeline cost: $0.5 + $0.1 + $1.0 = $1.60.
        Budget cap: $1.00 — below the total, so CSP marks the plan INFEASIBLE.

        CSP is advisory: the plan still executes (no execute callable → effects
        applied via defaults), but the CSP metadata must record the violation,
        proving the constraint was evaluated and not silently ignored.
        """
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"report_complete": True},
            constraints=[InterpretedConstraint(key="cost_usd", max=1.0)],
            reasoning="User set a $1 budget — impossible for this pipeline.",
        )
        llm = FakeStructuredModel(response=fake_response)
        graph = GoapGraph(actions=actions)

        result = graph.invoke_nl("Generate a report under $1", llm=llm)

        # CSP is advisory: plan executes despite constraint violation
        plan_obj = result.get("plan")
        assert plan_obj is not None, "graph must expose the executed plan"
        assert plan_obj.metadata.csp is not None, "CSP metadata must be attached"
        assert plan_obj.metadata.csp.status == CSPStatus.INFEASIBLE

    def test_nl_with_unreachable_goal_returns_no_plan(self) -> None:
        """NL request that produces an unreachable GoalSpec fails gracefully."""
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"impossible_condition": True},
            reasoning="LLM hallucinated a condition no action can produce.",
        )
        llm = FakeStructuredModel(response=fake_response)
        graph = GoapGraph(actions=actions)

        result = graph.invoke_nl("Do something impossible", llm=llm)

        assert result["status"] == "no_plan"

    @pytest.mark.asyncio
    async def test_ainvoke_nl_achieves_goal(self) -> None:
        """Async ainvoke_nl() one-liner achieves goal."""
        actions = _report_pipeline_actions()
        fake_response = InterpretedGoal(
            conditions={"report_complete": True},
            reasoning="Async convenience test.",
        )
        llm = FakeStructuredModel(response=fake_response)
        graph = GoapGraph(actions=actions)

        result = await graph.ainvoke_nl("Generate a report", llm=llm)

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["report_complete"] is True
