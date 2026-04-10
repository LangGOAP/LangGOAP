"""Integration test for the Hungry Agent tutorial (Tier 1 primer).

Exercises the full NL → GoalInterpreter → GoalSpec → A* → execution
loop.  A natural-language request ("I'm tired and hungry") is
translated into a GoalSpec via a FakeStructuredModel, and A* is
expected to pick the cheapest available eating action based on the
current world state.

Shared helpers live in
``examples/tutorials/tutorial_examples/hungry_agent.py``.
"""

from __future__ import annotations

from tutorial_examples.hungry_agent import (
    hungry_agent_actions,
    hungry_agent_start,
)

from langgoap import GoalInterpreter, GoapGraph, InterpretedGoal
from tests.conftest import FakeStructuredModel


def _fake_llm() -> FakeStructuredModel:
    """Return a FakeStructuredModel that maps 'tired and hungry' → the goal."""
    return FakeStructuredModel(
        response=InterpretedGoal(
            conditions={"hungry": False, "tired": False},
            reasoning="User is tired and hungry; needs to eat and rest.",
        ),
    )


class TestHungryAgent:
    """Hungry agent: NL intake + cost-driven action selection."""

    def test_snack_available_picks_cheapest_eat_and_sleep(self) -> None:
        """With a snack on hand, eat_snack (1) beats cook_meal (3) and delivery (5)."""
        actions = hungry_agent_actions()
        state = hungry_agent_start(has_snack=True, has_ingredients=True)

        llm = _fake_llm()
        interpreter = GoalInterpreter(llm=llm, actions=actions)
        goal = interpreter.interpret("I'm tired and hungry, figure it out")

        result = GoapGraph(actions=actions).invoke(goal=goal, world_state=state)

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["hungry"] is False
        assert result["world_state"]["tired"] is False

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert set(successful) == {"eat_snack", "sleep"}

    def test_no_snack_but_ingredients_picks_cook_meal(self) -> None:
        """Without a snack, cook_meal (3) is cheaper than order_delivery (5)."""
        actions = hungry_agent_actions()
        state = hungry_agent_start(has_snack=False, has_ingredients=True)

        llm = _fake_llm()
        graph = GoapGraph(actions=actions)
        result = graph.invoke_nl(
            "I'm tired and hungry, figure it out",
            llm=llm,
            world_state=state,
        )

        assert result["status"] == "goal_achieved"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert set(successful) == {"cook_meal", "sleep"}

    def test_nothing_at_home_falls_back_to_delivery(self) -> None:
        """With no snack and no ingredients, delivery is the only path."""
        actions = hungry_agent_actions()
        state = hungry_agent_start(has_snack=False, has_ingredients=False)

        llm = _fake_llm()
        graph = GoapGraph(actions=actions)
        result = graph.invoke_nl(
            "I'm tired and hungry, figure it out",
            llm=llm,
            world_state=state,
        )

        assert result["status"] == "goal_achieved"

        successful = [h.action_name for h in result["execution_history"] if h.success]
        assert set(successful) == {"order_delivery", "sleep"}

    def test_plan_total_cost_matches_expected(self) -> None:
        """Verify A* actually minimized cost: eat_snack + sleep = 2.0."""
        from langgoap import GoalSpec
        from langgoap.planner.astar import plan as astar_plan
        from langgoap.state import PlanningState

        actions = hungry_agent_actions()
        state = hungry_agent_start(has_snack=True, has_ingredients=True)
        goal = GoalSpec(conditions={"hungry": False, "tired": False})

        plan_obj = astar_plan(
            PlanningState.from_dict(state),
            goal,
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.total_cost == 2.0
        assert set(plan_obj.action_names) == {"eat_snack", "sleep"}
