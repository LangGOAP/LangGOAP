r"""Integration test for the Robot Navigation tutorial (Tier 1 primer).

Exercises the A* planner's path-finding on two small domains:

- ``linear_corridor`` — n sequential locations, single path solution.
- ``weighted_grid`` — 5-node bidirectional graph where the cheapest
  route is not the one with the fewest edges.

Shared helpers live in
``examples/tutorials/tutorial_examples/robot_navigation.py``.
"""

from __future__ import annotations

from tutorial_examples.robot_navigation import (
    linear_corridor_actions,
    linear_corridor_start,
    weighted_grid_actions,
    weighted_grid_start,
)

from langgoap import GoalSpec, GoapGraph


class TestLinearCorridor:
    """Linear corridor: robot walks from l0 to l{n}."""

    def test_three_step_corridor(self) -> None:
        """l0 → l1 → l2 → l3 (3 moves)."""
        actions = linear_corridor_actions(n_segments=3)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_l3": True}),
            world_state=linear_corridor_start(n_segments=3),
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["at_l3"] is True
        assert result["world_state"]["at_l0"] is False

        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert successful == ["move_l0_to_l1", "move_l1_to_l2", "move_l2_to_l3"]

    def test_ten_step_corridor(self) -> None:
        """l0 → … → l10 (10 moves)."""
        actions = linear_corridor_actions(n_segments=10)
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_l10": True}),
            world_state=linear_corridor_start(n_segments=10),
        )

        assert result["status"] == "goal_achieved"
        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert len(successful) == 10
        assert successful[0] == "move_l0_to_l1"
        assert successful[-1] == "move_l9_to_l10"

    def test_already_at_goal(self) -> None:
        """When the robot starts at the goal, no moves are executed."""
        actions = linear_corridor_actions(n_segments=3)
        state = linear_corridor_start(n_segments=3)
        state["at_l0"] = False
        state["at_l3"] = True

        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_l3": True}),
            world_state=state,
        )

        assert result["status"] == "goal_achieved"
        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert successful == []


class TestWeightedGrid:
    """Weighted grid: A* finds the cheapest route, not the shortest path."""

    def test_cheapest_route_wins(self) -> None:
        """A → B → C → E (cost 3) beats A → D → E (cost 5)."""
        actions = weighted_grid_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_E": True}),
            world_state=weighted_grid_start("A"),
        )

        assert result["status"] == "goal_achieved"
        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert successful == ["move_A_to_B", "move_B_to_C", "move_C_to_E"]

    def test_reverse_direction(self) -> None:
        """Bidirectional edges let the robot return E → A via the same cheap path."""
        actions = weighted_grid_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_A": True}),
            world_state=weighted_grid_start("E"),
        )

        assert result["status"] == "goal_achieved"
        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert successful == ["move_E_to_C", "move_C_to_B", "move_B_to_A"]

    def test_intermediate_goal(self) -> None:
        """Goal = reach C starting from A."""
        actions = weighted_grid_actions()
        result = GoapGraph(actions=actions).invoke(
            goal=GoalSpec(conditions={"at_C": True}),
            world_state=weighted_grid_start("A"),
        )

        assert result["status"] == "goal_achieved"
        successful = [
            h.action_name for h in result["execution_history"] if h.success
        ]
        assert successful == ["move_A_to_B", "move_B_to_C"]
