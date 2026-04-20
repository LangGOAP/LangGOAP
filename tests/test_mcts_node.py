"""Failing tests for :mod:`langgoap.planner.mcts` — node + UCB1 primitives.

Phase 4 of the advanced-planning research track lands an MCTS
planning strategy as an alternative to A* + CSP.  The node model and
UCB1 selection rule are adapted from LATS
(``research/repos/LATS/hotpot/lats.py``) — specifically the ``Node``
class and its ``uct`` method, which use the canonical c = √2
exploration constant with a running-mean value estimate.

LangGoap's node model differs from LATS in two ways:

1. **Action-edge, not state-node**: each child is reached by
   applying a concrete :class:`ActionSpec`, so the node stores the
   *action taken from its parent*, not a free-text state dict.
2. **Typed state**: the node carries a
   :class:`~langgoap.state.PlanningState` directly, aligning with
   :class:`AStarStrategy`'s contract and letting the compiled graph
   re-use the same satisfaction / apply machinery.

Pre-registered experiment:
``research/experiments/2026-04-20-mcts-vs-astar.md``.
"""

from __future__ import annotations

import math

import pytest

from langgoap.state import PlanningState


@pytest.fixture
def start_state() -> PlanningState:
    return PlanningState.from_dict({})


class TestMCTSNodeConstruction:
    def test_root_has_no_parent_or_action(self, start_state: PlanningState) -> None:
        from langgoap.planner.mcts import MCTSNode

        node = MCTSNode(state=start_state)
        assert node.parent is None
        assert node.action is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.depth == 0
        assert node.is_terminal is False

    def test_child_inherits_depth_plus_one(self, start_state: PlanningState) -> None:
        from langgoap.planner.mcts import MCTSNode

        root = MCTSNode(state=start_state)
        child = MCTSNode(state=start_state, parent=root, action=None)
        assert child.depth == 1

        grandchild = MCTSNode(state=start_state, parent=child, action=None)
        assert grandchild.depth == 2


class TestUCB1:
    """UCB1 matches LATS's canonical formulation with c = √2.

    Formula: ``value/visits + c · sqrt(ln(parent.visits) / visits)``.
    Children with zero visits return ``+inf`` so the selection policy
    expands them first.
    """

    def test_unvisited_child_returns_infinity(self, start_state: PlanningState) -> None:
        from langgoap.planner.mcts import MCTSNode, ucb1

        root = MCTSNode(state=start_state)
        root.visits = 10
        child = MCTSNode(state=start_state, parent=root, action=None)
        assert ucb1(child, c=math.sqrt(2)) == math.inf

    def test_visited_child_matches_lats_formula(
        self, start_state: PlanningState
    ) -> None:
        from langgoap.planner.mcts import MCTSNode, ucb1

        root = MCTSNode(state=start_state)
        root.visits = 10
        child = MCTSNode(state=start_state, parent=root, action=None)
        child.visits = 4
        child.value = 2.0
        c = math.sqrt(2)
        expected = 2.0 / 4 + c * math.sqrt(math.log(10) / 4)
        assert ucb1(child, c=c) == pytest.approx(expected)

    def test_exploration_constant_scales_uncertainty_term(
        self, start_state: PlanningState
    ) -> None:
        from langgoap.planner.mcts import MCTSNode, ucb1

        root = MCTSNode(state=start_state)
        root.visits = 10
        child = MCTSNode(state=start_state, parent=root, action=None)
        child.visits = 4
        child.value = 2.0
        low = ucb1(child, c=0.5)
        high = ucb1(child, c=2.0)
        assert high > low


class TestRunningMeanBackprop:
    """Backprop updates parents with an incremental mean, matching LATS."""

    def test_backpropagate_updates_mean_along_path(
        self, start_state: PlanningState
    ) -> None:
        from langgoap.planner.mcts import MCTSNode, backpropagate

        root = MCTSNode(state=start_state)
        child = MCTSNode(state=start_state, parent=root, action=None)
        grandchild = MCTSNode(state=start_state, parent=child, action=None)

        backpropagate(grandchild, reward=1.0)
        backpropagate(grandchild, reward=0.0)

        for node in (root, child, grandchild):
            assert node.visits == 2
            assert node.value == pytest.approx(0.5)

    def test_backpropagate_stops_at_root(self, start_state: PlanningState) -> None:
        from langgoap.planner.mcts import MCTSNode, backpropagate

        root = MCTSNode(state=start_state)
        backpropagate(root, reward=0.7)
        assert root.visits == 1
        assert root.value == pytest.approx(0.7)
