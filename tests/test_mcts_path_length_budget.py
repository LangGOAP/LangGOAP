"""Unit tests for ``MCTSStrategy.reuse.path_length_budget`` (Pepels BSc §4.1).

When actions carry variable ``cost`` (e.g. junction-graph corridor
traversals whose cost is the number of primitive steps), depth-in-
edges no longer bounds the search horizon sensibly.  Pepels introduces
a *path length budget* ``T_path`` — the maximum total action cost from
root to the deepest expanded leaf — to replace fixed edge-count
depth limits.  Default ``None`` preserves today's edge-count-only
semantics bit-identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from langgoap import ActionSpec, GoalSpec
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
)
from langgoap.state import PlanningState


@dataclass(frozen=True)
class _ChainActions:
    """Five actions forming a linear chain of cost-3 hops.

    State key ``pos`` advances from 0 → 5.  Goal is reached at
    ``pos == 5`` after five expansions totalling ``5 * 3 = 15`` cost
    units; with a budget of ``6`` no descent past two expansions is
    possible, so the planner must fail to reach the goal.
    """

    cost: float = 3.0

    def build(self) -> list[ActionSpec]:
        actions: list[ActionSpec] = []
        for i in range(5):
            actions.append(
                ActionSpec(
                    name=f"advance_{i}",
                    preconditions={"pos": i},
                    effects={"pos": i + 1},
                    cost=self.cost,
                )
            )
        return actions


def _start() -> PlanningState:
    return PlanningState.from_dict({"pos": 0})


def _goal() -> GoalSpec:
    return GoalSpec(conditions={"pos": 5})


class TestPathLengthBudgetDefaultPreservesBehaviour:
    def test_default_none_reaches_goal_on_chain(self) -> None:
        actions = _ChainActions().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=400, wall_clock_ms=0.0, seed=0)
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is not None
        assert [a.name for a in plan.actions] == [
            "advance_0",
            "advance_1",
            "advance_2",
            "advance_3",
            "advance_4",
        ]
        assert plan.total_cost == 15.0


class TestPathLengthBudgetStopsDescent:
    def test_budget_below_goal_cost_yields_no_plan(self) -> None:
        actions = _ChainActions().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=400, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=6),
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is None

    def test_budget_exactly_at_goal_cost_still_reaches_goal(self) -> None:
        actions = _ChainActions().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=1000, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=15),
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is not None
        assert plan.total_cost == 15.0

    def test_last_root_has_no_descendants_past_budget(self) -> None:
        actions = _ChainActions().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=400, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=6, anytime_fallback=True),
        )
        strategy.plan(_start(), _goal(), actions)
        root = strategy._last_root
        assert root is not None
        stack: list = list(root.children)
        max_depth_seen = 0
        while stack:
            node = stack.pop()
            chain_cost = 0.0
            cur = node
            while cur is not None and getattr(cur, "action", None) is not None:
                chain_cost += float(cur.action.cost)
                cur = cur.parent
                if cur is not None and hasattr(cur, "parent"):
                    cur = cur.parent
            max_depth_seen = max(max_depth_seen, int(chain_cost))
            stack.extend(node.children)
        assert max_depth_seen <= 6


class _StochasticIdentityModel:
    """Minimal non-deterministic model: behaves identically to the
    default deterministic one but fails the ``isinstance`` check, so
    :class:`MCTSStrategy` exercises its ``_select_stochastic`` and
    ``_expand_stochastic`` code paths.  Used to verify the
    path-length budget is honoured under both tree shapes."""

    divergence_policy = None

    def expected(
        self, state: Mapping[str, object], action: ActionSpec
    ) -> Mapping[str, object]:
        return action.get_effects(dict(state))

    def sample(
        self,
        state: Mapping[str, object],
        action: ActionSpec,
        rng: object,
    ) -> Mapping[str, object]:
        return action.get_effects(dict(state))


class TestPathLengthBudgetStochastic:
    def test_budget_respected_under_nondeterministic_model(self) -> None:
        actions = _ChainActions().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=400, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=6),
            transition_model=_StochasticIdentityModel(),
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is None


class _CallableCostChain:
    """Same five-hop chain, but each action's cost is a callable that
    returns ``3.0`` regardless of state.  Exercises the dynamic-cost
    branch of :func:`_path_cost`.
    """

    @staticmethod
    def _cost(_world: Mapping[str, object]) -> float:
        return 3.0

    def build(self) -> list[ActionSpec]:
        actions: list[ActionSpec] = []
        for i in range(5):
            actions.append(
                ActionSpec(
                    name=f"advance_{i}",
                    preconditions={"pos": i},
                    effects={"pos": i + 1},
                    cost=self._cost,
                )
            )
        return actions


class TestPathLengthBudgetCallableCost:
    def test_callable_cost_below_budget_yields_no_plan(self) -> None:
        actions = _CallableCostChain().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=400, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=6),
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is None

    def test_callable_cost_at_exact_budget_reaches_goal(self) -> None:
        actions = _CallableCostChain().build()
        strategy = MCTSStrategy(
            exploration=MCTSExploration(iterations=1000, wall_clock_ms=0.0, seed=0),
            reuse=MCTSReuseConfig(path_length_budget=15),
        )
        plan = strategy.plan(_start(), _goal(), actions)
        assert plan is not None
        assert plan.total_cost == 15.0
