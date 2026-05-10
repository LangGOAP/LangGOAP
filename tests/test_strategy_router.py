"""Tests for :mod:`langgoap.planner.router` \u2014 ``StrategyRouter`` +
``ProblemFeatures`` + ``RuleBasedClassifier``.

Gating invariant: the default classifier does **not** auto-promote
MCTS for ``is_stochastic=True``; it routes there only on explicit
opt-in (``prefer_mcts_for_stochastic=True``) or when the user
declared a risk-averse :class:`DivergencePolicy`.

See ``research/plans/strategy-router.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
)
from langgoap.planner.strategy import AStarStrategy, PlanningStrategy
from langgoap.planner.transitions import (
    DeterministicTransitionModel,
    DivergencePolicy,
    TransitionModel,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState
from langgoap.types import ObjectiveDirection

# --- helpers ---------------------------------------------------------------


def _actions(n: int = 1) -> list[ActionSpec]:
    return [
        ActionSpec(name=f"a{i}", preconditions={}, effects={"x": i + 1}, cost=1.0)
        for i in range(n)
    ]


def _goal(**conds: object) -> GoalSpec:
    return GoalSpec(conditions=dict(conds) or {"x": 1})


class _StochasticModel:
    """Minimal non-deterministic ``TransitionModel`` for router tests."""

    divergence_policy: DivergencePolicy | None = None

    def expected(self, state: PlanningState, action: ActionSpec) -> dict:
        return dict(action.get_effects(state.to_dict()))

    def sample(self, state: PlanningState, action: ActionSpec, rng: Random) -> dict:
        return self.expected(state, action)


# --- ProblemFeatures -------------------------------------------------------


class TestProblemFeatures:
    def test_deterministic_model_yields_strict_risk_profile(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(
            PlanningState.from_dict({}),
            _goal(x=1),
            _actions(3),
            transition_model=DeterministicTransitionModel(),
        )
        assert f.is_stochastic is False
        assert f.risk_profile == "strict"

    def test_stochastic_model_flags_is_stochastic(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(
            PlanningState.from_dict({}),
            _goal(),
            _actions(2),
            transition_model=_StochasticModel(),
        )
        assert f.is_stochastic is True
        assert f.risk_profile == "strict"  # no divergence policy declared

    def test_divergence_policy_kind_surfaces_as_risk_profile(self) -> None:
        from langgoap.planner.router import extract_features

        m = _StochasticModel()
        m.divergence_policy = DivergencePolicy(
            reason="CVaR planner", kind="risk-averse", extra={"cvar_alpha": 0.9}
        )
        f = extract_features(
            PlanningState.from_dict({}),
            _goal(),
            _actions(1),
            transition_model=m,
        )
        assert f.risk_profile == "risk-averse"

    def test_counts_and_constraints(self) -> None:
        from langgoap.planner.router import extract_features

        goal = GoalSpec(
            conditions={"a": 1, "b": 2},
            objectives={"cost": ObjectiveDirection.MINIMIZE},
            constraints=(ConstraintSpec(key="cost", max=10.0),),
        )
        f = extract_features(
            PlanningState.from_dict({"a": 1}),
            goal,
            _actions(5),
            transition_model=None,
        )
        assert f.action_count == 5
        assert f.goal_condition_count == 2
        assert f.has_hard_constraints is True
        assert f.has_soft_objectives is True
        assert f.admissible_h_at_start == 1  # "b" unsatisfied, "a" satisfied

    def test_default_transition_model_is_deterministic(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(
            PlanningState.from_dict({}),
            _goal(),
            _actions(1),
        )
        assert f.is_stochastic is False
        assert f.risk_profile == "strict"


# --- RuleBasedClassifier ---------------------------------------------------


class TestRuleBasedClassifier:
    def _features(self, **overrides: object) -> object:
        from langgoap.planner.router import ProblemFeatures

        defaults: dict[str, object] = dict(
            action_count=3,
            goal_condition_count=1,
            has_hard_constraints=False,
            has_soft_objectives=False,
            has_trajectory_metrics=False,
            is_stochastic=False,
            risk_profile="strict",
            admissible_h_at_start=1,
            horizon_estimate=3,
        )
        defaults.update(overrides)
        return ProblemFeatures(**defaults)

    def test_hard_constraints_route_to_csp_pipeline(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features(has_hard_constraints=True)) == "csp-pipeline"

    def test_soft_objectives_route_to_csp_pipeline(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features(has_soft_objectives=True)) == "csp-pipeline"

    def test_trajectory_metrics_route_to_csp_pipeline(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features(has_trajectory_metrics=True)) == "csp-pipeline"

    def test_risk_averse_routes_to_mcts_unconditionally(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()  # default: prefer_mcts_for_stochastic=False
        assert c(self._features(risk_profile="risk-averse")) == "mcts"

    def test_stochastic_default_gate_routes_to_astar(self) -> None:
        """Default gate: stochastic alone does NOT auto-promote MCTS."""
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features(is_stochastic=True)) == "astar"

    def test_stochastic_opt_in_routes_to_mcts(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier(prefer_mcts_for_stochastic=True)
        assert c(self._features(is_stochastic=True)) == "mcts"

    def test_large_branching_routes_to_mcts(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features(action_count=40, horizon_estimate=30)) == "mcts"

    def test_default_routes_to_astar(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier()
        assert c(self._features()) == "astar"

    def test_configurable_thresholds(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier

        c = RuleBasedClassifier(branching_threshold=5, horizon_threshold=2)
        assert c(self._features(action_count=6, horizon_estimate=3)) == "mcts"
        # Below threshold: back to astar.
        assert c(self._features(action_count=4, horizon_estimate=10)) == "astar"


# --- StrategyRouter --------------------------------------------------------


@dataclass
class _RecordingStrategy:
    name: str
    calls: list[tuple] = field(default_factory=list)

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        self.calls.append(
            (start, goal, tuple(actions), tuple(blacklisted_actions or ()))
        )
        return AStarStrategy().plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )


class TestStrategyRouter:
    def test_dispatches_to_classifier_choice(self) -> None:
        from langgoap.planner.router import StrategyRouter

        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=lambda f: "mcts",
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert len(mcts_rec.calls) == 1
        assert len(astar_rec.calls) == 0

    def test_unknown_strategy_raises_by_default(self) -> None:
        from langgoap.planner.router import StrategyRouter

        router = StrategyRouter(
            strategies={"astar": AStarStrategy()},
            classifier=lambda f: "nonexistent",
        )
        with pytest.raises(KeyError, match="nonexistent"):
            router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))

    def test_unknown_strategy_falls_back_when_on_unknown_default(self) -> None:
        from langgoap.planner.router import StrategyRouter

        fallback = _RecordingStrategy(name="astar")
        router = StrategyRouter(
            strategies={"astar": fallback},
            classifier=lambda f: "nonexistent",
            default="astar",
            on_unknown="default",
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert len(fallback.calls) == 1

    def test_satisfies_planning_strategy_protocol(self) -> None:
        from langgoap.planner.router import StrategyRouter

        router = StrategyRouter(
            strategies={"astar": AStarStrategy()},
            classifier=lambda f: "astar",
        )
        assert isinstance(router, PlanningStrategy)

    def test_exposes_last_chosen_strategy_name(self) -> None:
        """After each plan dispatch, the chosen strategy name is observable."""
        from langgoap.planner.router import StrategyRouter

        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        choices = iter(["astar", "mcts", "astar"])
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=lambda f: next(choices),
        )
        assert router.last_chosen is None
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert router.last_chosen == "astar"
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert router.last_chosen == "mcts"
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert router.last_chosen == "astar"

    def test_on_strategy_chosen_callback_fires_with_resolved_name(self) -> None:
        """The callback is invoked with the name actually dispatched to,
        including when ``on_unknown='default'`` rerouted an unknown pick."""
        from langgoap.planner.router import StrategyRouter

        seen: list[str] = []
        router = StrategyRouter(
            strategies={"astar": _RecordingStrategy(name="astar")},
            classifier=lambda f: "nonexistent",
            default="astar",
            on_unknown="default",
            on_strategy_chosen=seen.append,
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert seen == ["astar"]
        assert router.last_chosen == "astar"

    def test_blacklisted_actions_threaded_through(self) -> None:
        from langgoap.planner.router import StrategyRouter

        rec = _RecordingStrategy(name="astar")
        router = StrategyRouter(strategies={"astar": rec}, classifier=lambda f: "astar")
        router.plan(
            PlanningState.from_dict({}),
            _goal(x=1),
            _actions(3),
            blacklisted_actions=["a0"],
        )
        assert rec.calls[0][3] == ("a0",)

    def test_astar_compat_deterministic_problem(self) -> None:
        """With the default classifier on a deterministic small problem,
        the router's output must be bit-identical to ``AStarStrategy``."""
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )

        start = PlanningState.from_dict({})
        goal = _goal(x=1)
        actions = _actions(1)

        direct = AStarStrategy().plan(start, goal, actions)
        router = StrategyRouter(
            strategies={"astar": AStarStrategy()},
            classifier=RuleBasedClassifier(),
        )
        routed = router.plan(start, goal, actions)

        assert direct is not None and routed is not None
        assert tuple(a.name for a in routed.actions) == tuple(
            a.name for a in direct.actions
        )
        assert routed.total_cost == direct.total_cost


# --- End-to-end integration ------------------------------------------------


class TestStrategyRouterEndToEnd:
    def test_deterministic_problem_routes_to_astar(self) -> None:
        """Default classifier + deterministic model + no constraints \u2192 A*."""
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )

        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=RuleBasedClassifier(),
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(2))
        assert len(astar_rec.calls) == 1
        assert len(mcts_rec.calls) == 0

    def test_risk_averse_divergence_policy_routes_to_mcts(self) -> None:
        """User opt-in via ``DivergencePolicy`` \u2192 MCTS, even without
        ``prefer_mcts_for_stochastic``.  This is the 10%-user path."""
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )

        model = _StochasticModel()
        model.divergence_policy = DivergencePolicy(
            reason="CVaR alpha=0.9", kind="risk-averse"
        )
        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=RuleBasedClassifier(),  # default: no auto-promote
            transition_model=model,
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(1))
        assert len(mcts_rec.calls) == 1
        assert len(astar_rec.calls) == 0

    def test_stochastic_no_opt_in_routes_to_astar_by_default(self) -> None:
        """Default gate: stochastic alone is not enough."""
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )

        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=RuleBasedClassifier(),
            transition_model=_StochasticModel(),
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(2))
        assert len(astar_rec.calls) == 1
        assert len(mcts_rec.calls) == 0

    def test_stochastic_opt_in_routes_to_mcts_end_to_end(self) -> None:
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )

        astar_rec = _RecordingStrategy(name="astar")
        mcts_rec = _RecordingStrategy(name="mcts")
        router = StrategyRouter(
            strategies={"astar": astar_rec, "mcts": mcts_rec},
            classifier=RuleBasedClassifier(prefer_mcts_for_stochastic=True),
            transition_model=_StochasticModel(),
        )
        router.plan(PlanningState.from_dict({}), _goal(x=1), _actions(2))
        assert len(mcts_rec.calls) == 1
        assert len(astar_rec.calls) == 0

    def test_router_produces_real_plan_on_stochastic_gridworld(self) -> None:
        """Narrative test backing the ``stochastic_gridworld`` basics notebook.

        Wires a :class:`SlipperyTransitionModel` with a risk-averse
        :class:`DivergencePolicy`, routes through :class:`StrategyRouter`,
        and asserts the router actually produces an executable plan
        against ``frozen_lake_4x4`` \u2014 i.e. the full stack composes
        without the notebook having to patch anything.
        """
        from langgoap.planner.mcts import (
            MCTSExploration,
            MCTSReuseConfig,
            MCTSStrategy,
            MCTSTracingConfig,
        )
        from langgoap.planner.router import (
            RuleBasedClassifier,
            StrategyRouter,
        )
        from tests.fixtures.stochastic_gridworld import (
            SlipperyTransitionModel,
            frozen_lake_4x4,
            gridworld_goal,
            gridworld_start_state,
            make_gridworld_actions,
        )

        topology = frozen_lake_4x4()
        model = SlipperyTransitionModel(topology=topology, slip_prob=0.2)
        # Risk-averse opt-in \u2014 the user path the notebook documents.
        object.__setattr__(
            model,
            "divergence_policy",
            DivergencePolicy(reason="slippery MDP", kind="risk-averse"),
        )
        actions = make_gridworld_actions(topology)
        start = PlanningState.from_dict(gridworld_start_state(topology))

        router = StrategyRouter(
            strategies={
                "astar": AStarStrategy(),
                "mcts": MCTSStrategy(
                    exploration=MCTSExploration(
                        iterations=256, rollout_depth=12, wall_clock_ms=500.0, seed=7
                    ),
                    reuse=MCTSReuseConfig(anytime_fallback=True),
                    transition_model=model,
                ),
            },
            classifier=RuleBasedClassifier(),
            transition_model=model,
        )
        plan = router.plan(start, gridworld_goal(), actions)
        assert plan is not None
        assert len(plan.actions) >= 1
        for a in plan.actions:
            assert a.name in {"north", "south", "east", "west"}
