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


# --- replan contract (issue #10) ---------------------------------------------


class TestRouterReplanContract:
    """``StrategyRouter.plan`` must accept the advisory repair kwargs that
    ``GoapPlanner`` forwards on every replan (``prior_plan`` /
    ``current_step``), otherwise the first replan through a router raises
    ``TypeError``.  Regression tests for issue #10.
    """

    def _router(self) -> "StrategyRouter":
        from langgoap.planner.router import RuleBasedClassifier, StrategyRouter

        return StrategyRouter(
            strategies={"astar": AStarStrategy()},
            classifier=RuleBasedClassifier(),
        )

    def test_plan_accepts_prior_plan_and_current_step(self) -> None:
        router = self._router()
        actions = _actions(1)
        start = PlanningState.from_dict({})
        prior = router.plan(start, _goal(), actions)
        assert prior is not None
        again = router.plan(
            start,
            _goal(),
            actions,
            blacklisted_actions=[],
            prior_plan=prior,
            current_step=0,
        )
        assert again is not None
        assert again.action_names == prior.action_names

    def test_repair_kwargs_are_forwarded_only_when_prior_plan_set(self) -> None:
        from langgoap.planner.router import StrategyRouter

        seen: list[dict] = []

        class _Recorder:
            def plan(self, start, goal, actions, *, blacklisted_actions=None, **kw):
                seen.append(dict(kw))
                return Plan(actions=())

        router = StrategyRouter(
            strategies={"astar": _Recorder()}, classifier=lambda f: "astar"
        )
        start = PlanningState.from_dict({})
        router.plan(start, _goal(), _actions(1))
        prior = Plan(actions=())
        router.plan(start, _goal(), _actions(1), prior_plan=prior, current_step=2)
        assert seen[0] == {}
        assert seen[1] == {"prior_plan": prior, "current_step": 2}

    def test_router_inside_graph_survives_a_replan(self) -> None:
        """End-to-end: a router-driven graph must replan after an action fails."""
        from langgoap.graph.builder import GoapGraph
        from langgoap.planner.router import RuleBasedClassifier, StrategyRouter
        from langgoap.planner.strategy import TwoPhasePipelineStrategy

        calls = {"n": 0}

        def flaky(ws: dict) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first call fails")
            return {"b": True}

        actions = [
            ActionSpec(name="a_fast", effects={"b": True}, cost=1.0, execute=flaky),
            ActionSpec(
                name="a_slow",
                effects={"b": True},
                cost=5.0,
                execute=lambda ws: {"b": True},
            ),
            ActionSpec(
                name="finish",
                preconditions={"b": True},
                effects={"done": True},
                execute=lambda ws: {"done": True},
            ),
        ]
        router = StrategyRouter(
            strategies={
                "astar": AStarStrategy(),
                "csp-pipeline": TwoPhasePipelineStrategy(),
                "mcts": MCTSStrategy(),
            },
            classifier=RuleBasedClassifier(),
        )
        result = GoapGraph(actions=actions, strategy=router).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] == 1
        assert "a_fast" in result["blacklisted_actions"]


class TestStrategyProtocolConformance:
    """Every shipped strategy must accept the full ``PlanningStrategy.plan``
    keyword surface, including the advisory repair kwargs."""

    @staticmethod
    def _strategies() -> list[PlanningStrategy]:
        from langgoap.planner.repair import RepairStrategy
        from langgoap.planner.router import RuleBasedClassifier, StrategyRouter
        from langgoap.planner.strategy import (
            AnytimePlanningStrategy,
            LazyDecompositionStrategy,
            TwoPhasePipelineStrategy,
        )
        from langgoap.planner.utility import UtilityStrategy

        return [
            AStarStrategy(),
            TwoPhasePipelineStrategy(),
            LazyDecompositionStrategy(lookahead=1),
            AnytimePlanningStrategy(time_budget_ms=5_000),
            RepairStrategy(),
            UtilityStrategy(),
            MCTSStrategy(),
            StrategyRouter(
                strategies={"astar": AStarStrategy()},
                classifier=RuleBasedClassifier(),
            ),
        ]

    @pytest.mark.parametrize(
        "strategy", _strategies.__func__(), ids=lambda s: type(s).__name__
    )
    def test_plan_accepts_repair_kwargs(self, strategy: PlanningStrategy) -> None:
        actions = _actions(1)
        start = PlanningState.from_dict({})
        prior = Plan(actions=tuple(actions))
        result = strategy.plan(
            start,
            _goal(),
            actions,
            blacklisted_actions=[],
            prior_plan=prior,
            current_step=0,
        )
        assert result is None or isinstance(result, Plan)


# ---------------------------------------------------------------------------
# Horizon estimate from a relaxed plan (issue #24)
# ---------------------------------------------------------------------------


def _chain(length: int) -> list[ActionSpec]:
    """``s0 -> s1 -> ... -> s<length>``: a deterministic chain of ``length`` steps."""
    actions = [
        ActionSpec(name="step_1", preconditions={}, effects={"s1": True}, cost=1.0)
    ]
    for i in range(2, length + 1):
        actions.append(
            ActionSpec(
                name=f"step_{i}",
                preconditions={f"s{i - 1}": True},
                effects={f"s{i}": True},
                cost=1.0,
            )
        )
    return actions


def _noise(n: int) -> list[ActionSpec]:
    """Unrelated actions that inflate the catalog without touching the goal."""
    return [
        ActionSpec(
            name=f"noise_{i}", preconditions={}, effects={f"n{i}": True}, cost=1.0
        )
        for i in range(n)
    ]


class TestHorizonEstimate:
    def test_horizon_is_the_relaxed_plan_depth(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(PlanningState.from_dict({}), _goal(s5=True), _chain(5))
        assert f.horizon_estimate == 5

    def test_unrelated_actions_do_not_inflate_the_horizon(self) -> None:
        from langgoap.planner.router import extract_features

        catalog = _chain(5) + _noise(30)
        f = extract_features(PlanningState.from_dict({}), _goal(s5=True), catalog)
        assert f.action_count == 35
        assert f.horizon_estimate == 5

    def test_satisfied_goal_has_zero_horizon(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(
            PlanningState.from_dict({"s5": True}), _goal(s5=True), _chain(5)
        )
        assert f.horizon_estimate == 0

    def test_unreachable_goal_horizon_is_at_least_the_catalog_size(self) -> None:
        from langgoap.planner.router import extract_features

        f = extract_features(
            PlanningState.from_dict({}), _goal(missing=True), _chain(5)
        )
        assert f.horizon_estimate >= f.action_count

    def test_dynamic_effects_count_by_declared_keys(self) -> None:
        from langgoap.planner.router import extract_features

        dyn = ActionSpec(
            name="dyn",
            preconditions={"s1": True},
            effects=lambda ws: {"s2": True},
            effect_keys=frozenset({"s2"}),
        )
        f = extract_features(
            PlanningState.from_dict({}), _goal(s2=True), [_chain(1)[0], dyn]
        )
        assert f.horizon_estimate == 2

    def test_default_classifier_keeps_astar_for_short_chains_over_big_catalogs(
        self,
    ) -> None:
        from langgoap.planner.router import RuleBasedClassifier, extract_features

        f = extract_features(
            PlanningState.from_dict({}), _goal(s5=True), _chain(5) + _noise(30)
        )
        assert RuleBasedClassifier()(f) == "astar"

    def test_default_classifier_still_promotes_deep_chains_over_big_catalogs(
        self,
    ) -> None:
        from langgoap.planner.router import RuleBasedClassifier, extract_features

        f = extract_features(
            PlanningState.from_dict({}), _goal(s25=True), _chain(25) + _noise(10)
        )
        assert f.action_count >= 30 and f.horizon_estimate >= 20
        assert RuleBasedClassifier()(f) == "mcts"

    def test_router_solves_a_short_chain_in_a_big_catalog(self) -> None:
        from langgoap.planner.router import RuleBasedClassifier, StrategyRouter

        router = StrategyRouter(
            strategies={"astar": AStarStrategy(), "mcts": MCTSStrategy()},
            classifier=RuleBasedClassifier(),
        )
        result = router.plan(
            PlanningState.from_dict({}), _goal(s5=True), _chain(5) + _noise(30)
        )
        assert router.last_chosen == "astar"
        assert result is not None
        assert result.action_names == [f"step_{i}" for i in range(1, 6)]
