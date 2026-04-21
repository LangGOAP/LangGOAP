"""Strategy dispatcher.

``StrategyRouter`` reads a problem's features at plan time and
dispatches to the right :class:`~langgoap.planner.strategy.PlanningStrategy`
from a registered mapping.  The router itself satisfies
``PlanningStrategy`` so it composes transparently with existing
callers (``GoapPlanner(strategy=StrategyRouter(...))``).

Design: ``research/plans/strategy-router.md``.

Outcome-3 gate (from ``2026-04-20-mcts-on-stochastic.md``): the
default :class:`RuleBasedClassifier` does **not** auto-promote MCTS
for ``is_stochastic=True``.  Promotion requires either an explicit
:class:`~langgoap.planner.transitions.DivergencePolicy` of kind
``"risk-averse"`` (structured user opt-in) or the
``prefer_mcts_for_stochastic=True`` classifier flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.strategy import PlanningStrategy
from langgoap.planner.transitions import (
    DeterministicTransitionModel,
    TransitionModel,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState

RiskProfile = Literal[
    "strict", "risk-averse", "learned", "hierarchical", "other"
]


@dataclass(frozen=True, slots=True)
class ProblemFeatures:
    """Cheap-to-compute routing features extracted at plan time."""

    action_count: int
    goal_condition_count: int
    has_hard_constraints: bool
    has_soft_objectives: bool
    has_trajectory_metrics: bool
    is_stochastic: bool
    risk_profile: RiskProfile
    admissible_h_at_start: int
    horizon_estimate: int


def extract_features(
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
    *,
    transition_model: TransitionModel | None = None,
) -> ProblemFeatures:
    """Extract routing features.  Pure function, no side effects."""
    model = transition_model or DeterministicTransitionModel()
    is_stochastic = not isinstance(model, DeterministicTransitionModel)

    policy = getattr(model, "divergence_policy", None)
    risk_profile: RiskProfile = "strict" if policy is None else policy.kind

    start_dict = start.to_dict()
    admissible_h = sum(
        1 for k, v in goal.conditions.items() if start_dict.get(k) != v
    )

    # Coarse horizon proxy: unsatisfied conditions times action-space size.
    # Used only as a tiebreaker in the default classifier; the conservative
    # thresholds there keep common cases unambiguous.
    horizon_estimate = admissible_h * max(1, len(actions))

    return ProblemFeatures(
        action_count=len(actions),
        goal_condition_count=len(goal.conditions),
        has_hard_constraints=bool(goal.constraints),
        has_soft_objectives=goal.objectives is not None,
        has_trajectory_metrics=bool(goal.metrics),
        is_stochastic=is_stochastic,
        risk_profile=risk_profile,
        admissible_h_at_start=admissible_h,
        horizon_estimate=horizon_estimate,
    )


@runtime_checkable
class StrategyClassifier(Protocol):
    """Maps :class:`ProblemFeatures` to a registered strategy name."""

    def __call__(self, features: ProblemFeatures) -> str: ...


@dataclass
class RuleBasedClassifier:
    """Lexicographic classifier with conservative defaults.

    Order of precedence:
        1. Hard constraints / soft objectives / trajectory metrics
           \u2192 ``"csp-pipeline"``.
        2. ``risk_profile == "risk-averse"`` \u2192 ``"mcts"`` (explicit
           user opt-in via :class:`DivergencePolicy`).
        3. ``is_stochastic`` + ``prefer_mcts_for_stochastic=True``
           \u2192 ``"mcts"`` (Outcome-3 gated opt-in).
        4. Large branching (``action_count >= branching_threshold``)
           + deep horizon (``horizon_estimate >= horizon_threshold``)
           \u2192 ``"mcts"``.
        5. Default \u2192 ``"astar"``.
    """

    branching_threshold: int = 30
    horizon_threshold: int = 20
    prefer_mcts_for_stochastic: bool = False

    def __call__(self, f: ProblemFeatures) -> str:
        if (
            f.has_hard_constraints
            or f.has_soft_objectives
            or f.has_trajectory_metrics
        ):
            return "csp-pipeline"
        if f.risk_profile == "risk-averse":
            return "mcts"
        if f.is_stochastic and self.prefer_mcts_for_stochastic:
            return "mcts"
        if (
            f.action_count >= self.branching_threshold
            and f.horizon_estimate >= self.horizon_threshold
        ):
            return "mcts"
        return "astar"


@dataclass
class StrategyRouter:
    """Dispatches to a registered :class:`PlanningStrategy` by name.

    ``StrategyRouter`` itself satisfies the :class:`PlanningStrategy`
    Protocol, so callers that already accept a strategy (e.g.
    ``GoapPlanner(strategy=...)``) can opt in by passing a router
    with no other changes.

    Args:
        strategies: Mapping from strategy name to strategy instance.
        classifier: Callable from :class:`ProblemFeatures` to a
            registered strategy name.
        transition_model: Optional model consulted by
            :func:`extract_features` for the ``is_stochastic`` and
            ``risk_profile`` features.  Defaults to
            :class:`DeterministicTransitionModel`.
        default: Name of the fallback strategy used when
            ``on_unknown="default"`` and the classifier returns a
            name not registered in ``strategies``.
        on_unknown: ``"raise"`` (default) turns unknown classifier
            outputs into :class:`KeyError` \u2014 strict mode catches
            misconfigured routers loudly.  ``"default"`` silently
            falls back to ``strategies[default]``.
    """

    strategies: Mapping[str, PlanningStrategy]
    classifier: StrategyClassifier
    transition_model: TransitionModel | None = None
    default: str = "astar"
    on_unknown: Literal["raise", "default"] = "raise"

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        features = extract_features(
            start, goal, actions, transition_model=self.transition_model
        )
        chosen_name = self.classifier(features)
        chosen = self.strategies.get(chosen_name)
        if chosen is None:
            if self.on_unknown == "raise":
                raise KeyError(
                    f"Classifier returned {chosen_name!r}, not in "
                    f"registered strategies {sorted(self.strategies)}"
                )
            chosen = self.strategies[self.default]
        return chosen.plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )


__all__ = [
    "ProblemFeatures",
    "RiskProfile",
    "RuleBasedClassifier",
    "StrategyClassifier",
    "StrategyRouter",
    "extract_features",
]
