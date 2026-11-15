"""Strategy dispatcher.

``StrategyRouter`` reads a problem's features at plan time and
dispatches to the right :class:`~langgoap.planner.strategy.PlanningStrategy`
from a registered mapping.  The router itself satisfies
``PlanningStrategy`` so it composes transparently with existing
callers (``GoapPlanner(strategy=StrategyRouter(...))``).

The default :class:`RuleBasedClassifier` does **not** auto-promote
MCTS for ``is_stochastic=True``.  Promotion requires either an
explicit :class:`~langgoap.planner.transitions.DivergencePolicy`
of kind ``"risk-averse"`` (structured user opt-in) or the
``prefer_mcts_for_stochastic=True`` classifier flag.

See the internal "strategy-router" design note (not in the repository).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, cast, runtime_checkable

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.strategy import PlanningStrategy
from langgoap.planner.transitions import (
    DeterministicTransitionModel,
    TransitionModel,
)
from langgoap.planner.types import Plan
from langgoap.state import PlanningState

RiskProfile = Literal["strict", "risk-averse", "learned", "hierarchical", "other"]


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


def relaxed_plan_depth(
    start: PlanningState,
    goal: GoalSpec,
    actions: list[ActionSpec],
) -> int:
    """Depth of the relaxed planning graph from *start* to *goal*.

    Standard delete-relaxation: facts are only ever added, every action
    whose preconditions hold in the relaxed state fires in the same layer,
    and the depth is the number of layers until the goal holds.  It is a
    cheap lower-bound proxy for plan length that ignores catalog size, so
    thirty unrelated actions do not make a five-step chain look deep.
    Dynamic effects count through their declared ``effect_keys`` (any value).

    Returns ``0`` when the goal already holds and ``len(actions) + 1`` when
    the goal is unreachable even under relaxation.
    """
    known: set[tuple[str, Any]] = set(start.conditions)
    wild_keys: set[str] = set()

    def _holds(conditions: Mapping[str, Any]) -> bool:
        return all((k, v) in known or k in wild_keys for k, v in conditions.items())

    if _holds(goal.conditions):
        return 0
    remaining = list(actions)
    depth = 0
    while remaining:
        layer = [a for a in remaining if _holds(a.preconditions)]
        if not layer:
            break
        depth += 1
        for action in layer:
            if action.has_dynamic_effects:
                wild_keys.update(action.effect_key_set())
            else:
                known.update(cast(Mapping[str, Any], action.effects).items())
        remaining = [a for a in remaining if a not in layer]
        if _holds(goal.conditions):
            return depth
    return len(actions) + 1


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
    admissible_h = sum(1 for k, v in goal.conditions.items() if start_dict.get(k) != v)
    horizon_estimate = relaxed_plan_depth(start, goal, actions)

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
           \u2192 ``"mcts"`` (gated opt-in).
        4. Large branching (``action_count >= branching_threshold``)
           + deep horizon (``horizon_estimate >= horizon_threshold``)
           \u2192 ``"mcts"``.
        5. Default \u2192 ``"astar"``.
    """

    branching_threshold: int = 30
    horizon_threshold: int = 20
    prefer_mcts_for_stochastic: bool = False

    def __call__(self, f: ProblemFeatures) -> str:
        if f.has_hard_constraints or f.has_soft_objectives or f.has_trajectory_metrics:
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
        on_strategy_chosen: Optional callback invoked with the name of
            the strategy actually dispatched to (after ``on_unknown``
            fallback resolution).  Useful for observers that need to
            know which concrete strategy ran \u2014 e.g. a dashboard
            recorder that renders an A* tree for one turn and an MCTS
            tree for the next.
    """

    strategies: Mapping[str, PlanningStrategy]
    classifier: StrategyClassifier
    transition_model: TransitionModel | None = None
    default: str = "astar"
    on_unknown: Literal["raise", "default"] = "raise"
    on_strategy_chosen: Callable[[str], None] | None = None
    last_chosen: str | None = field(default=None, init=False, repr=False)

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Plan | None = None,
        current_step: int = 0,
    ) -> Plan | None:
        """Classify ``(start, goal, actions)`` and dispatch to the chosen strategy.

        ``prior_plan`` / ``current_step`` are the advisory repair kwargs of
        the :class:`~langgoap.planner.strategy.PlanningStrategy` Protocol.
        :class:`~langgoap.graph.nodes.GoapPlanner` forwards them on every
        replan, so the router must accept them; they are passed on to the
        chosen strategy only when a prior plan exists so legacy strategies
        whose ``plan`` signature predates the widened Protocol keep working
        on initial plans.
        """
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
            chosen_name = self.default
        self.last_chosen = chosen_name
        if self.on_strategy_chosen is not None:
            self.on_strategy_chosen(chosen_name)
        extras: dict[str, Any] = {}
        if prior_plan is not None:
            extras["prior_plan"] = prior_plan
            extras["current_step"] = current_step
        return chosen.plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            **extras,
        )


__all__ = [
    "ProblemFeatures",
    "RiskProfile",
    "RuleBasedClassifier",
    "StrategyClassifier",
    "StrategyRouter",
    "extract_features",
]
