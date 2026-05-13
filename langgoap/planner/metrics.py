"""Plan-quality metrics evaluated against a :class:`Plan`'s trajectory.

Adapted from ``unified-planning``'s ``PlanQualityMetric`` hierarchy
(``unified_planning/model/metrics.py``).  The ``unified-planning``
library separates metric *definition* (declarative) from metric
*evaluation* (solver-specific); LangGOAP keeps the same separation
so the :class:`PlanQualityMetric` Protocol is cheap to satisfy and
easy to extend.

Metric values flow into :class:`langgoap.score.HardSoftScore.soft`
at the CSP stage, weighted by :attr:`~PlanQualityMetric.weight`.
All metrics here are *minimise* semantics — lower is better — so a
positive metric value contributes a negative amount to ``soft``.
Clients that want a *maximise* metric can wrap an existing metric
with ``weight=-1.0`` or invert the expression.

The canonical use case is Pac-Man's ghost-proximity integral — a
:class:`MinimizeTrajectorySum` with ``expression=inverse_ghost_
distance`` penalises every plan step that ever passes within
striking distance of an adversary, not just plans whose terminal
state is unsafe.  See ``research/plans/trajectory-level-cost.md``
for the motivation and ``research/experiments/`` for the pre-
registered A/B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from langgoap.planner.types import Plan, PlanningState

__all__ = [
    "MinimizeFinalStateExpression",
    "MinimizeTrajectoryMax",
    "MinimizeTrajectorySum",
    "PlanQualityMetric",
]


@runtime_checkable
class PlanQualityMetric(Protocol):
    """Quality metric evaluated against a :class:`Plan`'s trajectory.

    Concrete metrics are minimisation-oriented: lower is better.  The
    contribution to :class:`~langgoap.score.HardSoftScore.soft` is
    ``-weight * evaluate(plan)``.
    """

    name: str
    """Human-readable metric label, surfaced in tracing / diagnostics."""

    weight: float
    """Linear combination weight.  Defaults to 1.0 for each concrete metric."""

    def evaluate(self, plan: Plan) -> float:
        """Return the metric value for ``plan``.

        The CSP pipeline invokes :meth:`evaluate` once per metric per
        candidate plan.  Implementations must be deterministic and
        side-effect-free so score comparisons remain stable across
        re-runs.  Empty trajectories (plans with no expected states)
        must return ``0.0`` rather than raising so trivial plans
        still participate in scoring.
        """
        ...


_StateExpression = Callable[[PlanningState], float]


@dataclass(frozen=True)
class MinimizeFinalStateExpression:
    """Minimise ``expression`` evaluated on the final expected state.

    Mirrors unified-planning's ``MinimizeExpressionOnFinalState``.
    """

    name: str
    expression: _StateExpression
    weight: float = 1.0

    def evaluate(self, plan: Plan) -> float:
        if not plan.expected_states:
            return 0.0
        return float(self.expression(plan.expected_states[-1]))


@dataclass(frozen=True)
class MinimizeTrajectorySum:
    """Minimise the sum of ``expression`` over every expected state.

    The canonical shape for *integral* trajectory metrics — e.g. the
    Pac-Man ghost-proximity integral
    ``sum(1 / distance_to_nearest_ghost(state) for state in plan)``
    which penalises any plan step that passes close to a ghost.
    """

    name: str
    expression: _StateExpression
    weight: float = 1.0

    def evaluate(self, plan: Plan) -> float:
        if not plan.expected_states:
            return 0.0
        return float(sum(self.expression(s) for s in plan.expected_states))


@dataclass(frozen=True)
class MinimizeTrajectoryMax:
    """Minimise the worst-case (maximum) value of ``expression``.

    Useful for *peak-threat* semantics: a plan whose integral
    proximity is low but which briefly spikes close to a ghost may
    be acceptable under :class:`MinimizeTrajectorySum` and
    unacceptable under :class:`MinimizeTrajectoryMax`.
    """

    name: str
    expression: _StateExpression
    weight: float = 1.0

    def evaluate(self, plan: Plan) -> float:
        if not plan.expected_states:
            return 0.0
        return float(max(self.expression(s) for s in plan.expected_states))
