"""Failing tests for :mod:`langgoap.planner.metrics`.

Phase 3 of the advanced-planning research track lands a small
:class:`PlanQualityMetric` hierarchy adapted from unified-planning's
``MinimizeExpressionOnFinalState`` / ``Oversubscription`` pattern, so
the CSP stage can re-rank candidate plans by whole-plan quality (not
just step-local cost).  Canonical use case: Pac-Man's ghost-proximity
integral over the expected trajectory, which penalises plans that
ever pass close to a ghost — not just plans whose terminal state is
unsafe.

The three concrete metrics landed here cover the patterns called out
in ``research/plans/trajectory-level-cost.md``:
* ``MinimizeFinalStateExpression`` — evaluate on the last state only.
* ``MinimizeTrajectorySum`` — integral over every expected state.
* ``MinimizeTrajectoryMax`` — worst-case state along the plan.

CSP integration is intentionally additive: the metrics contribute a
weighted minimisation term to ``HardSoftScore.soft`` alongside the
existing constraint / objective machinery.  ``hard`` is untouched
because these are preferences, not constraints.
"""

from __future__ import annotations

import pytest

from langgoap.planner.types import Plan, PlanningState

# ---------------------------------------------------------------------------
# Concrete metrics
# ---------------------------------------------------------------------------


def _plan_with_trajectory(values: list[dict[str, object]]) -> Plan:
    """Helper: build a :class:`Plan` whose expected states hold ``values``."""
    states = tuple(PlanningState.from_dict(v, keys=frozenset(v.keys())) for v in values)
    return Plan(actions=(), expected_states=states, total_cost=0.0)


class TestMinimizeFinalStateExpression:
    def test_evaluates_expression_on_last_state_only(self) -> None:
        from langgoap.planner.metrics import MinimizeFinalStateExpression

        m = MinimizeFinalStateExpression(
            name="terminal_threat",
            expression=lambda s: float(s.get("threat", 0.0)),
        )
        plan = _plan_with_trajectory(
            [{"threat": 5.0}, {"threat": 3.0}, {"threat": 1.0}]
        )
        assert m.evaluate(plan) == pytest.approx(1.0)

    def test_returns_zero_for_empty_trajectory(self) -> None:
        from langgoap.planner.metrics import MinimizeFinalStateExpression

        m = MinimizeFinalStateExpression(name="t", expression=lambda s: 42.0)
        empty = Plan(actions=(), expected_states=(), total_cost=0.0)
        assert m.evaluate(empty) == 0.0


class TestMinimizeTrajectorySum:
    def test_sums_expression_over_every_state(self) -> None:
        from langgoap.planner.metrics import MinimizeTrajectorySum

        m = MinimizeTrajectorySum(
            name="proximity_integral",
            expression=lambda s: float(s.get("d", 0.0)),
        )
        plan = _plan_with_trajectory([{"d": 1.0}, {"d": 2.0}, {"d": 3.0}])
        assert m.evaluate(plan) == pytest.approx(6.0)

    def test_returns_zero_for_empty_trajectory(self) -> None:
        from langgoap.planner.metrics import MinimizeTrajectorySum

        m = MinimizeTrajectorySum(name="t", expression=lambda s: 1.0)
        empty = Plan(actions=(), expected_states=(), total_cost=0.0)
        assert m.evaluate(empty) == 0.0


class TestMinimizeTrajectoryMax:
    def test_reports_worst_case_value_along_trajectory(self) -> None:
        from langgoap.planner.metrics import MinimizeTrajectoryMax

        m = MinimizeTrajectoryMax(
            name="peak_threat",
            expression=lambda s: float(s.get("d", 0.0)),
        )
        plan = _plan_with_trajectory([{"d": 2.0}, {"d": 5.0}, {"d": 1.0}])
        assert m.evaluate(plan) == pytest.approx(5.0)


class TestProtocolConformance:
    def test_concrete_metrics_satisfy_plan_quality_metric_protocol(self) -> None:
        from langgoap.planner.metrics import (
            MinimizeFinalStateExpression,
            MinimizeTrajectoryMax,
            MinimizeTrajectorySum,
            PlanQualityMetric,
        )

        for cls in (
            MinimizeFinalStateExpression,
            MinimizeTrajectorySum,
            MinimizeTrajectoryMax,
        ):
            m = cls(name="t", expression=lambda s: 0.0)
            assert isinstance(m, PlanQualityMetric)
            assert m.name == "t"
            assert m.weight == 1.0


# ---------------------------------------------------------------------------
# CSP / GoalSpec integration
# ---------------------------------------------------------------------------


class TestGoalSpecCarriesMetrics:
    def test_goal_spec_accepts_metrics_tuple(self) -> None:
        from langgoap.goals import GoalSpec
        from langgoap.planner.metrics import MinimizeTrajectorySum

        m = MinimizeTrajectorySum(name="proximity", expression=lambda s: 0.0)
        goal = GoalSpec(conditions={"done": True}, metrics=(m,))
        assert goal.metrics == (m,)

    def test_goal_spec_default_metrics_is_empty_tuple(self) -> None:
        from langgoap.goals import GoalSpec

        goal = GoalSpec(conditions={"done": True})
        assert goal.metrics == ()


class TestCSPScoresMetrics:
    def test_metric_value_contributes_to_soft_score_negatively(self) -> None:
        """Higher metric value (worse plan) → lower soft score."""
        from langgoap import ActionSpec, GoalSpec, GoapGraph
        from langgoap.planner.metrics import MinimizeFinalStateExpression

        actions = [
            ActionSpec(
                name="a",
                preconditions={},
                effects={"done": True, "penalty": 7.0},
                cost=1.0,
            ),
        ]
        m = MinimizeFinalStateExpression(
            name="terminal_penalty",
            expression=lambda s: float(s.get("penalty", 0.0)),
            weight=1.0,
        )
        goal = GoalSpec(conditions={"done": True}, metrics=(m,))
        graph = GoapGraph(actions).compile()
        result = graph.invoke({"world_state": {}, "goal": goal})
        assert result["status"] == "goal_achieved"
        plan = result["plan"]
        # After CSP re-scoring the soft term must carry the metric penalty.
        assert plan.score.soft == pytest.approx(-7.0)


class TestMetricsOnlyGoalShortCircuitsCSP:
    """Metrics-only goals skip ``enumerate_alternatives`` + CP-SAT.

    A goal with ``metrics`` but no ``constraints`` and no ``objectives``
    carries preferences, not constraints — there is nothing for CSP to
    validate.  The pipeline must return the primary A* plan verbatim
    with metrics folded into the soft score.  Prior to this fix
    ``validate_plan`` returned ``SKIPPED`` (its "no-op" sentinel),
    which the pipeline interpreted as "primary infeasible" and then
    ran ``enumerate_alternatives`` + ``optimize_plans`` — an
    unreachable selection path for metric-only goals because CP-SAT
    cannot express arbitrary Python metric callables.  On
    long-horizon problems with a low action-name cardinality (e.g.
    Pac-Man's four macro-actions) the name-blacklist alternatives
    search explodes into unbounded A* runs; this regression hung the
    trajectory-cost benchmark for 3+ hours before the fix.
    """

    def test_metrics_only_goal_does_not_invoke_alternatives_or_cpsat(
        self,
    ) -> None:
        from unittest.mock import patch

        from langgoap import ActionSpec, GoalSpec
        from langgoap.planner.metrics import MinimizeFinalStateExpression
        from langgoap.planner.pipeline import plan as pipeline_plan

        actions = [
            ActionSpec(
                name="a",
                preconditions={},
                effects={"done": True, "penalty": 3.0},
                cost=1.0,
            ),
        ]
        metric = MinimizeFinalStateExpression(
            name="terminal_penalty",
            expression=lambda s: float(s.get("penalty", 0.0)),
        )
        goal = GoalSpec(conditions={"done": True}, metrics=(metric,))

        with (
            patch("langgoap.planner.pipeline.enumerate_alternatives") as enum_spy,
            patch("langgoap.planner.pipeline.optimize_plans") as cpsat_spy,
        ):
            result = pipeline_plan({}, goal, actions)

        assert result is not None
        assert result.score.soft == pytest.approx(-3.0)
        enum_spy.assert_not_called()
        cpsat_spy.assert_not_called()
