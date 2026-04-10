"""Unit tests for the CSP optimizer module."""

from __future__ import annotations

from datetime import timedelta
from types import MappingProxyType
from typing import Any

import pytest

from langgoap import (
    ConstraintSpec,
    CSPMetadata,
    CSPStatus,
    GoalSpec,
    Maximize,
    Minimize,
    ResourceUsage,
    ScheduleEntry,
)
from langgoap.planner.csp import (
    build_dependency_graph,
    compute_resource_totals,
    optimize_plans,
    schedule_plan,
    validate_plan,
)
from langgoap.planner.types import Plan, PlanMetadata
from tests.conftest import make_action, make_plan

# ---------------------------------------------------------------------------
# CSPStatus
# ---------------------------------------------------------------------------


class TestCSPStatus:
    def test_all_values_exist(self) -> None:
        assert CSPStatus.OPTIMAL == "optimal"
        assert CSPStatus.FEASIBLE == "feasible"
        assert CSPStatus.INFEASIBLE == "infeasible"
        assert CSPStatus.ERROR == "error"
        assert CSPStatus.SKIPPED == "skipped"

    def test_is_string_enum(self) -> None:
        assert isinstance(CSPStatus.OPTIMAL, str)
        assert CSPStatus.OPTIMAL.upper() == "OPTIMAL"


# ---------------------------------------------------------------------------
# ResourceUsage
# ---------------------------------------------------------------------------


class TestResourceUsage:
    def test_creation(self) -> None:
        ru = ResourceUsage(key="tokens", total=500.0, constraint_max=1000.0)
        assert ru.key == "tokens"
        assert ru.total == 500.0
        assert ru.constraint_max == 1000.0
        assert ru.constraint_min is None
        assert ru.satisfied is True

    def test_unsatisfied(self) -> None:
        ru = ResourceUsage(
            key="cost", total=150.0, constraint_max=100.0, satisfied=False
        )
        assert ru.satisfied is False

    def test_frozen(self) -> None:
        ru = ResourceUsage(key="x", total=1.0)
        with pytest.raises(AttributeError):
            ru.total = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScheduleEntry
# ---------------------------------------------------------------------------


class TestScheduleEntry:
    def test_creation(self) -> None:
        se = ScheduleEntry(
            action_name="act",
            start=timedelta(seconds=0),
            duration=timedelta(seconds=5),
            end=timedelta(seconds=5),
        )
        assert se.action_name == "act"
        assert se.start == timedelta(seconds=0)
        assert se.duration == timedelta(seconds=5)
        assert se.end == timedelta(seconds=5)

    def test_end_equals_start_plus_duration(self) -> None:
        start = timedelta(seconds=10)
        dur = timedelta(seconds=3)
        se = ScheduleEntry(action_name="x", start=start, duration=dur, end=start + dur)
        assert se.end == se.start + se.duration

    def test_frozen(self) -> None:
        se = ScheduleEntry(
            action_name="x",
            start=timedelta(0),
            duration=timedelta(0),
            end=timedelta(0),
        )
        with pytest.raises(AttributeError):
            se.action_name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CSPMetadata
# ---------------------------------------------------------------------------


class TestCSPMetadata:
    def test_defaults(self) -> None:
        meta = CSPMetadata(status=CSPStatus.SKIPPED)
        assert meta.solver_time_ms == 0.0
        assert meta.resource_usage == ()
        assert meta.objective_values == MappingProxyType({})
        assert meta.schedule == ()
        assert meta.makespan is None
        assert meta.plans_evaluated == 0
        assert meta.scale_factor == 1000

    def test_frozen(self) -> None:
        meta = CSPMetadata(status=CSPStatus.OPTIMAL)
        with pytest.raises(AttributeError):
            meta.status = CSPStatus.ERROR  # type: ignore[misc]

    def test_objective_values_wrapped(self) -> None:
        meta = CSPMetadata(
            status=CSPStatus.FEASIBLE,
            objective_values={"cost": 42.0},  # type: ignore[arg-type]
        )
        assert isinstance(meta.objective_values, MappingProxyType)
        assert meta.objective_values["cost"] == 42.0

    def test_repr_skipped(self) -> None:
        meta = CSPMetadata(status=CSPStatus.SKIPPED)
        r = repr(meta)
        assert "skipped" in r

    def test_repr_with_details(self) -> None:
        meta = CSPMetadata(
            status=CSPStatus.OPTIMAL,
            solver_time_ms=1.5,
            resource_usage=(ResourceUsage(key="x", total=1.0),),
            objective_values=MappingProxyType({"cost": 10.0}),
            makespan=timedelta(seconds=5),
            plans_evaluated=3,
        )
        r = repr(meta)
        assert "optimal" in r
        assert "resources=1" in r
        assert "plans_evaluated=3" in r
        assert "makespan" in r


# ---------------------------------------------------------------------------
# compute_resource_totals
# ---------------------------------------------------------------------------


class TestResourceComputation:
    def test_sum_across_actions(self) -> None:
        a1 = make_action("a1", resources={"tokens": 100, "cost": 0.5})
        a2 = make_action("a2", resources={"tokens": 200, "cost": 1.0})
        totals = compute_resource_totals((a1, a2))
        assert totals == {"tokens": 300.0, "cost": 1.5}

    def test_actions_without_resources(self) -> None:
        a1 = make_action("a1")
        a2 = make_action("a2", resources={"tokens": 50})
        totals = compute_resource_totals((a1, a2))
        assert totals == {"tokens": 50.0}

    def test_empty_plan(self) -> None:
        totals = compute_resource_totals(())
        assert totals == {}

    def test_mixed_keys(self) -> None:
        a1 = make_action("a1", resources={"tokens": 100})
        a2 = make_action("a2", resources={"cost": 0.5})
        a3 = make_action("a3", resources={"tokens": 50, "cost": 0.3})
        totals = compute_resource_totals((a1, a2, a3))
        assert totals["tokens"] == 150.0
        assert totals["cost"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------


class TestDependencyGraph:
    def test_linear_chain(self) -> None:
        """A→B→C: B depends on A, C depends on B."""
        a = make_action("a", eff={"x": True})
        b = make_action("b", pre={"x": True}, eff={"y": True})
        c = make_action("c", pre={"y": True}, eff={"z": True})
        deps = build_dependency_graph((a, b, c))
        assert deps[0] == []
        assert deps[1] == [0]
        assert deps[2] == [1]

    def test_independent_actions(self) -> None:
        """No preconditions → no dependencies."""
        a = make_action("a", eff={"x": True})
        b = make_action("b", eff={"y": True})
        deps = build_dependency_graph((a, b))
        assert deps[0] == []
        assert deps[1] == []

    def test_diamond_dependency(self) -> None:
        """A produces x; B and C both need x; D needs both y and z."""
        a = make_action("a", eff={"x": True})
        b = make_action("b", pre={"x": True}, eff={"y": True})
        c = make_action("c", pre={"x": True}, eff={"z": True})
        d = make_action("d", pre={"y": True, "z": True}, eff={"done": True})
        deps = build_dependency_graph((a, b, c, d))
        assert deps[0] == []
        assert deps[1] == [0]
        assert deps[2] == [0]
        assert 2 in deps[3]  # depends on c for z
        assert 1 in deps[3]  # depends on b for y

    def test_closest_producer(self) -> None:
        """When multiple producers exist, picks the closest one before j."""
        a = make_action("a", eff={"x": True})
        b = make_action("b", eff={"x": True})  # also produces x
        c = make_action("c", pre={"x": True}, eff={"done": True})
        deps = build_dependency_graph((a, b, c))
        # c should depend on b (index 1), not a (index 0)
        assert deps[2] == [1]


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------


class TestValidatePlan:
    def _make_constrained_goal(
        self,
        constraints: list[ConstraintSpec] | None = None,
        objectives: dict[str, Any] | None = None,
    ) -> GoalSpec:
        return GoalSpec(
            conditions={"done": True},
            constraints=tuple(constraints) if constraints else (),
            objectives=objectives,
        )

    def test_no_constraints_skipped(self) -> None:
        """No constraints and no objectives → SKIPPED."""
        p = make_plan(make_action("a", eff={"done": True}))
        goal = GoalSpec(conditions={"done": True})
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.SKIPPED

    def test_within_constraints_feasible(self) -> None:
        """Resource usage within max → FEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"tokens": 100})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="tokens", max=200)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.FEASIBLE
        assert meta.plans_evaluated == 1

    def test_exceeds_max_infeasible(self) -> None:
        """Resource usage exceeds max → INFEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"tokens": 500})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="tokens", max=200)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.INFEASIBLE

    def test_below_min_infeasible(self) -> None:
        """Resource usage below min → INFEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"quality": 2})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="quality", min=5)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.INFEASIBLE

    def test_range_constraint_satisfied(self) -> None:
        """Resource between min and max → FEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"quality": 7})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="quality", min=5, max=10)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.FEASIBLE

    def test_resource_usage_details(self) -> None:
        """resource_usage contains per-key breakdown."""
        a = make_action("a", eff={"done": True}, resources={"tokens": 100, "cost": 0.5})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="tokens", max=200)]
        )
        meta = validate_plan(p, goal)
        token_usage = next(r for r in meta.resource_usage if r.key == "tokens")
        assert token_usage.total == 100.0
        assert token_usage.constraint_max == 200.0
        assert token_usage.satisfied is True
        # cost is unconstrained but still reported
        cost_usage = next(r for r in meta.resource_usage if r.key == "cost")
        assert cost_usage.total == 0.5
        assert cost_usage.constraint_max is None

    def test_missing_resource_key(self) -> None:
        """Constraint on a key with no action resources → total is 0."""
        a = make_action("a", eff={"done": True})  # no resources
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="tokens", max=100)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.FEASIBLE
        token_usage = next(r for r in meta.resource_usage if r.key == "tokens")
        assert token_usage.total == 0.0

    def test_objectives_compute_values(self) -> None:
        """Objectives compute values from resource totals."""
        a = make_action("a", eff={"done": True}, resources={"cost": 0.5, "quality": 8})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="cost", max=1.0)],
            objectives={"cost": Minimize, "quality": Maximize},
        )
        meta = validate_plan(p, goal)
        assert meta.objective_values["cost"] == 0.5
        assert meta.objective_values["quality"] == 8.0

    def test_pure_python_path_no_ortools_needed(self) -> None:
        """Resource-only validation (no durations) should work without CP-SAT."""
        a = make_action("a", eff={"done": True}, resources={"cost": 10})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="cost", max=20)]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.FEASIBLE
        assert meta.schedule == ()  # no scheduling performed

    def test_hard_violation_marks_infeasible(self) -> None:
        """Hard constraint violation → INFEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"cost": 500})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="cost", max=100, level="hard")]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.INFEASIBLE
        cost_usage = next(u for u in meta.resource_usage if u.key == "cost")
        assert cost_usage.level == "hard"
        assert cost_usage.satisfied is False

    def test_soft_violation_remains_feasible(self) -> None:
        """Soft constraint violation → FEASIBLE with level-tagged usage."""
        a = make_action("a", eff={"done": True}, resources={"cost": 500})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="cost", max=100, level="soft")]
        )
        meta = validate_plan(p, goal)
        # Soft violation does NOT mark infeasible
        assert meta.status == CSPStatus.FEASIBLE
        cost_usage = next(u for u in meta.resource_usage if u.key == "cost")
        assert cost_usage.level == "soft"
        assert cost_usage.satisfied is False

    def test_unconstrained_resource_keys_are_info(self) -> None:
        """Resource keys without matching constraints get level='info'."""
        a = make_action(
            "a", eff={"done": True}, resources={"tokens": 100, "latency_ms": 50}
        )
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[ConstraintSpec(key="tokens", max=200)]
        )
        meta = validate_plan(p, goal)
        latency_usage = next(u for u in meta.resource_usage if u.key == "latency_ms")
        assert latency_usage.level == "info"
        token_usage = next(u for u in meta.resource_usage if u.key == "tokens")
        assert token_usage.level == "hard"

    def test_hard_satisfied_with_soft_violated_is_feasible(self) -> None:
        """Mixed constraints: hard satisfied + soft violated → FEASIBLE."""
        a = make_action("a", eff={"done": True}, resources={"gpu": 1, "cost": 500})
        p = make_plan(a)
        goal = self._make_constrained_goal(
            constraints=[
                ConstraintSpec(key="gpu", max=10, level="hard"),
                ConstraintSpec(key="cost", max=100, level="soft"),
            ]
        )
        meta = validate_plan(p, goal)
        assert meta.status == CSPStatus.FEASIBLE


# ---------------------------------------------------------------------------
# schedule_plan
# ---------------------------------------------------------------------------


class TestSchedulePlan:
    def test_no_durations_skipped(self) -> None:
        """Actions without durations → SKIPPED."""
        a = make_action("a", eff={"x": True})
        p = make_plan(a)
        meta = schedule_plan(p)
        assert meta.status == CSPStatus.SKIPPED

    def test_linear_chain_schedule(self) -> None:
        """A→B→C linear chain: B starts after A finishes."""
        a = make_action("a", eff={"x": True}, duration=timedelta(seconds=2))
        b = make_action(
            "b",
            pre={"x": True},
            eff={"y": True},
            duration=timedelta(seconds=3),
        )
        c = make_action(
            "c",
            pre={"y": True},
            eff={"z": True},
            duration=timedelta(seconds=1),
        )
        p = make_plan(a, b, c)
        meta = schedule_plan(p)
        assert meta.status == CSPStatus.OPTIMAL
        assert meta.makespan is not None
        # Makespan = 2 + 3 + 1 = 6s for linear chain
        assert meta.makespan == timedelta(seconds=6)
        assert len(meta.schedule) == 3
        # Verify ordering
        entries = {e.action_name: e for e in meta.schedule}
        assert entries["b"].start >= entries["a"].end
        assert entries["c"].start >= entries["b"].end

    def test_independent_actions_parallelized(self) -> None:
        """Independent actions can run in parallel → makespan < sum of durations."""
        a = make_action("a", eff={"x": True}, duration=timedelta(seconds=3))
        b = make_action("b", eff={"y": True}, duration=timedelta(seconds=3))
        p = make_plan(a, b)
        meta = schedule_plan(p)
        assert meta.status == CSPStatus.OPTIMAL
        assert meta.makespan is not None
        # Independent → can run in parallel → makespan = max(3, 3) = 3
        assert meta.makespan == timedelta(seconds=3)

    def test_mixed_durations(self) -> None:
        """Some actions have durations, some don't (instantaneous)."""
        a = make_action("a", eff={"x": True}, duration=timedelta(seconds=2))
        b = make_action("b", pre={"x": True}, eff={"y": True})  # instantaneous
        c = make_action(
            "c",
            pre={"y": True},
            eff={"z": True},
            duration=timedelta(seconds=1),
        )
        p = make_plan(a, b, c)
        meta = schedule_plan(p)
        assert meta.status == CSPStatus.OPTIMAL
        assert meta.makespan is not None
        # a=2s, b=0s, c=1s → chain: 2+0+1=3s
        assert meta.makespan == timedelta(seconds=3)

    def test_makespan_equals_critical_path(self) -> None:
        """Diamond: A→(B,C)→D. Critical path is the longest branch."""
        a = make_action("a", eff={"x": True}, duration=timedelta(seconds=1))
        b = make_action(
            "b",
            pre={"x": True},
            eff={"y": True},
            duration=timedelta(seconds=5),
        )
        c = make_action(
            "c",
            pre={"x": True},
            eff={"z": True},
            duration=timedelta(seconds=2),
        )
        d = make_action(
            "d",
            pre={"y": True, "z": True},
            eff={"done": True},
            duration=timedelta(seconds=1),
        )
        p = make_plan(a, b, c, d)
        meta = schedule_plan(p)
        assert meta.status == CSPStatus.OPTIMAL
        assert meta.makespan is not None
        # Critical path: a(1) + b(5) + d(1) = 7s (not a+c+d = 4s)
        assert meta.makespan == timedelta(seconds=7)


# ---------------------------------------------------------------------------
# optimize_plans
# ---------------------------------------------------------------------------


class TestOptimizePlans:
    def test_selects_cheapest_feasible(self) -> None:
        """With MINIMIZE cost, selects the cheapest plan within constraints."""
        cheap = make_plan(
            make_action("a", eff={"done": True}, cost=1, resources={"cost": 10})
        )
        expensive = make_plan(
            make_action("b", eff={"done": True}, cost=5, resources={"cost": 50})
        )
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
            objectives={"cost": Minimize},
        )
        chosen, meta = optimize_plans([cheap, expensive], goal)
        assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)
        assert chosen.action_names == ["a"]

    def test_all_infeasible(self) -> None:
        """All plans exceed constraints → INFEASIBLE."""
        p1 = make_plan(make_action("a", eff={"done": True}, resources={"cost": 200}))
        p2 = make_plan(make_action("b", eff={"done": True}, resources={"cost": 300}))
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        _, meta = optimize_plans([p1, p2], goal)
        assert meta.status == CSPStatus.INFEASIBLE

    def test_single_plan(self) -> None:
        """Single feasible plan is selected."""
        p = make_plan(make_action("a", eff={"done": True}, resources={"cost": 10}))
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        chosen, meta = optimize_plans([p], goal)
        assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)
        assert chosen.action_names == ["a"]
        assert meta.plans_evaluated == 1

    def test_multi_objective_weighted(self) -> None:
        """Weighted multi-objective: minimize cost, maximize quality."""
        # Plan 1: low cost, low quality
        p1 = make_plan(
            make_action(
                "cheap",
                eff={"done": True},
                resources={"cost": 10, "quality": 2},
            )
        )
        # Plan 2: medium cost, high quality
        p2 = make_plan(
            make_action(
                "balanced",
                eff={"done": True},
                resources={"cost": 20, "quality": 50},
            )
        )
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
            objectives={"cost": Minimize, "quality": Maximize},
        )
        chosen, meta = optimize_plans([p1, p2], goal)
        assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)
        # CP-SAT minimizes (cost_scaled - quality_scaled) with equal weights.
        # p1 objective: 10*1000 - 2*1000 = 8_000
        # p2 objective: 20*1000 - 50*1000 = -30_000  ← wins (lower = better)
        # p2 wins because its quality advantage (Δ48 units) outweighs its
        # extra cost (Δ10 units) when scaled identically.
        assert chosen.action_names == ["balanced"]

    def test_empty_plans_raises(self) -> None:
        """Empty plans list raises ValueError."""
        goal = GoalSpec(conditions={"done": True})
        with pytest.raises(ValueError, match="No plans"):
            optimize_plans([], goal)

    def test_plans_evaluated_count(self) -> None:
        """plans_evaluated reflects the number of candidate plans."""
        plans = [
            make_plan(
                make_action(f"a{i}", eff={"done": True}, resources={"cost": i * 10})
            )
            for i in range(4)
        ]
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=100),),
        )
        _, meta = optimize_plans(plans, goal)
        assert meta.plans_evaluated == 4

    def test_soft_constraint_allows_violation_but_penalizes(self) -> None:
        """Soft max bound: a violating plan is still feasible and carries the
        penalty via the CP-SAT objective.  With only a soft budget and no
        objective, the solver should prefer the less-violating plan.
        """
        cheap = make_plan(
            make_action("cheap", eff={"done": True}, resources={"cost": 10})
        )
        expensive = make_plan(
            make_action("expensive", eff={"done": True}, resources={"cost": 50})
        )
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(ConstraintSpec(key="cost", max=5, weight=1.0, level="soft"),),
        )
        chosen, meta = optimize_plans([cheap, expensive], goal)
        # Soft budget → solver still returns a feasible status but minimizes
        # the penalty by preferring the plan with less violation.
        assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)
        assert chosen.action_names == ["cheap"]
        # The "cost" usage entry should be labelled soft and marked unsatisfied
        # (10 > 5) even though the plan is still returned.
        cost_usage = next(u for u in meta.resource_usage if u.key == "cost")
        assert cost_usage.level == "soft"
        assert cost_usage.satisfied is False

    def test_hard_and_soft_mix(self) -> None:
        """A plan with a hard constraint met and a soft constraint violated
        is chosen when both alternatives satisfy the hard bound."""
        low_gpu_high_cost = make_plan(
            make_action(
                "low_gpu_high_cost",
                eff={"done": True},
                resources={"gpu": 1, "cost": 20},
            )
        )
        high_gpu_low_cost = make_plan(
            make_action(
                "high_gpu_low_cost",
                eff={"done": True},
                resources={"gpu": 5, "cost": 5},
            )
        )
        goal = GoalSpec(
            conditions={"done": True},
            constraints=(
                ConstraintSpec(key="gpu", max=10, level="hard"),
                ConstraintSpec(key="cost", max=10, weight=1.0, level="soft"),
            ),
        )
        chosen, meta = optimize_plans([low_gpu_high_cost, high_gpu_low_cost], goal)
        # Both satisfy the hard GPU bound; the soft cost penalty picks the
        # plan with lower cost.
        assert meta.status in (CSPStatus.OPTIMAL, CSPStatus.FEASIBLE)
        assert chosen.action_names == ["high_gpu_low_cost"]
