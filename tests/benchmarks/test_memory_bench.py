"""Memory benchmarks for LangGoap's core frozen dataclasses.

Measures per-instance memory footprint for every hot-path dataclass:
PlanningState, ActionSpec, Plan, PlanMetadata, GoalSpec, CSPMetadata,
and the Score hierarchy.

``slots=True`` on frozen dataclasses removes the per-instance ``__dict__``
overhead (typically 48–112 bytes on CPython).  This benchmark records a
baseline so ``make benchmark-compare`` catches regressions if a future
refactor accidentally drops slots.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from types import MappingProxyType

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import ConstraintSpec, GoalSpec
from langgoap.planner.csp import CSPMetadata, CSPStatus, ResourceUsage, ScheduleEntry
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import BendableScore, HardSoftScore, SimpleScore
from langgoap.state import PlanningState

# ---------------------------------------------------------------------------
# Fixtures — representative instances
# ---------------------------------------------------------------------------


def _planning_state() -> PlanningState:
    return PlanningState.from_dict(
        {"a": True, "b": False, "c": True, "d": True, "e": False}
    )


def _action_spec() -> ActionSpec:
    return ActionSpec(
        name="gather_data",
        preconditions={"source": True},
        effects={"data_ready": True},
        cost=2.5,
        resources={"tokens": 500, "cost_usd": 0.02},
        duration=timedelta(seconds=3),
        metadata={"description": "Gather data from source"},
    )


def _goal_spec() -> GoalSpec:
    return GoalSpec(
        conditions={"report_done": True, "reviewed": True},
        constraints=(
            ConstraintSpec(key="tokens", max=10000.0, weight=1.0),
            ConstraintSpec(key="cost_usd", max=5.0, weight=2.0, level="soft"),
        ),
    )


def _plan() -> Plan:
    a1 = ActionSpec(name="step_a", effects={"a": True})
    a2 = ActionSpec(name="step_b", preconditions={"a": True}, effects={"b": True})
    s1 = PlanningState.from_dict({"a": True})
    s2 = PlanningState.from_dict({"a": True, "b": True})
    return Plan(
        actions=(a1, a2),
        expected_states=(s1, s2),
        total_cost=2.0,
        metadata=PlanMetadata(nodes_explored=15, planning_time_ms=1.2),
        score=SimpleScore(scalar=2.0),
    )


def _csp_metadata() -> CSPMetadata:
    return CSPMetadata(
        status=CSPStatus.FEASIBLE,
        solver_time_ms=42.5,
        resource_usage=(
            ResourceUsage(key="cpu", total=8.0, constraint_max=10.0, level="hard"),
            ResourceUsage(key="mem", total=3.0, constraint_max=5.0, level="soft"),
        ),
        objective_values=MappingProxyType({"cost": 4.5}),
        schedule=(
            ScheduleEntry(
                action_name="a",
                start=timedelta(0),
                duration=timedelta(seconds=2),
                end=timedelta(seconds=2),
            ),
        ),
        makespan=timedelta(seconds=5),
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.bench
class TestMemoryFootprint:
    """Benchmark per-instance ``sys.getsizeof`` for core dataclasses."""

    def test_planning_state(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = _planning_state()
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_action_spec(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = _action_spec()
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_goal_spec(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = _goal_spec()
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_plan(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = _plan()
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_plan_metadata(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = PlanMetadata(nodes_explored=100, planning_time_ms=5.0)
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_csp_metadata(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = _csp_metadata()
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_simple_score(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = SimpleScore(scalar=3.14)
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_hard_soft_score(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = HardSoftScore(hard=0.0, soft=-2.5)
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result

    def test_bendable_score(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        obj = BendableScore(
            hard_levels=(0.0, 0.0),
            soft_levels=(-1.0, -3.0, -0.5),
        )
        result = benchmark(sys.getsizeof, obj)
        benchmark.extra_info["bytes"] = result


@pytest.mark.bench
class TestSlotsRegression:
    """Assert every core frozen dataclass has ``__slots__`` and lacks ``__dict__``."""

    @pytest.mark.parametrize(
        "cls",
        [
            PlanningState,
            ActionSpec,
            GoalSpec,
            ConstraintSpec,
            Plan,
            PlanMetadata,
            CSPMetadata,
            ResourceUsage,
            ScheduleEntry,
            SimpleScore,
            HardSoftScore,
            BendableScore,
        ],
        ids=lambda c: c.__name__,
    )
    def test_has_slots(self, cls: type) -> None:
        assert hasattr(cls, "__slots__"), f"{cls.__name__} missing __slots__"

    @pytest.mark.parametrize(
        "factory",
        [
            _planning_state,
            _action_spec,
            _goal_spec,
            _plan,
            lambda: PlanMetadata(),
            _csp_metadata,
            lambda: SimpleScore(scalar=1.0),
            lambda: HardSoftScore(hard=0.0, soft=0.0),
            lambda: BendableScore(hard_levels=(0.0,), soft_levels=(0.0,)),
        ],
        ids=[
            "PlanningState",
            "ActionSpec",
            "GoalSpec",
            "Plan",
            "PlanMetadata",
            "CSPMetadata",
            "SimpleScore",
            "HardSoftScore",
            "BendableScore",
        ],
    )
    def test_no_dict(self, factory: object) -> None:
        obj = factory()  # type: ignore[operator]
        assert not hasattr(
            obj, "__dict__"
        ), f"{type(obj).__name__} has __dict__ — slots may be missing"


@pytest.mark.bench
class TestAllocationThroughput:
    """Benchmark allocation speed for hot-path objects."""

    def test_planning_state_alloc(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        d = {"a": True, "b": False, "c": True}
        benchmark(PlanningState.from_dict, d)

    def test_action_spec_alloc(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        benchmark(
            ActionSpec,
            name="test",
            preconditions={"x": True},
            effects={"y": True},
            cost=1.0,
        )

    def test_simple_score_alloc(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        benchmark(SimpleScore, scalar=2.0)
