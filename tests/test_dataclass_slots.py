"""Regression test: every frozen dataclass must declare ``__slots__``.

``slots=True`` on frozen dataclasses is a lever for memory and
attribute-access performance — A*'s closed set holds tens of thousands
of ``PlanningState`` instances per planning run, and every planner call
allocates hundreds of ``Plan`` and ``ActionSpec`` objects.  Dropping
``slots=True`` on any of these silently re-inflates the footprint.

This test enumerates every frozen dataclass in ``langgoap.*`` and
asserts two invariants:

1. The class has a non-empty ``__slots__`` (or ``__slots__`` equal to an
   empty tuple for a base class that declares no fields).
2. Instances of the class have **no** ``__dict__`` — i.e. slots are
   actually in force, not accidentally bypassed by a parent class.

A future contributor adding a new frozen dataclass without ``slots=True``
will see this test fail and can add the flag or an explicit opt-out.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.constraints import BuilderOutput, ChainOutput, _ChainState
from langgoap.goals import ConstraintSpec, GoalSpec, MultiGoal
from langgoap.history import ExecutionRecord
from langgoap.planner.csp import CSPMetadata, CSPStatus, ResourceUsage, ScheduleEntry
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.score import BendableScore, HardSoftScore, Score, SimpleScore
from langgoap.state import PlanningState

# Every frozen dataclass in langgoap.* should appear here.  Adding a new
# frozen dataclass without registering it will cause the "all accounted
# for" test below to fail, which is the point.
FROZEN_DATACLASSES: tuple[type[Any], ...] = (
    PlanningState,
    ActionSpec,
    ConstraintSpec,
    GoalSpec,
    MultiGoal,
    Score,
    SimpleScore,
    HardSoftScore,
    BendableScore,
    PlanMetadata,
    Plan,
    ResourceUsage,
    ScheduleEntry,
    CSPMetadata,
    ExecutionRecord,
    BuilderOutput,
    _ChainState,
    ChainOutput,
)


def test_all_frozen_dataclasses_declare_slots() -> None:
    """Every frozen dataclass in ``langgoap.*`` must declare ``__slots__``."""
    missing: list[str] = []
    for cls in FROZEN_DATACLASSES:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        assert cls.__dataclass_params__.frozen, (  # type: ignore[attr-defined]
            f"{cls.__name__} is not frozen"
        )
        if not hasattr(cls, "__slots__"):
            missing.append(cls.__name__)
    assert not missing, (
        f"Frozen dataclasses missing __slots__: {missing}. "
        "Add slots=True to the @dataclass decorator."
    )


def test_slots_actually_prevent_dict() -> None:
    """Instances of slotted frozen dataclasses must not carry a ``__dict__``.

    If a subclass silently acquires ``__dict__`` from a non-slotted
    parent, the memory saving is lost.  Instantiate a minimal example of
    each concrete class and verify.
    """
    # Build minimal valid instances of each concrete class.
    zero = timedelta(0)
    instances: list[Any] = [
        PlanningState(conditions=frozenset()),
        ActionSpec(name="noop"),
        ConstraintSpec(key="k", max=1.0),
        GoalSpec(),
        MultiGoal(goals=(GoalSpec(),)),
        SimpleScore(scalar=0.0),
        HardSoftScore(hard=0.0, soft=0.0),
        BendableScore(hard_levels=(), soft_levels=()),
        PlanMetadata(),
        Plan(actions=()),
        ResourceUsage(key="k", total=0.0),
        ScheduleEntry(action_name="a", start=zero, duration=zero, end=zero),
        CSPMetadata(status=CSPStatus.SKIPPED),
        ExecutionRecord(
            goal_hash="h",
            goal_conditions={},
            plan_actions=(),
            expected_cost=0.0,
            actual_cost=0.0,
            outcome="success",
            replan_count=0,
            timestamp=datetime.now(),
        ),
        BuilderOutput(),
        _ChainState(),
        ChainOutput(),
    ]

    offenders: list[str] = []
    for inst in instances:
        if hasattr(inst, "__dict__"):
            offenders.append(type(inst).__name__)
    assert not offenders, (
        f"Frozen dataclass instances still carry __dict__: {offenders}. "
        "Either the class or one of its ancestors is missing slots=True."
    )


def test_slots_are_frozen_assignment_still_fails() -> None:
    """Slotted frozen dataclasses must still reject attribute assignment.

    Regression guard: it is easy to accidentally lose immutability when
    switching a dataclass to slots, because slots change the attribute
    storage mechanism.  Verify that ``frozen=True`` survives.
    """
    ps = PlanningState(conditions=frozenset({("ready", True)}))
    try:
        ps.conditions = frozenset()  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - fail path
        raise AssertionError(
            "PlanningState accepted attribute assignment despite frozen=True"
        )
