"""Every JSON-native codec round-trips every field.

The audit of the v0.2.0 release found these fields serialised but never
asserted after a round trip: ``ActionResult.cost`` (only ``None`` was
covered), ``MultiGoal.on_infeasible`` and ``mode``, ``PlanMetadata``
counters, every ``CSPMetadata`` field, ``InfeasibilityExplanation``,
``ActionQos`` beyond ``max_attempts``, a typed-form ``ActionSpec``,
``NirvanaGoal`` and the legacy dicts without the newer keys.  Each test
here goes through LangGraph's ``JsonPlusSerializer`` (strict: no pickle
fallback), the serializer every checkpointer uses.
"""

from __future__ import annotations

from datetime import timedelta
from types import MappingProxyType
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from langgoap import (
    ActionQos,
    ActionSpec,
    ConstraintSpec,
    GoalPolicy,
    GoalSpec,
    MultiGoal,
    NirvanaGoal,
)
from langgoap.goals import goal_from_dict
from langgoap.graph.state import ActionResult
from langgoap.planner.csp import CSPMetadata, CSPStatus, ResourceUsage, ScheduleEntry
from langgoap.planner.explain import InfeasibilityExplanation, ResourceShortfall
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.types import ObjectiveDirection, ReplanStrategy

_SERDE = JsonPlusSerializer()


def _through_checkpointer(data: Any) -> Any:
    """What a checkpointer stores and reads back."""
    return _SERDE.loads_typed(_SERDE.dumps_typed(data))


class TestActionResult:
    def test_cost_round_trips(self) -> None:
        result = ActionResult(
            action_name="buy",
            success=True,
            state_before={"gold": 10},
            state_after={"gold": 7, "sword": True},
            cost=3.0,
        )
        restored = ActionResult.from_dict(_through_checkpointer(result.to_dict()))
        assert restored == result
        assert restored.cost == 3.0

    def test_failure_with_error_round_trips(self) -> None:
        result = ActionResult(
            action_name="buy",
            success=False,
            state_before={"gold": 1},
            state_after={"gold": 1},
            error="human_approval_denied: too dear",
            cost=3.0,
        )
        assert ActionResult.from_dict(_through_checkpointer(result.to_dict())) == result


class TestGoals:
    def test_multi_goal_mode_and_on_infeasible_round_trip(self) -> None:
        mg = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}, policy=GoalPolicy(priority=2)),
                GoalSpec(conditions={"b": True}),
            ),
            mode="sequential",
            on_infeasible="skip",
        )
        restored = goal_from_dict(_through_checkpointer(mg.to_dict()))
        assert isinstance(restored, MultiGoal)
        assert restored == mg
        assert restored.mode == "sequential"
        assert restored.on_infeasible == "skip"

    def test_legacy_multi_goal_dict_defaults(self) -> None:
        legacy = {"goals": [GoalSpec(conditions={"a": True}).to_dict()]}
        restored = goal_from_dict(legacy)
        assert isinstance(restored, MultiGoal)
        assert restored.mode == "sequential"
        assert restored.on_infeasible == "fail"

    def test_goal_policy_round_trips_every_field(self) -> None:
        policy = GoalPolicy(
            replan_strategy=ReplanStrategy.EVERY_ACTION,
            priority=7,
            max_replans=3,
            on_infeasible="execute",
            blacklist_fallback=True,
        )
        assert GoalPolicy.from_dict(_through_checkpointer(policy.to_dict())) == policy

    def test_legacy_goal_policy_dict_defaults(self) -> None:
        assert GoalPolicy.from_dict({}) == GoalPolicy()

    def test_nirvana_goal_round_trips_through_goal_from_dict(self) -> None:
        nirvana = NirvanaGoal(
            policy=GoalPolicy(max_replans=1),
            objectives={"cost_usd": ObjectiveDirection.MINIMIZE},
            value=2.5,
        )
        data = _through_checkpointer(nirvana.to_dict())
        assert data["kind"] == "nirvana"
        restored = goal_from_dict(data)
        assert isinstance(restored, NirvanaGoal)
        assert restored == nirvana
        assert dict(restored.conditions) == {}
        assert restored.value == 2.5


class TestPlanMetadata:
    @staticmethod
    def _csp() -> CSPMetadata:
        return CSPMetadata(
            status=CSPStatus.INFEASIBLE,
            solver_time_ms=12.5,
            resource_usage=(
                ResourceUsage(
                    key="cost_usd",
                    total=14.0,
                    constraint_max=10.0,
                    satisfied=False,
                    level="hard",
                ),
                ResourceUsage(key="tokens", total=300.0, level="info"),
            ),
            objective_values=MappingProxyType({"cost_usd": 14.0}),
            schedule=(
                ScheduleEntry(
                    action_name="a",
                    start=timedelta(0),
                    duration=timedelta(minutes=5),
                    end=timedelta(minutes=5),
                ),
            ),
            makespan=timedelta(minutes=5),
            plans_evaluated=4,
            scale_factor=100,
            explanation=InfeasibilityExplanation(
                conflicting_constraints=(ConstraintSpec(key="cost_usd", max=10.0),),
                resource_shortfalls=(
                    ResourceShortfall(
                        key="cost_usd", required=14.0, available_max=10.0, overrun=4.0
                    ),
                ),
                suggestion="raise the cap to 14",
            ),
        )

    def test_csp_metadata_round_trips_every_field(self) -> None:
        csp = self._csp()
        restored = CSPMetadata.from_dict(_through_checkpointer(csp.to_dict()))
        assert restored.status is CSPStatus.INFEASIBLE
        assert restored.solver_time_ms == 12.5
        assert restored.resource_usage == csp.resource_usage
        assert dict(restored.objective_values) == {"cost_usd": 14.0}
        assert restored.schedule == csp.schedule
        assert restored.makespan == timedelta(minutes=5)
        assert restored.plans_evaluated == 4
        assert restored.scale_factor == 100
        assert restored.explanation == csp.explanation

    def test_plan_metadata_counters_round_trip(self) -> None:
        meta = PlanMetadata(
            nodes_explored=17,
            applicable_count=5,
            planning_time_ms=3.25,
            actions_pruned=2,
            csp=self._csp(),
        )
        restored = PlanMetadata.from_dict(_through_checkpointer(meta.to_dict()))
        assert restored.nodes_explored == 17
        assert restored.applicable_count == 5
        assert restored.planning_time_ms == 3.25
        assert restored.actions_pruned == 2
        assert restored.csp is not None and restored.csp.plans_evaluated == 4

    def test_plan_round_trips_with_metadata(self) -> None:
        a = ActionSpec(name="a", preconditions={}, effects={"x": True}, cost=2.0)
        plan = Plan(actions=[a], total_cost=2.0, metadata=self._metadata())
        restored = Plan.from_dict(_through_checkpointer(plan.to_dict()), actions=[a])
        assert restored.action_names == ["a"]
        assert restored.total_cost == 2.0
        assert restored.metadata.applicable_count == 5
        assert restored.metadata.csp is not None
        assert restored.metadata.csp.status is CSPStatus.INFEASIBLE

    def _metadata(self) -> PlanMetadata:
        return PlanMetadata(
            nodes_explored=17, applicable_count=5, actions_pruned=2, csp=self._csp()
        )


class TestActionQos:
    def test_every_field_round_trips(self) -> None:
        qos = ActionQos(
            max_attempts=4,
            backoff_initial_ms=10.0,
            backoff_multiplier=3.0,
            backoff_max_ms=500.0,
            jitter=0.0,
            idempotent=True,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        data = _through_checkpointer(qos.to_dict())
        assert data["retryable_exceptions"] == ["ConnectionError", "TimeoutError"]
        restored = ActionQos.from_dict(data)
        assert restored.max_attempts == 4
        assert restored.backoff_initial_ms == 10.0
        assert restored.backoff_multiplier == 3.0
        assert restored.backoff_max_ms == 500.0
        assert restored.jitter == 0.0
        assert restored.idempotent is True
        # Exception classes cannot be serialised; the documented contract is
        # that they are recorded by name and reset to the default on load.
        assert restored.retryable_exceptions == (TimeoutError,)


class _Form(BaseModel):
    note: str


class TestActionSpec:
    def test_typed_form_gate_survives_as_a_boolean_gate(self) -> None:
        """A model class cannot be serialised; the documented contract is
        that the gate stays (``True``) and the schema is lost."""
        action = ActionSpec(
            name="confirm",
            preconditions={"ready": True},
            effects={"confirmed": True},
            require_human_approval=_Form,
            human_input_key="form",
            qos=ActionQos(max_attempts=2),
            can_rerun=False,
            max_retries=1,
        )
        data = _through_checkpointer(action.to_dict())
        assert data["require_human_approval"] is True
        restored = ActionSpec.from_dict(data)
        assert restored.require_human_approval is True
        assert restored.human_input_key == "form"
        assert restored.can_rerun is False
        assert restored.max_retries == 1
        assert restored.qos is not None and restored.qos.max_attempts == 2
