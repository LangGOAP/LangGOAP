"""Tests for the custom LangGOAP serializer (serde module).

Covers round-trip serialization for every LangGOAP frozen dataclass and
immutable container type that flows through LangGraph's checkpointer
wire format.  These are pure unit tests -- no Docker, no external services.
"""

from __future__ import annotations

from datetime import timedelta
from types import MappingProxyType
from typing import Any

import pytest

from langgoap import (
    ActionSpec,
    BendableScore,
    ConstraintSpec,
    GoalSpec,
    HardSoftScore,
    LangGoapSerializer,
    Plan,
    PlanMetadata,
    PlanningState,
    SimpleScore,
    install_langgoap_serde,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_trip(serde: LangGoapSerializer, obj: Any) -> Any:
    """Serialize then deserialize *obj* through the given serde."""
    type_tag, payload = serde.dumps_typed(obj)
    return serde.loads_typed((type_tag, payload))


# ---------------------------------------------------------------------------
# MappingProxyType round-trips
# ---------------------------------------------------------------------------


class TestMappingProxyTypeRoundTrip:
    def test_non_empty_mapping_proxy(self) -> None:
        serde = LangGoapSerializer()
        original = MappingProxyType({"a": True, "b": 42, "c": "hello"})
        restored = _round_trip(serde, original)
        # MappingProxyType decodes as a plain dict because the stock ext
        # hook has no special branch for it; the dataclass __post_init__
        # re-wraps it on construction.  The important thing is that the
        # *values* survive the round-trip.
        assert dict(restored) == dict(original)

    def test_empty_mapping_proxy(self) -> None:
        serde = LangGoapSerializer()
        original = MappingProxyType({})
        restored = _round_trip(serde, original)
        assert dict(restored) == {}


# ---------------------------------------------------------------------------
# frozenset[tuple[str, bool]] round-trips
# ---------------------------------------------------------------------------


class TestFrozensetTupleRoundTrip:
    def test_frozenset_of_tuples(self) -> None:
        """This is the critical path for PlanningState.conditions."""
        serde = LangGoapSerializer()
        original = frozenset({("has_data", True), ("ready", False)})
        restored = _round_trip(serde, original)
        assert isinstance(restored, frozenset)
        assert restored == original

    def test_empty_frozenset(self) -> None:
        serde = LangGoapSerializer()
        original: frozenset[tuple[str, bool]] = frozenset()
        restored = _round_trip(serde, original)
        assert isinstance(restored, frozenset)
        assert restored == original

    def test_frozenset_with_mixed_value_types(self) -> None:
        serde = LangGoapSerializer()
        original = frozenset({("count", 5), ("flag", True), ("label", "x")})
        restored = _round_trip(serde, original)
        assert isinstance(restored, frozenset)
        assert restored == original


# ---------------------------------------------------------------------------
# ActionSpec round-trips
# ---------------------------------------------------------------------------


def _sync_execute(state: dict[str, Any]) -> dict[str, Any]:
    return {"done": True}


async def _async_execute(state: dict[str, Any]) -> dict[str, Any]:
    return {"done": True}


def _validator(pre: dict[str, Any], post: dict[str, Any]) -> bool:
    return True


class TestActionSpecRoundTrip:
    def test_action_with_callables_serialized_as_none(self) -> None:
        serde = LangGoapSerializer()
        original = ActionSpec(
            name="test_action",
            preconditions={"a": True},
            effects={"b": True},
            cost=2.5,
            execute=_sync_execute,
            aexecute=_async_execute,
            effect_validator=_validator,
            max_retries=3,
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        assert restored.name == "test_action"
        assert dict(restored.preconditions) == {"a": True}
        assert dict(restored.effects) == {"b": True}
        assert restored.cost == 2.5
        # Callable fields serialize as None
        assert restored.execute is None
        assert restored.aexecute is None
        assert restored.effect_validator is None
        assert restored.max_retries == 3

    def test_action_with_all_none_callables(self) -> None:
        serde = LangGoapSerializer()
        original = ActionSpec(
            name="minimal",
            preconditions={},
            effects={"x": True},
        )
        assert original.execute is None
        assert original.aexecute is None
        assert original.effect_validator is None
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        assert restored.name == "minimal"
        assert restored.execute is None
        assert restored.aexecute is None
        assert restored.effect_validator is None

    def test_action_with_resources_and_duration(self) -> None:
        serde = LangGoapSerializer()
        original = ActionSpec(
            name="resource_action",
            effects={"done": True},
            resources={"tokens": 500.0, "cost_usd": 0.02},
            duration=timedelta(seconds=5),
            metadata={"description": "test action"},
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        assert restored.name == "resource_action"
        assert dict(restored.resources) == {"tokens": 500.0, "cost_usd": 0.02}  # type: ignore[arg-type]
        assert restored.duration == timedelta(seconds=5)
        assert dict(restored.metadata) == {"description": "test action"}  # type: ignore[arg-type]

    def test_action_with_dynamic_cost_function_serialized_as_none(self) -> None:
        """Dynamic cost functions are callables and must serialize as None."""
        serde = LangGoapSerializer()

        def cost_fn(ws: dict[str, Any]) -> float:
            return ws.get("difficulty", 1.0) * 2.0

        original = ActionSpec(name="dynamic_cost", effects={"x": True}, cost=cost_fn)
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        # The callable cost serializes as None because lambdas/functions
        # are not picklable across process boundaries.
        assert restored.cost is None

    def test_action_with_callable_effects_round_trips(self) -> None:
        """Dynamic-effect ActionSpec must survive a round-trip: the
        callable is lost (callables aren't portable across process
        boundaries), but effect_keys is preserved and a sentinel
        callable is installed so has_dynamic_effects stays True."""
        serde = LangGoapSerializer()

        def eat_here(state: dict[str, Any]) -> dict[str, Any]:
            return {"food": state["food"] - frozenset({state["location"]})}

        original = ActionSpec(
            name="eat",
            effects=eat_here,
            effect_keys=frozenset({"food"}),
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        assert restored.name == "eat"
        assert restored.effect_keys == frozenset({"food"})
        assert restored.has_dynamic_effects is True

    def test_restored_callable_effect_raises_on_invocation(self) -> None:
        """The sentinel installed by deserialization must raise a clear
        error if invoked before the caller re-binds the real callable."""
        serde = LangGoapSerializer()

        def eat_here(state: dict[str, Any]) -> dict[str, Any]:
            return {"food": frozenset()}

        original = ActionSpec(
            name="eat",
            effects=eat_here,
            effect_keys=frozenset({"food"}),
        )
        restored = _round_trip(serde, original)
        assert restored is not None
        with pytest.raises(RuntimeError, match="re-bound|rebind|checkpoint"):
            restored.get_effects({"food": frozenset({"a"}), "location": "a"})


# ---------------------------------------------------------------------------
# GoalSpec round-trips
# ---------------------------------------------------------------------------


class TestGoalSpecRoundTrip:
    def test_goal_with_constraints(self) -> None:
        serde = LangGoapSerializer()
        c1 = ConstraintSpec(key="tokens", max=1000.0)
        c2 = ConstraintSpec(key="cost_usd", max=0.10, level="soft", weight=2.0)
        original = GoalSpec(
            conditions={"report_ready": True},
            constraints=(c1, c2),
            max_replans=5,
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, GoalSpec)
        assert dict(restored.conditions) == {"report_ready": True}
        assert len(restored.constraints) == 2
        assert restored.constraints[0].key == "tokens"
        assert restored.constraints[0].max == 1000.0
        assert restored.constraints[1].key == "cost_usd"
        assert restored.constraints[1].level == "soft"
        assert restored.constraints[1].weight == 2.0
        assert restored.max_replans == 5

    def test_goal_minimal(self) -> None:
        serde = LangGoapSerializer()
        original = GoalSpec(conditions={"done": True})
        restored = _round_trip(serde, original)
        assert isinstance(restored, GoalSpec)
        assert dict(restored.conditions) == {"done": True}
        assert restored.constraints == ()


# ---------------------------------------------------------------------------
# Plan round-trips (nested dataclasses)
# ---------------------------------------------------------------------------


class TestPlanRoundTrip:
    def test_plan_with_actions_and_expected_states(self) -> None:
        serde = LangGoapSerializer()
        a1 = ActionSpec(name="gather", effects={"has_data": True}, cost=1.0)
        a2 = ActionSpec(
            name="analyze",
            preconditions={"has_data": True},
            effects={"analyzed": True},
            cost=2.0,
        )
        s1 = PlanningState.from_dict({"has_data": True})
        s2 = PlanningState.from_dict({"has_data": True, "analyzed": True})
        original = Plan(
            actions=(a1, a2),
            expected_states=(s1, s2),
            total_cost=3.0,
            metadata=PlanMetadata(nodes_explored=10, planning_time_ms=42.5),
            score=SimpleScore(scalar=3.0),
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, Plan)
        assert len(restored.actions) == 2
        assert restored.actions[0].name == "gather"
        assert restored.actions[1].name == "analyze"
        assert dict(restored.actions[0].effects) == {"has_data": True}
        assert dict(restored.actions[1].preconditions) == {"has_data": True}
        # Expected states
        assert len(restored.expected_states) == 2
        assert isinstance(restored.expected_states[0], PlanningState)
        assert restored.expected_states[0].to_dict() == {"has_data": True}
        assert restored.expected_states[1].to_dict() == {
            "has_data": True,
            "analyzed": True,
        }
        # Metadata
        assert isinstance(restored.metadata, PlanMetadata)
        assert restored.metadata.nodes_explored == 10
        assert restored.metadata.planning_time_ms == 42.5
        # Score
        assert isinstance(restored.score, SimpleScore)
        assert restored.score.scalar == 3.0
        # Total cost
        assert restored.total_cost == 3.0

    def test_empty_plan(self) -> None:
        serde = LangGoapSerializer()
        original = Plan.empty()
        restored = _round_trip(serde, original)
        assert isinstance(restored, Plan)
        assert len(restored.actions) == 0
        assert len(restored.expected_states) == 0
        assert restored.total_cost == 0.0


# ---------------------------------------------------------------------------
# PlanningState round-trips
# ---------------------------------------------------------------------------


class TestPlanningStateRoundTrip:
    def test_from_dict_round_trip(self) -> None:
        serde = LangGoapSerializer()
        original = PlanningState.from_dict({"has_data": True, "count": 5, "label": "x"})
        restored = _round_trip(serde, original)
        assert isinstance(restored, PlanningState)
        assert restored.to_dict() == original.to_dict()

    def test_empty_planning_state(self) -> None:
        serde = LangGoapSerializer()
        original = PlanningState.from_dict({})
        restored = _round_trip(serde, original)
        assert isinstance(restored, PlanningState)
        assert restored.to_dict() == {}

    def test_conditions_frozenset_preserved(self) -> None:
        """The conditions frozenset[tuple[str, Any]] must survive the round-trip."""
        serde = LangGoapSerializer()
        original = PlanningState.from_dict({"a": True, "b": False})
        restored = _round_trip(serde, original)
        assert isinstance(restored.conditions, frozenset)
        # Each item should be a tuple, not a list
        for item in restored.conditions:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}: {item}"


# ---------------------------------------------------------------------------
# timedelta round-trips
# ---------------------------------------------------------------------------


class TestTimedeltaRoundTrip:
    def test_timedelta_in_action_spec(self) -> None:
        """timedelta is used in ActionSpec.duration."""
        serde = LangGoapSerializer()
        original = ActionSpec(
            name="timed",
            effects={"done": True},
            duration=timedelta(minutes=2, seconds=30),
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, ActionSpec)
        assert restored.duration == timedelta(minutes=2, seconds=30)

    def test_timedelta_standalone(self) -> None:
        serde = LangGoapSerializer()
        original = timedelta(hours=1, minutes=30)
        restored = _round_trip(serde, original)
        assert restored == original


# ---------------------------------------------------------------------------
# Tuple identity preservation
# ---------------------------------------------------------------------------


class TestTuplePreservation:
    def test_plain_tuple_round_trips_as_tuple(self) -> None:
        serde = LangGoapSerializer()
        original = (1, "two", True, 4.0)
        restored = _round_trip(serde, original)
        assert isinstance(restored, tuple)
        assert restored == original

    def test_empty_tuple(self) -> None:
        serde = LangGoapSerializer()
        original: tuple[()] = ()
        restored = _round_trip(serde, original)
        assert isinstance(restored, tuple)
        assert restored == ()

    def test_nested_tuples(self) -> None:
        serde = LangGoapSerializer()
        original = ((1, 2), (3, 4))
        restored = _round_trip(serde, original)
        assert isinstance(restored, tuple)
        assert all(isinstance(item, tuple) for item in restored)
        assert restored == original

    def test_tuple_of_action_specs(self) -> None:
        """Plan.actions is tuple[ActionSpec, ...]; tuple identity must survive."""
        serde = LangGoapSerializer()
        a1 = ActionSpec(name="a", effects={"x": True})
        a2 = ActionSpec(name="b", effects={"y": True})
        original = (a1, a2)
        restored = _round_trip(serde, original)
        assert isinstance(restored, tuple)
        assert len(restored) == 2
        assert restored[0].name == "a"
        assert restored[1].name == "b"


# ---------------------------------------------------------------------------
# Score hierarchy round-trips
# ---------------------------------------------------------------------------


class TestScoreRoundTrip:
    def test_simple_score(self) -> None:
        serde = LangGoapSerializer()
        original = SimpleScore(scalar=42.0)
        restored = _round_trip(serde, original)
        assert isinstance(restored, SimpleScore)
        assert restored.scalar == 42.0

    def test_simple_score_default(self) -> None:
        serde = LangGoapSerializer()
        original = SimpleScore()
        restored = _round_trip(serde, original)
        assert isinstance(restored, SimpleScore)
        assert restored.scalar == 0.0

    def test_hard_soft_score(self) -> None:
        serde = LangGoapSerializer()
        original = HardSoftScore(hard=-5.0, soft=3.0)
        restored = _round_trip(serde, original)
        assert isinstance(restored, HardSoftScore)
        assert restored.hard == -5.0
        assert restored.soft == 3.0

    def test_hard_soft_score_feasible(self) -> None:
        serde = LangGoapSerializer()
        original = HardSoftScore(hard=0.0, soft=-2.5)
        restored = _round_trip(serde, original)
        assert isinstance(restored, HardSoftScore)
        assert restored.is_feasible() is True
        assert restored.soft == -2.5

    def test_bendable_score(self) -> None:
        serde = LangGoapSerializer()
        original = BendableScore(
            hard_levels=(0.0, -1.0),
            soft_levels=(-3.0, 2.0, -0.5),
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, BendableScore)
        assert restored.hard_levels == (0.0, -1.0)
        assert restored.soft_levels == (-3.0, 2.0, -0.5)

    def test_bendable_score_empty_levels(self) -> None:
        serde = LangGoapSerializer()
        original = BendableScore()
        restored = _round_trip(serde, original)
        assert isinstance(restored, BendableScore)
        assert restored.hard_levels == ()
        assert restored.soft_levels == ()


# ---------------------------------------------------------------------------
# install_langgoap_serde
# ---------------------------------------------------------------------------


class TestInstallLangGoapSerde:
    def test_idempotent_on_memory_saver(self) -> None:
        """Calling install_langgoap_serde twice must not error or double-wrap."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        result1 = install_langgoap_serde(cp)
        assert isinstance(result1.serde, LangGoapSerializer)

        result2 = install_langgoap_serde(cp)
        assert result2 is cp  # same object returned
        assert isinstance(result2.serde, LangGoapSerializer)

    def test_preserves_pickle_fallback(self) -> None:
        """install_langgoap_serde must carry over pickle_fallback from the original serde."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        cp = MemorySaver()
        cp.serde = JsonPlusSerializer(pickle_fallback=True)
        install_langgoap_serde(cp)
        assert isinstance(cp.serde, LangGoapSerializer)
        assert cp.serde.pickle_fallback is True


# ---------------------------------------------------------------------------
# Strict allowed_msgpack_modules by default
# ---------------------------------------------------------------------------


class TestAllowedMsgpackModules:
    """LangGoapSerializer must ship a strict default allowlist covering
    every LangGoap dataclass that flows through a checkpoint, so users
    never see the LangGraph 'unregistered type' deprecation warning."""

    _LANGGOAP_LOGGER = "langgraph.checkpoint.serde.jsonplus"

    def test_default_allowlist_is_strict_not_legacy(self) -> None:
        """Default must not be the legacy ``True`` (allow-all)."""
        serde = LangGoapSerializer()
        assert serde._allowed_msgpack_modules is not True
        assert serde._allowed_msgpack_modules is not None

    def test_round_trip_of_core_types_emits_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every LangGoap dataclass routinely checkpointed must round-trip
        without an 'unregistered type' log on the langgraph serde logger."""
        serde = LangGoapSerializer()
        samples: list[Any] = [
            ActionSpec(name="a", effects={"x": True}),
            GoalSpec(conditions={"x": True}),
            ConstraintSpec(key="tokens", max=100.0),
            PlanningState.from_dict({"x": True}),
            SimpleScore(scalar=1.0),
            HardSoftScore(hard=0.0, soft=1.0),
            BendableScore(hard_levels=(0.0,), soft_levels=(1.0,)),
        ]
        with caplog.at_level("WARNING", logger=self._LANGGOAP_LOGGER):
            for obj in samples:
                _round_trip(serde, obj)
        unregistered = [
            rec for rec in caplog.records if "unregistered" in rec.message.lower()
        ]
        assert unregistered == [], (
            "LangGoapSerializer should register every LangGoap dataclass in "
            "its default allowed_msgpack_modules. Got warnings: "
            f"{[r.getMessage() for r in unregistered]}"
        )

    def test_user_allowlist_is_merged_with_langgoap_defaults(self) -> None:
        """Users extending the allowlist must still get LangGoap defaults."""
        import dataclasses

        @dataclasses.dataclass
        class UserType:
            x: int

        serde = LangGoapSerializer(
            allowed_msgpack_modules=[(UserType.__module__, UserType.__name__)]
        )
        allow = serde._allowed_msgpack_modules
        assert isinstance(allow, set)
        assert ("langgoap.actions", "ActionSpec") in allow
        assert (UserType.__module__, UserType.__name__) in allow

    def test_legacy_true_opt_in_still_works(self) -> None:
        """Users who explicitly pass ``True`` keep legacy allow-all behavior."""
        serde = LangGoapSerializer(allowed_msgpack_modules=True)
        assert serde._allowed_msgpack_modules is True

    def test_install_langgoap_serde_uses_strict_default(self) -> None:
        """install_langgoap_serde on a bare checkpointer must produce a
        serde with the strict LangGoap allowlist, not legacy ``True``."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        install_langgoap_serde(cp)
        assert isinstance(cp.serde, LangGoapSerializer)
        assert cp.serde._allowed_msgpack_modules is not True
        assert isinstance(cp.serde._allowed_msgpack_modules, set)
        assert ("langgoap.actions", "ActionSpec") in cp.serde._allowed_msgpack_modules


# ---------------------------------------------------------------------------
# Special values: None, bytes, bytearray
# ---------------------------------------------------------------------------


class TestSpecialValues:
    def test_none(self) -> None:
        serde = LangGoapSerializer()
        type_tag, payload = serde.dumps_typed(None)
        assert type_tag == "null"
        assert payload == b""
        restored = serde.loads_typed((type_tag, payload))
        assert restored is None

    def test_bytes(self) -> None:
        serde = LangGoapSerializer()
        original = b"hello bytes"
        type_tag, payload = serde.dumps_typed(original)
        assert type_tag == "bytes"
        assert payload == original
        restored = serde.loads_typed((type_tag, payload))
        assert restored == original

    def test_bytearray(self) -> None:
        serde = LangGoapSerializer()
        original = bytearray(b"hello bytearray")
        type_tag, payload = serde.dumps_typed(original)
        assert type_tag == "bytearray"
        restored = serde.loads_typed((type_tag, payload))
        assert restored == original


# ---------------------------------------------------------------------------
# Edge cases: deeply nested structures
# ---------------------------------------------------------------------------


class TestNestedDataclasses:
    def test_plan_containing_action_containing_mapping_proxy(self) -> None:
        """End-to-end: Plan -> ActionSpec -> MappingProxyType fields."""
        serde = LangGoapSerializer()
        action = ActionSpec(
            name="nested_action",
            preconditions={"source_ready": True},
            effects={"output_ready": True},
            cost=1.5,
            resources={"tokens": 200.0, "api_calls": 1.0},
            metadata={"tag": "test"},
        )
        state = PlanningState.from_dict({"output_ready": True})
        plan = Plan(
            actions=(action,),
            expected_states=(state,),
            total_cost=1.5,
            metadata=PlanMetadata(nodes_explored=5, planning_time_ms=10.0),
            score=HardSoftScore(hard=0.0, soft=-1.5),
        )
        restored = _round_trip(serde, plan)
        assert isinstance(restored, Plan)
        assert len(restored.actions) == 1
        ra = restored.actions[0]
        assert ra.name == "nested_action"
        assert dict(ra.preconditions) == {"source_ready": True}
        assert dict(ra.effects) == {"output_ready": True}
        assert ra.cost == 1.5
        assert dict(ra.resources) == {"tokens": 200.0, "api_calls": 1.0}  # type: ignore[arg-type]
        assert dict(ra.metadata) == {"tag": "test"}  # type: ignore[arg-type]
        # Score
        assert isinstance(restored.score, HardSoftScore)
        assert restored.score.hard == 0.0
        assert restored.score.soft == -1.5
        # Expected states
        assert restored.expected_states[0].to_dict() == {"output_ready": True}

    def test_constraint_spec_round_trip(self) -> None:
        serde = LangGoapSerializer()
        original = ConstraintSpec(
            key="budget", max=100.0, min=10.0, weight=1.5, level="soft"
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, ConstraintSpec)
        assert restored.key == "budget"
        assert restored.max == 100.0
        assert restored.min == 10.0
        assert restored.weight == 1.5
        assert restored.level == "soft"

    def test_plan_metadata_round_trip(self) -> None:
        serde = LangGoapSerializer()
        original = PlanMetadata(
            nodes_explored=42,
            planning_time_ms=123.456,
            actions_pruned=3,
        )
        restored = _round_trip(serde, original)
        assert isinstance(restored, PlanMetadata)
        assert restored.nodes_explored == 42
        assert restored.planning_time_ms == 123.456
        assert restored.actions_pruned == 3
        assert restored.csp is None


# ---------------------------------------------------------------------------
# Sets (plain set, not just frozenset)
# ---------------------------------------------------------------------------


class TestSetRoundTrip:
    def test_plain_set_round_trips(self) -> None:
        serde = LangGoapSerializer()
        original = {1, 2, 3}
        restored = _round_trip(serde, original)
        assert isinstance(restored, set)
        assert restored == original

    def test_set_of_tuples(self) -> None:
        serde = LangGoapSerializer()
        original = {("a", 1), ("b", 2)}
        restored = _round_trip(serde, original)
        assert isinstance(restored, set)
        assert restored == original


# ---------------------------------------------------------------------------
# Dict / list pass-through (should not be mangled)
# ---------------------------------------------------------------------------


class TestStockTypesPassThrough:
    def test_plain_dict(self) -> None:
        serde = LangGoapSerializer()
        original = {"key": "value", "number": 42}
        restored = _round_trip(serde, original)
        assert restored == original

    def test_plain_list(self) -> None:
        serde = LangGoapSerializer()
        original = [1, 2, 3, "four"]
        restored = _round_trip(serde, original)
        assert restored == original

    def test_scalar_int(self) -> None:
        serde = LangGoapSerializer()
        assert _round_trip(serde, 42) == 42

    def test_scalar_float(self) -> None:
        serde = LangGoapSerializer()
        assert _round_trip(serde, 3.14) == pytest.approx(3.14)

    def test_scalar_string(self) -> None:
        serde = LangGoapSerializer()
        assert _round_trip(serde, "hello") == "hello"

    def test_scalar_bool(self) -> None:
        serde = LangGoapSerializer()
        assert _round_trip(serde, True) is True
        assert _round_trip(serde, False) is False
