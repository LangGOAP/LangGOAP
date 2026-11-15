"""JSON-native ``GoapState`` (issue #17).

Every channel value LangGraph persists must be plain JSON so that *any*
``BaseCheckpointSaver`` and *any* serializer round-trips a GOAP run.  These
tests pin:

* the ``to_dict`` / ``from_dict`` codecs on every type that used to live in
  the state as a dataclass;
* the typed <-> JSON edges (``GoapGraph.invoke`` returns typed objects, a
  compiled graph exposes JSON, ``rehydrate`` converts back);
* a full plan / execute / replan / interrupt cycle through ``MemorySaver``
  with LangGraph's **stock** serializer in strict-msgpack mode, which
  refuses any non-registered type;
* the in-process registry that keeps callable goal fields (metrics,
  callable ``value``) alive across a checkpoint round-trip, and the
  documented degradation when the registry cannot serve a token.
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
from datetime import timedelta
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from langgoap import (
    ActionSpec,
    GoalPolicy,
    GoalSpec,
    GoapGraph,
    MultiGoal,
    successful_action_names,
)
from langgoap.goals import ConstraintSpec, SoftGoal, goal_from_dict
from langgoap.graph.state import (
    ActionResult,
    goal_from_state,
    goal_to_state,
    history_from_state,
    history_to_state,
    plan_from_state,
    plan_to_state,
    rehydrate_state,
    serialize_updates,
)
from langgoap.planner.csp import CSPMetadata, CSPStatus, ResourceUsage, ScheduleEntry
from langgoap.planner.explain import InfeasibilityExplanation, ResourceShortfall
from langgoap.planner.metrics import MinimizeFinalStateExpression
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.qos import ActionQos
from langgoap.score import BendableScore, HardSoftScore, SimpleScore, score_from_dict
from langgoap.state import PlanningState
from langgoap.types import ObjectiveDirection, ReplanStrategy

JSON_SCALARS = (str, int, float, bool, type(None))


def _assert_json_native(value: Any, path: str = "$") -> None:
    """Fail unless *value* is built from dicts, lists and JSON scalars only."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"{path}: non-string key {k!r}"
            _assert_json_native(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_json_native(v, f"{path}[{i}]")
    else:
        assert isinstance(
            value, JSON_SCALARS
        ), f"{path}: {type(value).__name__} is not JSON-native ({value!r})"


class _Form(BaseModel):
    reason: str


def _plan_fixture() -> tuple[Plan, list[ActionSpec]]:
    live = [
        ActionSpec(
            name="fetch",
            effects={"fetched": True},
            cost=lambda ws: 2.0,
            execute=lambda ws: {"fetched": True},
            duration=timedelta(minutes=5),
            resources={"tokens": 100.0},
            qos=ActionQos(max_attempts=3, idempotent=True),
        ),
        ActionSpec(
            name="summarise",
            preconditions={"fetched": True},
            effects=lambda ws: {"summary": True},
            effect_keys=frozenset({"summary"}),
            cost=3.0,
            max_retries=2,
            can_rerun=False,
            read_only=True,
            require_human_approval=_Form,
            human_input_key="review",
            metadata={"owner": "docs"},
        ),
    ]
    explanation = InfeasibilityExplanation(
        conflicting_constraints=(ConstraintSpec(key="tokens", max=50.0),),
        resource_shortfalls=(
            ResourceShortfall(
                key="tokens", required=100.0, available_max=50.0, overrun=50.0
            ),
        ),
        suggestion="raise the token budget",
    )
    csp = CSPMetadata(
        status=CSPStatus.INFEASIBLE,
        solver_time_ms=1.5,
        resource_usage=(
            ResourceUsage(
                key="tokens",
                total=100.0,
                constraint_max=50.0,
                satisfied=False,
                level="hard",
            ),
        ),
        objective_values={"tokens": 100.0},
        schedule=(
            ScheduleEntry(
                action_name="fetch",
                start=timedelta(0),
                duration=timedelta(minutes=5),
                end=timedelta(minutes=5),
            ),
        ),
        makespan=timedelta(minutes=5),
        plans_evaluated=3,
        explanation=explanation,
    )
    plan = Plan(
        actions=tuple(live),
        expected_states=(
            PlanningState.from_dict({"fetched": True}),
            PlanningState.from_dict({"fetched": True, "summary": True}),
        ),
        total_cost=5.0,
        metadata=PlanMetadata(nodes_explored=7, planning_time_ms=0.25, csp=csp),
        score=HardSoftScore(hard=-1.0, soft=-2.5),
    )
    return plan, live


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


class TestGoalCodec:
    def test_full_goal_round_trips_and_is_json(self) -> None:
        goal = GoalSpec(
            conditions={"done": True, "count": 3, "label": "x"},
            policy=GoalPolicy(
                replan_strategy=ReplanStrategy.EVERY_ACTION,
                priority=2,
                max_replans=4,
                on_infeasible="execute",
                blacklist_fallback=True,
            ),
            objectives={
                "cost": ObjectiveDirection.MINIMIZE,
                "quality": ObjectiveDirection.MAXIMIZE,
            },
            constraints=[
                ConstraintSpec(key="cost", max=10.0),
                ConstraintSpec(key="quality", min=0.5, weight=2.0, level="soft"),
            ],
            soft_goals=[SoftGoal(conditions={"bonus": True}, weight=1.5, name="bonus")],
            value=7.0,
        )
        data = goal.to_dict()
        _assert_json_native(data)
        json.dumps(data)
        restored = GoalSpec.from_dict(json.loads(json.dumps(data)))
        assert restored == goal
        assert goal_from_dict(data) == goal

    def test_callable_fields_are_declared_not_persisted(self) -> None:
        metric = MinimizeFinalStateExpression(
            name="penalty",
            expression=lambda s: float(s.get("penalty", 0.0)),
            weight=2.0,
        )
        goal = GoalSpec(
            conditions={"done": True}, metrics=(metric,), value=lambda ws: 3.0
        )
        data = goal.to_dict()
        _assert_json_native(data)
        assert data["value"] is None
        assert data["metrics"] == [
            {"name": "penalty", "weight": 2.0, "type": "MinimizeFinalStateExpression"}
        ]
        restored = GoalSpec.from_dict(data)
        assert restored.value == 1.0
        assert restored.metrics == ()
        assert restored.conditions == goal.conditions

    def test_multi_goal_round_trips(self) -> None:
        multi = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}),
                GoalSpec(conditions={"b": True}, value=2.0),
            ),
            mode="best_value",
        )
        data = multi.to_dict()
        _assert_json_native(data)
        assert data["kind"] == "multi"
        assert MultiGoal.from_dict(data) == multi
        assert goal_from_dict(data) == multi

    def test_goal_from_dict_defaults_to_single_goal_without_kind(self) -> None:
        assert goal_from_dict({"conditions": {"x": 1}}) == GoalSpec(conditions={"x": 1})


class TestActionAndPlanCodec:
    def test_action_static_fields_round_trip(self) -> None:
        _, live = _plan_fixture()
        fetch, summarise = live
        data = fetch.to_dict()
        _assert_json_native(data)
        assert data["cost"] is None, "callable cost is not portable"
        assert data["duration"] == 300.0
        assert data["qos"]["max_attempts"] == 3
        restored = ActionSpec.from_dict(json.loads(json.dumps(data)))
        assert restored.name == "fetch"
        assert restored.cost == 1.0
        assert restored.execute is None
        assert restored.duration == timedelta(minutes=5)
        assert dict(restored.resources or {}) == {"tokens": 100.0}
        assert restored.qos is not None and restored.qos.max_attempts == 3

        data = summarise.to_dict()
        _assert_json_native(data)
        assert data["effects"] is None and data["effect_keys"] == ["summary"]
        assert data["require_human_approval"] is True
        restored = ActionSpec.from_dict(data)
        assert restored.has_dynamic_effects
        assert restored.effect_key_set() == frozenset({"summary"})
        assert restored.max_retries == 2 and restored.can_rerun is False
        assert restored.read_only is True and restored.human_input_key == "review"
        assert dict(restored.metadata or {}) == {"owner": "docs"}

    def test_plan_round_trips_with_csp_metadata(self) -> None:
        plan, live = _plan_fixture()
        data = plan.to_dict()
        _assert_json_native(data)
        wire = json.loads(json.dumps(data))

        restored = Plan.from_dict(wire)
        assert restored.action_names == ["fetch", "summarise"]
        assert restored.total_cost == 5.0
        assert [s.to_dict() for s in restored.expected_states] == [
            {"fetched": True},
            {"fetched": True, "summary": True},
        ]
        assert restored.score == HardSoftScore(hard=-1.0, soft=-2.5)
        meta = restored.metadata
        assert meta.nodes_explored == 7 and meta.planning_time_ms == 0.25
        csp = meta.csp
        assert csp is not None
        assert csp.status is CSPStatus.INFEASIBLE
        assert csp.makespan == timedelta(minutes=5)
        assert csp.schedule[0].end == timedelta(minutes=5)
        assert csp.resource_usage[0].satisfied is False
        assert dict(csp.objective_values) == {"tokens": 100.0}
        assert isinstance(csp.explanation, InfeasibilityExplanation)
        assert csp.explanation.conflicting_constraints[0].key == "tokens"
        assert csp.explanation.resource_shortfalls[0].overrun == 50.0

    def test_plan_from_dict_rebinds_live_actions_by_name(self) -> None:
        plan, live = _plan_fixture()
        wire = json.loads(json.dumps(plan.to_dict()))
        rebound = Plan.from_dict(wire, actions=live)
        assert rebound.actions[0] is live[0]
        assert rebound.actions[1] is live[1]
        assert rebound.actions[0].execute is not None
        assert rebound.actions[1].require_human_approval is _Form
        # Mapping form and partial live lists are accepted too.
        partial = Plan.from_dict(wire, actions={"fetch": live[0]})
        assert partial.actions[0] is live[0]
        assert partial.actions[1] is not live[1]
        assert partial.actions[1].name == "summarise"

    def test_score_codecs(self) -> None:
        for score in (
            SimpleScore(scalar=4.0),
            HardSoftScore(hard=0.0, soft=-3.0),
            BendableScore(hard_levels=(0.0, -1.0), soft_levels=(-2.0,)),
        ):
            data = score.to_dict()
            _assert_json_native(data)
            assert score_from_dict(json.loads(json.dumps(data))) == score
        assert score_from_dict({}) == SimpleScore(scalar=0.0)
        with pytest.raises(ValueError):
            score_from_dict({"type": "mystery"})

    def test_action_result_round_trips(self) -> None:
        result = ActionResult(
            action_name="fetch",
            success=False,
            state_before={"a": 1},
            state_after={"a": 2},
            error="boom",
        )
        data = result.to_dict()
        _assert_json_native(data)
        assert ActionResult.from_dict(json.loads(json.dumps(data))) == result


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


class TestStateHelpers:
    def test_successful_action_names_accepts_objects_and_dicts(self) -> None:
        typed = {
            "execution_history": [
                ActionResult(action_name="a", success=True),
                ActionResult(action_name="b", success=False),
            ]
        }
        raw = {"execution_history": history_to_state(typed["execution_history"])}
        assert successful_action_names(typed) == ["a"]
        assert successful_action_names(raw) == ["a"]
        assert (
            history_from_state(raw["execution_history"]) == typed["execution_history"]
        )

    def test_rehydrate_and_serialize_are_inverse(self) -> None:
        plan, live = _plan_fixture()
        goal = GoalSpec(conditions={"summary": True})
        typed = {
            "goal": goal,
            "plan": plan,
            "execution_history": [ActionResult(action_name="fetch", success=True)],
            "status": "executing",
        }
        raw = serialize_updates(typed)
        _assert_json_native(raw)
        back = rehydrate_state(raw, live)
        assert back["goal"] == goal
        assert isinstance(back["plan"], Plan)
        assert back["plan"].actions[0] is live[0]
        assert back["execution_history"] == typed["execution_history"]
        assert back["status"] == "executing"

    def test_empty_channel_values_rehydrate_to_none(self) -> None:
        assert goal_from_state({}) is None
        assert goal_from_state(None) is None
        assert plan_from_state({}) is None
        assert plan_from_state(None) is None
        assert plan_to_state(None) is None
        assert goal_to_state(None) is None

    def test_typed_values_pass_through(self) -> None:
        goal = GoalSpec(conditions={"x": True})
        assert goal_from_state(goal) is goal
        plan = Plan.empty()
        assert plan_from_state(plan) is plan
        assert rehydrate_state({"goal": goal, "plan": plan})["plan"] is plan

    def test_live_registry_keeps_callables_in_process(self) -> None:
        metric = MinimizeFinalStateExpression(
            name="penalty", expression=lambda s: float(s.get("penalty", 0.0))
        )
        goal = GoalSpec(conditions={"done": True}, metrics=(metric,))
        data = goal_to_state(goal)
        assert data is not None
        _assert_json_native(data)
        assert goal_from_state(json.loads(json.dumps(data))) is goal

    def test_registry_miss_degrades_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        goal = GoalSpec(conditions={"done": True}, value=lambda ws: 9.0)
        data = goal_to_state(goal)
        assert data is not None
        data["_live"] = "0" * 32  # token from another process
        with caplog.at_level(logging.WARNING, logger="langgoap.graph.state"):
            restored = goal_from_state(data)
        assert isinstance(restored, GoalSpec)
        assert restored.value == 1.0
        assert "could not be restored" in caplog.text

    def test_plain_goal_has_no_registry_token(self) -> None:
        data = goal_to_state(GoalSpec(conditions={"done": True}))
        assert data is not None and "_live" not in data


# ---------------------------------------------------------------------------
# Graph edges: typed in, JSON inside, typed out
# ---------------------------------------------------------------------------


def _linear_actions(calls: dict[str, int] | None = None) -> list[ActionSpec]:
    calls = calls if calls is not None else {}

    def _exec(name: str, effects: dict[str, Any]):
        def run(ws: dict[str, Any]) -> dict[str, Any]:
            calls[name] = calls.get(name, 0) + 1
            return effects

        return run

    return [
        ActionSpec(
            name="gather",
            effects={"data": True},
            execute=_exec("gather", {"data": True}),
        ),
        ActionSpec(
            name="process",
            preconditions={"data": True},
            effects={"done": True},
            execute=_exec("process", {"done": True}),
        ),
    ]


class TestGraphEdges:
    def test_goap_graph_invoke_returns_typed_objects(self) -> None:
        graph = GoapGraph(_linear_actions())
        result = graph.invoke(goal=GoalSpec(conditions={"done": True}), world_state={})
        assert result["status"] == "goal_achieved"
        assert isinstance(result["goal"], GoalSpec)
        assert isinstance(result["plan"], Plan)
        assert result["plan"].action_names == ["gather", "process"]
        assert all(isinstance(h, ActionResult) for h in result["execution_history"])
        assert successful_action_names(result) == ["gather", "process"]

    def test_compiled_graph_exposes_json_state_and_accepts_goal_objects(self) -> None:
        graph = GoapGraph(_linear_actions())
        compiled = graph.compile()
        result = compiled.invoke(
            {"goal": GoalSpec(conditions={"done": True}), "world_state": {}}
        )
        assert result["status"] == "goal_achieved"
        _assert_json_native(result)
        json.dumps(result)
        assert result["plan"]["actions"][0]["name"] == "gather"
        assert successful_action_names(result) == ["gather", "process"]
        typed = graph.rehydrate(result)
        assert isinstance(typed["plan"], Plan)
        assert typed["plan"].actions[0].execute is not None
        assert isinstance(typed["goal"], GoalSpec)

    def test_initial_state_is_json_native(self) -> None:
        graph = GoapGraph(_linear_actions())
        state = graph.initial_state(GoalSpec(conditions={"done": True}), {"seed": 1})
        _assert_json_native(state)
        assert state["world_state"] == {"seed": 1}
        assert graph.initial_state({"conditions": {"done": True}})["goal"] == {
            "conditions": {"done": True}
        }


@pytest.fixture
def strict_msgpack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make LangGraph's stock serializer refuse unregistered types."""
    import langgraph.checkpoint.serde._msgpack as lg_msgpack

    monkeypatch.setattr(lg_msgpack, "STRICT_MSGPACK_ENABLED", True)


class TestStockCheckpointerStrict:
    """A GOAP run persists through MemorySaver with the stock serializer."""

    def test_plan_execute_replan_through_checkpointer(
        self, strict_msgpack: None
    ) -> None:
        calls: dict[str, int] = {}

        def flaky(ws: dict[str, Any]) -> dict[str, Any]:
            calls["flaky"] = calls.get("flaky", 0) + 1
            if calls["flaky"] == 1:
                raise RuntimeError("transient")
            return {"data": True}

        actions = [
            ActionSpec(
                name="flaky", effects={"data": True}, execute=flaky, max_retries=1
            ),
            ActionSpec(
                name="process", preconditions={"data": True}, effects={"done": True}
            ),
        ]
        graph = GoapGraph(actions)
        saver = MemorySaver()
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "t-replan"}}
        result = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert result["status"] == "goal_achieved"
        assert result["replan_count"] >= 1
        _assert_json_native(result)
        values = compiled.get_state(config).values
        _assert_json_native(values)
        json.dumps(values)
        assert successful_action_names(values) == ["flaky", "process"]
        typed = graph.rehydrate(values)
        assert isinstance(typed["plan"], Plan)
        assert typed["plan"].actions[0].execute is flaky

    def test_interrupt_before_executor_and_resume(self, strict_msgpack: None) -> None:
        graph = GoapGraph(_linear_actions())
        compiled = graph.compile(
            checkpointer=MemorySaver(), interrupt_before=["executor"]
        )
        config = {"configurable": {"thread_id": "t-resume"}}
        round1 = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert round1["plan"]["actions"][0]["name"] == "gather"
        assert compiled.get_state(config).next == ("executor",)
        round2 = compiled.invoke(None, config=config)
        assert round2["world_state"]["data"] is True and round2["current_step"] == 1
        round3 = compiled.invoke(None, config=config)
        assert round3["status"] == "goal_achieved"
        assert compiled.get_state(config).next == ()

    def test_human_approval_interrupt_and_resume(self, strict_msgpack: None) -> None:
        actions = [
            ActionSpec(name="gather", effects={"data": True}),
            ActionSpec(
                name="deploy",
                preconditions={"data": True},
                effects={"done": True},
                require_human_approval=True,
            ),
        ]
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-hitl"}}
        paused = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert "__interrupt__" in paused
        resumed = compiled.invoke(Command(resume=True), config=config)
        assert resumed["status"] == "goal_achieved"
        assert successful_action_names(resumed) == ["gather", "deploy"]

    def test_typed_form_approval_through_checkpointer(
        self, strict_msgpack: None
    ) -> None:
        actions = [
            ActionSpec(
                name="review",
                effects={"done": True},
                require_human_approval=_Form,
                human_input_key="review",
            ),
        ]
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-form"}}
        paused = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert "__interrupt__" in paused
        resumed = compiled.invoke(
            Command(resume={"reason": "looks good"}), config=config
        )
        assert resumed["status"] == "goal_achieved"
        assert resumed["world_state"]["review"] == {"reason": "looks good"}

    def test_multi_goal_sequential_through_checkpointer(
        self, strict_msgpack: None
    ) -> None:
        actions = [
            ActionSpec(name="a", effects={"a": True}),
            ActionSpec(name="b", preconditions={"a": True}, effects={"b": True}),
        ]
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-multi"}}
        multi = MultiGoal(
            goals=(GoalSpec(conditions={"a": True}), GoalSpec(conditions={"b": True}))
        )
        result = compiled.invoke(graph.initial_state(multi), config=config)
        assert result["status"] == "goal_achieved"
        assert successful_action_names(result) == ["a", "b"]
        assert graph.rehydrate(result)["goal"] == multi

    def test_metrics_survive_checkpoint_in_process(self, strict_msgpack: None) -> None:
        actions = [ActionSpec(name="a", effects={"done": True, "penalty": 3.0})]
        metric = MinimizeFinalStateExpression(
            name="penalty", expression=lambda s: float(s.get("penalty", 0.0))
        )
        goal = GoalSpec(conditions={"done": True}, metrics=(metric,))
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-metrics"}}
        result = compiled.invoke(graph.initial_state(goal), config=config)
        assert result["status"] == "goal_achieved"
        typed = graph.rehydrate(result)
        assert typed["goal"] is goal
        assert typed["plan"].score.soft == pytest.approx(-3.0)

    @pytest.mark.asyncio
    async def test_async_loop_through_checkpointer(self, strict_msgpack: None) -> None:
        async def agather(ws: dict[str, Any]) -> dict[str, Any]:
            return {"data": True}

        actions = [
            ActionSpec(name="gather", effects={"data": True}, aexecute=agather),
            ActionSpec(
                name="process", preconditions={"data": True}, effects={"done": True}
            ),
        ]
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-async"}}
        result = await compiled.ainvoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert result["status"] == "goal_achieved"
        _assert_json_native((await compiled.aget_state(config)).values)


class TestDeprecatedSerdeShims:
    def test_install_langgoap_serde_is_a_warning_no_op(self) -> None:
        from langgoap.serde import (
            LANGGOAP_ALLOWED_MSGPACK_TYPES,
            install_langgoap_serde,
        )

        saver = MemorySaver()
        stock = saver.serde
        with pytest.warns(DeprecationWarning):
            assert install_langgoap_serde(saver) is saver
        assert saver.serde is stock
        assert LANGGOAP_ALLOWED_MSGPACK_TYPES == ()

    def test_allowed_types_constant_is_a_silent_empty_tuple(self) -> None:
        """A module constant cannot warn on access; the CHANGELOG says so too."""
        import warnings

        import langgoap

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert langgoap.LANGGOAP_ALLOWED_MSGPACK_TYPES == ()
        changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(
            encoding="utf-8"
        )
        assert "kept for import compatibility" in " ".join(changelog.split())

    def test_langgoap_serializer_is_a_warning_shim(self) -> None:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        from langgoap.serde import LangGOAPSerializer

        with pytest.warns(DeprecationWarning):
            serde = LangGOAPSerializer()
        assert isinstance(serde, JsonPlusSerializer)
        kind, payload = serde.dumps_typed({"a": 1})
        assert serde.loads_typed((kind, payload)) == {"a": 1}


class TestNirvanaGoalCodec:
    def test_nirvana_goal_keeps_its_type_through_the_state(self) -> None:
        from langgoap.planner.utility import NirvanaGoal, is_nirvana

        goal = NirvanaGoal(policy=GoalPolicy(max_replans=3))
        data = goal_to_state(goal)
        assert data is not None and data["kind"] == "nirvana"
        _assert_json_native(data)
        restored = goal_from_state(json.loads(json.dumps(data)))
        assert is_nirvana(restored)
        assert restored.policy.max_replans == 3
        assert dict(restored.conditions) == {}

    def test_nirvana_goal_repr_names_the_type(self) -> None:
        from langgoap.planner.utility import NirvanaGoal

        assert repr(NirvanaGoal()) == "NirvanaGoal()"
        assert repr(NirvanaGoal(policy=GoalPolicy(max_replans=3))).startswith(
            "NirvanaGoal(policy="
        )
