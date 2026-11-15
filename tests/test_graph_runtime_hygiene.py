"""Graph-runtime hygiene pinned by issue #64.

Each class covers one finding: dead helpers are gone, a non-dict
``execute`` return is reported instead of silently dropped, QoS back-off
jitter draws from its own RNG so seeded stochastic runs stay
reproducible across retries, an observer built without action specs
honours the ``max_retries`` carried by the plan, an any-mode
``MultiGoal`` that cannot be planned explains itself, and a final
``no_plan`` is not misreported as ``max_replans_exceeded``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Any

import pytest
from langgraph.graph import END

from langgoap.goals import GoalPolicy, GoalSpec, MultiGoal
from langgoap.graph.nodes import (
    GoapExecutor,
    GoapObserver,
    GoapPlanner,
    ParallelGoapExecutor,
)
from langgoap.graph.state import ActionResult, GoapState
from langgoap.qos import ActionQos
from langgoap.stuck import StuckHandlerResult, StuckHandlingResultCode
from tests.conftest import make_action, make_plan


def _exec_state(action: Any, world: dict[str, Any] | None = None) -> GoapState:
    return {
        "world_state": dict(world or {}),
        "plan": make_plan(action),
        "current_step": 0,
    }


class TestDeadHelpersAreGone:
    def test_safe_tracer_helpers_removed(self) -> None:
        import langgoap.graph.nodes as nodes
        import langgoap.graph.nodes._helpers as helpers

        assert not hasattr(helpers, "_safe_tracer_call")
        assert not hasattr(helpers, "_safe_tracer_acall")
        assert not hasattr(nodes, "_safe_tracer_call")
        assert "_safe_tracer_call" not in nodes.__all__

    def test_sequential_subgoal_helper_removed(self) -> None:
        assert not hasattr(GoapPlanner, "_resolve_sequential_subgoal")


class TestNonDictExecuteReturn:
    def test_non_dict_return_is_reported_and_declared_effects_apply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        action = make_action("step", eff={"a": True}, execute=lambda ws: "ok")
        with caplog.at_level(logging.WARNING):
            result = GoapExecutor()(_exec_state(action))

        assert result["world_state"]["a"] is True
        assert result["execution_history"][0]["success"] is True
        messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("step" in m and "str" in m for m in messages), messages

    def test_none_return_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        action = make_action("step", eff={"a": True}, execute=lambda ws: None)
        with caplog.at_level(logging.WARNING):
            GoapExecutor()(_exec_state(action))

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    async def test_async_path_reports_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def execute(ws: dict[str, Any]) -> Any:
            return 42

        action = make_action("step", eff={"a": True}, execute=execute)
        with caplog.at_level(logging.WARNING):
            result = await GoapExecutor().acall(_exec_state(action))

        assert result["world_state"]["a"] is True
        messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("step" in m and "int" in m for m in messages), messages


class _Noisy:
    """A transition model whose sampled effect reveals the RNG stream."""

    def expected(self, state: Any, action: Any) -> dict[str, Any]:
        return dict(action.get_effects(dict(state)))

    def sample(self, state: Any, action: Any, rng: random.Random) -> dict[str, Any]:
        return {"a": rng.random()}


def _fail_once() -> Any:
    calls = {"n": 0}

    def execute(ws: dict[str, Any]) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient")
        return None

    return execute


_QOS = ActionQos(
    max_attempts=2,
    backoff_initial_ms=1.0,
    backoff_max_ms=1.0,
    jitter=0.5,
    idempotent=True,
)


class TestSeparateRngStreams:
    """A retry's jittered back-off must not consume the sampling RNG."""

    @staticmethod
    def _sampled(executor_cls: Any, action: Any) -> float:
        executor = executor_cls(transition_model=_Noisy(), rng=random.Random(0))
        return executor(_exec_state(action))["world_state"]["a"]

    def test_sequential_executor(self) -> None:
        steady = replace(make_action("step", eff={"a": 0.0}), qos=_QOS)
        flaky = replace(
            make_action("step", eff={"a": 0.0}, execute=_fail_once()), qos=_QOS
        )
        assert self._sampled(GoapExecutor, flaky) == self._sampled(GoapExecutor, steady)

    def test_parallel_executor(self) -> None:
        steady = replace(make_action("step", eff={"a": 0.0}), qos=_QOS)
        flaky = replace(
            make_action("step", eff={"a": 0.0}, execute=_fail_once()), qos=_QOS
        )
        assert self._sampled(ParallelGoapExecutor, flaky) == self._sampled(
            ParallelGoapExecutor, steady
        )

    async def test_async_executor(self) -> None:
        steady = replace(make_action("step", eff={"a": 0.0}), qos=_QOS)
        flaky = replace(
            make_action("step", eff={"a": 0.0}, execute=_fail_once()), qos=_QOS
        )

        async def sampled(action: Any) -> float:
            executor = GoapExecutor(transition_model=_Noisy(), rng=random.Random(0))
            return (await executor.acall(_exec_state(action)))["world_state"]["a"]

        assert await sampled(flaky) == await sampled(steady)


class TestObserverWithoutActionSpecs:
    def test_max_retries_comes_from_the_plan_when_no_actions_given(self) -> None:
        flaky = make_action("flaky", eff={"x": True}, max_retries=2)
        observer = GoapObserver()  # manual graph: no action list
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"x": True}),
            "plan": make_plan(flaky),
            "current_step": 0,
            "status": "action_failed",
            "execution_history": [
                ActionResult(action_name="flaky", success=False, error="err"),
            ],
        }
        cmd = observer(state)

        assert cmd.goto == "planner"
        assert cmd.update["action_failure_counts"] == {"flaky": 1}
        assert cmd.update["blacklisted_actions"] == []


class TestAnyModeNoPlanExplanation:
    _actions = [make_action("noop", eff={"z": True})]
    _goal = MultiGoal(
        goals=(GoalSpec(conditions={"a": True}), GoalSpec(conditions={"b": True})),
        mode="any",
    )

    def test_no_plan_carries_a_per_subgoal_explanation(self) -> None:
        updates = GoapPlanner(self._actions)({"world_state": {}, "goal": self._goal})

        assert updates["status"] == "no_plan"
        explanation = updates["no_plan_explanation"]
        assert explanation is not None
        assert explanation["mode"] == "any"
        assert [s["unreachable_conditions"] for s in explanation["subgoals"]] == [
            ["a"],
            ["b"],
        ]
        assert "sub-goal" in explanation["suggestion"]

    def test_stuck_handler_receives_the_reason(self) -> None:
        seen: list[Any] = []

        class Recording:
            def handle_stuck(self, state: Any, reason: Any) -> StuckHandlerResult:
                seen.append(reason)
                return StuckHandlerResult(
                    code=StuckHandlingResultCode.NO_RESOLUTION,
                    message="noted",
                    handler_name="recording",
                )

        planner = GoapPlanner(self._actions, stuck_handlers=[Recording()])
        planner({"world_state": {}, "goal": self._goal})

        assert seen and seen[0] is not None
        assert seen[0]["mode"] == "any"


class TestNoPlanIsNotMisreported:
    def test_final_no_plan_at_the_replan_limit_stays_no_plan(self) -> None:
        state: GoapState = {
            "world_state": {},
            "goal": GoalSpec(conditions={"a": True}, policy=GoalPolicy(max_replans=3)),
            "plan": None,
            "current_step": 0,
            "status": "no_plan",
            "replan_count": 3,
        }
        cmd = GoapObserver()(state)

        assert cmd.goto == END
        assert cmd.update["status"] == "no_plan"


class TestSpentActionsAreExplainedAsSpent:
    """A ``can_rerun=False`` action that already ran is not a transient failure."""

    @staticmethod
    def _state(**extra: Any) -> dict[str, Any]:
        return {
            "world_state": {},
            "goal": GoalSpec(conditions={"bought": True}),
            "blacklisted_actions": ["buy"],
            **extra,
        }

    def test_spent_action_gets_its_own_suggestion(self) -> None:
        buy = replace(make_action("buy", eff={"bought": True}), can_rerun=False)
        updates = GoapPlanner([buy])(self._state(action_failure_counts={}))

        assert updates["status"] == "no_plan"
        explanation = updates["no_plan_explanation"]
        assert explanation["spent_actions"] == ["buy"]
        assert "already ran" in explanation["suggestion"]
        assert "can_rerun" in explanation["suggestion"]
        assert "transient" not in explanation["suggestion"]

    def test_failed_action_keeps_the_retry_suggestion(self) -> None:
        buy = make_action("buy", eff={"bought": True})
        updates = GoapPlanner([buy])(self._state(action_failure_counts={"buy": 1}))

        explanation = updates["no_plan_explanation"]
        assert "spent_actions" not in explanation
        assert "max_retries" in explanation["suggestion"]
