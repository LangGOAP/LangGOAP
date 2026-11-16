"""``ainvoke`` produces the same result as ``invoke`` for every control path.

The audit of the v0.2.0 release found termination policies, the three
``MultiGoal`` modes (and the strict sequential default, #35), stuck
recovery, failure counts, handled tool errors (#37), the parallel
executor's wave merge (#16) and the blacklist fallback (#15) exercised on
one path only.  Each test here runs the same scenario on both paths and
compares everything but timing.  The last classes pin three documented
behaviours the audit found unasserted: the goal-object-with-checkpointer
error (#17), the compile-option ``TypeError`` wording (#19) and the report
formatter's key cap (#57).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from langgoap import (
    ActionSpec,
    FunctionalStuckHandler,
    GoalPolicy,
    GoalSpec,
    GoapGraph,
    MaxActionsPolicy,
    MaxCostPolicy,
    MultiGoal,
    OnStuckPolicy,
    StuckHandlerResult,
    StuckHandlingResultCode,
)
from langgoap.graph.state import ActionResult
from langgoap.integrations import (
    GoapSubgraph,
    add_goap_subgraph,
    create_goap_agent,
    goapify_tool,
)
from langgoap.integrations.deepagents import format_goap_result
from langgoap.planner.types import Plan

# --------------------------------------------------------------------------- helpers


def _comparable(result: dict[str, Any]) -> dict[str, Any]:
    """Everything a caller can observe, minus timing."""
    history = [
        (
            (h.action_name, h.success, h.error)
            if isinstance(h, ActionResult)
            else (h["action_name"], h["success"], h["error"])
        )
        for h in result.get("execution_history", [])
    ]
    plan = result.get("plan")
    plan_names = (
        plan.action_names
        if isinstance(plan, Plan)
        else [a["name"] for a in plan["actions"]] if plan else None
    )
    goal = result.get("goal")
    goal_dict = goal.to_dict() if hasattr(goal, "to_dict") else goal
    return {
        "status": result.get("status"),
        "history": history,
        "world_state": result.get("world_state"),
        "plan": plan_names,
        "replan_count": result.get("replan_count"),
        "replan_reason": result.get("replan_reason"),
        "blacklisted_actions": result.get("blacklisted_actions"),
        "action_failure_counts": result.get("action_failure_counts"),
        "skipped_subgoals": result.get("skipped_subgoals"),
        "current_subgoal_index": result.get("current_subgoal_index"),
        "no_plan_explanation": result.get("no_plan_explanation"),
        "goal": goal_dict,
    }


def _both(
    build: Callable[[], GoapGraph], goal: Any, world_state: dict[str, Any]
) -> None:
    """Run ``build()`` on both paths and assert identical observable results."""
    sync = build().invoke(goal=goal, world_state=dict(world_state))
    coro = build().ainvoke(goal=goal, world_state=dict(world_state))
    async_ = asyncio.run(coro)
    assert _comparable(async_) == _comparable(sync)
    return None


def _chain(n: int = 3, usd_each: float = 0.0) -> list[ActionSpec]:
    actions: list[ActionSpec] = []
    prev: dict[str, Any] = {}
    for i in range(n):
        key = f"s{i}"

        def execute(ws: dict[str, Any], key: str = key) -> dict[str, Any]:
            out: dict[str, Any] = {key: True}
            if usd_each:
                out["total_cost_usd"] = ws.get("total_cost_usd", 0.0) + usd_each
            return out

        actions.append(
            ActionSpec(
                name=key, preconditions=prev, effects={key: True}, execute=execute
            )
        )
        prev = {key: True}
    return actions


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


# --------------------------------------------------------------------------- termination


class TestTerminationPolicies:
    def test_max_actions(self) -> None:
        _both(
            lambda: GoapGraph(
                _chain(3), termination_policies=[MaxActionsPolicy(max_actions=2)]
            ),
            GoalSpec(conditions={"s2": True}),
            {},
        )
        result = GoapGraph(
            _chain(3), termination_policies=[MaxActionsPolicy(max_actions=2)]
        ).invoke(goal=GoalSpec(conditions={"s2": True}), world_state={})
        assert result["status"] == "terminated"
        assert len(result["execution_history"]) == 2

    def test_max_cost(self) -> None:
        build = lambda: GoapGraph(  # noqa: E731
            _chain(3, usd_each=1.0), termination_policies=[MaxCostPolicy(usd=1.5)]
        )
        _both(build, GoalSpec(conditions={"s2": True}), {})
        result = build().invoke(goal=GoalSpec(conditions={"s2": True}), world_state={})
        assert result["status"] == "terminated"
        assert result["world_state"]["total_cost_usd"] == 2.0

    def test_on_stuck_turns_no_plan_into_terminated(self) -> None:
        build = lambda: GoapGraph(
            _chain(1), termination_policies=[OnStuckPolicy()]
        )  # noqa: E731
        _both(build, GoalSpec(conditions={"never": True}), {})
        assert (
            build().invoke(goal=GoalSpec(conditions={"never": True}), world_state={})[
                "status"
            ]
            == "terminated"
        )


# --------------------------------------------------------------------------- multi-goal


def _two_goal_actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="cheap", effects={"a": True}, cost=1.0, execute=lambda ws: {"a": True}
        ),
        ActionSpec(
            name="dear", effects={"b": True}, cost=9.0, execute=lambda ws: {"b": True}
        ),
    ]


class TestMultiGoalModes:
    def test_any_mode(self) -> None:
        goal = MultiGoal(
            goals=(GoalSpec(conditions={"b": True}), GoalSpec(conditions={"a": True})),
            mode="any",
        )
        _both(lambda: GoapGraph(_two_goal_actions()), goal, {})
        result = GoapGraph(_two_goal_actions()).invoke(goal=goal, world_state={})
        assert [h.action_name for h in result["execution_history"]] == ["cheap"]

    def test_best_value_mode(self) -> None:
        goal = MultiGoal(
            goals=(
                GoalSpec(conditions={"a": True}, value=1.0),
                GoalSpec(conditions={"b": True}, value=50.0),
            ),
            mode="best_value",
        )
        _both(lambda: GoapGraph(_two_goal_actions()), goal, {})
        result = GoapGraph(_two_goal_actions()).invoke(goal=goal, world_state={})
        assert [h.action_name for h in result["execution_history"]] == ["dear"]

    def test_sequential_is_strict_by_default(self) -> None:
        goal = MultiGoal(
            goals=(
                GoalSpec(conditions={"never": True}),
                GoalSpec(conditions={"a": True}),
            )
        )
        _both(lambda: GoapGraph(_two_goal_actions()), goal, {})
        result = asyncio.run(
            GoapGraph(_two_goal_actions()).ainvoke(goal=goal, world_state={})
        )
        assert result["status"] == "no_plan"
        assert result["execution_history"] == []

    def test_sequential_skip(self) -> None:
        goal = MultiGoal(
            goals=(
                GoalSpec(conditions={"never": True}),
                GoalSpec(conditions={"a": True}),
            ),
            on_infeasible="skip",
        )
        _both(lambda: GoapGraph(_two_goal_actions()), goal, {})
        result = asyncio.run(
            GoapGraph(_two_goal_actions()).ainvoke(goal=goal, world_state={})
        )
        assert result["status"] == "partial"
        assert result["skipped_subgoals"] == [0]
        assert [h.action_name for h in result["execution_history"]] == ["cheap"]


# --------------------------------------------------------------------------- stuck recovery


class TestStuckRecovery:
    def test_relaxed_goal_is_applied_on_both_paths(self) -> None:
        def relax(state: Any, reason: Any) -> StuckHandlerResult:
            return StuckHandlerResult(
                code=StuckHandlingResultCode.REPLAN,
                message="settle for a",
                new_goal=GoalSpec(conditions={"a": True}),
            )

        build = lambda: GoapGraph(  # noqa: E731
            _two_goal_actions(), stuck_handlers=[FunctionalStuckHandler("relax", relax)]
        )
        _both(build, GoalSpec(conditions={"never": True}), {})
        result = asyncio.run(
            build().ainvoke(goal=GoalSpec(conditions={"never": True}), world_state={})
        )
        assert result["status"] == "goal_achieved"
        assert dict(result["goal"].conditions) == {"a": True}


# --------------------------------------------------------------------------- failure counts


def _flaky_then_gated(calls: dict[str, int]) -> list[ActionSpec]:
    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        calls["A"] += 1
        if calls["A"] == 1:
            raise RuntimeError("transient")
        return {"a": True}

    return [
        ActionSpec(name="A", effects={"a": True}, execute=flaky, max_retries=1),
        ActionSpec(
            name="B",
            preconditions={"a": True},
            effects={"done": True},
            execute=lambda ws: {"done": True},
            require_human_approval=True,
        ),
    ]


class TestFailureCountsUnderAinvoke:
    def _run_sync(self) -> dict[str, Any]:
        calls = {"A": 0}
        graph = GoapGraph(_flaky_then_gated(calls))
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config()
        compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        return compiled.invoke(Command(resume=False), config=config)

    async def _run_async(self) -> dict[str, Any]:
        calls = {"A": 0}
        graph = GoapGraph(_flaky_then_gated(calls))
        compiled = graph.compile(checkpointer=MemorySaver())
        config = _config()
        await compiled.ainvoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        return await compiled.ainvoke(Command(resume=False), config=config)

    @pytest.mark.asyncio
    async def test_retried_failure_and_denial_counts_match(self) -> None:
        sync = self._run_sync()
        async_ = await self._run_async()
        assert _comparable(async_) == _comparable(sync)
        assert async_["status"] == "no_plan"
        assert async_["action_failure_counts"]["A"] == 1
        assert async_["action_failure_counts"]["B"] >= 1
        assert async_["blacklisted_actions"] == ["B"]


# --------------------------------------------------------------------------- #37 handled tool errors


def _boom(x: int) -> str:
    raise ToolException("kaboom")


class TestHandledToolErrorsOnBothPaths:
    def test_handled_tool_error_is_a_failure_under_ainvoke_too(self) -> None:
        def build() -> GoapGraph:
            tool = StructuredTool.from_function(
                func=_boom,
                name="boom",
                description="always fails",
                handle_tool_error=True,
            )
            return GoapGraph(
                [goapify_tool(tool, effects={"done": True}, result_key="out")]
            )

        _both(build, GoalSpec(conditions={"done": True}), {"x": 1})
        result = asyncio.run(
            build().ainvoke(
                goal=GoalSpec(conditions={"done": True}), world_state={"x": 1}
            )
        )
        assert result["status"] != "goal_achieved"
        failures = [h for h in result["execution_history"] if not h.success]
        assert failures and "kaboom" in (failures[0].error or "")
        assert "out" not in result["world_state"]


# --------------------------------------------------------------------------- #16 parallel wave merge


class TestParallelWaveMergeOnBothPaths:
    @staticmethod
    def _independent() -> list[ActionSpec]:
        return [
            ActionSpec(
                name="left",
                effects={"x": True},
                execute=lambda ws: {"x": True, "left_ran": 1},
            ),
            ActionSpec(
                name="right",
                effects={"y": True},
                execute=lambda ws: {"y": True, "right_ran": 1},
            ),
            ActionSpec(
                name="join",
                preconditions={"x": True, "y": True},
                effects={"done": True},
                execute=lambda ws: {"done": True},
            ),
        ]

    def test_each_actions_delta_is_merged(self) -> None:
        build = lambda: GoapGraph(
            self._independent(), executor="parallel"
        )  # noqa: E731
        _both(build, GoalSpec(conditions={"done": True}), {"untouched": "keep"})
        result = build().invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={"untouched": "keep"}
        )
        assert result["world_state"] == {
            "untouched": "keep",
            "x": True,
            "left_ran": 1,
            "y": True,
            "right_ran": 1,
            "done": True,
        }
        assert {h.action_name for h in result["execution_history"][:2]} == {
            "left",
            "right",
        }


# --------------------------------------------------------------------------- #15 blacklist fallback


class TestBlacklistFallbackOnBothPaths:
    def test_opt_in_fallback_readmits_the_only_route(self) -> None:
        def build_with(calls: dict[str, int]) -> GoapGraph:
            def flaky(ws: dict[str, Any]) -> dict[str, Any]:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("once")
                return {"done": True}

            return GoapGraph(
                [ActionSpec(name="only", effects={"done": True}, execute=flaky)]
            )

        goal = GoalSpec(
            conditions={"done": True}, policy=GoalPolicy(blacklist_fallback=True)
        )
        sync = build_with({"n": 0}).invoke(goal=goal, world_state={})
        async_ = asyncio.run(build_with({"n": 0}).ainvoke(goal=goal, world_state={}))
        assert _comparable(async_) == _comparable(sync)
        assert sync["status"] == "goal_achieved"
        assert [(h.action_name, h.success) for h in sync["execution_history"]] == [
            ("only", False),
            ("only", True),
        ]
        strict = build_with({"n": 0}).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        assert strict["status"] == "no_plan"
        assert "blacklist_fallback" in strict["no_plan_explanation"]["suggestion"]


# --------------------------------------------------------------------------- #17 goal object + checkpointer


class TestGoalObjectWithACheckpointer:
    def test_is_rejected_by_the_serializer_and_initial_state_is_the_way(self) -> None:
        graph = GoapGraph(_chain(1))
        compiled = graph.compile(checkpointer=MemorySaver())
        with pytest.raises(TypeError, match="GoalSpec"):
            compiled.invoke(
                {"goal": GoalSpec(conditions={"s0": True}), "world_state": {}},
                config=_config(),
            )
        ok = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"s0": True})), config=_config()
        )
        assert ok["status"] == "goal_achieved"

    @pytest.mark.asyncio
    async def test_async_path_rejects_it_the_same_way(self) -> None:
        graph = GoapGraph(_chain(1))
        compiled = graph.compile(checkpointer=MemorySaver())
        with pytest.raises(TypeError, match="GoalSpec"):
            await compiled.ainvoke(
                {"goal": GoalSpec(conditions={"s0": True}), "world_state": {}},
                config=_config(),
            )


# --------------------------------------------------------------------------- #19 compile-option errors


class _ParentState(TypedDict, total=False):
    world_state: dict[str, Any]
    plan_result: dict[str, Any]


class TestCompileOptionErrorsNameBothOptionSets:
    def test_create_goap_agent(self) -> None:
        tool = StructuredTool.from_function(
            func=lambda x: "ok", name="t", description="t"
        )
        with pytest.raises(TypeError) as exc:
            create_goap_agent(
                tools=[tool],
                goal=GoalSpec(conditions={"done": True}),
                effects={"t": {"done": True}},
                bogus=1,
            )
        message = str(exc.value)
        assert (
            "'bogus'" in message
            and "checkpointer" in message
            and "stuck_handlers" in message
        )

    def test_goap_subgraph_compile(self) -> None:
        sub = GoapSubgraph(_chain(1), GoalSpec(conditions={"s0": True}))
        with pytest.raises(TypeError, match="'tracer'"):
            sub.compile(tracer=object())

    def test_add_goap_subgraph(self) -> None:
        with pytest.raises(TypeError, match="'interrupt_befor'"):
            add_goap_subgraph(
                StateGraph(_ParentState),
                name="goap",
                actions=_chain(1),
                goal=GoalSpec(conditions={"s0": True}),
                compile_kwargs={"interrupt_befor": ["executor"]},
            )


# --------------------------------------------------------------------------- #57 report key cap


class TestReportKeyCap:
    def test_more_than_forty_keys_are_summarised(self) -> None:
        result = GoapGraph(_chain(1)).invoke(
            goal=GoalSpec(conditions={"s0": True}),
            world_state={f"k{i:02d}": i for i in range(45)},
        )
        report = format_goap_result(result)
        assert "… 6 more key(s)" in report  # 45 inputs + s0 = 46 keys, 40 shown
        assert "k39" in report and "k44" not in report
