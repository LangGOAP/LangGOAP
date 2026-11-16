"""A run resumed by a new process picks up exactly where the old one stopped.

Every earlier resume test re-used the same compiled graph.  Here every
round builds a **new** ``GoapGraph`` and compiles it against the same
checkpointer and thread, which is what a crash and restart looks like, and
asserts the parts of the state the audit found unasserted: plan identity,
execution history, replan count, failure counts and blacklist, sub-goal
index and skipped sub-goals, the wall-clock anchor, the no-plan
explanation and reflections, and that a finished thread stays finished.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgoap import ActionSpec, GoalSpec, GoapGraph, MultiGoal
from langgoap.reflexion import ReflexionTracer


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _flaky_chain(calls: dict[str, int] | None = None) -> list[ActionSpec]:
    """A fails once (retried after a re-plan), then B, then C.

    ``calls`` is the external system's memory of how often A was tried.  A
    crashed-and-restarted process rebuilds its actions but not the world,
    so the tests share one counter across every rebuilt graph.
    """
    calls = {"a": 0} if calls is None else calls

    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        calls["a"] += 1
        if calls["a"] == 1:
            raise RuntimeError("transient")
        return {"a": True}

    return [
        ActionSpec(name="A", effects={"a": True}, execute=flaky, max_retries=1),
        ActionSpec(
            name="B",
            preconditions={"a": True},
            effects={"b": True},
            execute=lambda ws: {"b": True},
        ),
        ActionSpec(
            name="C",
            preconditions={"b": True},
            effects={"c": True},
            execute=lambda ws: {"c": True},
        ),
    ]


GOAL = GoalSpec(conditions={"c": True})


def _run_with_a_crash_before_every_step(
    build: Callable[[], GoapGraph], goal: Any, world_state: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compile with ``interrupt_before=["executor"]`` and, at every pause,
    throw the graph away and resume with a freshly built one."""
    saver = MemorySaver()
    config = _config()
    first = build()
    compiled = first.compile(checkpointer=saver, interrupt_before=["executor"])
    snapshots = [compiled.invoke(first.initial_state(goal, world_state), config=config)]
    del first, compiled
    for _ in range(50):
        fresh = build().compile(checkpointer=saver, interrupt_before=["executor"])
        if not fresh.get_state(config).next:
            break
        snapshots.append(fresh.invoke(None, config=config))
    final = build().compile(checkpointer=saver, interrupt_before=["executor"])
    assert final.get_state(config).next == ()
    return final.get_state(config).values, snapshots


class TestCrashBetweenSteps:
    def test_history_plan_and_counters_survive_every_crash(self) -> None:
        calls = {"a": 0}
        values, snapshots = _run_with_a_crash_before_every_step(
            lambda: GoapGraph(_flaky_chain(calls)), GOAL, {}
        )
        assert values["status"] == "goal_achieved"
        history = [
            (h["action_name"], h["success"]) for h in values["execution_history"]
        ]
        assert history == [("A", False), ("A", True), ("B", True), ("C", True)]
        assert values["replan_count"] == 1
        assert values["action_failure_counts"] == {"A": 1}
        assert values["blacklisted_actions"] == []
        assert values["replan_reason"] == "action_failed"
        assert values["world_state"] == {"a": True, "b": True, "c": True}

    def test_plan_identity_is_kept_across_a_crash(self) -> None:
        calls = {"a": 0}
        values, snapshots = _run_with_a_crash_before_every_step(
            lambda: GoapGraph(_flaky_chain(calls)), GOAL, {}
        )
        plans = [
            [a["name"] for a in s["plan"]["actions"]]
            for s in snapshots
            if s.get("plan")
        ]
        # The first plan and the re-plan are the same action sequence; a
        # resumed process executes the saved plan, it does not plan again.
        assert plans and all(p == ["A", "B", "C"] for p in plans)
        assert [s["current_step"] for s in snapshots] == [0, 0, 1, 2, 3]
        rehydrated = GoapGraph(_flaky_chain(calls)).rehydrate(values)
        assert rehydrated["plan"] is not None
        assert rehydrated["plan"].action_names == ["A", "B", "C"]

    def test_wall_clock_anchor_is_not_reset_by_a_new_process(self) -> None:
        calls = {"a": 0}
        values, snapshots = _run_with_a_crash_before_every_step(
            lambda: GoapGraph(_flaky_chain(calls)), GOAL, {}
        )
        anchors = {s["wall_clock_started_at"] for s in snapshots}
        assert len(anchors) == 1
        assert values["wall_clock_started_at"] == anchors.pop()

    def test_finished_thread_stays_finished(self) -> None:
        saver = MemorySaver()
        config = _config()
        graph = GoapGraph(_flaky_chain())
        done = graph.compile(checkpointer=saver).invoke(
            graph.initial_state(GOAL, {}), config=config
        )
        assert done["status"] == "goal_achieved"
        again = GoapGraph(_flaky_chain()).compile(checkpointer=saver)
        assert again.get_state(config).next == ()
        after = again.invoke(None, config=config)
        assert after["status"] == "goal_achieved"
        assert len(after["execution_history"]) == len(done["execution_history"])

    def test_no_plan_thread_keeps_its_explanation(self) -> None:
        saver = MemorySaver()
        config = _config()
        graph = GoapGraph(_flaky_chain())
        stuck = graph.compile(checkpointer=saver).invoke(
            graph.initial_state(GoalSpec(conditions={"never": True}), {}),
            config=config,
        )
        assert stuck["status"] == "no_plan"
        explanation = stuck["no_plan_explanation"]
        assert explanation and "never" in str(explanation)
        again = GoapGraph(_flaky_chain()).compile(checkpointer=saver)
        assert again.get_state(config).next == ()
        values = again.get_state(config).values
        assert values["no_plan_explanation"] == explanation
        typed = GoapGraph(_flaky_chain()).rehydrate(values)
        assert typed["status"] == "no_plan" and typed["plan"] is None

    @pytest.mark.asyncio
    async def test_async_resume_in_a_new_process(self) -> None:
        saver = MemorySaver()
        config = _config()
        calls = {"a": 0}
        first = GoapGraph(_flaky_chain(calls))
        compiled = first.compile(checkpointer=saver, interrupt_before=["executor"])
        await compiled.ainvoke(first.initial_state(GOAL, {}), config=config)
        anchor = (await compiled.aget_state(config)).values["wall_clock_started_at"]
        del first, compiled
        for _ in range(50):
            fresh = GoapGraph(_flaky_chain(calls)).compile(
                checkpointer=saver, interrupt_before=["executor"]
            )
            if not (await fresh.aget_state(config)).next:
                break
            await fresh.ainvoke(None, config=config)
        values = (
            await GoapGraph(_flaky_chain())
            .compile(checkpointer=saver)
            .aget_state(config)
        ).values
        assert values["status"] == "goal_achieved"
        assert [h["action_name"] for h in values["execution_history"]] == [
            "A",
            "A",
            "B",
            "C",
        ]
        assert values["replan_count"] == 1
        assert values["action_failure_counts"] == {"A": 1}
        assert values["wall_clock_started_at"] == anchor


class TestCrashWhileAHumanDecides:
    @staticmethod
    def _gated() -> list[ActionSpec]:
        return [
            ActionSpec(
                name="pay",
                effects={"paid": True},
                execute=lambda ws: {"paid": True},
                require_human_approval=True,
            ),
            ActionSpec(
                name="pay_by_invoice",
                effects={"paid": True},
                execute=lambda ws: {"paid": True},
                cost=5.0,
            ),
        ]

    def test_denial_after_a_crash_is_recorded_and_replanned_around(self) -> None:
        saver = MemorySaver()
        config = _config()
        goal = GoalSpec(conditions={"paid": True})
        first = GoapGraph(self._gated())
        paused = first.compile(checkpointer=saver).invoke(
            first.initial_state(goal, {}), config=config
        )
        assert paused["__interrupt__"][0].value["action"] == "pay"
        del first
        result = (
            GoapGraph(self._gated())
            .compile(checkpointer=saver)
            .invoke(
                Command(resume={"approved": False, "reason": "wrong card"}),
                config=config,
            )
        )
        assert result["status"] == "goal_achieved"
        assert result["blacklisted_actions"] == ["pay"]
        assert result["action_failure_counts"]["pay"] >= 1
        assert result["replan_count"] == 1
        assert result["replan_reason"] == "action_failed"
        names = [(h["action_name"], h["success"]) for h in result["execution_history"]]
        assert names == [("pay", False), ("pay_by_invoice", True)]
        assert "wrong card" in result["execution_history"][0]["error"]


class TestCrashInAMultiGoal:
    @staticmethod
    def _actions() -> list[ActionSpec]:
        return [
            ActionSpec(
                name="do_b", effects={"b": True}, execute=lambda ws: {"b": True}
            ),
        ]

    def test_skipped_subgoal_and_index_survive_a_crash(self) -> None:
        goal = MultiGoal(
            goals=(
                GoalSpec(conditions={"unreachable": True}),
                GoalSpec(conditions={"b": True}),
            ),
            mode="sequential",
            on_infeasible="skip",
        )
        values, snapshots = _run_with_a_crash_before_every_step(
            lambda: GoapGraph(self._actions()), goal, {}
        )
        assert snapshots[0]["skipped_subgoals"] == [0]
        assert snapshots[0]["current_subgoal_index"] == 1
        assert values["skipped_subgoals"] == [0]
        assert values["status"] == "partial"
        assert [h["action_name"] for h in values["execution_history"]] == ["do_b"]
        assert values["world_state"]["b"] is True


class TestReflectionsSurviveACrash:
    def test_reflection_context_is_kept_when_the_new_process_has_none(self) -> None:
        build = lambda: GoapGraph(
            _flaky_chain(), tracer=ReflexionTracer()
        )  # noqa: E731
        values, snapshots = _run_with_a_crash_before_every_step(build, GOAL, {})
        # The planner wrote the reflection on the re-plan after A failed...
        written = [
            s["reflection_context"] for s in snapshots if s.get("reflection_context")
        ]
        assert written, "the reflexion tracer produced no reflection on the failure"
        assert any("A" in line for line in written[0])
        # ...and a later process whose fresh tracer has no reflections did
        # not blank it out.
        assert values["reflection_context"] == written[0]
