"""Guard blocks, denials and invalid forms merge into ``action_failure_counts``
instead of replacing it (issue #33).

Those three paths set the failing action's count to ``max_retries + 1`` so
the observer blacklists it immediately.  They used to write a fresh
one-entry dict, wiping every other action's accumulated failures and
therefore their remaining retry budgets.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.graph.nodes import GoapExecutor
from langgoap.guards import FunctionalGuard, GuardResult, GuardSeverity
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


def _block(name: str) -> FunctionalGuard:
    return FunctionalGuard(
        "block",
        lambda action, ws: GuardResult(
            passed=action.name != name, message="blocked", severity=GuardSeverity.BLOCK
        ),
    )


def _plan(*actions: ActionSpec) -> Plan:
    states = []
    state = PlanningState.from_dict({})
    for a in actions:
        state = state.apply(a.effects)  # type: ignore[arg-type]
        states.append(state)
    return Plan(
        actions=actions, expected_states=tuple(states), total_cost=float(len(actions))
    )


class TestExecutorMergesCounts:
    def test_guard_block_keeps_other_actions_counts(self) -> None:
        b = ActionSpec(name="B", effects={"b": True}, execute=lambda ws: {"b": True})
        executor = GoapExecutor(guards=[_block("B")])
        result = executor(
            {
                "world_state": {},
                "plan": _plan(b),
                "current_step": 0,
                "action_failure_counts": {"A": 1},
            }
        )
        assert result["status"] == "action_failed"
        assert result["action_failure_counts"] == {"A": 1, "B": 1}


class _Form(BaseModel):
    reason: str


def _flaky_then_gated(gate: Any) -> tuple[list[ActionSpec], dict[str, int]]:
    calls = {"A": 0}

    def flaky(ws: dict[str, Any]) -> dict[str, Any]:
        calls["A"] += 1
        if calls["A"] == 1:
            raise RuntimeError("transient")
        return {"a": True}

    actions = [
        ActionSpec(name="A", effects={"a": True}, execute=flaky, max_retries=1),
        ActionSpec(
            name="B",
            preconditions={"a": True},
            effects={"done": True},
            execute=lambda ws: {"done": True},
            require_human_approval=gate,
        ),
    ]
    return actions, calls


class TestGraphKeepsEarlierFailures:
    def test_denial_after_a_retried_failure_keeps_both_counts(self) -> None:
        actions, calls = _flaky_then_gated(True)
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-denial-counts"}}
        paused = compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        assert "__interrupt__" in paused
        result = compiled.invoke(
            Command(resume={"approved": False, "reason": "no"}), config=config
        )
        assert calls["A"] == 2
        assert result["status"] == "no_plan"
        counts = result["action_failure_counts"]
        assert counts["A"] == 1, "A's earlier failure must survive B's denial"
        assert counts["B"] >= 1
        assert result["blacklisted_actions"] == ["B"]

    def test_denied_form_after_a_retried_failure_keeps_both_counts(self) -> None:
        actions, _ = _flaky_then_gated(_Form)
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-form-counts"}}
        compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        result = compiled.invoke(Command(resume=False), config=config)
        assert result["status"] == "no_plan"
        counts = result["action_failure_counts"]
        assert counts["A"] == 1 and counts["B"] >= 1

    def test_invalid_form_after_a_retried_failure_is_asked_again(self) -> None:
        """An invalid submission is not a failure: it is re-asked, and the
        earlier action's count is untouched in the meantime."""
        actions, _ = _flaky_then_gated(_Form)
        graph = GoapGraph(actions)
        compiled = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-form-reask"}}
        compiled.invoke(
            graph.initial_state(GoalSpec(conditions={"done": True})), config=config
        )
        paused = compiled.invoke(Command(resume={"not_the_field": 1}), config=config)
        assert paused["__interrupt__"][0].value["attempt"] == 2
        counts = paused["action_failure_counts"]
        assert counts == {"A": 1}
        assert [
            h["action_name"] for h in paused["execution_history"] if not h["success"]
        ] == ["A"]

    def test_guard_block_after_a_retried_failure_keeps_both_counts(self) -> None:
        actions, _ = _flaky_then_gated(False)
        graph = GoapGraph(actions, guards=[_block("B")])
        result = graph.invoke(goal=GoalSpec(conditions={"done": True}), world_state={})
        assert result["status"] == "no_plan"
        counts = result["action_failure_counts"]
        assert counts["A"] == 1 and counts["B"] >= 1
