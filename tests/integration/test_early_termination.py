"""Integration tests for ``TerminationPolicy`` in the observer.

Mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
com/embabel/agent/core/EarlyTerminationPolicyTest.kt`` and
``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/agent/core/EarlyTerminationPolicy.kt``:

* ``shouldTerminate(agentProcess) -> EarlyTermination | None`` —
  returned non-null means terminate.
* Built-ins: ``maxActions``, ``maxTokens``, ``hardBudgetLimit``,
  ``ON_STUCK``, ``firstOf`` composite.

LangGOAP additions:

* ``MaxWallClockPolicy`` — kept first-class because GoapState carries a
  start timestamp.  Embabel offloads wall-clock to Spring's
  ``Scheduler`` rather than embedding it in the policy abstraction.
* ``AllOfPolicy`` — composes by AND (every policy must fire) for
  symmetry with ``FirstOfPolicy``.

The observer (``GoapObserver``) consults the policies between every
node-to-node routing decision so termination short-circuits every
other observer logic (replan, execute, deviation check).
"""

from __future__ import annotations

import time
from typing import Any

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.termination import (
    AllOfPolicy,
    EarlyTermination,
    FirstOfPolicy,
    MaxActionsPolicy,
    MaxCostPolicy,
    MaxLLMCallsPolicy,
    MaxTokensPolicy,
    MaxWallClockPolicy,
    OnStuckPolicy,
    TerminationPolicy,
)
from tests.conftest import make_action as _action


def _trivial_actions(n: int) -> list[ActionSpec]:
    """``n``-step linear chain of trivial actions."""
    actions: list[ActionSpec] = []
    for i in range(n):
        pre = {f"step{i}": True} if i > 0 else {}
        actions.append(_action(f"step_{i}", pre=pre, eff={f"step{i + 1}": True}))
    return actions


def _final_goal(n: int) -> GoalSpec:
    return GoalSpec(conditions={f"step{n}": True})


# ---------------------------------------------------------------------------
# MaxActionsPolicy — the canonical Embabel scenario
# ---------------------------------------------------------------------------


class TestMaxActionsPolicy:
    def test_max_actions_terminates_after_threshold(self) -> None:
        """Halts after the second action even though the goal needs five.

        Mirrors Embabel's ``test MaxActions`` (history.size == 3, threshold
        == 2 → termination).
        """
        graph = GoapGraph(
            _trivial_actions(5),
            termination_policies=[MaxActionsPolicy(2)],
        )

        result = graph.invoke(
            goal=_final_goal(5),
            world_state={},
        )

        assert result["status"] == "terminated"
        history = result.get("execution_history", [])
        assert len(history) == 2, f"expected 2 executed actions, got {len(history)}"

    def test_max_actions_threshold_above_plan_length_completes(self) -> None:
        """A generous limit must not interfere with a normal-completion run."""
        graph = GoapGraph(
            _trivial_actions(3),
            termination_policies=[MaxActionsPolicy(100)],
        )

        result = graph.invoke(
            goal=_final_goal(3),
            world_state={},
        )

        assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# Wall-clock policy
# ---------------------------------------------------------------------------


class TestMaxWallClockPolicy:
    def test_wall_clock_terminates_long_runs(self) -> None:
        """Slow action sleeps past the budget; the observer halts before
        the second action runs."""

        def slow_execute(world_state: dict[str, Any]) -> dict[str, Any]:
            time.sleep(0.05)
            return {"step1": True}

        slow_action = ActionSpec(
            name="slow",
            preconditions={},
            effects={"step1": True},
            execute=slow_execute,
        )
        next_action = _action("step_1", pre={"step1": True}, eff={"step2": True})

        graph = GoapGraph(
            [slow_action, next_action],
            termination_policies=[
                MaxWallClockPolicy(seconds=0.01),
            ],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"step2": True}),
            world_state={},
        )

        assert result["status"] == "terminated"
        # First action ran (so wall-clock had something to measure); second
        # was aborted by the policy.
        assert len(result.get("execution_history", [])) == 1


# ---------------------------------------------------------------------------
# Cost / token / LLM-call accounting policies
# ---------------------------------------------------------------------------


class TestAccountingPolicies:
    """Cost/token/llm-call metrics are user-supplied keys on world_state.

    LangGOAP ships the policies; users wire the bookkeeping via their
    actions or via a LangChain callback handler.  These tests use action
    closures that update the keys explicitly to keep the assertion
    chain deterministic and dependency-free.
    """

    def test_max_cost_policy_terminates_when_budget_exceeded(self) -> None:
        def expensive_execute(world_state: dict[str, Any]) -> dict[str, Any]:
            current = float(world_state.get("total_cost_usd", 0.0))
            return {"step1": True, "total_cost_usd": current + 0.30}

        actions = [
            ActionSpec(
                name="expensive",
                preconditions={},
                effects={"step1": True, "total_cost_usd": 0.0},
                execute=expensive_execute,
            ),
            _action("step_1", pre={"step1": True}, eff={"step2": True}),
        ]
        graph = GoapGraph(
            actions,
            termination_policies=[MaxCostPolicy(usd=0.20)],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"step2": True}),
            world_state={"total_cost_usd": 0.0},
        )

        assert result["status"] == "terminated"
        assert len(result.get("execution_history", [])) == 1

    def test_max_tokens_policy_terminates_when_threshold_exceeded(self) -> None:
        def token_burn_execute(world_state: dict[str, Any]) -> dict[str, Any]:
            current = int(world_state.get("total_tokens", 0))
            return {"step1": True, "total_tokens": current + 1500}

        actions = [
            ActionSpec(
                name="token_burn",
                preconditions={},
                effects={"step1": True, "total_tokens": 0},
                execute=token_burn_execute,
            ),
            _action("step_1", pre={"step1": True}, eff={"step2": True}),
        ]
        graph = GoapGraph(
            actions,
            termination_policies=[MaxTokensPolicy(tokens=1000)],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"step2": True}),
            world_state={"total_tokens": 0},
        )

        assert result["status"] == "terminated"
        assert len(result.get("execution_history", [])) == 1

    def test_max_llm_calls_policy_terminates_when_threshold_exceeded(self) -> None:
        def llm_call_execute(world_state: dict[str, Any]) -> dict[str, Any]:
            current = int(world_state.get("llm_call_count", 0))
            return {"step1": True, "llm_call_count": current + 1}

        actions = [
            ActionSpec(
                name="llm_call",
                preconditions={},
                effects={"step1": True, "llm_call_count": 0},
                execute=llm_call_execute,
            ),
            _action("step_1", pre={"step1": True}, eff={"step2": True}),
        ]
        graph = GoapGraph(
            actions,
            termination_policies=[MaxLLMCallsPolicy(max_calls=1)],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"step2": True}),
            world_state={"llm_call_count": 0},
        )

        assert result["status"] == "terminated"
        assert len(result.get("execution_history", [])) == 1


# ---------------------------------------------------------------------------
# OnStuckPolicy
# ---------------------------------------------------------------------------


class TestOnStuckPolicy:
    def test_on_stuck_short_circuits_no_plan(self) -> None:
        """When the planner emits ``no_plan``, OnStuckPolicy converts the
        terminal status to ``terminated`` (with error=False, mirroring
        Embabel's "stuck without error" semantics)."""
        unreachable_action = _action(
            "needs_unicorn",
            pre={"unicorn_present": True},
            eff={"goal": True},
        )
        graph = GoapGraph(
            [unreachable_action],
            termination_policies=[OnStuckPolicy()],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"goal": True}),
            world_state={},
        )

        assert result["status"] == "terminated"


# ---------------------------------------------------------------------------
# Composite policies
# ---------------------------------------------------------------------------


class TestFirstOfPolicy:
    def test_first_of_short_circuits_on_first_match(self) -> None:
        """Mirrors Embabel test
        ``test first of MaxAction and budget terminates``: any inner
        policy firing terminates."""
        graph = GoapGraph(
            _trivial_actions(5),
            termination_policies=[
                FirstOfPolicy(MaxActionsPolicy(100), MaxActionsPolicy(2)),
            ],
        )

        result = graph.invoke(
            goal=_final_goal(5),
            world_state={},
        )

        assert result["status"] == "terminated"
        assert len(result.get("execution_history", [])) == 2

    def test_first_of_does_not_terminate_when_all_policies_pass(self) -> None:
        """Mirrors Embabel test
        ``test first of MaxAction and budget does not terminate``."""
        graph = GoapGraph(
            _trivial_actions(3),
            termination_policies=[
                FirstOfPolicy(
                    MaxActionsPolicy(100),
                    MaxCostPolicy(usd=100.00),
                ),
            ],
        )

        result = graph.invoke(
            goal=_final_goal(3),
            world_state={"total_cost_usd": 0.0},
        )

        assert result["status"] == "goal_achieved"


class TestAllOfPolicy:
    def test_all_of_terminates_only_when_all_inner_policies_fire(self) -> None:
        """Even though MaxActionsPolicy(2) fires after the second action,
        AllOf does not terminate because the cost policy hasn't fired."""
        graph = GoapGraph(
            _trivial_actions(5),
            termination_policies=[
                AllOfPolicy(MaxActionsPolicy(2), MaxCostPolicy(usd=1000.00)),
            ],
        )

        result = graph.invoke(
            goal=_final_goal(5),
            world_state={"total_cost_usd": 0.0},
        )

        # Goal achieved because the AllOf composite never fires.
        assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# EarlyTermination event surface
# ---------------------------------------------------------------------------


class TestEarlyTerminationEvent:
    def test_termination_carries_policy_name_and_reason(self) -> None:
        graph = GoapGraph(
            _trivial_actions(5),
            termination_policies=[MaxActionsPolicy(1)],
        )

        result = graph.invoke(
            goal=_final_goal(5),
            world_state={},
        )

        assert result["status"] == "terminated"
        # Reason and policy name surfaced in the GoapState delta on END.
        assert "MaxActionsPolicy" in (result.get("replan_reason") or "")


# ---------------------------------------------------------------------------
# Custom-policy extensibility (Protocol conformance)
# ---------------------------------------------------------------------------


class TestCustomPolicy:
    def test_user_policy_can_terminate_on_arbitrary_condition(self) -> None:
        class TerminateOnFlag:
            name = "TerminateOnFlag"

            def should_terminate(
                self, state: dict[str, Any]
            ) -> EarlyTermination | None:
                if state.get("world_state", {}).get("kill_switch") is True:
                    return EarlyTermination(
                        policy_name=self.name,
                        reason="kill switch flipped",
                        error=False,
                    )
                return None

        assert isinstance(TerminateOnFlag(), TerminationPolicy)

        actions = [
            ActionSpec(
                name="flip_switch",
                preconditions={},
                effects={"step1": True, "kill_switch": True},
                execute=lambda ws: {"step1": True, "kill_switch": True},
            ),
            _action("step_1", pre={"step1": True}, eff={"step2": True}),
        ]
        graph = GoapGraph(
            actions,
            termination_policies=[TerminateOnFlag()],
        )

        result = graph.invoke(
            goal=GoalSpec(conditions={"step2": True}),
            world_state={},
        )

        assert result["status"] == "terminated"
        assert "kill switch flipped" in (result.get("replan_reason") or "")
