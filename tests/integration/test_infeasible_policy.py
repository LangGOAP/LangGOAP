"""``GoalPolicy.on_infeasible`` through the full GOAP loop (issue #13).

A plan that violates a *hard* constraint must not run by default.  The
planner rejects it, the run ends with ``status="no_plan"`` and the
``no_plan_explanation`` carries the resource shortfalls so a stuck
handler can relax the constraint.  ``on_infeasible="execute"`` restores
the pre-0.2 advisory behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest

from langgoap import (
    ActionSpec,
    ConstraintSpec,
    FunctionalStuckHandler,
    GoalPolicy,
    GoalSpec,
    GoapGraph,
    StuckHandlerResult,
)
from langgoap.planner.csp import CSPStatus

CAP = 2.0


def _burn_action(calls: dict[str, int]) -> ActionSpec:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        calls["burn"] = calls.get("burn", 0) + 1
        return {"done": True}

    return ActionSpec(
        name="burn",
        effects={"done": True},
        resources={"cost_usd": 50.0},
        execute=execute,
    )


def _capped_goal(**policy: Any) -> GoalSpec:
    return GoalSpec(
        conditions={"done": True},
        constraints=(ConstraintSpec(key="cost_usd", max=CAP, level="hard"),),
        policy=GoalPolicy(**policy),
    )


class TestInfeasibleFailsByDefault:
    def test_default_policy_is_fail(self) -> None:
        assert GoalPolicy().on_infeasible == "fail"

    def test_hard_cap_violation_ends_as_no_plan_without_executing(self) -> None:
        calls: dict[str, int] = {}
        result = GoapGraph([_burn_action(calls)]).invoke(
            goal=_capped_goal(), world_state={}
        )
        assert result["status"] == "no_plan"
        assert calls == {}
        assert result["execution_history"] == []
        assert result["plan"] is None
        expl = result["no_plan_explanation"]
        assert expl["infeasible_plan"] == ["burn"]
        shortfall = expl["resource_shortfalls"][0]
        assert shortfall["key"] == "cost_usd"
        assert shortfall["required"] == 50.0
        assert shortfall["available_max"] == CAP
        assert shortfall["overrun"] == pytest.approx(48.0)
        assert "cost_usd" in expl["suggestion"]

    async def test_async_path_rejects_too(self) -> None:
        calls: dict[str, int] = {}
        result = await GoapGraph([_burn_action(calls)]).ainvoke(
            goal=_capped_goal(), world_state={}
        )
        assert result["status"] == "no_plan"
        assert calls == {}

    def test_feasible_plan_is_unaffected(self) -> None:
        calls: dict[str, int] = {}
        cheap = ActionSpec(
            name="cheap",
            effects={"done": True},
            resources={"cost_usd": 0.5},
            execute=lambda ws: {"done": True},
        )
        result = GoapGraph([cheap, _burn_action(calls)]).invoke(
            goal=_capped_goal(), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert result["plan"].action_names == ["cheap"]
        assert calls == {}


class TestInfeasibleExecuteOptOut:
    def test_execute_runs_the_plan_and_keeps_the_violation_visible(self) -> None:
        calls: dict[str, int] = {}
        result = GoapGraph([_burn_action(calls)]).invoke(
            goal=_capped_goal(on_infeasible="execute"), world_state={}
        )
        assert result["status"] == "goal_achieved"
        assert calls == {"burn": 1}
        csp = result["plan"].metadata.csp
        assert csp is not None
        assert csp.status == CSPStatus.INFEASIBLE

    def test_invalid_value_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="on_infeasible"):
            GoalPolicy(on_infeasible="maybe")  # type: ignore[arg-type]


class TestStuckHandlerCanRelaxTheConstraint:
    def test_handler_swaps_in_a_relaxed_goal(self) -> None:
        calls: dict[str, int] = {}
        seen: list[dict[str, Any]] = []

        def relax(state: Any, reason: Any) -> StuckHandlerResult:
            seen.append(reason)
            relaxed = GoalSpec(
                conditions={"done": True},
                constraints=(ConstraintSpec(key="cost_usd", max=100.0),),
            )
            return StuckHandlerResult.replan(
                handler_name="relax_budget",
                message="raise the cap to 100",
                new_goal=relaxed,
            )

        graph = GoapGraph(
            [_burn_action(calls)],
            stuck_handlers=[FunctionalStuckHandler("relax_budget", relax)],
        )
        result = graph.invoke(goal=_capped_goal(), world_state={})
        assert result["status"] == "goal_achieved"
        assert calls == {"burn": 1}
        # The handler saw the infeasibility payload, not a generic no-plan.
        assert seen and seen[0]["infeasible_plan"] == ["burn"]
        assert seen[0]["resource_shortfalls"][0]["key"] == "cost_usd"
