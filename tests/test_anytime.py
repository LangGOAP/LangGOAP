"""Anytime planning (issue #23).

``plan(..., time_budget_ms=...)`` must return the best *complete* plan found
within the budget.  A* stops at the first goal it pops, so a plain search
never has a "best so far" to return on a deadline: the anytime path runs
weighted A* with a tightening weight schedule, which yields a complete
(bounded-suboptimal) plan early and only improves it afterwards.
``AnytimePlanningStrategy`` must hand A* the deadline natively and must never
block the caller past its budget when wrapping a black-box strategy.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import pytest

from langgoap import ActionSpec, GoalPolicy, GoalSpec
from langgoap.planner import astar
from langgoap.planner.astar import plan as astar_plan
from langgoap.planner.strategy import AnytimePlanningStrategy
from langgoap.planner.types import Plan
from langgoap.state import PlanningState

START = PlanningState.from_dict({})
GOAL = GoalSpec(conditions={"g": True})


def _actions() -> list[ActionSpec]:
    """Weighted A* (w >= 2) pops ``direct`` first; plain A* finds the cheaper chain.

    h(start) = 1 (one unsatisfied condition, every action establishes at most
    one goal condition, cheapest action costs 1).  f(direct) = 2.5 while
    f(x1) = 1 + w * 1, so any w > 1.5 prefers ``direct``.
    """
    return [
        ActionSpec(name="direct", effects={"g": True}, cost=2.5),
        ActionSpec(name="x1", effects={"p1": True}, cost=1.0),
        ActionSpec(
            name="x2", preconditions={"p1": True}, effects={"g": True}, cost=1.0
        ),
    ]


class _FakeClock:
    """A monotonic clock under test control."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def expire(self) -> None:
        self.now += 1e9


class TestWeightedSearch:
    def test_weight_one_is_plain_optimal_astar(self) -> None:
        result, _ = astar._search(
            START, GOAL.conditions, _actions(), 0.0, None, weight=1.0
        )
        assert result is not None
        assert result.action_names == ["x1", "x2"]

    def test_large_weight_returns_a_bounded_suboptimal_plan(self) -> None:
        result, _ = astar._search(
            START, GOAL.conditions, _actions(), 0.0, None, weight=5.0
        )
        assert result is not None
        assert result.action_names == ["direct"]
        assert result.total_cost <= 5.0 * 2.0  # weighted A* bound: w * optimal


class TestAnytimePlan:
    def test_without_budget_result_is_optimal(self) -> None:
        plan_obj = astar_plan(START, GOAL, _actions())
        assert plan_obj is not None and plan_obj.action_names == ["x1", "x2"]

    def test_ample_budget_result_is_optimal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(astar, "_clock", _FakeClock())
        plan_obj = astar_plan(START, GOAL, _actions(), time_budget_ms=1000.0)
        assert plan_obj is not None and plan_obj.action_names == ["x1", "x2"]

    def test_deadline_after_first_iteration_returns_best_complete_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(astar, "_clock", clock)
        real_search = astar._search
        weights: list[float] = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            result = real_search(*args, **kwargs)
            weights.append(kwargs.get("weight", 1.0))
            clock.expire()  # the budget runs out right after this iteration
            return result

        monkeypatch.setattr(astar, "_search", spy)
        plan_obj = astar_plan(START, GOAL, _actions(), time_budget_ms=1000.0)
        assert plan_obj is not None
        assert plan_obj.action_names == ["direct"], "best complete plan so far"
        assert weights == [weights[0]] and weights[0] > 1.0

    def test_expired_before_any_plan_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every clock read advances one second, so a 1 ms budget is gone
        # before the first node is popped.
        monkeypatch.setattr(astar, "_clock", _FakeClock(step=1.0))
        assert astar_plan(START, GOAL, _actions(), time_budget_ms=1.0) is None

    def test_anytime_plan_metadata_reports_total_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(astar, "_clock", _FakeClock())
        single, n_single = astar._search(START, GOAL.conditions, _actions(), 0.0, None)
        plan_obj = astar_plan(START, GOAL, _actions(), time_budget_ms=1000.0)
        assert plan_obj is not None
        # Several weighted iterations ran; the metadata counts all of them.
        assert plan_obj.metadata.nodes_explored > n_single


class _SearchCompleteRecorder:
    def __init__(self) -> None:
        self.nodes: list[int] = []

    def on_search_complete(
        self, nodes_explored: int, duration_ms: float, found: bool
    ) -> None:
        self.nodes.append(nodes_explored)


class TestFallbackNodeAccounting:
    def _domain(self) -> tuple[list[ActionSpec], GoalSpec]:
        actions = [
            ActionSpec(name="only_g", effects={"g": True}),
            ActionSpec(name="make_q", effects={"q": True}),
            # Covers ``g`` for the reachability pre-check but can never run.
            ActionSpec(
                name="fake_g", preconditions={"never": True}, effects={"g": True}
            ),
        ]
        goal = GoalSpec(
            conditions={"g": True, "q": True},
            policy=GoalPolicy(blacklist_fallback=True),
        )
        return actions, goal

    def test_fallback_plan_counts_both_searches(self) -> None:
        actions, goal = self._domain()
        _, n_first = astar._search(
            START,
            goal.conditions,
            [a for a in actions if a.name != "only_g"],
            0.0,
            None,
        )
        assert n_first > 0, "the blacklisted search must explore before failing"
        rec = _SearchCompleteRecorder()
        plan_obj = astar_plan(
            START,
            goal,
            actions,
            blacklisted_actions=["only_g"],
            tracer=rec,
            record_expansions=True,
        )
        assert plan_obj is not None
        assert rec.nodes == [plan_obj.metadata.nodes_explored]
        unblacklisted = astar_plan(START, goal, actions)
        assert unblacklisted is not None
        assert (
            plan_obj.metadata.nodes_explored
            == n_first + unblacklisted.metadata.nodes_explored
        )


class _SlowStrategy:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def plan(
        self,
        start: Any,
        goal: Any,
        actions: Any,
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Any = None,
        current_step: int = 0,
    ) -> Plan | None:
        time.sleep(self.seconds)
        return Plan.empty()


class TestAnytimePlanningStrategy:
    def test_black_box_inner_does_not_block_past_the_budget(self) -> None:
        strategy = AnytimePlanningStrategy(time_budget_ms=50, inner=_SlowStrategy(1.0))
        started = time.monotonic()
        with pytest.warns(RuntimeWarning, match="budget"):
            result = strategy.plan(START, GOAL, _actions())
        elapsed = time.monotonic() - started
        assert result is None
        assert elapsed < 0.6, "the caller must not wait for the 1 s inner strategy"

    def test_fast_inner_result_is_returned(self) -> None:
        strategy = AnytimePlanningStrategy(
            time_budget_ms=5000, inner=_SlowStrategy(0.0)
        )
        assert strategy.plan(START, GOAL, _actions()) == Plan.empty()

    def test_default_inner_runs_astar_natively_with_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_threads(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "A* honours the deadline itself; no worker thread needed"
            )

        monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", no_threads)
        strategy = AnytimePlanningStrategy(time_budget_ms=5000)
        plan_obj = strategy.plan(START, GOAL, _actions())
        assert plan_obj is not None and plan_obj.action_names == ["x1", "x2"]
