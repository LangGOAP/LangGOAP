"""Benchmarks for planning with dynamic (callable) costs.

The domain is the branching case of the read-amplification diagnostic: eight
required steps, seven optional flags that together enable an expensive route,
optimum 8.  Variants:

- ``static``: every cost a number.
- ``undeclared``: callable costs that read a context key (constant during the
  search) without saying so.
- ``constant``: the same, declared with ``cost_keys=()``.
- ``keyed``: callable costs that read a planning key set by ``step_0``,
  declared with ``cost_keys=("g0",)``.
- ``anytime``: ``undeclared`` and ``keyed`` under a time budget, so the
  anytime schedule runs the search up to four times.

``cost_calls_per_plan`` in each benchmark's ``extra_info`` is the number of
cost-function calls one ``plan()`` made.  On a library without ``cost_keys``
the declared variants run undeclared, which is the baseline.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan

HAS_COST_KEYS = "cost_keys" in {f.name for f in dataclasses.fields(ActionSpec)}


def _spec(**kw: Any) -> ActionSpec:
    if not HAS_COST_KEYS:
        kw.pop("cost_keys", None)
    return ActionSpec(**kw)


def _domain(kind: str) -> tuple[list[ActionSpec], GoalSpec, dict[str, Any], list[int]]:
    calls = [0]

    def quote(ws: dict[str, Any]) -> float:
        calls[0] += 1
        return float(ws["quote"])

    def keyed(ws: dict[str, Any]) -> float:
        calls[0] += 1
        return float(ws["quote"]) * (1.0 if ws.get("g0") else 1.5)

    def cost(_name: str) -> dict[str, Any]:
        if kind == "static":
            return {"cost": 1.0}
        if kind == "constant":
            return {"cost": quote, "cost_keys": ()}
        if kind == "keyed":
            return {"cost": keyed, "cost_keys": ("g0",)}
        return {"cost": quote}

    actions = [
        _spec(
            name=f"step_{i}",
            preconditions={} if i == 0 else {f"g{i - 1}": True},
            effects={f"g{i}": True},
            **cost(f"step_{i}"),
        )
        for i in range(8)
    ]
    actions += [
        _spec(
            name=f"option_{i}",
            preconditions={},
            effects={f"x{i}": True},
            **cost(f"option_{i}"),
        )
        for i in range(7)
    ]
    actions.append(
        _spec(
            name="expensive_route",
            preconditions={f"x{i}": True for i in range(7)},
            effects={"g7": True},
            cost=100.0,
        )
    )
    goal = GoalSpec(conditions={f"g{i}": True for i in range(8)})
    return actions, goal, {"quote": 1.0}, calls


@pytest.mark.parametrize("kind", ["static", "undeclared", "constant", "keyed"])
def test_bench_plan_dynamic_costs(benchmark: Any, kind: str) -> None:
    actions, goal, start, calls = _domain(kind)
    first = plan(dict(start), goal, actions)
    assert first is not None and first.action_names == [f"step_{i}" for i in range(8)]
    assert first.total_cost == (8.5 if kind == "keyed" else 8.0)
    benchmark.extra_info["cost_calls_per_plan"] = calls[0]
    benchmark.extra_info["nodes_explored"] = first.metadata.nodes_explored
    result = benchmark(plan, dict(start), goal, actions)
    assert result is not None


@pytest.mark.parametrize("kind", ["undeclared", "keyed"])
def test_bench_plan_dynamic_costs_anytime(benchmark: Any, kind: str) -> None:
    actions, goal, start, calls = _domain(kind)
    first = plan(dict(start), goal, actions, time_budget_ms=10_000)
    assert first is not None and first.total_cost == (8.5 if kind == "keyed" else 8.0)
    benchmark.extra_info["cost_calls_per_plan"] = calls[0]
    benchmark.extra_info["nodes_explored"] = first.metadata.nodes_explored
    result = benchmark(plan, dict(start), goal, actions, time_budget_ms=10_000)
    assert result is not None
