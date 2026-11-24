"""Dynamic costs: what they declare, how often they run, what pruning keeps (#118, #119).

A cost function is a function of the state it is given.  For one planning call
the planner asks a cost that declares ``cost_keys`` once per distinct value of
those keys; one that declares nothing is asked at every evaluation, as before.
A declared cost that reads nothing the search changes keeps the heuristic
alive, and the pruning passes treat the declared keys as read by the action.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from langgoap import ActionSpec, GoalSpec, GoapAction, GoapGraph, goap_action
from langgoap.planner.astar import plan

GOAL8 = GoalSpec(conditions={f"g{i}": True for i in range(8)})
STEPS = [f"step_{i}" for i in range(8)]


class Calls:
    def __init__(self) -> None:
        self.n = 0
        self.inputs: Counter[tuple[str, tuple[Any, ...]]] = Counter()

    def wrap(self, name: str, fn: Any) -> Any:
        def counted(ws: dict[str, Any]) -> float:
            self.n += 1
            self.inputs[(name, tuple(sorted(ws.items())))] += 1
            return fn(ws)

        return counted


def _quote(ws: dict[str, Any]) -> float:
    return float(ws["quote"])


def _keyed(ws: dict[str, Any]) -> float:
    return float(ws["quote"]) * (1.0 if ws.get("g0") else 1.5)


def _branching(
    calls: Calls | None = None,
    *,
    cost: Any = None,
    cost_keys: Any = None,
    static: bool = False,
) -> list[ActionSpec]:
    """Eight steps, seven optional flags that together enable an expensive route; optimum 8."""

    def spec(name: str, pre: dict[str, Any], eff: dict[str, Any]) -> ActionSpec:
        if static:
            return ActionSpec(name, pre, eff, cost=1.0)
        fn = calls.wrap(name, cost) if calls is not None else cost
        kw = {} if cost_keys is None else {"cost_keys": cost_keys}
        return ActionSpec(name, pre, eff, cost=fn, **kw)

    actions = [
        spec(f"step_{i}", {} if i == 0 else {f"g{i - 1}": True}, {f"g{i}": True})
        for i in range(8)
    ]
    actions += [spec(f"option_{i}", {}, {f"x{i}": True}) for i in range(7)]
    actions.append(
        ActionSpec(
            "expensive_route",
            {f"x{i}": True for i in range(7)},
            {"g7": True},
            cost=100.0,
        )
    )
    return actions


class TestDeclaration:
    def test_keys_become_a_frozenset(self) -> None:
        a = ActionSpec("a", {}, {"x": True}, cost=lambda ws: 1.0, cost_keys=["k", "j"])
        assert a.cost_keys == frozenset({"k", "j"})

    def test_no_keys_is_a_declaration_too(self) -> None:
        a = ActionSpec("a", {}, {"x": True}, cost=lambda ws: 1.0, cost_keys=())
        assert a.cost_keys == frozenset()

    def test_undeclared_is_none(self) -> None:
        assert ActionSpec("a", {}, {"x": True}, cost=lambda ws: 1.0).cost_keys is None

    def test_a_bare_string_is_refused(self) -> None:
        with pytest.raises(TypeError, match="cost_keys"):
            ActionSpec("a", {}, {"x": True}, cost=lambda ws: 1.0, cost_keys="k")

    def test_keys_on_a_number_are_refused(self) -> None:
        with pytest.raises(ValueError, match="cost_keys"):
            ActionSpec("a", {}, {"x": True}, cost=2.0, cost_keys=("k",))

    def test_the_decorator_carries_them(self) -> None:
        @goap_action(effects={"x": True}, cost=lambda ws: 1.0, cost_keys={"k"})
        def act(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        assert act.cost_keys == frozenset({"k"})

    def test_the_class_carries_them(self) -> None:
        class Act(GoapAction):
            effects = {"x": True}
            cost_keys = {"k"}

            def cost(self, world_state: dict[str, Any]) -> float:
                return 1.0 if world_state.get("k") else 2.0

        spec = Act().to_spec()
        assert callable(spec.cost)
        assert spec.cost_keys == frozenset({"k"})

    def test_a_class_that_keeps_the_default_cost_has_a_static_one(self) -> None:
        class Act(GoapAction):
            effects = {"x": True}

        spec = Act().to_spec()
        assert spec.cost == 1.0 and not callable(spec.cost)
        assert spec.cost_keys is None

    def test_to_dict_carries_them_and_from_dict_drops_them_with_the_function(
        self,
    ) -> None:
        a = ActionSpec("a", {}, {"x": True}, cost=lambda ws: 1.0, cost_keys=("k", "j"))
        d = a.to_dict()
        assert d["cost"] is None and d["cost_keys"] == ["j", "k"]
        back = ActionSpec.from_dict(d)
        assert back.cost == 1.0 and back.cost_keys is None
        assert ActionSpec("b", {}, {"x": True}, cost=2.0).to_dict()["cost_keys"] is None


class TestOnePlanningCall:
    def test_a_constant_cost_is_asked_once_per_action(self) -> None:
        calls = Calls()
        result = plan(
            {"quote": 1.0}, GOAL8, _branching(calls, cost=_quote, cost_keys=())
        )
        assert result is not None
        assert result.action_names == STEPS and result.total_cost == 8.0
        assert calls.n == 15

    def test_a_constant_cost_keeps_the_heuristic(self) -> None:
        static = plan({}, GOAL8, _branching(static=True))
        constant = plan({"quote": 1.0}, GOAL8, _branching(cost=_quote, cost_keys=()))
        assert static is not None and constant is not None
        assert static.metadata.nodes_explored == constant.metadata.nodes_explored == 9

    def test_a_keyed_cost_is_asked_once_per_value(self) -> None:
        calls = Calls()
        actions = _branching(calls, cost=_keyed, cost_keys=("g0",))
        result = plan({"quote": 1.0}, GOAL8, actions)
        assert result is not None
        assert result.action_names == STEPS and result.total_cost == 8.5
        assert calls.n <= 2 * 15  # g0 unset, then True
        assert max(calls.inputs.values()) == 1

    def test_an_undeclared_cost_is_asked_at_every_evaluation(self) -> None:
        """Nothing is remembered without a declaration: the anytime schedule asks again."""
        calls = Calls()
        actions = _branching(calls, cost=_keyed)
        result = plan({"quote": 1.0}, GOAL8, actions, time_budget_ms=10_000)
        assert result is not None
        assert result.action_names == STEPS and result.total_cost == 8.5
        single = plan({"quote": 1.0}, GOAL8, _branching(cost=_keyed))
        assert single is not None
        assert (
            result.metadata.nodes_explored > single.metadata.nodes_explored
        )  # several searches
        assert max(calls.inputs.values()) > 1

    def test_a_declared_cost_is_remembered_across_the_anytime_schedule(self) -> None:
        calls = Calls()
        actions = _branching(calls, cost=_keyed, cost_keys=("g0",))
        result = plan({"quote": 1.0}, GOAL8, actions, time_budget_ms=10_000)
        assert result is not None
        assert result.action_names == STEPS and result.total_cost == 8.5
        assert calls.n <= 2 * 15

    def test_the_memo_ends_with_the_call(self) -> None:
        calls = Calls()
        actions = _branching(calls, cost=_quote, cost_keys=())
        first = plan({"quote": 1.0}, GOAL8, actions)
        second = plan({"quote": 3.0}, GOAL8, actions)
        assert first is not None and second is not None
        assert (first.total_cost, second.total_cost) == (8.0, 24.0)
        assert calls.n == 30


class TestPruning:
    GOAL = GoalSpec(conditions={"done": True})

    @staticmethod
    def _actions(declared: bool, with_route: bool) -> list[ActionSpec]:
        kw: dict[str, Any] = {"cost_keys": ("discounted",)} if declared else {}
        actions = [
            ActionSpec("discount", {}, {"discounted": True}, cost=1.0),
            ActionSpec(
                "buy",
                {},
                {"done": True},
                cost=lambda ws: 1.0 if ws.get("discounted") else 10.0,
                **kw,
            ),
        ]
        if with_route:
            actions.append(
                ActionSpec(
                    "expensive_finish", {"discounted": True}, {"done": True}, cost=100.0
                )
            )
        return actions

    @pytest.mark.parametrize("with_route", [False, True])
    def test_a_declared_dependency_keeps_the_enabler(self, with_route: bool) -> None:
        result = plan({}, self.GOAL, self._actions(True, with_route))
        assert result is not None
        assert result.action_names == ["discount", "buy"] and result.total_cost == 2.0

    def test_the_shortener_never_makes_a_found_plan_dearer(self) -> None:
        result = plan({}, self.GOAL, self._actions(False, True))
        assert result is not None
        assert result.action_names == ["discount", "buy"] and result.total_cost == 2.0

    def test_an_undeclared_dependency_is_not_searched(self) -> None:
        """Without cost_keys what a cost reads is unknown: the relevance filter drops the enabler."""
        result = plan({}, self.GOAL, self._actions(False, False))
        assert result is not None
        assert result.action_names == ["buy"] and result.total_cost == 10.0


class TestClassBasedActions:
    def test_the_default_cost_does_not_disable_the_heuristic(self) -> None:
        def act(name: str, pre: dict[str, Any], eff: dict[str, Any]) -> ActionSpec:
            cls = type(name, (GoapAction,), {"preconditions": pre, "effects": eff})
            return cls().to_spec()

        actions = [
            act(f"step_{i}", {} if i == 0 else {f"g{i - 1}": True}, {f"g{i}": True})
            for i in range(8)
        ]
        actions += [act(f"option_{i}", {}, {f"x{i}": True}) for i in range(7)]
        actions.append(
            ActionSpec(
                "expensive_route",
                {f"x{i}": True for i in range(7)},
                {"g7": True},
                cost=100.0,
            )
        )
        result = plan({}, GOAL8, actions)
        assert result is not None
        assert result.action_names == STEPS and result.metadata.nodes_explored == 9


class TestOptimalityAndAccounting:
    GOAL = GoalSpec(conditions={"done": True})

    @pytest.mark.parametrize("cheap", [0.1, 0.4, 0.75])
    @pytest.mark.parametrize("declared", ["undeclared", "constant", "keyed"])
    def test_a_callable_cost_below_one_keeps_optimality(
        self, cheap: float, declared: str
    ) -> None:
        """The heuristic's floor must follow a declared cost below the static minimum."""
        kw: dict[str, Any] = {}
        if declared == "constant":
            kw = {"cost_keys": ()}
        elif declared == "keyed":
            kw = {"cost_keys": ("ready",)}
        result = plan(
            {},
            self.GOAL,
            [
                ActionSpec("direct", {}, {"done": True}, cost=1.0),
                ActionSpec("prepare", {}, {"ready": True}, cost=lambda ws: cheap, **kw),
                ActionSpec(
                    "finish",
                    {"ready": True},
                    {"done": True},
                    cost=lambda ws: cheap,
                    **kw,
                ),
            ],
        )
        assert result is not None
        assert result.total_cost == pytest.approx(min(1.0, 2 * cheap))
        assert result.action_names == (
            ["prepare", "finish"] if 2 * cheap < 1.0 else ["direct"]
        )

    def test_an_irrelevant_action_s_cost_is_never_asked(self) -> None:
        calls = [0]

        def counted(ws: dict[str, Any]) -> float:
            calls[0] += 1
            return 1.0

        actions = [ActionSpec("finish", {}, {"done": True}, cost=1.0)]
        actions += [
            ActionSpec(f"unrelated_{i}", {}, {f"unused_{i}": True}, cost=counted)
            for i in range(100)
        ]
        result = plan({}, self.GOAL, actions)
        assert result is not None
        assert result.action_names == ["finish"] and result.metadata.nodes_explored == 2
        assert calls[0] == 0

    @pytest.mark.parametrize("declared", [False, True])
    def test_history_costs_sum_to_the_plan_s_total(self, declared: bool) -> None:
        """The realised cost is evaluated once per executed step, outside the memo."""
        length, calls = 4, [0]

        def two(ws: dict[str, Any]) -> float:
            calls[0] += 1
            return 2.0

        kw: dict[str, Any] = {"cost_keys": ()} if declared else {}
        actions = [
            ActionSpec(
                f"step_{i}",
                {"stage": i},
                {"stage": i + 1},
                cost=two,
                execute=lambda ws: None,  # the declared effects apply
                **kw,
            )
            for i in range(length)
        ]
        result = GoapGraph(actions).invoke(
            GoalSpec(conditions={"stage": length}), world_state={"stage": 0}
        )
        assert result.succeeded
        assert result["plan"].total_cost == 2.0 * length
        assert sum(entry.cost for entry in result["execution_history"]) == 2.0 * length
        if declared:
            assert (
                calls[0] == 2 * length
            )  # once per action for the search, once per step executed
        else:
            assert calls[0] <= 3 * length  # search, plan building, execution
