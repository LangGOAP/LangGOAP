"""A plan rebuilt from its JSON form materialises expected states only when they are read.

Every graph node rehydrates the plan from the state channel; rebuilding one
``PlanningState`` per step of the plan on every node made a run quadratic in
its length. The rebuilt sequence behaves like the tuple it replaces.
"""

from __future__ import annotations

import langgoap.state as state_module
from langgoap import ActionSpec, GoalSpec, plan
from langgoap.planner.types import LazyStates, Plan
from langgoap.state import PlanningState


def _plan(n: int) -> Plan:
    actions = [
        ActionSpec(f"a{i}", {f"s{i}": True}, {f"s{i + 1}": True}) for i in range(n)
    ]
    start = {"s0": True, **{f"s{i}": False for i in range(1, n + 1)}}
    result = plan(start, GoalSpec(conditions={f"s{n}": True}), actions)
    assert result is not None
    return result


def test_round_trip_is_equal_and_lazy(monkeypatch) -> None:
    original = _plan(30)
    data = original.to_dict()
    built = 0
    real = PlanningState.from_dict

    def counting(d, **kw):
        nonlocal built
        built += 1
        return real(d, **kw)

    monkeypatch.setattr(state_module.PlanningState, "from_dict", staticmethod(counting))
    restored = Plan.from_dict(data, actions=original.actions)
    assert built == 0, "nothing should be rebuilt until a state is read"
    assert isinstance(restored.expected_states, LazyStates)
    assert len(restored.expected_states) == 30
    assert restored.expected_states[-1] == original.expected_states[-1]
    assert built == 1
    assert restored.expected_states[4] == original.expected_states[4]
    assert restored == original  # materialises the rest for the comparison
    assert restored.to_dict() == data


def test_lazy_states_behave_like_a_tuple() -> None:
    original = _plan(5)
    lazy = Plan.from_dict(original.to_dict(), actions=original.actions).expected_states
    assert list(lazy) == list(original.expected_states)
    assert tuple(lazy[1:3]) == original.expected_states[1:3]
    assert lazy[0] in lazy and original.expected_states[2] in lazy
    assert bool(lazy) and not LazyStates(())
    assert (
        original.expected_states[:1] + lazy[1:]
    ) == original.expected_states  # repair.py does this
    assert (
        lazy + () == original.expected_states and () + lazy == original.expected_states
    )
    assert hash(lazy) == hash(original.expected_states)
    assert lazy == original.expected_states and original.expected_states == lazy
    assert lazy != original.expected_states[:-1]
    assert repr(lazy).startswith("LazyStates(")


def test_apply_keeps_dropping_unhashable_values_with_a_warning(caplog) -> None:
    state = PlanningState.from_dict({"a": 1})
    with caplog.at_level("WARNING"):
        new = state.apply({"b": [1, 2], "c": 3})
    assert new.to_dict() == {"a": 1, "c": 3}
    assert "unhashable" in caplog.text.lower()
    assert state.apply({"a": 2}).to_dict() == {"a": 2}
    assert state.apply({"a": 1}) == state
