"""The execution history is rebuilt from the state channel only when it is read."""

from __future__ import annotations

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.graph.state import (
    ActionResult,
    LazyHistory,
    history_from_state,
    rehydrate_state,
)


def _entries(n: int) -> list[dict]:
    return [
        ActionResult(f"a{i}", True, {"k": i}, {"k": i + 1}, None, 1.0).to_dict()
        for i in range(n)
    ]


def test_history_is_rebuilt_only_when_read(monkeypatch) -> None:
    import langgoap.graph.state as module

    built = 0
    real = module._as_action_result

    def counting(entry):
        nonlocal built
        built += 1
        return real(entry)

    monkeypatch.setattr(module, "_as_action_result", counting)
    state = rehydrate_state({"execution_history": _entries(50)})
    history = state["execution_history"]
    assert isinstance(history, LazyHistory) and len(history) == 50 and built == 0
    assert history[-1].action_name == "a49" and built == 1
    assert [h.action_name for h in reversed(history)][:2] == ["a49", "a48"]
    assert history_from_state(history) is history  # idempotent


def test_lazy_history_behaves_like_the_list_it_replaces() -> None:
    history = history_from_state(_entries(4))
    eager = [ActionResult.from_dict(e) for e in _entries(4)]
    assert list(history) == eager and history == eager and eager == history
    assert history[1:3] == eager[1:3] and history + [eager[0]] == eager + [eager[0]]
    assert [] + history == eager and bool(history) and not LazyHistory([])
    assert repr(history) == "LazyHistory(4 entries)"
    assert [h.action_name for h in history if h.success] == ["a0", "a1", "a2", "a3"]


def test_the_final_state_still_reads_as_before() -> None:
    actions = [
        ActionSpec(f"a{i}", {f"s{i}": True}, {f"s{i + 1}": True}) for i in range(5)
    ]
    goal = GoalSpec(conditions={"s5": True})
    state = GoapGraph(actions).invoke(goal, {"s0": True})
    assert state["status"] == "goal_achieved"
    assert [h.action_name for h in state["execution_history"]] == [
        f"a{i}" for i in range(5)
    ]
    assert len(state["execution_history"]) == 5
