"""The planning-key set is remembered per action list instead of recomputed by every node."""

from __future__ import annotations

from langgoap import ActionSpec, GoalSpec
from langgoap.graph.nodes._helpers import _ACTION_KEYS, _planning_keys


def test_planning_keys_are_remembered_per_action_list() -> None:
    actions = [
        ActionSpec("a", {"x": True}, {"y": True}),
        ActionSpec("b", {"y": True}, {"z": True}),
    ]
    _ACTION_KEYS.clear()
    first = _planning_keys(actions, GoalSpec(conditions={"z": True}))
    assert first == {"x", "y", "z"} and id(actions) in _ACTION_KEYS
    assert _planning_keys(actions, GoalSpec(conditions={"w": True})) == {
        "x",
        "y",
        "z",
        "w",
    }
    actions.append(
        ActionSpec("c", {"q": True}, {"r": True})
    )  # the same list object, new names
    assert _planning_keys(actions, GoalSpec(conditions={})) == {"x", "y", "z", "q", "r"}
