"""Duplicate action names are rejected at construction (issue #48).

Blacklists, failure counts, history records and executor rebinding are all
keyed by ``ActionSpec.name``; two actions sharing a name make the planner
plan one and the executor run the other.
"""

from __future__ import annotations

import pytest

from langgoap import ActionSpec, GoapGraph
from langgoap.graph.nodes import GoapExecutor, GoapPlanner, ParallelGoapExecutor


def _dupes() -> list[ActionSpec]:
    return [
        ActionSpec(name="do", effects={"x": True}, cost=1.0),
        ActionSpec(name="other", effects={"y": True}),
        ActionSpec(name="do", effects={"x": True}, cost=9.0),
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda actions: GoapGraph(actions),
        lambda actions: GoapPlanner(actions),
        lambda actions: GoapExecutor(actions=actions),
        lambda actions: ParallelGoapExecutor(actions=actions),
    ],
    ids=["GoapGraph", "GoapPlanner", "GoapExecutor", "ParallelGoapExecutor"],
)
def test_duplicate_names_are_rejected(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="'do'"):
        build(_dupes())


def test_unique_names_are_accepted() -> None:
    actions = [a for a in _dupes() if a.cost == 1.0 or a.name == "other"]
    assert GoapGraph(actions).actions == actions
    assert GoapExecutor(actions=actions) is not None
