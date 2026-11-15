"""The wall-clock anchor survives a resume in another process (issue #50).

``time.monotonic()`` is process-local, so an anchor persisted by one process
is meaningless to another.  The planner now stores epoch seconds
(``time.time()``) and ``MaxWallClockPolicy`` compares against the same clock.
"""

from __future__ import annotations

import time

from langgoap import ActionSpec, GoalSpec, GoapGraph
from langgoap.termination import MaxWallClockPolicy


def _state(anchor: float | None) -> dict:
    return {"wall_clock_started_at": anchor, "execution_history": [], "world_state": {}}


class TestEpochAnchor:
    def test_planner_stores_epoch_seconds(self) -> None:
        result = GoapGraph([ActionSpec(name="a", effects={"done": True})]).invoke(
            goal=GoalSpec(conditions={"done": True}), world_state={}
        )
        anchor = result["wall_clock_started_at"]
        assert anchor is not None
        assert (
            abs(anchor - time.time()) < 60
        ), "anchor is an epoch timestamp, not a monotonic reading"

    def test_persisted_anchor_from_another_process_fires(self) -> None:
        # A run started 100 s ago (in any process) is over a 10 s budget.
        decision = MaxWallClockPolicy(seconds=10.0).should_terminate(
            _state(time.time() - 100)
        )
        assert decision is not None and "exceeded" in decision.reason

    def test_fresh_anchor_does_not_fire(self) -> None:
        assert (
            MaxWallClockPolicy(seconds=10.0).should_terminate(_state(time.time()))
            is None
        )

    def test_missing_anchor_is_not_a_termination(self) -> None:
        assert MaxWallClockPolicy(seconds=0.001).should_terminate(_state(None)) is None
