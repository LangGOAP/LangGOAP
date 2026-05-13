"""Integration tests for ``ActionSpec.can_rerun`` / ``read_only`` flags.

Mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/main/kotlin/
com/embabel/agent/core/Action.kt``:

* ``canRerun: Boolean`` — when ``False``, the planner must not include
  the action twice in a single plan.  Required to allow looping
  behaviour in agents whose actions are explicitly marked otherwise.
* ``readOnly: Boolean`` — informational metadata: the action has no
  external side effects (no API/DB/file mutation).  Used by tooling
  for catch-up / learning replays; not consumed by the planner.

Embabel does not ship a dedicated test file for these flags (they are
contract-level on the ``Action`` interface), so the assertions below
are derived from the source contract per Rule 3 in the gap-closure
plan.  When at least one action in the action set is ``can_rerun=False``
A* must avoid revisiting that action on any path; ``read_only`` must
survive serialization round-trips so executor tooling can reason about
it after a checkpoint reload.
"""

from __future__ import annotations

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.astar import plan
from langgoap.serde import LangGOAPSerializer
from langgoap.state import PlanningState
from tests.conftest import make_action as _action

# ---------------------------------------------------------------------------
# can_rerun
# ---------------------------------------------------------------------------


class TestCanRerun:
    def test_default_can_rerun_allows_repetition(self) -> None:
        """Without can_rerun=False the planner happily uses the same
        action twice when needed (counter-decrement style)."""
        decrement = ActionSpec(
            name="decrement",
            preconditions={},
            effects=lambda ws: {"counter": int(ws.get("counter", 2)) - 1},
            effect_keys=frozenset({"counter"}),
            cost=1.0,
        )
        start = PlanningState.from_dict({"counter": 2})
        goal = GoalSpec(conditions={"counter": 0})

        result = plan(start, goal, [decrement])

        assert result is not None
        # Two applications of the same action — proves default can_rerun=True.
        assert result.action_names == ["decrement", "decrement"]

    def test_can_rerun_false_excludes_action_after_use(self) -> None:
        """A fire-once action is excluded from successor expansion if it
        is already in the plan path; the planner picks an alternate path
        even when reusing the cheap action would be optimal."""
        single_use = ActionSpec(
            name="cheap_single",
            preconditions={},
            effects=lambda ws: {"counter": int(ws.get("counter", 2)) - 1},
            effect_keys=frozenset({"counter"}),
            cost=1.0,
            can_rerun=False,
        )
        backup = ActionSpec(
            name="reusable_backup",
            preconditions={},
            effects=lambda ws: {"counter": int(ws.get("counter", 2)) - 1},
            effect_keys=frozenset({"counter"}),
            cost=10.0,  # deliberately more expensive
        )
        start = PlanningState.from_dict({"counter": 2})
        goal = GoalSpec(conditions={"counter": 0})

        result = plan(start, goal, [single_use, backup])

        assert result is not None
        # cheap_single fires at most once; the second decrement uses the
        # more expensive reusable_backup because cheap_single is locked out.
        assert result.action_names.count("cheap_single") == 1
        assert "reusable_backup" in result.action_names

    def test_can_rerun_false_used_once_when_sufficient(self) -> None:
        """When a single use of a can_rerun=False action satisfies the
        goal, the plan still includes it (the flag forbids reuse, not
        first use)."""
        actions = [
            _action("setup", eff={"a": True}),
            ActionSpec(
                name="finalize_once",
                preconditions={"a": True},
                effects={"goal": True},
                cost=1.0,
                can_rerun=False,
            ),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["setup", "finalize_once"]

    def test_can_rerun_default_is_true(self) -> None:
        """Backwards compatibility: actions without can_rerun set behave
        exactly as today (rerun allowed)."""
        action = ActionSpec(name="legacy", effects={"a": True})
        assert action.can_rerun is True


# ---------------------------------------------------------------------------
# read_only
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_read_only_default_is_false(self) -> None:
        action = ActionSpec(name="default", effects={"a": True})
        assert action.read_only is False

    def test_read_only_flag_preserved_through_serialization(self) -> None:
        """msgpack round-trip must preserve the flag for tooling that
        consumes it after a checkpoint reload."""
        action = ActionSpec(
            name="analyze_only",
            preconditions={"data_loaded": True},
            effects={"analysis_complete": True},
            cost=1.0,
            read_only=True,
        )
        serializer = LangGOAPSerializer()

        kind, payload = serializer.dumps_typed(action)
        restored = serializer.loads_typed((kind, payload))

        assert restored.name == "analyze_only"
        assert restored.read_only is True

    def test_read_only_does_not_affect_planning(self) -> None:
        """``read_only`` is informational; the planner must not treat
        read-only actions any differently from mutating ones."""
        actions = [
            _action("setup", eff={"a": True}),
            ActionSpec(
                name="ro_finalize",
                preconditions={"a": True},
                effects={"goal": True},
                cost=1.0,
                read_only=True,
            ),
        ]
        start = PlanningState.from_dict({})
        goal = GoalSpec(conditions={"goal": True})

        result = plan(start, goal, actions)

        assert result is not None
        assert result.action_names == ["setup", "ro_finalize"]


# ---------------------------------------------------------------------------
# Combined: both flags on the same action
# ---------------------------------------------------------------------------


class TestBothFlags:
    def test_can_rerun_false_and_read_only_true_compose(self) -> None:
        action = ActionSpec(
            name="single_audit",
            preconditions={"data_loaded": True},
            effects={"audit_logged": True},
            cost=1.0,
            can_rerun=False,
            read_only=True,
        )
        assert action.can_rerun is False
        assert action.read_only is True

    def test_serialization_preserves_both_flags(self) -> None:
        action = ActionSpec(
            name="single_audit",
            preconditions={"data_loaded": True},
            effects={"audit_logged": True},
            cost=1.0,
            can_rerun=False,
            read_only=True,
        )
        serializer = LangGOAPSerializer()
        kind, payload = serializer.dumps_typed(action)
        restored = serializer.loads_typed((kind, payload))
        assert restored.can_rerun is False
        assert restored.read_only is True
