"""Integration tests for the ``StuckHandler`` protocol.

Mirrors Embabel's contract on
``research/repos/embabel-agent/embabel-agent-api/src/test/kotlin/
com/embabel/agent/api/common/MulticastStuckHandlerTest.kt``:

* ``StuckHandler.handle_stuck(state, reason) -> StuckHandlerResult`` —
  called when the planner cannot find a plan.
* ``StuckHandlerResult.code`` is ``REPLAN`` (handler resolved the
  situation; planner should retry) or ``NO_RESOLUTION`` (handler
  could not help; planner moves on).
* ``MulticastStuckHandler(handlers)`` — tries each in order, returns
  the first ``REPLAN`` result.  If every handler returns
  ``NO_RESOLUTION``, returns an aggregated ``NO_RESOLUTION``.

LangGOAP additions:

* The handler returns optional ``state_updates`` (merged into world
  state on REPLAN) and an optional ``new_goal`` (replaces the goal on
  REPLAN), so the planner can attempt a real second plan in the same
  invocation.
* ``GoapPlanner`` consults the configured handlers when a planning
  attempt fails, looping at most ``max_stuck_iterations`` times before
  surfacing the original ``no_plan`` result.
"""

from __future__ import annotations

from typing import Any

from langgoap.goals import GoalPolicy, GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.stuck import (
    FunctionalStuckHandler,
    MulticastStuckHandler,
    StuckHandler,
    StuckHandlerResult,
    StuckHandlingResultCode,
)
from tests.conftest import make_action as _action


def _actions() -> list[Any]:
    """Two actions: the second only fires once a 'license_acquired' flag is set."""
    return [
        _action("setup", eff={"data_loaded": True}),
        _action(
            "license_protected",
            pre={"data_loaded": True, "license_acquired": True},
            eff={"published": True},
        ),
    ]


def _publish_goal() -> GoalSpec:
    return GoalSpec(conditions={"published": True}, policy=GoalPolicy(max_replans=0))


# ---------------------------------------------------------------------------
# StuckHandlerResult
# ---------------------------------------------------------------------------


class TestStuckHandlerResult:
    def test_replan_with_state_updates_resolves_planning(self) -> None:
        """A handler that fills in the missing ``license_acquired`` flag
        unblocks planning on the immediate retry."""

        license_handler = FunctionalStuckHandler(
            name="acquire_license",
            fn=lambda state, reason: StuckHandlerResult.replan(
                handler_name="acquire_license",
                message="license acquired out-of-band",
                state_updates={"license_acquired": True},
            ),
        )
        graph = GoapGraph(_actions(), stuck_handlers=[license_handler])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},  # no license_acquired — first plan will fail
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["published"] is True
        assert result["world_state"]["license_acquired"] is True

    def test_no_resolution_surfaces_no_plan(self) -> None:
        """If every handler returns NO_RESOLUTION the planner emits the
        same no_plan response as before stuck handling existed."""

        no_op_handler = FunctionalStuckHandler(
            name="no_op",
            fn=lambda state, reason: StuckHandlerResult.no_resolution(
                handler_name="no_op", message="cannot help"
            ),
        )
        graph = GoapGraph(_actions(), stuck_handlers=[no_op_handler])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "no_plan"

    def test_replan_with_new_goal_relaxes_target(self) -> None:
        """A handler can return a relaxed ``new_goal`` that the planner
        substitutes into the next planning attempt."""

        relax_handler = FunctionalStuckHandler(
            name="relax_to_data_loaded",
            fn=lambda state, reason: StuckHandlerResult.replan(
                handler_name="relax_to_data_loaded",
                message="dropped publish requirement",
                new_goal=GoalSpec(
                    conditions={"data_loaded": True},
                    policy=GoalPolicy(max_replans=0),
                ),
            ),
        )
        graph = GoapGraph(_actions(), stuck_handlers=[relax_handler])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["data_loaded"] is True
        # Original publish goal was relaxed away — published flag never set.
        assert "published" not in result["world_state"] or not result[
            "world_state"
        ].get("published")


# ---------------------------------------------------------------------------
# MulticastStuckHandler — semantics from Embabel
# ---------------------------------------------------------------------------


class TestMulticastStuckHandler:
    def test_returns_first_replan_result_short_circuiting(self) -> None:
        """When the first handler returns REPLAN, downstream handlers must
        not be consulted (mirrors Embabel test
        ``should return first successful resolution``)."""
        h2_called = [False]

        h1 = FunctionalStuckHandler(
            name="h1",
            fn=lambda s, r: StuckHandlerResult.replan(
                handler_name="h1",
                message="h1 fix",
                state_updates={"license_acquired": True},
            ),
        )

        def h2_fn(s: Any, r: Any) -> StuckHandlerResult:
            h2_called[0] = True
            return StuckHandlerResult.no_resolution(
                handler_name="h2", message="should not be called"
            )

        h2 = FunctionalStuckHandler(name="h2", fn=h2_fn)

        multicast = MulticastStuckHandler([h1, h2])
        graph = GoapGraph(_actions(), stuck_handlers=[multicast])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert h2_called[0] is False

    def test_tries_handlers_in_order_until_replan(self) -> None:
        """Mirrors Embabel test ``should try all handlers in order until success``."""
        call_order: list[str] = []

        def make_handler(name: str, decision: StuckHandlerResult) -> StuckHandler:
            def fn(s: Any, r: Any) -> StuckHandlerResult:
                call_order.append(name)
                return decision

            return FunctionalStuckHandler(name=name, fn=fn)

        h1 = make_handler(
            "h1",
            StuckHandlerResult.no_resolution(handler_name="h1", message="no"),
        )
        h2 = make_handler(
            "h2",
            StuckHandlerResult.no_resolution(handler_name="h2", message="also no"),
        )
        h3 = make_handler(
            "h3",
            StuckHandlerResult.replan(
                handler_name="h3",
                message="finally yes",
                state_updates={"license_acquired": True},
            ),
        )

        multicast = MulticastStuckHandler([h1, h2, h3])
        graph = GoapGraph(_actions(), stuck_handlers=[multicast])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "goal_achieved"
        assert call_order == ["h1", "h2", "h3"]

    def test_aggregates_no_resolution_when_all_handlers_fail(self) -> None:
        """Mirrors Embabel test
        ``should return NO_RESOLUTION when all handlers fail``."""
        h1 = FunctionalStuckHandler(
            name="h1",
            fn=lambda s, r: StuckHandlerResult.no_resolution(
                handler_name="h1", message="no"
            ),
        )
        h2 = FunctionalStuckHandler(
            name="h2",
            fn=lambda s, r: StuckHandlerResult.no_resolution(
                handler_name="h2", message="also no"
            ),
        )

        multicast = MulticastStuckHandler([h1, h2])
        result = multicast.handle_stuck(state={}, reason=None)

        assert result.code is StuckHandlingResultCode.NO_RESOLUTION
        assert "No stuck handler could resolve" in result.message
        assert "h1" in result.message
        assert "h2" in result.message
        assert result.handler_name is None

    def test_empty_list_returns_no_resolution(self) -> None:
        """Mirrors Embabel test ``should handle empty handlers list``."""
        multicast = MulticastStuckHandler([])
        result = multicast.handle_stuck(state={}, reason=None)
        assert result.code is StuckHandlingResultCode.NO_RESOLUTION
        assert result.handler_name is None

    def test_handler_exception_is_caught_as_no_resolution(self) -> None:
        """LangGOAP-specific: an exception in one handler must not crash
        the multicast — observability/resilience invariant matching the
        tracer-never-raises rule."""

        def boom(s: Any, r: Any) -> StuckHandlerResult:
            raise RuntimeError("oops")

        h1 = FunctionalStuckHandler(name="exploding", fn=boom)
        h2 = FunctionalStuckHandler(
            name="recover",
            fn=lambda s, r: StuckHandlerResult.replan(
                handler_name="recover",
                message="recovered after sibling crash",
                state_updates={"license_acquired": True},
            ),
        )

        multicast = MulticastStuckHandler([h1, h2])
        graph = GoapGraph(_actions(), stuck_handlers=[multicast])

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# Bounded retry budget
# ---------------------------------------------------------------------------


class TestStuckIterationBudget:
    def test_max_stuck_iterations_caps_retries(self) -> None:
        """A handler that returns REPLAN with no useful updates cannot
        cause an infinite loop — ``max_stuck_iterations`` must apply."""
        attempts = [0]

        def always_replan_no_op(s: Any, r: Any) -> StuckHandlerResult:
            attempts[0] += 1
            return StuckHandlerResult.replan(
                handler_name="no_op_replan",
                message="claims to have fixed nothing",
                # no state_updates → next plan will also fail
            )

        handler = FunctionalStuckHandler(name="no_op_replan", fn=always_replan_no_op)
        graph = GoapGraph(
            _actions(),
            stuck_handlers=[handler],
            max_stuck_iterations=3,
        )

        result = graph.invoke(
            goal=_publish_goal(),
            world_state={},
        )

        assert result["status"] == "no_plan"
        # Initial planning + 3 stuck retries = 3 handler invocations
        # (handler is consulted at most max_stuck_iterations times).
        assert attempts[0] == 3
