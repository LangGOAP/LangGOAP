r"""Hierarchical product launch — Tier 3 ``MultiGoal`` sequential tutorial.

A SaaS product launch decomposes naturally into three stages that must
run in order: you cannot build what you have not specified, and you
cannot announce what you have not built.  LangGOAP expresses this with
a :class:`~langgoap.MultiGoal` in ``"sequential"`` mode — the observer
plans and executes sub-goal 0 to completion, then uses the resulting
world state as the starting state for sub-goal 1, and so on.

The tutorial spotlights four features at once:

1. **``MultiGoal`` sequential decomposition** — three sub-goals in one
   ``graph.invoke`` call; the observer chains the world state between
   stages with no user-side coordination.
2. **Per-stage A* planning** — each sub-goal plans independently, so
   the plan for stage 2 never mentions stage 3's actions (the planner
   literally never sees them during stage 2).
3. **State hand-off** — effects from stage 1 (``prd_approved=True``)
   become preconditions for stage 2's actions; effects from stage 2
   (``qa_passed=True``) satisfy stage 3's launch preconditions.
4. **Per-sub-goal accounting reset** — ``replan_count``,
   ``blacklisted_actions``, and ``action_failure_counts`` all reset at
   every sub-goal advance, so a transient failure while writing the
   PRD does not eat into the build stage's replan budget.

Everything is hermetic: the tutorial defines deterministic
``execute`` functions that mutate nothing outside the world state.
The notebook wraps these into a standard :class:`~langgoap.GoapGraph`
and drives it with :func:`product_launch_goal`.

Stage breakdown
---------------

- **Stage 1 — Discovery** (``prd_approved=True``).
  ``research_market`` unlocks ``write_prd``, which locks the product
  requirements doc.
- **Stage 2 — Build** (``qa_passed=True``).
  ``implement_features`` requires ``prd_approved`` (handed off from
  stage 1); ``qa_test`` gates the build on passing tests.
- **Stage 3 — Launch** (``launched=True``).
  ``prepare_marketing`` requires the QA gate; ``announce_launch``
  requires the marketing gate and cements ``launched=True``.

Action catalog
--------------

+---------------------+-----------------------------------+----------------------------------+------+
| Action              | Preconditions                     | Effects                          | Cost |
+=====================+===================================+==================================+======+
| research_market     | (none)                            | market_data=True                 | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| write_prd           | market_data=True                  | prd_approved=True                | 3.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| implement_features  | prd_approved=True                 | code_written=True                | 5.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| qa_test             | code_written=True                 | qa_passed=True                   | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| prepare_marketing   | qa_passed=True                    | marketing_ready=True             | 2.0  |
+---------------------+-----------------------------------+----------------------------------+------+
| announce_launch     | marketing_ready=True,             | launched=True                    | 1.0  |
|                     | qa_passed=True                    |                                  |      |
+---------------------+-----------------------------------+----------------------------------+------+
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec, GoalSpec, MultiGoal

# ---------------------------------------------------------------------------
# Execute functions — deterministic, world-state-only mutation
# ---------------------------------------------------------------------------


def _research_market(ws: dict[str, Any]) -> dict[str, Any]:
    del ws  # starting-state inspection not needed
    return {"market_data": True}


def _write_prd(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"prd_approved": True}


def _implement_features(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"code_written": True}


def _qa_test(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"qa_passed": True}


def _prepare_marketing(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"marketing_ready": True}


def _announce_launch(ws: dict[str, Any]) -> dict[str, Any]:
    del ws
    return {"launched": True}


# ---------------------------------------------------------------------------
# Action catalog factory
# ---------------------------------------------------------------------------


def product_launch_actions() -> list[ActionSpec]:
    """Return the six-action catalog that spans all three launch stages.

    Every action is deterministic and its ``execute`` function touches
    only the world state — no side effects, no I/O.  Costs reflect
    rough relative effort so ``Plan.total_cost`` is a meaningful
    readout: discovery 5, build 7, launch 3 (sum 15).
    """
    return [
        ActionSpec(
            name="research_market",
            preconditions={},
            effects={"market_data": True},
            cost=2.0,
            execute=_research_market,
        ),
        ActionSpec(
            name="write_prd",
            preconditions={"market_data": True},
            effects={"prd_approved": True},
            cost=3.0,
            execute=_write_prd,
        ),
        ActionSpec(
            name="implement_features",
            preconditions={"prd_approved": True},
            effects={"code_written": True},
            cost=5.0,
            execute=_implement_features,
        ),
        ActionSpec(
            name="qa_test",
            preconditions={"code_written": True},
            effects={"qa_passed": True},
            cost=2.0,
            execute=_qa_test,
        ),
        ActionSpec(
            name="prepare_marketing",
            preconditions={"qa_passed": True},
            effects={"marketing_ready": True},
            cost=2.0,
            execute=_prepare_marketing,
        ),
        ActionSpec(
            name="announce_launch",
            preconditions={"marketing_ready": True, "qa_passed": True},
            effects={"launched": True},
            cost=1.0,
            execute=_announce_launch,
        ),
    ]


# ---------------------------------------------------------------------------
# Starting state and goal
# ---------------------------------------------------------------------------


def product_launch_start() -> dict[str, Any]:
    """Clean-slate world state: no milestones reached yet.

    All six boolean flags are seeded ``False`` so the planner has an
    unambiguous starting point and the notebook can flip individual
    flags to ``True`` to demonstrate pre-satisfied sub-goal advance.
    """
    return {
        "market_data": False,
        "prd_approved": False,
        "code_written": False,
        "qa_passed": False,
        "marketing_ready": False,
        "launched": False,
    }


def product_launch_goal() -> MultiGoal:
    """Return the three-stage sequential ``MultiGoal`` for the launch.

    Sub-goal 0 locks the PRD, sub-goal 1 ships a QA-passing build,
    sub-goal 2 announces the launch.  Each sub-goal is a plain
    :class:`~langgoap.GoalSpec` with a single boolean condition — the
    observer chains them in order.
    """
    return MultiGoal(
        goals=(
            GoalSpec(conditions={"prd_approved": True}),  # Stage 1 — Discovery
            GoalSpec(conditions={"qa_passed": True}),  # Stage 2 — Build
            GoalSpec(conditions={"launched": True}),  # Stage 3 — Launch
        ),
        mode="sequential",
    )
