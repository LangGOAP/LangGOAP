r"""Temporal Match Cellar — Tier 3 tutorial showcasing durative actions,
CSP temporal scheduling, parallelization, and Gantt visualization.

Translates the classic *match cellar* temporal planning example from
unified-planning/03 into LangGOAP.  In the original problem an agent
must mend a set of fuses in a dark cellar, and mending requires light
from a match that burns for a fixed time — introducing a temporal
overlap constraint that is the hallmark of durative-action planning.

LangGOAP's CSP phase cannot express the *over-all* "light must be on
during mend" constraint directly (the scheduler uses strict
end-before-start precedence, not interval containment).  The domain
is therefore recast into a form LangGOAP can represent exactly:

- Lighting a match takes 6 seconds and *leaves behind* a
  ``light_N_ready`` flag.  This models "the match has been lit and
  there is still enough light to start mending".
- Mending a fuse takes 5 seconds and *requires* the matching
  ``light_N_ready`` flag.  It also requires the previous fuse to be
  mended, which encodes the single-hand mutex chain.

The simplification makes the temporal story visible to LangGOAP's
scheduler: lighting matches is **independent work** that can run in
parallel, and mending fuses is **sequential work** because of the
hand mutex.  Every notebook assertion pins exactly what the pipeline
returns — the schedule is not approximate.

What this tutorial spotlights
-----------------------------

1. **Durative actions via ``ActionSpec.duration``** — every action
   carries a :class:`datetime.timedelta` duration that CP-SAT's
   temporal scheduler uses to build ``IntervalVar`` decision
   variables.

2. **Parallelization from the dependency graph** — three
   ``light_match`` actions have no precedence between them and the
   CP-SAT scheduler places all three in parallel at ``t=0``.  This
   saves 12 seconds of wall-clock time versus the naive serial plan.

3. **Mutex via precondition chains** — each ``mend_fuse_N`` has
   ``fuse_(N-1)_mended=True`` as a precondition, which translates
   into a dependency-graph edge that the scheduler honors by
   serializing the mends.  This is the *right* way to express a
   single-hand mutex in a LangGOAP action model: encode it in the
   preconditions, not in resource totals.

4. **Gantt visualization** — :func:`~langgoap.viz.render_ascii_gantt`
   renders the resulting schedule as an ASCII bar chart directly
   from :attr:`CSPMetadata.schedule`, which makes the
   parallel-then-sequential staircase immediately obvious.

5. **Fluent ``ConstraintBuilder`` twin** — the goal also has a
   builder-built variant and the integration test pins that the two
   produce structurally identical :class:`GoalSpec` instances.

GOAP modelling
--------------

**World state flags** (all ``False`` at start):

- ``light_1_ready`` / ``light_2_ready`` / ``light_3_ready`` — the
  matching match has been lit and the work window is open.
- ``match_1_used`` / ``match_2_used`` / ``match_3_used`` — consumable
  resource flag (prevents relighting the same match).
- ``fuse_1_mended`` / ``fuse_2_mended`` / ``fuse_3_mended`` — the
  target goal facts.

**Action catalog** (6 actions):

+---------------+--------------------------------+----------------------+-----+------+
| Action        | Pre                            | Effects              | Cst | Dur  |
+===============+================================+======================+=====+======+
| light_match_1 | (none)                         | light_1_ready,       | 1.0 | 6s   |
|               |                                | match_1_used         |     |      |
+---------------+--------------------------------+----------------------+-----+------+
| light_match_2 | (none)                         | light_2_ready,       | 1.0 | 6s   |
|               |                                | match_2_used         |     |      |
+---------------+--------------------------------+----------------------+-----+------+
| light_match_3 | (none)                         | light_3_ready,       | 1.0 | 6s   |
|               |                                | match_3_used         |     |      |
+---------------+--------------------------------+----------------------+-----+------+
| mend_fuse_1   | light_1_ready                  | fuse_1_mended        | 1.0 | 5s   |
+---------------+--------------------------------+----------------------+-----+------+
| mend_fuse_2   | light_2_ready, fuse_1_mended   | fuse_2_mended        | 1.0 | 5s   |
+---------------+--------------------------------+----------------------+-----+------+
| mend_fuse_3   | light_3_ready, fuse_2_mended   | fuse_3_mended        | 1.0 | 5s   |
+---------------+--------------------------------+----------------------+-----+------+

Every action also carries a ``duration_seconds`` resource equal to
its :class:`~datetime.timedelta` duration in seconds, which the goal
then attaches a ``MINIMIZE`` objective to.  The objective exists
solely to route the goal through CSP (pure A\\* has no schedule
information) — the real temporal win comes from the schedule the
scheduler computes, not from the objective itself.

Expected plan and schedule
--------------------------

A\\* finds the 6-action plan with total cost ``6.0``.  The CSP
scheduler then places it on a time axis:

- ``light_match_1``, ``light_match_2``, ``light_match_3`` — all
  three run in **parallel** from ``t=0`` to ``t=6``.
- ``mend_fuse_1`` — starts at ``t=6`` (waits for ``light_1_ready``),
  ends at ``t=11``.
- ``mend_fuse_2`` — starts at ``t=11`` (waits for
  ``fuse_1_mended``; ``light_2_ready`` has been available since
  ``t=6``), ends at ``t=16``.
- ``mend_fuse_3`` — starts at ``t=16`` (waits for
  ``fuse_2_mended``), ends at ``t=21``.

**Makespan = 21 seconds.**  The naive serial execution would take
``3*6 + 3*5 = 33`` seconds — parallel lights save **12 seconds** of
wall-clock time.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgoap import ActionSpec, GoalSpec
from langgoap.constraints import ConstraintBuilder
from langgoap.goals import ObjectiveDirection

# ---------------------------------------------------------------------------
# Domain constants — used by both the action factory and the integration
# test so that any change to the numbers shows up in a single place.
# ---------------------------------------------------------------------------

LIGHT_DURATION_SECONDS = 6.0
MEND_DURATION_SECONDS = 5.0
NUM_PAIRS = 3


# ---------------------------------------------------------------------------
# Execute helpers — deterministic, world-state-only mutation
# ---------------------------------------------------------------------------


def _make_execute(effects: dict[str, Any]) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        del ws  # effects are static
        return dict(effects)

    return execute


# ---------------------------------------------------------------------------
# Action catalog factory
# ---------------------------------------------------------------------------


def match_cellar_actions(num_pairs: int = NUM_PAIRS) -> list[ActionSpec]:
    """Return the match-cellar action catalog (``2 * num_pairs`` actions).

    Args:
        num_pairs: Number of (match, fuse) pairs.  Defaults to 3,
            matching the classic unified-planning example.  The
            integration test pins behavior for ``num_pairs=3`` and
            ``num_pairs=2``.
    """
    actions: list[ActionSpec] = []

    # Lighting: independent, can run in parallel.
    for i in range(1, num_pairs + 1):
        effects = {f"light_{i}_ready": True, f"match_{i}_used": True}
        actions.append(
            ActionSpec(
                name=f"light_match_{i}",
                preconditions={},
                effects=effects,
                cost=1.0,
                resources={"duration_seconds": LIGHT_DURATION_SECONDS},
                duration=timedelta(seconds=LIGHT_DURATION_SECONDS),
                execute=_make_execute(effects),
            )
        )

    # Mending: chained via prior fuse_(i-1)_mended for hand mutex.
    for i in range(1, num_pairs + 1):
        pre: dict[str, Any] = {f"light_{i}_ready": True}
        if i > 1:
            pre[f"fuse_{i - 1}_mended"] = True
        eff = {f"fuse_{i}_mended": True}
        actions.append(
            ActionSpec(
                name=f"mend_fuse_{i}",
                preconditions=pre,
                effects=eff,
                cost=1.0,
                resources={"duration_seconds": MEND_DURATION_SECONDS},
                duration=timedelta(seconds=MEND_DURATION_SECONDS),
                execute=_make_execute(eff),
            )
        )

    return actions


def match_cellar_start(num_pairs: int = NUM_PAIRS) -> dict[str, Any]:
    """Clean-slate world state — no matches used, no fuses mended."""
    state: dict[str, Any] = {}
    for i in range(1, num_pairs + 1):
        state[f"light_{i}_ready"] = False
        state[f"match_{i}_used"] = False
        state[f"fuse_{i}_mended"] = False
    return state


# ---------------------------------------------------------------------------
# Goal factories — hand-rolled and fluent-builder variants
# ---------------------------------------------------------------------------


def match_cellar_goal(num_pairs: int = NUM_PAIRS) -> GoalSpec:
    """Mend every fuse, with ``duration_seconds`` MINIMIZE routing the
    goal through CSP so the scheduler runs and ``plan.metadata.csp`` is
    populated with a full schedule and a ``HardSoftScore``.
    """
    conditions = {f"fuse_{i}_mended": True for i in range(1, num_pairs + 1)}
    return GoalSpec(
        conditions=conditions,
        objectives={"duration_seconds": ObjectiveDirection.MINIMIZE},
    )


def match_cellar_goal_fluent(num_pairs: int = NUM_PAIRS) -> GoalSpec:
    """Fluent-builder twin of :func:`match_cellar_goal`.

    Exercises :class:`~langgoap.constraints.ConstraintBuilder` as a
    ``ConstraintProvider`` analogue.  The integration test pins that
    this produces a :class:`GoalSpec` structurally identical to the
    hand-rolled form.
    """
    conditions = {f"fuse_{i}_mended": True for i in range(1, num_pairs + 1)}
    output = ConstraintBuilder.build(
        ConstraintBuilder.for_plan()
        .sum_resource("duration_seconds")
        .minimize()
        .as_objective("duration_seconds"),
    )
    return GoalSpec.from_builder(conditions=conditions, builder_output=output)


__all__ = [
    "LIGHT_DURATION_SECONDS",
    "MEND_DURATION_SECONDS",
    "NUM_PAIRS",
    "match_cellar_actions",
    "match_cellar_goal",
    "match_cellar_goal_fluent",
    "match_cellar_start",
]
