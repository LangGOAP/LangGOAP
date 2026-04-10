# OptaPlanner ↔ LangGoap Concept Mapping

LangGoap draws on OptaPlanner's vocabulary but does **not** reimplement
OptaPlanner.  OR-Tools CP-SAT already provides branch-and-bound search,
propagation, no-good learning, restarts, and the full family of moves
(change, swap, list-change) that OptaPlanner exposes as Java classes.
Reimplementing those constructs on top of CP-SAT would be strictly
worse: slower, less tested, and with no gain in expressiveness.

Instead, LangGoap exposes a **Python-facing hierarchy** that makes the
OptaPlanner concepts legible to users while leaving the heavy lifting
to CP-SAT.  This document lists the mapping so OptaPlanner users can
find their bearings quickly.

## Concept table

| OptaPlanner concept         | LangGoap equivalent                          | Location                              |
| --------------------------- | -------------------------------------------- | ------------------------------------- |
| `Solver`                    | `GoapGraph` (the compiled LangGraph)         | `langgoap.graph.builder`              |
| `SolverFactory`             | `GoapGraph.__init__` + `.compile()`          | `langgoap.graph.builder`              |
| `Phase`                     | `PlanningStrategy` (Protocol)                | `langgoap.planner.strategy`           |
| Construction heuristic      | `AStarStrategy`                              | `langgoap.planner.strategy`           |
| Local search phase          | `CSPRefinementStrategy`                      | `langgoap.planner.strategy`           |
| Composite solver            | `TwoPhasePipelineStrategy` (the default)     | `langgoap.planner.strategy`           |
| `Move` / `ChangeMove` / …   | CP-SAT internal (branch-and-bound)           | `ortools.sat.python.cp_model`         |
| `MoveSelector` / tabu lists | CP-SAT internal (propagation, learning)      | `ortools.sat.python.cp_model`         |
| `Score`                     | `Score` base class                           | `langgoap.score`                      |
| `SimpleScore`               | `SimpleScore`                                | `langgoap.score`                      |
| `HardSoftScore`             | `HardSoftScore`                              | `langgoap.score`                      |
| `BendableScore`             | `BendableScore`                              | `langgoap.score`                      |
| `ScoreDirector`             | `_score_from_csp()` + `CSPMetadata`          | `langgoap.planner.pipeline`           |
| `ConstraintProvider`        | `ConstraintBuilder` (fluent)                 | `langgoap.constraints`                |
| `@PlanningEntity`           | `ActionSpec`                                 | `langgoap.actions`                    |
| `@PlanningVariable`         | `ActionSpec.effects` (implicit)              | `langgoap.actions`                    |
| `@PlanningSolution`         | `Plan` (immutable result)                    | `langgoap.planner.types`              |
| Constraint weights          | `ConstraintSpec.weight` + `.level`           | `langgoap.goals`                      |
| Hard constraint             | `ConstraintSpec(level="hard")`               | `langgoap.goals`                      |
| Soft constraint             | `ConstraintSpec(level="soft")`               | `langgoap.goals`                      |
| Best solution events        | `PlanningTracer` hooks (Epic 5)              | `langgoap.tracing`                    |
| `SolutionPartitioner`       | *Not supported in v0.1.0*                    | —                                     |
| Custom move factories       | Write a `PlanningStrategy` (see below)       | `langgoap.planner.strategy`           |

## Score hierarchy at a glance

```text
Score
├── SimpleScore(scalar: float)           — A*-only plans
├── HardSoftScore(hard, soft: float)      — CSP-refined plans
└── BendableScore(hard_levels, soft_levels: tuple[float, ...])
```

Sign convention:

- `hard` is `<= 0`.  A feasible plan has `hard == 0.0`.  Every hard
  constraint violation subtracts its weighted penalty from `hard`.
- `soft` has **no** sign restriction.  Minimize objectives and soft
  violations push `soft` negative; maximize objectives push it
  positive.  This matches OptaPlanner's `penalize(amount)` /
  `reward(amount)` convention.

Comparison is lexicographic within one concrete subclass; mixing
subclasses raises `TypeError` to force callers to stay within one
score type per planning run.

## Where CP-SAT replaces OptaPlanner machinery

OptaPlanner ships dozens of `Move` subclasses and selector strategies
(`SelectionOrder.RANDOM`, `SelectionFilter`, tabu, late acceptance,
simulated annealing, …).  CP-SAT subsumes **every one of these** via:

1. Branch-and-bound with learning — reimplements tabu / late
   acceptance / simulated annealing as a single coherent solver.
2. Constraint propagation — eliminates dead branches without
   requiring a `SelectionFilter`.
3. No-good learning — records why each failure happened and avoids
   repeating it, replacing OptaPlanner's tabu list.
4. Restart + random search — replaces the manual phase-switching
   OptaPlanner users write to escape local minima.
5. `IntervalVar` + `NoOverlap` — replaces OptaPlanner's
   `ChainedGraphMove` and `ListChangeMove` for scheduling problems.

Writing a `ChangeMove` class in Python would be strictly slower than
letting CP-SAT decide how to mutate the assignment.  LangGoap
therefore exposes CP-SAT at the `csp.py` level and does not duplicate
the moves as Python classes.

## Writing a custom planning strategy

The `PlanningStrategy` Protocol is the extension point for users who
want a solver different from A* or the two-phase pipeline.  Any class
with the signature below can be passed to `GoapPlanner(..., strategy=
...)`:

```python
from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.planner.strategy import PlanningStrategy
from langgoap.planner.types import Plan
from langgoap.state import PlanningState


class GreedyFirstActionStrategy:
    """Toy strategy that picks the first applicable action until the goal is met.

    Only useful as a teaching example — real usage should compose
    ``AStarStrategy`` with ``CSPRefinementStrategy`` via
    ``TwoPhasePipelineStrategy``.
    """

    def plan(
        self,
        start: PlanningState,
        goal: GoalSpec,
        actions: list[ActionSpec],
        *,
        blacklisted_actions: list[str] | None = None,
    ) -> Plan | None:
        from langgoap.planner.astar import plan as astar_plan

        # Delegate to A* for this toy example; the point is that any
        # class implementing .plan(...) is a valid strategy.
        return astar_plan(
            start, goal, actions, blacklisted_actions=blacklisted_actions
        )


assert isinstance(GreedyFirstActionStrategy(), PlanningStrategy)
```

Pass the strategy to the planner node:

```python
from langgoap.graph.nodes import GoapPlanner

planner = GoapPlanner(actions, strategy=GreedyFirstActionStrategy())
```

The default when no `strategy` is passed is auto-routing: pure A* for
plain goals, the `TwoPhasePipelineStrategy` when the goal has
constraints or objectives.

## What is **not** mapped

The following OptaPlanner features are explicitly out of scope for
LangGoap v0.1.0:

- **`SolutionPartitioner`** — the partitioned search is an OptaPlanner-
  specific mechanism for splitting a single problem across threads.
  CP-SAT parallelizes search internally; a user-facing partitioner is
  unnecessary.
- **Annotation-based API (`@PlanningEntity`, `@PlanningVariable`,
  `@PlanningSolution`)** — these rely on JVM reflection.  Python
  users get explicit `ActionSpec`, `GoalSpec`, and `Plan` dataclasses
  instead, which are easier to type-check and compose.
- **Benchmarker** — OptaPlanner ships a solution-quality benchmarker.
  Not yet in LangGoap; tracked for a post-v0.1.0 release.
- **Custom `Move` classes** — see above.  Write a `PlanningStrategy`
  if you need to inject non-CP-SAT search behavior.

## Further reading

- `langgoap/score.py` — Score hierarchy and sign convention.
- `langgoap/constraints.py` — Fluent `ConstraintBuilder` API.
- `langgoap/planner/strategy.py` — `PlanningStrategy` Protocol and
  the three concrete strategies.
- `langgoap/planner/csp.py` — Where CP-SAT actually runs.
- `langgoap/planner/pipeline.py` — `_score_from_csp()` formula and
  the A* → CSP refinement pipeline.
