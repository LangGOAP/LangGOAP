# API Reference

Every symbol listed below is part of the stable `v0.1.0` public surface
re-exported from the top-level `langgoap` package. Anything **not**
listed here is internal and subject to change without notice.

## Core

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.GoapGraph
   langgoap.ActionSpec
   langgoap.GoapAction
   langgoap.goap_action
   langgoap.GoalSpec
   langgoap.Goal
   langgoap.MultiGoal
   langgoap.ConstraintSpec
   langgoap.PlanningState
   langgoap.ActionResult
   langgoap.GoapState
```

## Planning

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.planner.astar.plan
   langgoap.pipeline_plan
   langgoap.Plan
   langgoap.PlanMetadata
   langgoap.PlanningStrategy
   langgoap.AStarStrategy
   langgoap.CSPRefinementStrategy
   langgoap.TwoPhasePipelineStrategy
```

## Constraints, scoring, and CSP

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.Score
   langgoap.SimpleScore
   langgoap.HardSoftScore
   langgoap.BendableScore
   langgoap.ConstraintBuilder
   langgoap.ConstraintChain
   langgoap.ChainOutput
   langgoap.BuilderOutput
   langgoap.CSPMetadata
   langgoap.CSPStatus
   langgoap.ResourceUsage
   langgoap.ScheduleEntry
```

## Integrations (low-code LangGraph on-ramp)

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.create_goap_agent
   langgoap.goapify_tool
   langgoap.GoapSubgraph
   langgoap.add_goap_subgraph
```

## Natural-language goal interpreter

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.GoalInterpreter
   langgoap.InterpretedGoal
   langgoap.InterpretedConstraint
   langgoap.InterpretedObjective
```

## Graph nodes

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.GoapPlanner
   langgoap.GoapExecutor
   langgoap.GoapObserver
```

## Tracing

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.PlanningTracer
   langgoap.NullTracer
   langgoap.LoggingTracer
   langgoap.MultiTracer
```

## Execution history

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.ExecutionRecord
   langgoap.StoreExecutionHistory
   langgoap.compute_goal_hash
```

## Types

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.CostFunction
   langgoap.Maximize
   langgoap.Minimize
   langgoap.ObjectiveDirection
   langgoap.ReplanStrategy
```

## Visualization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   langgoap.render_mermaid
   langgoap.render_mermaid_gantt
   langgoap.render_dot
   langgoap.render_ascii
   langgoap.render_ascii_gantt
   langgoap.visualize
```

```{toctree}
:maxdepth: 2
:hidden:
```
