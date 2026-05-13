"""LangGOAP: Goal-Oriented Action Planning framework for LangGraph."""

import logging

from langgoap._version import __version__


def set_log_level(level: str | int) -> None:
    """Set the log level for all langgoap loggers.

    Convenience function replacing the common 2-line pattern::

        import logging
        logging.getLogger("langgoap").setLevel(logging.ERROR)

    With::

        import langgoap
        langgoap.set_log_level("ERROR")

    Args:
        level: Log level as a string (e.g. ``"ERROR"``, ``"DEBUG"``)
            or an ``int`` (e.g. ``logging.WARNING``).
    """
    logging.getLogger("langgoap").setLevel(level)


from langgoap.actions import ActionSpec, EffectFunction, GoapAction, goap_action
from langgoap.callbacks import DEFAULT_COST_PER_1K_TOKENS, CostAccumulator
from langgoap.conditions import (
    AsyncConditionResolver,
    ConditionResolver,
    ConditionStatus,
    FunctionalConditionResolver,
    PromptConditionResolver,
    aresolve_conditions,
    resolve_conditions,
)
from langgoap.constraints import (
    BuilderOutput,
    ChainOutput,
    ConstraintBuilder,
    ConstraintChain,
)
from langgoap.goals import (
    ConstraintSpec,
    Goal,
    GoalPolicy,
    GoalSpec,
    MultiGoal,
    SoftGoal,
)
from langgoap.graph.builder import GoapGraph
from langgoap.graph.nodes import (
    GoapExecutor,
    GoapObserver,
    GoapPlanner,
    ParallelGoapExecutor,
)
from langgoap.graph.state import ActionResult, GoapState, successful_action_names
from langgoap.guards import (
    ActionGuard,
    AsyncActionGuard,
    FunctionalGuard,
    GuardResult,
    GuardSeverity,
    has_blocking_failure,
    run_guards_async,
    run_guards_sync,
)
from langgoap.history import (
    ExecutionRecord,
    StoreExecutionHistory,
    compute_goal_hash,
)
from langgoap.integrations import (
    GoapSubgraph,
    add_goap_subgraph,
    create_goap_agent,
    create_goap_subagent,
    create_goap_tool,
    format_goap_result,
    goapify_tool,
    scaffold_deployment,
)
from langgoap.interpreter import (
    GoalInterpreter,
    InterpretedConstraint,
    InterpretedGoal,
    InterpretedObjective,
)
from langgoap.planner.astar import plan
from langgoap.planner.csp import (
    CSPMetadata,
    CSPStatus,
    ResourceUsage,
    ScheduleEntry,
    pareto_plans,
)
from langgoap.planner.explain import (
    InfeasibilityExplanation,
    NoPlanExplanation,
    ResourceShortfall,
    explain_infeasibility,
    explain_no_plan,
)
from langgoap.planner.mcts import (
    MCTSExploration,
    MCTSReuseConfig,
    MCTSStrategy,
    MCTSTracingConfig,
)
from langgoap.planner.pipeline import plan as pipeline_plan
from langgoap.planner.repair import RepairStrategy
from langgoap.planner.router import (
    ProblemFeatures,
    RuleBasedClassifier,
    StrategyClassifier,
    StrategyRouter,
    extract_features,
)
from langgoap.planner.strategy import (
    AnytimePlanningStrategy,
    AStarStrategy,
    CSPRefinementStrategy,
    LazyDecompositionStrategy,
    PlanningStrategy,
    TwoPhasePipelineStrategy,
)
from langgoap.planner.transitions import (
    DeterministicTransitionModel,
    DivergencePolicy,
    TransitionModel,
)
from langgoap.planner.types import Plan, PlanMetadata
from langgoap.planner.utility import (
    NIRVANA_NAME,
    NirvanaGoal,
    UtilityStrategy,
    is_nirvana,
)
from langgoap.qos import DEFAULT_RETRY_POLICY, FIRE_ONCE, ActionQos
from langgoap.reflexion import Reflection, ReflexionTracer
from langgoap.score import BendableScore, HardSoftScore, Score, SimpleScore
from langgoap.sensors import (
    AsyncSensor,
    FunctionalSensor,
    Sensor,
    run_sensors_async,
    run_sensors_sync,
)
from langgoap.serde import (
    LANGGOAP_ALLOWED_MSGPACK_TYPES,
    LangGOAPSerializer,
    install_langgoap_serde,
)
from langgoap.state import PlanningState, infer_start_state
from langgoap.stuck import (
    FunctionalStuckHandler,
    MulticastStuckHandler,
    StuckHandler,
    StuckHandlerResult,
    StuckHandlingResultCode,
)
from langgoap.termination import (
    AllOfPolicy,
    EarlyTermination,
    FirstOfPolicy,
    MaxActionsPolicy,
    MaxCostPolicy,
    MaxLLMCallsPolicy,
    MaxTokensPolicy,
    MaxWallClockPolicy,
    OnStuckPolicy,
    TerminationPolicy,
)
from langgoap.tracing import (
    LangSmithTracer,
    LoggingTracer,
    MultiTracer,
    NullTracer,
    PlanningTracer,
)
from langgoap.types import (
    CostFunction,
    Maximize,
    Minimize,
    ObjectiveDirection,
    ReplanStrategy,
)
from langgoap.viz import (
    render_ascii,
    render_ascii_gantt,
    render_dot,
    render_mermaid,
    render_mermaid_gantt,
    visualize,
)

__all__ = [
    "__version__",
    "set_log_level",
    # Actions
    "ActionSpec",
    "EffectFunction",
    "GoapAction",
    "goap_action",
    # Conditions (three-valued logic + LLM-evaluated conditions)
    "AsyncConditionResolver",
    "ConditionResolver",
    "ConditionStatus",
    "FunctionalConditionResolver",
    "PromptConditionResolver",
    "aresolve_conditions",
    "resolve_conditions",
    # Goals
    "ConstraintSpec",
    "Goal",
    "GoalSpec",
    "GoalPolicy",
    "MultiGoal",
    "SoftGoal",
    # Constraint builder
    "BuilderOutput",
    "ChainOutput",
    "ConstraintBuilder",
    "ConstraintChain",
    # Integrations (low-code LangGraph on-ramp)
    "create_goap_agent",
    "create_goap_tool",
    "create_goap_subagent",
    "format_goap_result",
    "goapify_tool",
    "GoapSubgraph",
    "add_goap_subgraph",
    "scaffold_deployment",
    # Interpreter
    "GoalInterpreter",
    "InterpretedConstraint",
    "InterpretedGoal",
    "InterpretedObjective",
    # Graph
    "GoapGraph",
    "GoapPlanner",
    "GoapExecutor",
    "GoapObserver",
    "ParallelGoapExecutor",
    "GoapState",
    "ActionResult",
    "successful_action_names",
    # Planning
    "plan",
    "pipeline_plan",
    "Plan",
    "PlanMetadata",
    "PlanningState",
    "infer_start_state",
    # Planning strategies
    "PlanningStrategy",
    "AStarStrategy",
    "AnytimePlanningStrategy",
    "CSPRefinementStrategy",
    "LazyDecompositionStrategy",
    "MCTSStrategy",
    "MCTSExploration",
    "MCTSReuseConfig",
    "MCTSTracingConfig",
    "RepairStrategy",
    "TwoPhasePipelineStrategy",
    # Utility (greedy) planner
    "NIRVANA_NAME",
    "NirvanaGoal",
    "UtilityStrategy",
    "is_nirvana",
    # Router
    "ProblemFeatures",
    "RuleBasedClassifier",
    "StrategyClassifier",
    "StrategyRouter",
    "extract_features",
    # Transition models
    "DeterministicTransitionModel",
    "DivergencePolicy",
    "TransitionModel",
    # Guards
    "ActionGuard",
    "AsyncActionGuard",
    "FunctionalGuard",
    "GuardResult",
    "GuardSeverity",
    "has_blocking_failure",
    "run_guards_async",
    "run_guards_sync",
    # LLM cost / token accounting
    "CostAccumulator",
    "DEFAULT_COST_PER_1K_TOKENS",
    # Action QoS / retry policy
    "ActionQos",
    "DEFAULT_RETRY_POLICY",
    "FIRE_ONCE",
    # Stuck-handler protocol
    "FunctionalStuckHandler",
    "MulticastStuckHandler",
    "StuckHandler",
    "StuckHandlerResult",
    "StuckHandlingResultCode",
    # Early-termination policies
    "AllOfPolicy",
    "EarlyTermination",
    "FirstOfPolicy",
    "MaxActionsPolicy",
    "MaxCostPolicy",
    "MaxLLMCallsPolicy",
    "MaxTokensPolicy",
    "MaxWallClockPolicy",
    "OnStuckPolicy",
    "TerminationPolicy",
    # Scores
    "Score",
    "SimpleScore",
    "HardSoftScore",
    "BendableScore",
    # CSP
    "CSPMetadata",
    "CSPStatus",
    "ResourceUsage",
    "ScheduleEntry",
    "pareto_plans",
    # Plan explanation
    "InfeasibilityExplanation",
    "NoPlanExplanation",
    "ResourceShortfall",
    "explain_infeasibility",
    "explain_no_plan",
    # Types
    "CostFunction",
    "Maximize",
    "Minimize",
    "ObjectiveDirection",
    "ReplanStrategy",
    # Visualization
    "render_mermaid",
    "render_mermaid_gantt",
    "render_dot",
    "render_ascii",
    "render_ascii_gantt",
    "visualize",
    # Tracing
    "PlanningTracer",
    "NullTracer",
    "LoggingTracer",
    "MultiTracer",
    "LangSmithTracer",
    # Execution history
    "ExecutionRecord",
    "StoreExecutionHistory",
    "compute_goal_hash",
    # Sensors
    "Sensor",
    "AsyncSensor",
    "FunctionalSensor",
    "run_sensors_sync",
    "run_sensors_async",
    # Reflexion
    "Reflection",
    "ReflexionTracer",
    # Serde
    "LANGGOAP_ALLOWED_MSGPACK_TYPES",
    "LangGOAPSerializer",
    "install_langgoap_serde",
]
