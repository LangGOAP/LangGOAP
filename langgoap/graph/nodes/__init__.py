"""LangGraph nodes for GOAP planning, execution, and observation.

These nodes form the core GOAP loop:
  planner \u2192 executor \u2192 observer \u2192 (executor | planner | END)

This module re-exports the same names that the previous single-file
``langgoap.graph.nodes`` module exposed so existing imports keep
working unchanged.  The implementations live in:

* :mod:`langgoap.graph.nodes._helpers` \u2014 cross-module helpers
  (``_safe_tracer_call``, ``_planning_keys``, ``_is_better_plan``).
* :mod:`langgoap.graph.nodes._execution` \u2014 executor result
  builders (``_apply_result``, ``_build_*``, ``_is_approved``,
  ``_check_human_approval``, ``_prepare_execution``).
* :mod:`langgoap.graph.nodes.planner` \u2014 :class:`GoapPlanner`.
* :mod:`langgoap.graph.nodes.executor` \u2014 :class:`GoapExecutor` and
  :func:`async_execute_action`.
* :mod:`langgoap.graph.nodes.parallel_executor` \u2014
  :class:`ParallelGoapExecutor` and :func:`_find_parallel_group`.
* :mod:`langgoap.graph.nodes.observer` \u2014 :class:`GoapObserver` and
  its routing helpers.
"""

from __future__ import annotations

# Internal helpers re-exported because tests and a few internal callers
# (e.g. ``test_graph_nodes.py``) import them directly.
from langgoap.graph.nodes._execution import (
    _apply_result,
    _build_failure,
    _build_guard_block,
    _build_success,
    _check_human_approval,
    _is_approved,
    _prepare_execution,
)
from langgoap.graph.nodes._helpers import (
    _is_better_plan,
    _planning_keys,
    _safe_tracer_acall,
    _safe_tracer_call,
)

# Public API \u2014 the names users (and tests) import.
from langgoap.graph.nodes.executor import GoapExecutor, async_execute_action
from langgoap.graph.nodes.observer import (
    GoapObserver,
    _get_last_failed_action,
    _get_max_retries,
    _is_goal_satisfied,
    _outcome_from_status,
    _resolve_active_goal,
    _route_multi_goal,
    _route_replan_limit,
)
from langgoap.graph.nodes.parallel_executor import (
    ParallelGoapExecutor,
    _find_parallel_group,
)
from langgoap.graph.nodes.planner import GoapPlanner

__all__ = [
    "GoapPlanner",
    "GoapExecutor",
    "ParallelGoapExecutor",
    "GoapObserver",
    "async_execute_action",
    "_find_parallel_group",
    "_is_better_plan",
    "_planning_keys",
    "_safe_tracer_call",
    "_safe_tracer_acall",
    "_apply_result",
    "_build_success",
    "_build_failure",
    "_build_guard_block",
    "_is_approved",
    "_check_human_approval",
    "_prepare_execution",
    "_get_last_failed_action",
    "_get_max_retries",
    "_is_goal_satisfied",
    "_outcome_from_status",
    "_resolve_active_goal",
    "_route_multi_goal",
    "_route_replan_limit",
]
