"""JSON / describe helpers shared by tracer implementations.

Internal module \u2014 not part of the public ``langgoap.tracing`` surface.
:class:`~langgoap.tracing.langsmith.LangSmithTracer` is currently the
only consumer; future LangChain-style tracers (OpenTelemetry, etc.) can
reuse the same helpers without taking a hard dependency on the
LangSmith adapter module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    """Timezone-aware ``datetime.utcnow`` replacement.

    LangSmith's HTTP wire format expects ISO-8601 timestamps with
    explicit UTC offsets; naive datetimes trigger a server-side
    warning that is easy to miss.  Every timestamp the tracer passes
    to ``client.create_run`` / ``client.update_run`` goes through
    this helper.
    """
    return datetime.now(timezone.utc)


def jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary values to JSON-friendly types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    # Frozen dataclasses, enums, etc. -- repr is always a safe last
    # resort.
    return repr(value)


def describe_goal(goal: Any) -> Any:
    """Render a goal for LangSmith inputs.

    :class:`~langgoap.goals.GoalSpec` and
    :class:`~langgoap.goals.MultiGoal` are frozen dataclasses; falling
    back to ``repr`` always produces *something* readable even for
    bare dicts or strings.
    """
    if goal is None:
        return None
    for attr in ("conditions", "goals"):
        value = getattr(goal, attr, None)
        if value is not None:
            return {attr: jsonable(value)}
    return jsonable(goal)


def describe_plan(plan: Any) -> Any:
    """Render a :class:`~langgoap.planner.types.Plan` as a plain dict."""
    if plan is None:
        return None
    action_names = getattr(plan, "action_names", None)
    total_cost = getattr(plan, "total_cost", None)
    if action_names is None and total_cost is None:
        return jsonable(plan)
    return {
        "action_names": list(action_names) if action_names is not None else [],
        "total_cost": total_cost,
    }


def describe_action(action: Any) -> Any:
    """Render an :class:`~langgoap.actions.ActionSpec` as a plain dict."""
    if action is None:
        return None
    name = getattr(action, "name", None)
    if name is None:
        return jsonable(action)
    return {
        "name": name,
        "preconditions": jsonable(getattr(action, "preconditions", None)),
        "effects": jsonable(getattr(action, "effects", None)),
        "cost": getattr(action, "cost", None),
    }


def describe_result(result: Any) -> Any:
    """Render an ``ActionResult`` as a plain dict."""
    if result is None:
        return None
    fields = ("action_name", "success", "state_after", "error")
    payload = {name: jsonable(getattr(result, name, None)) for name in fields}
    if all(v is None for v in payload.values()):
        return jsonable(result)
    return payload


__all__ = [
    "utc_now",
    "jsonable",
    "describe_goal",
    "describe_plan",
    "describe_action",
    "describe_result",
]
