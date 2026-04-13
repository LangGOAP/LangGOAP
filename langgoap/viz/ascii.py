"""ASCII-art renderers for Plan objects.

These renderers never touch the terminal directly; they return plain
strings so callers can log, pipe to files, or embed them in notebooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgoap.planner.csp import build_dependency_graph

if TYPE_CHECKING:
    from langgoap.planner.types import Plan


def render_ascii(
    plan: Plan,
    *,
    show_resources: bool = True,
    show_schedule: bool = True,
) -> str:
    """Render a Plan as a top-to-bottom ASCII tree.

    Each action is one line prefixed with its index.  Dependency arrows
    are drawn to the left.  When a schedule is attached, entries at the
    same start time are grouped into a ``┬── parallel:`` block.

    Args:
        plan: Plan to render.
        show_resources: If True, append a resource usage table.
        show_schedule: If True, render parallel groups from the schedule.

    Returns:
        ASCII string representation of the plan.
    """
    if len(plan) == 0:
        return "Plan: (empty — goal already satisfied)\n"

    lines: list[str] = [
        f"Plan ({len(plan)} step{'s' if len(plan) != 1 else ''}, "
        f"cost={plan.total_cost:g})"
    ]

    schedule = None
    if show_schedule and plan.metadata.csp is not None and plan.metadata.csp.schedule:
        schedule = plan.metadata.csp.schedule

    if schedule is not None and len(schedule) == len(plan.actions):
        by_start: dict[float, list[int]] = {}
        for i, entry in enumerate(schedule):
            by_start.setdefault(entry.start.total_seconds(), []).append(i)
        for start_t in sorted(by_start):
            indices = by_start[start_t]
            if len(indices) > 1:
                lines.append(f"├── t={start_t:g}s (parallel):")
                for i in indices:
                    dur = schedule[i].duration.total_seconds()
                    lines.append(
                        f"│     • [{i}] {plan.actions[i].name}  " f"duration={dur:g}s"
                    )
            else:
                i = indices[0]
                dur = schedule[i].duration.total_seconds()
                dur_str = f"  duration={dur:g}s" if dur > 0 else ""
                lines.append(
                    f"├── t={start_t:g}s  [{i}] {plan.actions[i].name}{dur_str}"
                )
    else:
        deps = build_dependency_graph(plan.actions)
        for i, action in enumerate(plan.actions):
            dep_indices = deps[i]
            dep_str = f"  (after: {dep_indices})" if dep_indices else ""
            lines.append(f"├── [{i}] {action.name}{dep_str}")

    # Resources
    if (
        show_resources
        and plan.metadata.csp is not None
        and plan.metadata.csp.resource_usage
    ):
        lines.append("")
        lines.append("Resources:")
        for usage in plan.metadata.csp.resource_usage:
            bound = ""
            if usage.constraint_max is not None:
                bound = f" / {usage.constraint_max:g}"
            elif usage.constraint_min is not None:
                bound = f" >= {usage.constraint_min:g}"
            marker = "OK" if usage.satisfied else "VIOLATED"
            lines.append(f"  {usage.key}: {usage.total:g}{bound}  [{marker}]")
        if plan.metadata.csp.makespan is not None:
            lines.append(f"  makespan: {plan.metadata.csp.makespan.total_seconds():g}s")

    # Infeasibility explanation
    if plan.metadata.csp is not None and plan.metadata.csp.explanation is not None:
        explanation = plan.metadata.csp.explanation
        lines.append("")
        lines.append("Infeasibility Explanation:")
        if explanation.conflicting_constraints:
            lines.append("  Conflicting constraints:")
            for c in explanation.conflicting_constraints:
                bound = ""
                if c.max is not None:
                    bound = f"max={c.max:g}"
                if c.min is not None:
                    bound = f"min={c.min:g}" if not bound else f"{bound}, min={c.min:g}"
                lines.append(f"    - {c.key} ({bound}, weight={c.weight:g})")
        if explanation.resource_shortfalls:
            lines.append("  Resource shortfalls:")
            for sf in explanation.resource_shortfalls:
                if sf.available_max is not None:
                    lines.append(
                        f"    - {sf.key}: {sf.required:g} / {sf.available_max:g} "
                        f"(overrun: {sf.overrun:g})"
                    )
                elif sf.available_min is not None:
                    lines.append(
                        f"    - {sf.key}: {sf.required:g} >= {sf.available_min:g} "
                        f"(shortfall: {sf.overrun:g})"
                    )
        if explanation.suggestion:
            lines.append(f"  Suggestion: {explanation.suggestion}")

    return "\n".join(lines) + "\n"


def render_ascii_gantt(plan: Plan, *, width: int = 60) -> str:
    """Render the plan's schedule as an ASCII Gantt chart.

    Args:
        plan: Plan whose ``metadata.csp.schedule`` will be rendered.
        width: Number of characters for the time axis (excluding labels).

    Returns:
        ASCII Gantt block as a plain string.

    Raises:
        ValueError: If the plan has no schedule.
    """
    csp = plan.metadata.csp
    if csp is None or not csp.schedule:
        raise ValueError(
            "render_ascii_gantt requires a plan with a CSP schedule "
            "(plan.metadata.csp.schedule is empty)."
        )
    if width < 10:
        raise ValueError(f"width must be >= 10, got {width}")

    max_end = max(entry.end.total_seconds() for entry in csp.schedule)
    if max_end <= 0:
        max_end = 1.0
    scale = width / max_end

    name_width = max(len(entry.action_name) for entry in csp.schedule)
    lines: list[str] = []
    lines.append(f"{'action'.ljust(name_width)}  |{'-' * width}|  " f"0..{max_end:g}s")
    for entry in csp.schedule:
        start_col = int(entry.start.total_seconds() * scale)
        end_col = max(start_col + 1, int(entry.end.total_seconds() * scale))
        bar = [" "] * width
        for col in range(start_col, min(end_col, width)):
            bar[col] = "#"
        lines.append(
            f"{entry.action_name.ljust(name_width)}  |{''.join(bar)}|  "
            f"{entry.start.total_seconds():g}-{entry.end.total_seconds():g}s"
        )
    return "\n".join(lines) + "\n"
