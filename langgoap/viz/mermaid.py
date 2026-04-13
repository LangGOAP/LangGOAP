"""Mermaid flowchart and gantt renderers for Plan objects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langgoap.planner.csp import build_dependency_graph

if TYPE_CHECKING:
    from langgoap.planner.csp import ScheduleEntry
    from langgoap.planner.types import Plan


_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _node_id(index: int, name: str) -> str:
    """Return a Mermaid-safe node id.

    Mermaid node ids must match ``[A-Za-z0-9_]+``; action names that
    contain dots or spaces would otherwise break the diagram.
    """
    safe = _ID_SAFE.sub("_", name)
    return f"a{index}_{safe}"


def _escape_label(text: str) -> str:
    """Escape a label so Mermaid parses it as a single token."""
    return text.replace('"', "'")


def render_mermaid(
    plan: Plan,
    *,
    show_resources: bool = True,
    show_schedule: bool = True,
) -> str:
    """Render a Plan as a Mermaid ``flowchart TD`` diagram.

    Nodes are actions; edges are effect→precondition dependencies from
    :func:`langgoap.planner.csp.build_dependency_graph`.  When
    ``plan.metadata.csp`` carries a schedule, actions at the same
    start time are grouped into ``subgraph`` blocks to highlight
    parallelism.  Resource totals are appended as a Mermaid comment
    block at the bottom when ``show_resources`` is True.

    Args:
        plan: Plan to render.
        show_resources: If True, append resource usage summary.
        show_schedule: If True, group parallel actions into subgraphs
            and highlight the critical path.

    Returns:
        Mermaid source code as a plain string.
    """
    if len(plan) == 0:
        return 'flowchart TD\n    empty["(empty plan)"]\n'

    lines: list[str] = ["flowchart TD"]

    node_ids = [_node_id(i, a.name) for i, a in enumerate(plan.actions)]
    schedule = None
    if show_schedule and plan.metadata.csp is not None and plan.metadata.csp.schedule:
        schedule = plan.metadata.csp.schedule

    # Group actions by start time when a schedule is available.
    if schedule is not None and len(schedule) == len(plan.actions):
        by_start: dict[float, list[int]] = {}
        for i, entry in enumerate(schedule):
            key = entry.start.total_seconds()
            by_start.setdefault(key, []).append(i)
        sorted_starts = sorted(by_start)
        for t_idx, start_t in enumerate(sorted_starts):
            indices = by_start[start_t]
            if len(indices) > 1:
                lines.append(f'    subgraph slot_{t_idx}["t={start_t:g}s"]')
                for i in indices:
                    label = _format_node_label(plan, i, schedule[i])
                    lines.append(f'        {node_ids[i]}["{label}"]')
                lines.append("    end")
            else:
                i = indices[0]
                label = _format_node_label(plan, i, schedule[i])
                lines.append(f'    {node_ids[i]}["{label}"]')
    else:
        for i, action in enumerate(plan.actions):
            label = _escape_label(action.name)
            cost = action.get_cost({})
            if cost != 1.0:
                label = f"{label}\\ncost={cost:g}"
            lines.append(f'    {node_ids[i]}["{label}"]')

    # Dependency edges
    deps = build_dependency_graph(plan.actions)
    has_any_edge = False
    for j in range(len(plan.actions)):
        for i in deps[j]:
            lines.append(f"    {node_ids[i]} --> {node_ids[j]}")
            has_any_edge = True

    # If no dependency edges were produced, fall back to sequential edges.
    if not has_any_edge and len(plan.actions) > 1:
        for i in range(len(plan.actions) - 1):
            lines.append(f"    {node_ids[i]} -.-> {node_ids[i + 1]}")

    # Resource summary
    if (
        show_resources
        and plan.metadata.csp is not None
        and plan.metadata.csp.resource_usage
    ):
        lines.append("")
        lines.append("    %% Resource usage")
        for usage in plan.metadata.csp.resource_usage:
            bound = ""
            if usage.constraint_max is not None:
                bound = f" / {usage.constraint_max:g}"
            elif usage.constraint_min is not None:
                bound = f" >= {usage.constraint_min:g}"
            status = "OK" if usage.satisfied else "VIOLATED"
            lines.append(f"    %%   {usage.key}: {usage.total:g}{bound} [{status}]")

    # Infeasibility explanation as a note
    if plan.metadata.csp is not None and plan.metadata.csp.explanation is not None:
        explanation = plan.metadata.csp.explanation
        note_lines = ["INFEASIBLE PLAN"]
        for sf in explanation.resource_shortfalls:
            if sf.available_max is not None:
                note_lines.append(
                    f"{sf.key}: {sf.required:g}/{sf.available_max:g} "
                    f"(overrun: {sf.overrun:g})"
                )
            elif sf.available_min is not None:
                note_lines.append(
                    f"{sf.key}: {sf.required:g}>={sf.available_min:g} "
                    f"(shortfall: {sf.overrun:g})"
                )
        if explanation.suggestion:
            note_lines.append(explanation.suggestion)
        # Build the Mermaid note block anchored to the first action
        if node_ids:
            note_content = "<br/>".join(note_lines)
            lines.append("")
            lines.append(f"    note right of {node_ids[0]}: {note_content}")

    return "\n".join(lines) + "\n"


def _format_node_label(plan: Plan, index: int, schedule_entry: ScheduleEntry) -> str:
    action = plan.actions[index]
    label = _escape_label(action.name)
    duration_s = schedule_entry.duration.total_seconds()
    if duration_s > 0:
        label = f"{label}\\nduration={duration_s:g}s"
    return label


def render_mermaid_gantt(plan: Plan) -> str:
    """Render a Plan's schedule as a Mermaid ``gantt`` chart.

    Requires ``plan.metadata.csp.schedule`` to be populated.  Returns a
    minimal ``gantt`` block whose entries have integer-millisecond
    start/duration values.

    Raises:
        ValueError: If the plan has no schedule attached.
    """
    csp = plan.metadata.csp
    if csp is None or not csp.schedule:
        raise ValueError(
            "render_mermaid_gantt requires a plan with CSP schedule metadata "
            "(plan.metadata.csp.schedule is empty or absent)."
        )

    lines: list[str] = [
        "gantt",
        "    title Plan Schedule",
        "    dateFormat  x",
        "    axisFormat  %Lms",
    ]

    # Group by identical start for visual grouping (optional); here we
    # emit every entry on its own line for clarity.
    lines.append("    section Plan")
    for entry in csp.schedule:
        start_ms = int(entry.start.total_seconds() * 1000)
        dur_ms = max(int(entry.duration.total_seconds() * 1000), 1)
        safe_name = _escape_label(entry.action_name)
        # task_name :id, start_ms, duration_ms
        lines.append(
            f"    {safe_name} :task_{start_ms}_{dur_ms}_{safe_name}, {start_ms}, {dur_ms}ms"
        )
    return "\n".join(lines) + "\n"
