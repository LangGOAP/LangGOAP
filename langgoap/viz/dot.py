"""Graphviz DOT renderer for Plan objects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langgoap.planner.csp import build_dependency_graph

if TYPE_CHECKING:
    from langgoap.planner.csp import ScheduleEntry
    from langgoap.planner.types import Plan


_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _node_id(index: int, name: str) -> str:
    return f"a{index}_{_ID_SAFE.sub('_', name)}"


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def render_dot(
    plan: Plan,
    *,
    show_resources: bool = True,
    show_schedule: bool = True,
) -> str:
    """Render a Plan as a Graphviz DOT digraph.

    Args:
        plan: Plan to render.
        show_resources: If True, append a legend node with resource usage.
        show_schedule: If True and the plan has a schedule, group
            actions with equal start time into clusters.

    Returns:
        DOT source code as a plain string.
    """
    if len(plan) == 0:
        return 'digraph Plan {\n    rankdir=TB;\n    empty [label="(empty plan)"];\n}\n'

    lines: list[str] = [
        "digraph Plan {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fillcolor="#e8f4ff"];',
    ]

    node_ids = [_node_id(i, a.name) for i, a in enumerate(plan.actions)]

    schedule = None
    if show_schedule and plan.metadata.csp is not None and plan.metadata.csp.schedule:
        schedule = plan.metadata.csp.schedule

    # Emit nodes (clustered by start-time when schedule is present).
    if schedule is not None and len(schedule) == len(plan.actions):
        by_start: dict[float, list[int]] = {}
        for i, entry in enumerate(schedule):
            by_start.setdefault(entry.start.total_seconds(), []).append(i)
        for cluster_idx, start_t in enumerate(sorted(by_start)):
            indices = by_start[start_t]
            if len(indices) > 1:
                lines.append(f"    subgraph cluster_{cluster_idx} {{")
                lines.append(f'        label="t={start_t:g}s";')
                lines.append('        style="dashed";')
                for i in indices:
                    label = _format_label(plan, i, schedule[i])
                    lines.append(f'        {node_ids[i]} [label="{label}"];')
                lines.append("    }")
            else:
                i = indices[0]
                label = _format_label(plan, i, schedule[i])
                lines.append(f'    {node_ids[i]} [label="{label}"];')
    else:
        for i, action in enumerate(plan.actions):
            label = _escape_dot(action.name)
            cost = action.get_cost({})
            if cost != 1.0:
                label = f"{label}\\ncost={cost:g}"
            lines.append(f'    {node_ids[i]} [label="{label}"];')

    # Dependency edges
    deps = build_dependency_graph(plan.actions)
    has_any_edge = False
    for j in range(len(plan.actions)):
        for i in deps[j]:
            lines.append(f"    {node_ids[i]} -> {node_ids[j]};")
            has_any_edge = True

    if not has_any_edge and len(plan.actions) > 1:
        for i in range(len(plan.actions) - 1):
            lines.append(f"    {node_ids[i]} -> {node_ids[i + 1]} [style=dashed];")

    # Resource legend
    if (
        show_resources
        and plan.metadata.csp is not None
        and plan.metadata.csp.resource_usage
    ):
        legend_lines = ["Resources:"]
        for usage in plan.metadata.csp.resource_usage:
            bound = ""
            if usage.constraint_max is not None:
                bound = f" / {usage.constraint_max:g}"
            elif usage.constraint_min is not None:
                bound = f" >= {usage.constraint_min:g}"
            status = "OK" if usage.satisfied else "VIOLATED"
            legend_lines.append(f"{usage.key}: {usage.total:g}{bound} [{status}]")
        legend = "\\l".join(_escape_dot(line) for line in legend_lines) + "\\l"
        lines.append(
            '    legend [shape=note, fillcolor="#fff8dc", ' f'label="{legend}"];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _format_label(plan: Plan, index: int, schedule_entry: ScheduleEntry) -> str:
    action = plan.actions[index]
    label = _escape_dot(action.name)
    duration_s = schedule_entry.duration.total_seconds()
    if duration_s > 0:
        label = f"{label}\\nduration={duration_s:g}s"
    return label
