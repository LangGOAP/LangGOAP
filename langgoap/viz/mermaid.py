"""Mermaid flowchart and gantt renderers for Plan objects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

from langgoap.planner.csp import build_dependency_graph

if TYPE_CHECKING:
    from langgoap.planner.csp import CSPMetadata, ScheduleEntry
    from langgoap.planner.explain import InfeasibilityExplanation
    from langgoap.planner.types import Plan


_ID_SAFE = re.compile(r"[^A-Za-z0-9_]")

# -- Brand styling ------------------------------------------------------------
# Consistent visual identity across all LangGOAP plan diagrams.  Action nodes
# get a light background with a distinctive blue border.

_CLASSDEFS = [
    "    classDef action fill:#f8f9fa,stroke:#4a90d9,stroke-width:2px,color:#1a1a2e",
    "    classDef cost fill:#d4edda,stroke:#2d8a4e,stroke-width:1px,color:#155724,font-size:10px",
]

# Font Awesome icon used inside cost badge nodes.
_COST_ICON = "fa:fa-dollar-sign"


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

    has_costed = any(a.get_cost({}) != 1.0 for a in plan.actions)

    lines: list[str] = []
    if has_costed:
        lines.append('%%{ init: { "flowchart": { "nodeSpacing": 5 } } }%%')
    lines.append("flowchart TD")
    lines.extend(_CLASSDEFS)

    node_ids = [_node_id(i, a.name) for i, a in enumerate(plan.actions)]
    csp = plan.metadata.csp
    schedule = (
        csp.schedule if (show_schedule and csp is not None and csp.schedule) else None
    )

    if schedule is not None and len(schedule) == len(plan.actions):
        lines.extend(_emit_scheduled_nodes(plan, node_ids, schedule))
    else:
        lines.extend(_emit_unscheduled_nodes(plan, node_ids))

    lines.extend(_emit_edges(plan, node_ids))

    if show_resources and csp is not None and csp.resource_usage:
        lines.extend(_emit_resource_summary(csp))

    if csp is not None and csp.explanation is not None and node_ids:
        lines.extend(_emit_infeasibility_note(csp.explanation, node_ids[0]))

    return "\n".join(lines) + "\n"


def _emit_scheduled_nodes(
    plan: Plan, node_ids: list[str], schedule: Sequence[ScheduleEntry]
) -> list[str]:
    """Emit nodes grouped by start time, parallel entries wrapped in subgraphs."""
    by_start: dict[float, list[int]] = {}
    for i, entry in enumerate(schedule):
        by_start.setdefault(entry.start.total_seconds(), []).append(i)
    out: list[str] = []
    for t_idx, start_t in enumerate(sorted(by_start)):
        indices = by_start[start_t]
        if len(indices) > 1:
            out.append(f'    subgraph slot_{t_idx}["t={start_t:g}s"]')
            for i in indices:
                label = _format_node_label(plan, i, schedule[i])
                out.append(f'        {node_ids[i]}["{label}"]:::action')
            out.append("    end")
        else:
            i = indices[0]
            label = _format_node_label(plan, i, schedule[i])
            out.append(f'    {node_ids[i]}["{label}"]:::action')
    return out


def _emit_unscheduled_nodes(plan: Plan, node_ids: list[str]) -> list[str]:
    """Emit action nodes with brand styling and cost badge pills.

    Actions with non-default costs get a separate stadium-shaped pill
    node (green ``cost`` classDef) containing a ``fa:fa-comment-dollar``
    icon and the cost value.  The action and its pill are grouped
    side-by-side inside a transparent ``subgraph`` with ``direction LR``.
    """
    out: list[str] = []
    badge_subgraphs: list[str] = []
    for i, action in enumerate(plan.actions):
        label = _escape_label(action.name)
        cost = action.get_cost({})
        if cost != 1.0:
            sg_id = f"sg_{node_ids[i]}"
            badge_id = f"c{i}"
            out.append(f'    subgraph {sg_id}[" "]')
            out.append("        direction LR")
            out.append(f'        {node_ids[i]}["{label}"]:::action')
            out.append(f'        {badge_id}(["{_COST_ICON} {cost:g}"]):::cost')
            out.append("    end")
            badge_subgraphs.append(sg_id)
        else:
            out.append(f'    {node_ids[i]}["{label}"]:::action')
    for sg_id in badge_subgraphs:
        out.append(f"    style {sg_id} fill:none,stroke:none")
    return out


def _emit_edges(plan: Plan, node_ids: list[str]) -> list[str]:
    """Emit dependency edges, falling back to sequential when none are derived."""
    deps = build_dependency_graph(plan.actions)
    out: list[str] = []
    has_any_edge = False
    for j in range(len(plan.actions)):
        for i in deps[j]:
            out.append(f"    {node_ids[i]} --> {node_ids[j]}")
            has_any_edge = True
    if not has_any_edge and len(plan.actions) > 1:
        for i in range(len(plan.actions) - 1):
            out.append(f"    {node_ids[i]} -.-> {node_ids[i + 1]}")
    return out


def _emit_resource_summary(csp: CSPMetadata) -> list[str]:
    """Emit the resource-usage summary as Mermaid comments."""
    out = ["", "    %% Resource usage"]
    for usage in csp.resource_usage:
        bound = ""
        if usage.constraint_max is not None:
            bound = f" / {usage.constraint_max:g}"
        elif usage.constraint_min is not None:
            bound = f" >= {usage.constraint_min:g}"
        status = "OK" if usage.satisfied else "VIOLATED"
        out.append(f"    %%   {usage.key}: {usage.total:g}{bound} [{status}]")
    return out


def _emit_infeasibility_note(
    explanation: InfeasibilityExplanation, anchor_id: str
) -> list[str]:
    """Emit a Mermaid note anchored to the first action describing infeasibility."""
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
    note_content = "<br/>".join(note_lines)
    return ["", f"    note right of {anchor_id}: {note_content}"]


def _format_node_label(plan: Plan, index: int, schedule_entry: ScheduleEntry) -> str:
    action = plan.actions[index]
    label = _escape_label(action.name)
    duration_s = schedule_entry.duration.total_seconds()
    if duration_s > 0:
        label = f"{label}\\n\u23f1 {duration_s:g}s"
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


def draw_mermaid_png(plan: Plan, **kwargs: Any) -> bytes:
    """Render ``plan`` as a PNG image via Mermaid.

    Delegates to ``langchain_core``'s ``draw_mermaid_png()`` which uses
    the mermaid.ink API by default.  All keyword arguments are forwarded
    (e.g. ``background_color``, ``draw_method``).
    """
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png as _draw

    return _draw(render_mermaid(plan), **kwargs)


def draw_gantt_png(plan: Plan, **kwargs: Any) -> bytes:
    """Render ``plan``'s schedule as a PNG Gantt chart via Mermaid.

    Requires ``plan.metadata.csp.schedule`` to be populated; otherwise
    :func:`render_mermaid_gantt` raises ``ValueError`` first.
    """
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png as _draw

    return _draw(render_mermaid_gantt(plan), **kwargs)
