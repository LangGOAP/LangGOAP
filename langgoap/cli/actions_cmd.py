"""``langgoap actions`` command."""

from __future__ import annotations

import json
from typing import Any

import click

from langgoap.cli._loader import load_actions


@click.command("actions")
@click.argument("actions_ref")
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all action details.",
)
def actions(
    actions_ref: str,
    fmt: str,
    verbose: bool,
) -> None:
    """List actions in an action library.

    ACTIONS_REF is a Python module:variable reference pointing to a
    list[ActionSpec] (e.g. my_module:ACTIONS).
    """
    try:
        actions_list = load_actions(actions_ref)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if fmt == "json":
        click.echo(_render_json(actions_list, verbose))
    else:
        click.echo(_render_table(actions_list, verbose))


def _render_table(actions_list: list[Any], verbose: bool) -> str:
    """Render actions as an ASCII table."""
    if not actions_list:
        return "(no actions)"

    lines: list[str] = []

    # Compute column widths.
    name_w = max(len(a.name) for a in actions_list)
    name_w = max(name_w, 4)  # minimum header width
    cost_w = 6

    header = f"{'Name':<{name_w}}  {'Cost':>{cost_w}}  Preconditions  Effects"
    lines.append(header)
    lines.append("-" * len(header))

    for a in actions_list:
        pre = dict(a.preconditions) if a.preconditions else {}
        eff = dict(a.effects) if a.effects else {}
        line = f"{a.name:<{name_w}}  {a.cost:>{cost_w}.1f}  {pre!s:<13}  {eff!s}"
        lines.append(line)

        if verbose:
            if a.resources:
                lines.append(f"  resources: {dict(a.resources)}")
            if a.duration:
                lines.append(f"  duration: {a.duration}")
            if a.metadata:
                lines.append(f"  metadata: {dict(a.metadata)}")

    lines.append(f"\n{len(actions_list)} action(s)")
    return "\n".join(lines)


def _render_json(actions_list: list[Any], verbose: bool) -> str:
    """Render actions as JSON."""
    items = []
    for a in actions_list:
        entry: dict[str, Any] = {
            "name": a.name,
            "cost": a.cost,
            "preconditions": dict(a.preconditions) if a.preconditions else {},
            "effects": dict(a.effects) if a.effects else {},
        }
        if verbose:
            if a.resources:
                entry["resources"] = dict(a.resources)
            if a.duration:
                entry["duration"] = str(a.duration)
            if a.metadata:
                entry["metadata"] = dict(a.metadata)
        items.append(entry)
    return json.dumps(items, indent=2)
