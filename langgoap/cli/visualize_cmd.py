"""``langgoap visualize`` command."""

from __future__ import annotations

import click

from langgoap.cli._loader import load_python_variable
from langgoap.planner.types import Plan


@click.command("visualize")
@click.argument("plan_ref")
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["ascii", "mermaid", "dot", "gantt"]),
    default="ascii",
    show_default=True,
    help="Visualization format.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to a file instead of stdout.",
)
def visualize(
    plan_ref: str,
    fmt: str,
    output_path: str | None,
) -> None:
    """Visualize an existing Plan object.

    PLAN_REF is a Python module:variable reference pointing to a Plan
    instance (e.g. my_module:MY_PLAN).
    """
    try:
        plan_obj = load_python_variable(plan_ref)
    except (ValueError, ImportError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    if not isinstance(plan_obj, Plan):
        raise click.ClickException(
            f"Expected a Plan from {plan_ref!r}, got {type(plan_obj).__name__}."
        )

    output = _render(plan_obj, fmt)
    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        click.echo(f"Visualization written to {output_path}")
    else:
        click.echo(output)


def _render(plan_obj: Plan, fmt: str) -> str:
    """Render the plan in the requested format."""
    if fmt == "ascii":
        return plan_obj.to_ascii()
    elif fmt == "mermaid":
        return plan_obj.to_mermaid()
    elif fmt == "dot":
        return plan_obj.to_dot()
    elif fmt == "gantt":
        return plan_obj.to_gantt()
    return str(plan_obj)
