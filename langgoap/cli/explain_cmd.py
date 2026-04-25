"""``langgoap explain`` command."""

from __future__ import annotations

import click

from langgoap.cli._loader import load_actions, load_goal, load_world_state


@click.command("explain")
@click.argument("actions_ref")
@click.argument("goal_ref")
@click.option(
    "-w",
    "--world-state",
    "world_state_ref",
    default=None,
    help="Module:variable or JSON file for initial world state.",
)
def explain(
    actions_ref: str,
    goal_ref: str,
    world_state_ref: str | None,
) -> None:
    """Explain why a goal cannot be achieved.

    ACTIONS_REF and GOAL_REF are Python module:variable references
    (e.g. my_module:ACTIONS, my_module:GOAL).
    """
    try:
        actions_list = load_actions(actions_ref)
        goal = load_goal(goal_ref)
        ws = load_world_state(world_state_ref)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    from langgoap.planner.explain import explain_no_plan
    from langgoap.state import PlanningState

    start = PlanningState.from_dict(ws)
    explanation = explain_no_plan(start, goal, actions_list)

    click.echo(f"Goal conditions: {dict(goal.conditions)}")
    click.echo(f"Start state: {start.to_dict()}")
    click.echo()

    if explanation.unreachable_conditions:
        click.echo("Unreachable conditions (no action produces these):")
        for cond in explanation.unreachable_conditions:
            click.echo(f"  - {cond}")
    else:
        click.echo("All goal conditions are producible by the action set.")

    if explanation.missing_preconditions:
        click.echo("\nMissing preconditions (required but never produced):")
        for pre in explanation.missing_preconditions:
            click.echo(f"  - {pre}")

    if explanation.suggestion:
        click.echo(f"\nSuggestion: {explanation.suggestion}")
