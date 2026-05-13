"""``langgoap plan`` command."""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgoap.planner.types import Plan

import click

from langgoap.cli._loader import load_actions, load_goal, load_world_state

logger = logging.getLogger(__name__)


@click.command("plan")
@click.argument("actions_ref")
@click.argument("goal_ref")
@click.option(
    "-w",
    "--world-state",
    "world_state_ref",
    default=None,
    help="Module:variable or JSON file for initial world state.",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["ascii", "mermaid", "dot", "json", "gantt"]),
    default="ascii",
    show_default=True,
    help="Output format for the plan.",
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["astar", "pipeline", "mcts"]),
    default="astar",
    show_default=True,
    help="Planning strategy.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to a file instead of stdout.",
)
def plan(
    actions_ref: str,
    goal_ref: str,
    world_state_ref: str | None,
    fmt: str,
    strategy: str,
    output_path: str | None,
) -> None:
    """Plan a sequence of actions to achieve a goal.

    ACTIONS_REF and GOAL_REF are Python module:variable references
    (e.g. my_module:ACTIONS, my_module:GOAL).
    """
    try:
        actions_list = load_actions(actions_ref)
        goal = load_goal(goal_ref)
        ws = load_world_state(world_state_ref)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise click.ClickException(str(exc)) from exc

    strategy_obj = _resolve_strategy(strategy)

    from langgoap.graph.builder import GoapGraph

    graph = GoapGraph(actions=actions_list, strategy=strategy_obj)
    result = graph.invoke(goal=goal, world_state=ws)

    status = result.get("status", "unknown")
    result_plan = result.get("plan")

    if status == "no_plan" or result_plan is None:
        click.echo(f"No plan found (status={status}).", err=True)
        _auto_explain(actions_list, goal, ws)
        sys.exit(1)

    output = _render_plan(result_plan, fmt, dict(result))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        click.echo(f"Plan written to {output_path}")
    else:
        click.echo(output)


def _resolve_strategy(name: str) -> Any:
    """Resolve a strategy name to a strategy instance."""
    if name == "astar":
        return None  # default
    if name == "pipeline":
        from langgoap.planner.strategy import TwoPhasePipelineStrategy

        return TwoPhasePipelineStrategy()
    if name == "mcts":
        from langgoap.planner.mcts import MCTSStrategy

        return MCTSStrategy()
    return None


def _render_plan(plan_obj: "Plan", fmt: str, result: dict[str, Any]) -> str:
    """Render a plan in the requested format."""
    if fmt == "ascii":
        return plan_obj.to_ascii()
    elif fmt == "mermaid":
        return plan_obj.to_mermaid()
    elif fmt == "dot":
        return plan_obj.to_dot()
    elif fmt == "gantt":
        return plan_obj.to_gantt()
    elif fmt == "json":
        return json.dumps(
            {
                "status": result.get("status"),
                "actions": plan_obj.action_names,
                "total_cost": plan_obj.total_cost,
                "steps": len(plan_obj),
            },
            indent=2,
        )
    return str(plan_obj)


def _auto_explain(
    actions_list: list[Any],
    goal: Any,
    ws: dict[str, Any],
) -> None:
    """Print an explanation when no plan is found."""
    try:
        from langgoap.planner.explain import explain_no_plan
        from langgoap.state import PlanningState

        start = PlanningState.from_dict(ws)
        explanation = explain_no_plan(start, goal, actions_list)
        click.echo("\nExplanation:", err=True)
        if explanation.unreachable_conditions:
            click.echo(
                f"  Unreachable conditions: {list(explanation.unreachable_conditions)}",
                err=True,
            )
        if explanation.missing_preconditions:
            click.echo(
                f"  Missing preconditions: {list(explanation.missing_preconditions)}",
                err=True,
            )
        if explanation.suggestion:
            click.echo(f"  Suggestion: {explanation.suggestion}", err=True)
    except Exception as exc:
        # Best-effort explainer — the CLI must keep going if the helper
        # hits an edge case.  Surface the cause under DEBUG logging.
        logger.debug("auto-explain failed: %s", exc, exc_info=True)
