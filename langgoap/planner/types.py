"""Types for GOAP planning results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langgoap.actions import ActionSpec
from langgoap.score import Score, SimpleScore
from langgoap.state import PlanningState

if TYPE_CHECKING:
    from langgoap.planner.csp import CSPMetadata


@dataclass(frozen=True, slots=True)
class PlanMetadata:
    """Metadata about how a plan was generated.

    Attributes:
        nodes_explored: Number of A* / MCTS / pipeline nodes expanded
            during search.  Search-style strategies populate this; the
            greedy :class:`~langgoap.planner.utility.UtilityStrategy`
            leaves it at ``0`` and reports its per-tick branching
            factor in :attr:`applicable_count` instead.
        applicable_count: Number of actions whose preconditions were
            satisfied during this planning round.  Populated by the
            utility planner; ``0`` for the search planners.
        planning_time_ms: Wall-clock time spent planning in milliseconds.
        actions_pruned: Number of actions removed by optimization passes.
        csp: Results from CSP constraint validation/optimization, or ``None``
            when no constraints or objectives were specified.
    """

    nodes_explored: int = 0
    applicable_count: int = 0
    planning_time_ms: float = 0.0
    actions_pruned: int = 0
    csp: CSPMetadata | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """A sequence of actions that achieves a goal from a given start state.

    Attributes:
        actions: Ordered list of ActionSpecs to execute.
        expected_states: The world state expected after each action.
        total_cost: Sum of action costs along the plan.
        metadata: Planning algorithm statistics.
        score: :class:`~langgoap.score.Score` for the plan.  A*-only plans
            carry a :class:`~langgoap.score.SimpleScore` equal to
            ``total_cost``; after CSP evaluation, the pipeline replaces
            this with a :class:`~langgoap.score.HardSoftScore` whose
            ``hard`` level is 0.0 for feasible plans and whose ``soft``
            level aggregates weighted objectives and soft penalties.
    """

    actions: tuple[ActionSpec, ...]
    expected_states: tuple[PlanningState, ...] = ()
    total_cost: float = 0.0
    metadata: PlanMetadata = field(default_factory=PlanMetadata)
    score: Score = field(default_factory=lambda: SimpleScore(scalar=0.0))

    @property
    def action_names(self) -> list[str]:
        """Return the names of all actions in the plan."""
        return [a.name for a in self.actions]

    def __len__(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        return (
            f"Plan(actions={self.action_names!r}, "
            f"total_cost={self.total_cost:.4g}, "
            f"steps={len(self)})"
        )

    @classmethod
    def empty(cls) -> Plan:
        """Create an empty plan (goal already satisfied)."""
        return cls(actions=(), expected_states=(), total_cost=0.0)

    def net_value(
        self, goal: Any, world_state: Any | None = None
    ) -> float:
        """Return ``goal.value - total_cost`` for ``MultiGoal`` selection.

        ``goal.value`` may be a static float or a callable resolved
        against ``world_state`` if supplied (preferred — preserves
        rich, non-planning context that's stripped from the planner's
        :class:`PlanningState`); otherwise it falls back to the plan's
        expected end state, then to an empty mapping.

        Used by ``MultiGoal(mode='best_value')`` to pick the goal with
        the highest net utility.
        """
        value_attr = getattr(goal, "value", 1.0)
        if callable(value_attr):
            if world_state is not None:
                ctx: Any = world_state
            elif self.expected_states:
                ctx = self.expected_states[-1].to_dict()
            else:
                ctx = {}
            value = float(value_attr(ctx))
        else:
            value = float(value_attr)
        return value - self.total_cost

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_mermaid(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as a Mermaid ``flowchart TD`` diagram.

        Returns:
            Mermaid source code as a plain string.
        """
        from langgoap.viz.mermaid import render_mermaid

        return render_mermaid(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def to_dot(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as Graphviz DOT source.

        Returns:
            DOT source code as a plain string.
        """
        from langgoap.viz.dot import render_dot

        return render_dot(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def to_ascii(
        self,
        *,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str:
        """Render this plan as an ASCII tree.

        Returns:
            Plain-text ASCII representation of the plan.
        """
        from langgoap.viz.ascii import render_ascii

        return render_ascii(
            self,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def visualize(
        self,
        *,
        format: Literal[
            "auto", "mermaid", "dot", "ascii", "gantt", "ascii_gantt"
        ] = "auto",
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> str | Any:
        """Display-aware visualization helper.

        When IPython is available, ``format="auto"`` returns an
        ``IPython.display.Markdown`` wrapping the Mermaid source so
        Jupyter renders it as a diagram.  When IPython is absent, returns
        the ASCII rendering as a plain string.

        See :func:`langgoap.viz.jupyter.visualize` for the full format
        table.
        """
        from langgoap.viz.jupyter import visualize as _visualize

        return _visualize(
            self,
            format=format,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )

    def to_gantt(self) -> str:
        """Render this plan's schedule as a Mermaid ``gantt`` chart.

        Requires ``plan.metadata.csp.schedule`` to be populated.

        Returns:
            Mermaid gantt source code as a plain string.

        Raises:
            ValueError: If the plan has no schedule attached.
        """
        from langgoap.viz.mermaid import render_mermaid_gantt

        return render_mermaid_gantt(self)

    def draw_gantt_png(self, **kwargs: Any) -> bytes:
        """Render this plan's schedule as a PNG Gantt chart via Mermaid.

        Thin shim over :func:`langgoap.viz.mermaid.draw_gantt_png`.
        """
        from langgoap.viz.mermaid import draw_gantt_png

        return draw_gantt_png(self, **kwargs)

    def draw_mermaid_png(self, **kwargs: Any) -> bytes:
        """Render this plan as a PNG image via Mermaid.

        Thin shim over :func:`langgoap.viz.mermaid.draw_mermaid_png`.
        """
        from langgoap.viz.mermaid import draw_mermaid_png

        return draw_mermaid_png(self, **kwargs)

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Jupyter rich display \u2014 thin shim over
        :func:`langgoap.viz.jupyter.repr_mimebundle`.
        """
        from langgoap.viz.jupyter import repr_mimebundle

        return repr_mimebundle(self, **kwargs)

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["mermaid", "dot", "ascii"] | None = None,
        show_resources: bool = True,
        show_schedule: bool = True,
    ) -> Path:
        """Write a rendered representation of this plan to ``path``.

        Thin shim over :func:`langgoap.viz.save.save_plan` \u2014 see that
        function for the format-inference rules.
        """
        from langgoap.viz.save import save_plan

        return save_plan(
            self,
            path,
            format=format,
            show_resources=show_resources,
            show_schedule=show_schedule,
        )
