"""Common ``RunResult`` returned by all four screencast versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunResult:
    """Outcome of one screencast version's run.

    Attributes:
        name: Stable identifier (``react_baseline``,
            ``langgraph_routed``, ``langgoap_planned``, ``disrupted``).
        status: ``ok`` on completion; ``error`` when an exception
            escaped; ``budget_exceeded`` when a hard cost cap fired;
            ``goal_not_reached`` when the GOAP loop ran out of replans.
        cost_summary: ``CostMeter.snapshot()`` result — USD, tokens,
            call counts.
        summary_text: The cohort summary the agent produced
            (empty when ``status != 'ok'``).
        elapsed_s: Wall-clock seconds.
        error: Exception summary when ``status == 'error'``.
        replans: How many times GOAP replanned during the run
            (``0`` for non-GOAP versions).
        path_taken: Action names executed in order
            (empty for ``react_baseline``).
    """

    name: str
    status: str
    cost_summary: dict[str, Any]
    summary_text: str
    elapsed_s: float
    error: str | None = None
    replans: int = 0
    path_taken: list[str] | None = None
