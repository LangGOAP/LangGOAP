"""File-output helpers for plan visualisations.

Centralises the format inference + dispatch + write loop so the
:class:`~langgoap.planner.types.Plan` value type does not own filesystem
concerns.  :meth:`langgoap.planner.types.Plan.save` is a thin shim over
:func:`save_plan`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langgoap.viz.ascii import render_ascii
from langgoap.viz.dot import render_dot
from langgoap.viz.mermaid import render_mermaid

if TYPE_CHECKING:
    from langgoap.planner.types import Plan


SaveFormat = Literal["mermaid", "dot", "ascii"]


def _infer_format(suffix: str) -> SaveFormat:
    suffix = suffix.lower().lstrip(".")
    if suffix in {"mmd", "mermaid"}:
        return "mermaid"
    if suffix in {"dot", "gv"}:
        return "dot"
    return "ascii"


def save_plan(
    plan: Plan,
    path: str | Path,
    *,
    format: SaveFormat | None = None,
    show_resources: bool = True,
    show_schedule: bool = True,
) -> Path:
    """Write a rendered representation of ``plan`` to ``path``.

    The format is inferred from the path suffix when ``format`` is
    ``None``: ``.mmd``/``.mermaid`` \u2192 Mermaid, ``.dot``/``.gv`` \u2192
    DOT, anything else \u2192 ASCII.

    Args:
        plan: Plan to render.
        path: Destination file path.
        format: Explicit format override.
        show_resources: Forwarded to the renderer.
        show_schedule: Forwarded to the renderer.

    Returns:
        The :class:`~pathlib.Path` that was written.
    """
    dest = Path(path)
    fmt: SaveFormat = format if format is not None else _infer_format(dest.suffix)

    if fmt == "mermaid":
        source = render_mermaid(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )
    elif fmt == "dot":
        source = render_dot(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )
    else:
        source = render_ascii(
            plan, show_resources=show_resources, show_schedule=show_schedule
        )

    dest.write_text(source, encoding="utf-8")
    return dest


__all__ = ["save_plan", "SaveFormat"]
