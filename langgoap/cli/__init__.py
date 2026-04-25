"""LangGOAP CLI — standalone command-line interface.

Requires the ``cli`` optional dependency::

    pip install langgoap[cli]

or::

    uv pip install langgoap[cli]
"""

from __future__ import annotations

try:
    import click as _click  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "The langgoap CLI requires the 'click' package. "
        "Install it with: pip install langgoap[cli]"
    ) from exc
