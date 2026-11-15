"""Deprecated: LangGOAP no longer ships a custom checkpoint serializer.

Since 0.2.0 :class:`~langgoap.graph.state.GoapState` is JSON-native, so
every LangGraph ``BaseCheckpointSaver`` persists a GOAP run with its stock
serializer and no LangGOAP-specific wiring.  The names below are kept for
one release so existing imports keep working; each one emits a
``DeprecationWarning`` and changes nothing.  They will be removed in 0.3.0.

Migration:

* Delete calls to :func:`install_langgoap_serde`; pass the checkpointer to
  ``GoapGraph.compile`` as before.
* Build graph input with ``GoapGraph.initial_state(goal, world_state)`` (or
  ``goal.to_dict()``) when a checkpointer is attached, and use
  ``GoapGraph.rehydrate`` to turn a raw state back into typed objects.
"""

from __future__ import annotations

import warnings
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

__all__ = [
    "LANGGOAP_ALLOWED_MSGPACK_TYPES",
    "LangGOAPSerializer",
    "install_langgoap_serde",
]

_DEPRECATION = (
    "{name} is deprecated and does nothing: since LangGOAP 0.2.0 GoapState is "
    "JSON-native and every LangGraph checkpointer works with its stock "
    "serializer. Remove the usage; this shim will be deleted in 0.3.0."
)

LANGGOAP_ALLOWED_MSGPACK_TYPES: tuple[type, ...] = ()
"""Deprecated and always empty: no LangGOAP type is stored in a checkpoint."""


class LangGOAPSerializer(JsonPlusSerializer):
    """Deprecated no-op subclass of LangGraph's ``JsonPlusSerializer``."""

    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        allowed_json_modules: Any = None,
        allowed_msgpack_modules: Any = None,
        **_ignored: Any,
    ) -> None:
        warnings.warn(
            _DEPRECATION.format(name="LangGOAPSerializer"),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            pickle_fallback=pickle_fallback,
            allowed_json_modules=allowed_json_modules,
            allowed_msgpack_modules=allowed_msgpack_modules,
        )


def install_langgoap_serde(
    checkpointer: BaseCheckpointSaver,
    *,
    extra_allowed_types: tuple[type, ...] = (),
) -> BaseCheckpointSaver:
    """Deprecated no-op: returns *checkpointer* unchanged."""
    name = "install_langgoap_serde"
    if extra_allowed_types:
        name += f" (ignoring {len(extra_allowed_types)} extra_allowed_types)"
    warnings.warn(
        _DEPRECATION.format(name=name),
        DeprecationWarning,
        stacklevel=2,
    )
    return checkpointer
