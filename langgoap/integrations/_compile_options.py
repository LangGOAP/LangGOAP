"""The keyword arguments every on-ramp forwards to :meth:`GoapGraph.compile`.

Shared by :func:`~langgoap.integrations.prebuilt.create_goap_agent`,
:meth:`~langgoap.integrations.subgraph.GoapSubgraph.compile` and
:func:`~langgoap.integrations.subgraph.add_goap_subgraph`, so a typo or a
``GoapGraph.__init__`` option passed through ``**compile_kwargs`` raises the
same ``TypeError`` naming both option sets everywhere (issue #19).
"""

from __future__ import annotations

from typing import Any

#: Keyword arguments that :meth:`GoapGraph.compile` accepts.  Anything else
#: passed through ``**compile_kwargs`` is a typo or a ``GoapGraph.__init__``
#: option that must be passed by its explicit name.
COMPILE_OPTIONS: frozenset[str] = frozenset(
    {"checkpointer", "store", "interrupt_before", "interrupt_after"}
)


def _check_compile_kwargs(func_name: str, compile_kwargs: dict[str, Any]) -> None:
    unknown = sorted(set(compile_kwargs) - COMPILE_OPTIONS)
    if unknown:
        raise TypeError(
            f"{func_name}() got an unexpected keyword argument "
            f"{unknown[0]!r}. Compile options are "
            f"{sorted(COMPILE_OPTIONS)}; GoapGraph options (tracer, strategy, "
            "history, sensors, guards, resolvers, transition_model, rng, "
            "stuck_handlers, max_stuck_iterations, termination_policies, "
            "record_expansions, executor, history_state) are explicit keyword arguments."
        )


__all__ = ["COMPILE_OPTIONS", "_check_compile_kwargs"]
