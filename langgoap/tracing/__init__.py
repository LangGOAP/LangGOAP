"""Observability hooks for the GOAP planning loop.

Public surface:

* :class:`PlanningTracer` \u2014 the Protocol implemented by every tracer.
* :class:`NullTracer` \u2014 zero-overhead default (every hook is a pass).
* :class:`LoggingTracer` \u2014 stdlib-``logging``-backed tracer for local
  debugging.
* :class:`MultiTracer` \u2014 fan a single event out to multiple tracers.
* :class:`LangSmithTracer` \u2014 LangSmith adapter that emits each
  planner/executor cycle as a structured run tree.

This module re-exports the same names that the previous single-file
``langgoap.tracing`` module exposed; ``from langgoap.tracing import X``
continues to work for ``X`` in the list above.
"""

from __future__ import annotations

from langgoap.tracing._protocol import PlanningTracer
from langgoap.tracing._safe_proxy import SafeTracerProxy
from langgoap.tracing.langsmith import LangSmithTracer
from langgoap.tracing.logging import LoggingTracer
from langgoap.tracing.multi import MultiTracer
from langgoap.tracing.null import NullTracer

__all__ = [
    "PlanningTracer",
    "NullTracer",
    "LoggingTracer",
    "MultiTracer",
    "LangSmithTracer",
    "SafeTracerProxy",
]
