"""A* search-event accumulator with a truncation cap and flush batching.

Extracted from :class:`~langgoap.tracing.langsmith.LangSmithTracer` so
the bookkeeping (capped append, in-place truncation marker, batched
flush threshold) lives in one focused, unit-testable place.

The buffer is **not** thread-safe on its own \u2014 callers must serialise
mutations.  :class:`LangSmithTracer` already holds an internal
``threading.Lock`` for its per-invocation state, so wrapping every
buffer method in another lock would be redundant.
"""

from __future__ import annotations

from typing import Any


class SearchEventBuffer:
    """Accumulates A* search events for a single LangSmith root run.

    Lifecycle:

    * :meth:`reset` \u2014 start of a new root run; clears state.
    * :meth:`add_expand` \u2014 append a ``search_expand`` event subject to
      the ``max_events`` cap.  Returns ``True`` when the caller should
      flush the pending list (``flush_every`` boundary hit).
    * :meth:`add_uncapped` \u2014 append a low-cardinality event
      (``search_dead_end``) that bypasses the cap.  Returns ``True``
      when the caller should flush.
    * :meth:`drain` \u2014 snapshot the pending event list and clear the
      flush counter; caller is responsible for actually uploading.

    The cap is enforced via a single ``search_truncated`` marker whose
    ``dropped`` counter is updated in place once subsequent expansions
    arrive.  The expansion counter is maintained incrementally so each
    add is O(1).
    """

    __slots__ = (
        "_max_events",
        "_flush_every",
        "_events",
        "_events_since_flush",
        "_expand_count",
        "_dropped_expansions",
        "_truncation_recorded",
        "_truncation_index",
    )

    def __init__(self, *, max_events: int, flush_every: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        self._max_events = max_events
        self._flush_every = flush_every
        self._events: list[dict[str, Any]] = []
        self._events_since_flush = 0
        self._expand_count = 0
        self._dropped_expansions = 0
        self._truncation_recorded = False
        self._truncation_index: int | None = None

    def reset(self) -> None:
        """Clear all per-root state.  Call at the start of a new root run."""
        self._events = []
        self._events_since_flush = 0
        self._expand_count = 0
        self._dropped_expansions = 0
        self._truncation_recorded = False
        self._truncation_index = None

    def add_expand(self, event: dict[str, Any]) -> bool:
        """Append a ``search_expand`` event subject to the cap.

        Returns ``True`` when ``flush_every`` is reached and the caller
        should drain + upload.
        """
        if self._expand_count >= self._max_events:
            self._dropped_expansions += 1
            if not self._truncation_recorded:
                self._truncation_index = len(self._events)
                self._events.append(
                    {
                        "kind": "search_truncated",
                        "dropped": self._dropped_expansions,
                    }
                )
                self._truncation_recorded = True
            elif self._truncation_index is not None:
                self._events[self._truncation_index][
                    "dropped"
                ] = self._dropped_expansions
        else:
            self._events.append(event)
            self._expand_count += 1
        self._events_since_flush += 1
        return self._events_since_flush >= self._flush_every

    def add_uncapped(self, event: dict[str, Any]) -> bool:
        """Append a low-cardinality event (bypasses the cap).

        Returns ``True`` when ``flush_every`` is reached and the caller
        should drain + upload.
        """
        self._events.append(event)
        self._events_since_flush += 1
        return self._events_since_flush >= self._flush_every

    def drain(self) -> list[dict[str, Any]] | None:
        """Snapshot pending events and reset the flush counter.

        Returns the snapshot list (a copy, safe to upload outside any
        caller-held lock) or ``None`` when nothing is pending.  Always
        zeros ``events_since_flush`` so a no-op flush still moves the
        threshold counter.
        """
        self._events_since_flush = 0
        if not self._events:
            return None
        return list(self._events)


__all__ = ["SearchEventBuffer"]
