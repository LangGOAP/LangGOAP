"""Unit tests for :class:`langgoap.tracing._search_buffer.SearchEventBuffer`.

The buffer was extracted from :class:`LangSmithTracer` so the
state-machine logic (cap, in-place truncation, batched flush) can be
tested in isolation without LangSmith mocks.
"""

from __future__ import annotations

import pytest

from langgoap.tracing._search_buffer import SearchEventBuffer


def _expand(node_id: int) -> dict[str, object]:
    return {"kind": "search_expand", "node_id": node_id}


class TestConstruction:
    def test_rejects_non_positive_max(self) -> None:
        with pytest.raises(ValueError, match="max_events"):
            SearchEventBuffer(max_events=0, flush_every=1)

    def test_rejects_non_positive_flush(self) -> None:
        with pytest.raises(ValueError, match="flush_every"):
            SearchEventBuffer(max_events=10, flush_every=0)


class TestFlushThreshold:
    def test_returns_false_until_threshold_hit(self) -> None:
        buf = SearchEventBuffer(max_events=100, flush_every=3)
        assert buf.add_expand(_expand(1)) is False
        assert buf.add_expand(_expand(2)) is False
        # Third event trips the threshold.
        assert buf.add_expand(_expand(3)) is True

    def test_threshold_resets_after_drain(self) -> None:
        buf = SearchEventBuffer(max_events=100, flush_every=2)
        buf.add_expand(_expand(1))
        assert buf.add_expand(_expand(2)) is True
        assert buf.drain() is not None
        # New cycle: count restarts from zero.
        assert buf.add_expand(_expand(3)) is False
        assert buf.add_expand(_expand(4)) is True

    def test_drain_returns_none_when_empty(self) -> None:
        buf = SearchEventBuffer(max_events=100, flush_every=1)
        assert buf.drain() is None

    def test_drain_zeroes_flush_counter_even_on_none(self) -> None:
        """Drain on an empty buffer still resets the threshold counter
        \u2014 callers rely on this so a no-op flush after a forced
        terminal lifecycle event does not leave a stale count.
        """
        buf = SearchEventBuffer(max_events=100, flush_every=2)
        buf.add_expand(_expand(1))
        buf.drain()  # discards the one event, zeroes counter
        assert buf.add_expand(_expand(2)) is False  # counter starts at 1, not 2


class TestCapAndTruncation:
    def test_cap_records_single_truncation_marker(self) -> None:
        buf = SearchEventBuffer(max_events=2, flush_every=100)
        buf.add_expand(_expand(1))
        buf.add_expand(_expand(2))
        buf.add_expand(_expand(3))  # over cap
        buf.add_expand(_expand(4))  # over cap
        snapshot = buf.drain()
        assert snapshot is not None
        kinds = [e["kind"] for e in snapshot]
        # 2 retained expansions + 1 truncation marker = 3 entries.
        assert kinds == ["search_expand", "search_expand", "search_truncated"]
        # Marker carries the cumulative dropped count, updated in place.
        truncation = snapshot[-1]
        assert truncation["dropped"] == 2

    def test_uncapped_events_bypass_cap(self) -> None:
        buf = SearchEventBuffer(max_events=1, flush_every=100)
        buf.add_expand(_expand(1))
        buf.add_expand(_expand(2))  # capped
        buf.add_uncapped({"kind": "search_dead_end", "reason": "exhausted"})
        snapshot = buf.drain()
        assert snapshot is not None
        kinds = [e["kind"] for e in snapshot]
        assert "search_dead_end" in kinds
        assert "search_truncated" in kinds
        # The dead_end event itself is retained even when expansions are capped.
        dead_ends = [e for e in snapshot if e["kind"] == "search_dead_end"]
        assert len(dead_ends) == 1


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        buf = SearchEventBuffer(max_events=2, flush_every=100)
        for i in range(5):
            buf.add_expand(_expand(i))
        # Sanity: cap fired and truncation marker present.
        snap_before = buf.drain()
        assert snap_before is not None
        assert any(e["kind"] == "search_truncated" for e in snap_before)

        buf.reset()
        # After reset the buffer behaves as if freshly constructed.
        assert buf.drain() is None
        assert buf.add_expand(_expand(99)) is False
        snap_after = buf.drain()
        assert snap_after is not None
        assert len(snap_after) == 1
        assert snap_after[0]["node_id"] == 99
