"""USD cost tracking for the research-agent screencast.

Wraps :class:`langgoap.CostAccumulator` so all four versions
(react, langgraph_routed, langgoap_planned, disrupted) report cost
the same way. The accumulator handles LLM token cost; we add manual
bookkeeping for Tavily / DuckDuckGo per-call charges so the meter
reflects what a real OpenAI + Tavily bill would show.

``TAVILY_USD_PER_CALL`` is Tavily's published per-search rate as of
2026-05. Override via ``CostMeter(tavily_per_call=...)`` if your plan
differs. DuckDuckGo is free.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from langgoap import CostAccumulator

TAVILY_USD_PER_CALL: float = 0.008

REVEAL_ENV = "LANGGOAP_REVEAL_COST_LIVE"


@contextmanager
def reveal_cost_live() -> Iterator[None]:
    """Enable per-event cost printing for the duration of the block.

    Sets ``LANGGOAP_REVEAL_COST_LIVE=1`` on entry and restores the prior
    value on exit. :class:`CostMeter` checks this var on every charge.
    """
    prev = os.environ.get(REVEAL_ENV)
    os.environ[REVEAL_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(REVEAL_ENV, None)
        else:
            os.environ[REVEAL_ENV] = prev


@dataclass
class CostMeter:
    """Composable cost meter for the screencast.

    Mutates a ``world_state``-shaped dict so :class:`MaxCostPolicy`
    and :class:`ConstraintSpec(key="cost_usd")` see live values
    without further wiring.

    Attributes:
        world_state: The dict that LangGOAP / the screencast scripts
            also read from. Mutated in place.
        tavily_per_call: USD cost per Tavily search call.
        events: Timestamped (event_type, usd_delta) tuples — used to
            render cost-over-time charts in the notebook.
    """

    world_state: dict[str, Any]
    tavily_per_call: float = TAVILY_USD_PER_CALL
    events: list[tuple[float, str, float]] = field(default_factory=list)
    _t0: float = field(default_factory=time.monotonic)
    _llm_handler: CostAccumulator | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.world_state.setdefault("total_cost_usd", 0.0)
        self.world_state.setdefault("total_tokens", 0)
        self.world_state.setdefault("llm_call_count", 0)
        self.world_state.setdefault("tavily_call_count", 0)
        self.world_state.setdefault("ddg_call_count", 0)
        # Build the LLM cost handler now so it shares our dict.
        self._llm_handler = _MeteredCostAccumulator(self)

    @property
    def llm_callback(self) -> CostAccumulator:
        """Pass this to ``ChatOpenAI(callbacks=[meter.llm_callback])``."""
        assert self._llm_handler is not None
        return self._llm_handler

    def record_tavily_call(self) -> None:
        """Add one Tavily call's USD cost to the running total."""
        self._charge("tavily", self.tavily_per_call)
        self.world_state["tavily_call_count"] = (
            int(self.world_state.get("tavily_call_count", 0)) + 1
        )

    def record_ddg_call(self) -> None:
        """Record a DuckDuckGo call (zero cost, but counted)."""
        self._charge("ddg", 0.0)
        self.world_state["ddg_call_count"] = (
            int(self.world_state.get("ddg_call_count", 0)) + 1
        )

    def _charge(self, kind: str, usd: float) -> None:
        if usd:
            self.world_state["total_cost_usd"] = (
                float(self.world_state.get("total_cost_usd", 0.0)) + usd
            )
        self.events.append((time.monotonic() - self._t0, kind, usd))
        self._reveal(kind)

    def _reveal(self, kind: str) -> None:
        """Print a one-line live status if ``LANGGOAP_REVEAL_COST_LIVE`` is set."""
        if not os.environ.get(REVEAL_ENV):
            return
        s = self.snapshot()
        line = (
            f"  \u00b7 t={s['elapsed_s']:5.1f}s  "
            f"cost=${s['total_cost_usd']:.4f}  "
            f"tok={s['total_tokens']:>6,}  "
            f"calls={s['llm_call_count']:>3}  "
            f"tav={s['tavily_call_count']:>2}  "
            f"ddg={s['ddg_call_count']:>2}  ({kind})"
        )
        print(line, file=sys.stderr, flush=True)

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the meter's current totals."""
        ws = self.world_state
        return {
            "total_cost_usd": float(ws.get("total_cost_usd", 0.0)),
            "total_tokens": int(ws.get("total_tokens", 0)),
            "llm_call_count": int(ws.get("llm_call_count", 0)),
            "tavily_call_count": int(ws.get("tavily_call_count", 0)),
            "ddg_call_count": int(ws.get("ddg_call_count", 0)),
            "elapsed_s": time.monotonic() - self._t0,
        }

    def format_summary(self) -> str:
        s = self.snapshot()
        return (
            f"  cost:        ${s['total_cost_usd']:.4f}\n"
            f"  tokens:      {s['total_tokens']:,}\n"
            f"  LLM calls:   {s['llm_call_count']}\n"
            f"  Tavily:      {s['tavily_call_count']}\n"
            f"  DuckDuckGo:  {s['ddg_call_count']}\n"
            f"  wall-clock:  {s['elapsed_s']:.1f}s"
        )


class _MeteredCostAccumulator(CostAccumulator):
    """``CostAccumulator`` variant that also records LLM events."""

    def __init__(self, meter: CostMeter) -> None:
        super().__init__(meter.world_state)
        self._meter = meter
        self._last_total = float(meter.world_state.get("total_cost_usd", 0.0))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        super().on_llm_end(response, **kwargs)
        new_total = float(self._meter.world_state.get("total_cost_usd", 0.0))
        delta = new_total - self._last_total
        self._last_total = new_total
        self._meter.events.append((time.monotonic() - self._meter._t0, "llm", delta))
        self._meter._reveal("llm")
