"""The 10 tickers the screencast agent researches.

Kept small enough for ~$1–$3 of recording cost per take, large enough
that replanning visibly improves on the naive baseline. Mix of
big-tech and non-tech so the agent cannot hard-code a sector
heuristic — it must actually fetch.

``TICKER_HINTS`` is consumed only by the mocked integration test (and
by ``--offline`` runs); production code paths ignore it and read
search results from Tavily / DuckDuckGo.
"""

from __future__ import annotations

from typing import TypedDict

TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "WMT",
    "NFLX",
]


class TickerHint(TypedDict):
    """Reference data used by the offline / mocked path only."""

    company: str
    eps_reported: float
    eps_consensus: float
    quarter: str


TICKER_HINTS: dict[str, TickerHint] = {
    "AAPL": {
        "company": "Apple Inc.",
        "eps_reported": 1.64,
        "eps_consensus": 1.60,
        "quarter": "Q1 2026",
    },
    "MSFT": {
        "company": "Microsoft Corporation",
        "eps_reported": 3.46,
        "eps_consensus": 3.32,
        "quarter": "Q1 2026",
    },
    "GOOG": {
        "company": "Alphabet Inc.",
        "eps_reported": 2.31,
        "eps_consensus": 2.22,
        "quarter": "Q4 2025",
    },
    "AMZN": {
        "company": "Amazon.com Inc.",
        "eps_reported": 1.43,
        "eps_consensus": 1.31,
        "quarter": "Q4 2025",
    },
    "META": {
        "company": "Meta Platforms Inc.",
        "eps_reported": 7.14,
        "eps_consensus": 6.78,
        "quarter": "Q4 2025",
    },
    "NVDA": {
        "company": "NVIDIA Corporation",
        "eps_reported": 0.81,
        "eps_consensus": 0.85,
        "quarter": "Q4 FY2026",
    },
    "TSLA": {
        "company": "Tesla, Inc.",
        "eps_reported": 0.66,
        "eps_consensus": 0.74,
        "quarter": "Q4 2025",
    },
    "JPM": {
        "company": "JPMorgan Chase & Co.",
        "eps_reported": 4.81,
        "eps_consensus": 4.62,
        "quarter": "Q4 2025",
    },
    "WMT": {
        "company": "Walmart Inc.",
        "eps_reported": 0.66,
        "eps_consensus": 0.65,
        "quarter": "Q4 FY2026",
    },
    "NFLX": {
        "company": "Netflix, Inc.",
        "eps_reported": 4.27,
        "eps_consensus": 4.19,
        "quarter": "Q4 2025",
    },
}
