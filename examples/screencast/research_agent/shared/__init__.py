"""Shared utilities for the research-agent screencast.

Re-exports the tickers fixture, cost meter, and tool factories so the
four scripts (`react_baseline`, `langgraph_routed`, `langgoap_planned`,
`disrupted`) each compose the same primitives.

Importing this package also loads a project-root ``.env`` (if present
and ``python-dotenv`` is installed) so ``OPENAI_API_KEY`` /
``TAVILY_API_KEY`` are available without an explicit ``source .env``.
"""

try:  # pragma: no cover - optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Keep the screencast console clean. LangGraph >=1.0 prints a
# pending-deprecation banner for ``create_react_agent``; the import
# itself still works and we deliberately want to compare against it.
import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r".*create_react_agent has been moved.*",
)
_warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
)

from examples.screencast.research_agent.shared.cost_meter import (
    TAVILY_USD_PER_CALL,
    CostMeter,
)
from examples.screencast.research_agent.shared.result import RunResult
from examples.screencast.research_agent.shared.tickers import (
    TICKER_HINTS,
    TICKERS,
)
from examples.screencast.research_agent.shared.tools import (
    build_research_tools,
)

__all__ = [
    "TICKERS",
    "TICKER_HINTS",
    "TAVILY_USD_PER_CALL",
    "CostMeter",
    "RunResult",
    "build_research_tools",
]
