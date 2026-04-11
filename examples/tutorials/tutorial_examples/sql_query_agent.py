r"""Text-to-SQL agent — Tier 2 tutorial translating deepagents' example.

Adapts the workflow sketched in
``research/repos/deepagents/examples/text-to-sql-agent`` (schema
exploration → query drafting → execution) into a GOAP plan driven by
LangGoap.  The tutorial spotlights three features at once:

1. **Action sequencing** — ``explore_schema`` must fire before any draft
   action because drafting preconditions on ``schema_known=True``.
2. **Replanning on failure** — a cheap ``fast_query_attempt`` drafts a
   SQL string with a deliberate typo.  Running it raises
   ``sqlite3.OperationalError``, so the executor blacklists the action
   and the planner falls back on the more expensive
   ``careful_query_attempt``.
3. **``effect_validator`` as a soundness check** — the fallback action
   attaches a validator that verifies the query produced at least one
   row.  The same validator is exercised in the test suite against an
   empty database to show the failure path.

The tutorial uses :mod:`sqlite3` with an in-memory connection seeded
from :mod:`tutorial_examples.data.sql_query_instance` so the notebook
and tests share a single source of truth and never touch disk.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgoap import ActionSpec, GoalSpec

from .data.sql_query_instance import (
    CUSTOMER_ROWS,
    INVOICE_ROWS,
    SCHEMA_DDL,
)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------


def build_in_memory_db(*, seed: bool = True) -> sqlite3.Connection:
    """Return a fresh in-memory SQLite connection seeded from the fixture.

    ``check_same_thread=False`` is set so the async executor can run
    actions from a worker thread — the tutorial never issues concurrent
    writes so this relaxation is safe.

    Args:
        seed: When True (default), insert the fixture rows.  Pass
            ``False`` to exercise the empty-database branch in tests.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for ddl in SCHEMA_DDL:
        conn.execute(ddl)
    if seed:
        conn.executemany(
            "INSERT INTO Customer VALUES (?, ?, ?, ?)", CUSTOMER_ROWS
        )
        conn.executemany(
            "INSERT INTO Invoice VALUES (?, ?, ?)", INVOICE_ROWS
        )
        conn.commit()
    return conn


# ---------------------------------------------------------------------------
# SQL fragments the actions emit.
# ---------------------------------------------------------------------------

# Deliberately misspelled column (``Frst_Name``) so sqlite raises
# ``OperationalError: no such column`` when the executor runs it.
FAST_QUERY_SQL = """
SELECT c.Frst_Name, SUM(i.Total) AS Spent
FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId
GROUP BY c.CustomerId
ORDER BY Spent DESC
LIMIT 3
""".strip()

CAREFUL_QUERY_SQL = """
SELECT c.FirstName || ' ' || c.LastName AS Name,
       ROUND(SUM(i.Total), 2) AS Spent
FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId
GROUP BY c.CustomerId
ORDER BY Spent DESC
LIMIT 3
""".strip()


# ---------------------------------------------------------------------------
# Execute callables
# ---------------------------------------------------------------------------


def _make_explore_schema(conn: sqlite3.Connection) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = [r[0] for r in rows]
        return {
            "schema_known": True,
            "tables": tables,
        }

    return execute


def _make_fast_query_attempt(conn: sqlite3.Connection) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        # Running the deliberately broken SQL raises sqlite3.OperationalError
        # which bubbles up to the executor and triggers the blacklist.
        rows = conn.execute(FAST_QUERY_SQL).fetchall()
        return {
            "answer_ready": True,
            "sql": FAST_QUERY_SQL,
            "results": rows,
        }

    return execute


def _make_careful_query_attempt(conn: sqlite3.Connection) -> Any:
    def execute(ws: dict[str, Any]) -> dict[str, Any]:
        rows = conn.execute(CAREFUL_QUERY_SQL).fetchall()
        return {
            "answer_ready": True,
            "sql": CAREFUL_QUERY_SQL,
            "results": [(name, spent) for (name, spent) in rows],
        }

    return execute


def _results_non_empty(
    pre_state: dict[str, Any], post_state: dict[str, Any]
) -> bool:
    """Effect validator: the drafted query must produce at least one row."""
    results = post_state.get("results")
    if not isinstance(results, list):
        return False
    return len(results) > 0


# ---------------------------------------------------------------------------
# Action / goal factories
# ---------------------------------------------------------------------------


def sql_query_agent_actions(conn: sqlite3.Connection) -> list[ActionSpec]:
    """Return the GOAP action list bound to *conn*.

    Three actions are returned:

    - ``explore_schema`` (cost 1) — introspects ``sqlite_master`` and
      records the table list in world state.
    - ``fast_query_attempt`` (cost 1) — drafts a query with a deliberate
      typo; raises ``sqlite3.OperationalError`` when executed so the
      executor blacklists it.
    - ``careful_query_attempt`` (cost 3) — drafts a correct aggregated
      query and ships with an ``effect_validator`` that verifies the
      result set is non-empty.
    """
    return [
        ActionSpec(
            name="explore_schema",
            preconditions={"schema_known": False},
            effects={"schema_known": True},
            cost=1.0,
            execute=_make_explore_schema(conn),
        ),
        ActionSpec(
            name="fast_query_attempt",
            preconditions={"schema_known": True, "answer_ready": False},
            effects={"answer_ready": True},
            cost=1.0,
            execute=_make_fast_query_attempt(conn),
        ),
        ActionSpec(
            name="careful_query_attempt",
            preconditions={"schema_known": True, "answer_ready": False},
            effects={"answer_ready": True},
            cost=3.0,
            execute=_make_careful_query_attempt(conn),
            effect_validator=_results_non_empty,
        ),
    ]


def sql_query_agent_start() -> dict[str, Any]:
    """Initial world state: nothing explored, no answer yet."""
    return {
        "schema_known": False,
        "answer_ready": False,
    }


def sql_query_agent_goal() -> GoalSpec:
    """Goal: produce an answer to the user's question."""
    return GoalSpec(conditions={"answer_ready": True})
