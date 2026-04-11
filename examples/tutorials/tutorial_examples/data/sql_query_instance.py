r"""Minimal Chinook subset for the Text-to-SQL tutorial.

Provenance
----------
Structurally adapted from deepagents'
``research/repos/deepagents/examples/text-to-sql-agent`` which targets the
full `Chinook <https://github.com/lerocha/chinook-database>`_ sample
database.  For a runnable tutorial we compact the schema to just two
tables — ``Customer`` and ``Invoice`` — and seed them with a handful of
rows hand-picked so that the aggregated "top spenders" answer is
deterministic and easy to verify.

The tutorial uses :mod:`sqlite3` with an in-memory connection so nothing
is written to disk.

Schema
------
::

    Customer(CustomerId, FirstName, LastName, Country)
    Invoice(InvoiceId, CustomerId, Total)

The Invoice totals below are chosen so the ranked top-3 spenders are:

- Helena Holý (Czech Republic) — 49.62
- Richard Cunningham (USA)     — 47.62
- Luís Gonçalves (Brazil)      — 39.62
"""

from __future__ import annotations

CUSTOMER_ROWS: tuple[tuple[int, str, str, str], ...] = (
    (1, "Luís", "Gonçalves", "Brazil"),
    (2, "Leonie", "Köhler", "Germany"),
    (3, "François", "Tremblay", "Canada"),
    (4, "Helena", "Holý", "Czech Republic"),
    (5, "Richard", "Cunningham", "USA"),
)

INVOICE_ROWS: tuple[tuple[int, int, float], ...] = (
    # (InvoiceId, CustomerId, Total)
    (1, 1, 15.99),
    (2, 1, 11.99),
    (3, 1, 8.91),
    (4, 1, 2.73),
    (5, 2, 8.91),
    (6, 2, 5.94),
    (7, 3, 8.91),
    (8, 3, 1.98),
    (9, 4, 25.86),
    (10, 4, 14.91),
    (11, 4, 8.85),
    (12, 5, 21.86),
    (13, 5, 17.91),
    (14, 5, 7.85),
)


SCHEMA_DDL: tuple[str, ...] = (
    """
    CREATE TABLE Customer (
        CustomerId INTEGER PRIMARY KEY,
        FirstName  TEXT NOT NULL,
        LastName   TEXT NOT NULL,
        Country    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE Invoice (
        InvoiceId  INTEGER PRIMARY KEY,
        CustomerId INTEGER NOT NULL,
        Total      REAL    NOT NULL,
        FOREIGN KEY (CustomerId) REFERENCES Customer(CustomerId)
    )
    """,
)


# Expected top-3 result set — used by tests to pin the correct answer.
EXPECTED_TOP_SPENDERS: tuple[tuple[str, float], ...] = (
    ("Helena Holý", 49.62),
    ("Richard Cunningham", 47.62),
    ("Luís Gonçalves", 39.62),
)
