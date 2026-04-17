"""Cloud balancing instance derived from standard benchmark data.

Provenance
----------
Derived from a standard 2-computer / 6-process cloud-balancing
benchmark instance.

The full benchmark has 6 processes; this fixture keeps 4 of them
(p0, p1, p2, p5) so the CSP pipeline solves in well under a second
while still exercising forced assignments (p1 requires more memory
than ``server_small`` can provide, so it is pinned to
``server_big``) and flexible choices (p0, p2, p5 all fit on either
server).

The per-action ``cost_usd`` value is a *simplification* for the
tutorial: the original base-cost-per-computer model would require
non-linear activation constraints that CP-SAT cannot express
directly.  Each ``assign`` action instead charges an amortized
per-process rate — ``480`` on ``server_big``, ``66`` on
``server_small`` — obtained by dividing the original ``cost`` fields
by 10.  The relative ordering (big is pricier than small) is
preserved so the optimizer still has a meaningful objective to
minimize.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Computer:
    name: str
    cpu: int
    memory: int
    network: int
    cost_usd_per_proc: float


@dataclass(frozen=True)
class Process:
    name: str
    cpu: int
    memory: int
    network: int


# ---------------------------------------------------------------------------
# 2-computer / 4-process instance (subset of the standard 2x6 benchmark)
# ---------------------------------------------------------------------------

SERVERS: tuple[Computer, ...] = (
    Computer(
        name="server_big",
        cpu=24,
        memory=96,
        network=16,
        cost_usd_per_proc=480.0,
    ),
    Computer(
        name="server_small",
        cpu=6,
        memory=4,
        network=6,
        cost_usd_per_proc=66.0,
    ),
)


PROCESSES: tuple[Process, ...] = (
    Process(name="p0", cpu=1, memory=1, network=1),
    # p1 requires 6 memory — forced onto server_big (server_small has mem=4).
    Process(name="p1", cpu=3, memory=6, network=1),
    Process(name="p2", cpu=1, memory=1, network=3),
    Process(name="p5", cpu=1, memory=1, network=5),
)
