r"""Task Assigning instance derived from standard benchmark data.

Provenance
----------
Derived from reference benchmark data for the task-assigning problem
(24 tasks x 8 employees x 6 skills x 4 task types, scored with a
``BendableScore`` of 1 hard level + 4 soft levels).

We collapse the problem to **6 tasks x 3 employees x 4 skills x 4
task types** and squash the multi-level soft score into a single
weighted ``weighted_delay`` resource so the current LangGoap
``HardSoftScore`` covers it.  The mapping is:

- **Hard constraint — no missing skills** → skill matching at
  action-build time (unqualified employees have no action to take
  the task, exactly like the nurse rostering tutorial).
- **Soft level 0 — critical priority end time** and
  **soft level 2 — major priority end time** and
  **soft level 3 — minor priority end time** →
  flattened into one ``weighted_delay`` resource via the
  ``priority × base_duration × affinity`` cost formula.  Critical
  tasks are weighted 4×, major 2×, minor 1×, so the planner
  naturally front-loads high-priority work.
- **Soft level 1 — minimize makespan** → modelled as a
  per-employee ``workload_<name>`` resource that can carry a soft
  cap driving load balancing (see
  :func:`task_assigning.task_assigning_goal_with_workload_cap`).

Multi-mode scheduling, per-period resource capacities, and true
task chaining (per-employee ordered execution) are out of scope
for this tutorial, exactly as documented in the project job
scheduling fixture.

Entities
--------
::

    Employees                Skills
      alice                   problem_solving, risk_management, strategic_planning
      bob                     problem_solving, creative_thinking
      carol                   risk_management, strategic_planning, creative_thinking

    Task types (req. skill → base duration)
      sales_strategy           strategic_planning  →  4h
      compliance               risk_management     →  3h
      brand_story              creative_thinking   →  5h
      root_cause               problem_solving     →  4h

    Tasks (6 total, customer × priority mix)
      t1  steel  sales_strategy  CRITICAL
      t2  paper  compliance      MAJOR
      t3  stone  brand_story     CRITICAL
      t4  wood   root_cause      MAJOR
      t5  steel  compliance      MINOR
      t6  paper  root_cause      MINOR

Affinity (employee -> task type, using the affinity model):
  HIGH=1 (natural fit), MEDIUM=2, LOW=3, NONE=no entry (skill filter already
  blocks unqualified pairs so NONE is effectively unreachable here).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskType:
    name: str
    required_skill: str
    base_duration_hours: int


@dataclass(frozen=True)
class Task:
    name: str
    task_type: str  # matches TaskType.name
    customer: str
    priority: str  # "critical" | "major" | "minor"


@dataclass(frozen=True)
class Employee:
    name: str
    skills: frozenset[str]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

SKILLS: tuple[str, ...] = (
    "problem_solving",
    "risk_management",
    "strategic_planning",
    "creative_thinking",
)


# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------

TASK_TYPES: tuple[TaskType, ...] = (
    TaskType(name="sales_strategy", required_skill="strategic_planning", base_duration_hours=4),
    TaskType(name="compliance", required_skill="risk_management", base_duration_hours=3),
    TaskType(name="brand_story", required_skill="creative_thinking", base_duration_hours=5),
    TaskType(name="root_cause", required_skill="problem_solving", base_duration_hours=4),
)

TASK_TYPE_BY_NAME: dict[str, TaskType] = {tt.name: tt for tt in TASK_TYPES}


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

EMPLOYEES: tuple[Employee, ...] = (
    Employee(
        name="alice",
        skills=frozenset({"problem_solving", "risk_management", "strategic_planning"}),
    ),
    Employee(
        name="bob",
        skills=frozenset({"problem_solving", "creative_thinking"}),
    ),
    Employee(
        name="carol",
        skills=frozenset({"risk_management", "strategic_planning", "creative_thinking"}),
    ),
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS: tuple[Task, ...] = (
    Task(name="t1", task_type="sales_strategy", customer="steel", priority="critical"),
    Task(name="t2", task_type="compliance", customer="paper", priority="major"),
    Task(name="t3", task_type="brand_story", customer="stone", priority="critical"),
    Task(name="t4", task_type="root_cause", customer="wood", priority="major"),
    Task(name="t5", task_type="compliance", customer="steel", priority="minor"),
    Task(name="t6", task_type="root_cause", customer="paper", priority="minor"),
)


# ---------------------------------------------------------------------------
# Priority weights
# ---------------------------------------------------------------------------

# The original benchmark uses separate soft levels per priority.  We flatten
# that into a single weighted_delay resource by multiplying the effective
# duration by a priority weight — critical tasks are the most expensive to delay.
PRIORITY_WEIGHT: dict[str, int] = {
    "critical": 4,
    "major": 2,
    "minor": 1,
}


# ---------------------------------------------------------------------------
# Affinity (employee -> task type -> duration multiplier)
# ---------------------------------------------------------------------------

# HIGH=1, MEDIUM=2, LOW=3; no entry = NONE (unreachable here because the skill
# filter drops every unqualified (employee, task) pair before the multiplier
# is ever read).
AFFINITY: dict[tuple[str, str], int] = {
    # alice — strongest at sales_strategy, moderate at compliance, weak at root_cause.
    ("alice", "sales_strategy"): 1,
    ("alice", "compliance"): 2,
    ("alice", "root_cause"): 3,
    # bob — creative/analytical generalist, great at brand_story and root_cause.
    ("bob", "brand_story"): 1,
    ("bob", "root_cause"): 1,
    # carol — compliance specialist, decent branding, weak at sales_strategy.
    ("carol", "sales_strategy"): 3,
    ("carol", "compliance"): 1,
    ("carol", "brand_story"): 2,
}
