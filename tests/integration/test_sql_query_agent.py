r"""Integration test for the Text-to-SQL agent tutorial (Tier 2).

Exercises LangGOAP against an in-memory SQLite database seeded from the
compact Chinook subset in
``examples/tutorials/tutorial_examples/data/sql_query_instance.py``.
The battery verifies four pieces of the GOAP loop at once:

- **Action sequencing** — ``explore_schema`` must fire before any draft
  action because every draft preconditions on ``schema_known=True``.
- **Cheap-first selection** — A\* picks ``explore_schema ->
  fast_query_attempt`` (total cost 2) before trying the more expensive
  ``careful_query_attempt``.
- **Replanning on execution failure** — the fast draft uses a
  deliberately misspelled column so sqlite raises ``OperationalError``;
  the executor blacklists the action and replanning falls back on the
  correct draft.
- **``effect_validator`` as a soundness check** —
  :func:`_results_non_empty` is invoked directly against synthetic pre-
  and post-states so the failure path is covered without relying on
  state pollution through the executor.

The tutorial file is
``examples/tutorials/tutorial_examples/sql_query_agent.py``.
"""

from __future__ import annotations

import sqlite3

from tutorial_examples.data.sql_query_instance import EXPECTED_TOP_SPENDERS
from tutorial_examples.sql_query_agent import (
    CAREFUL_QUERY_SQL,
    FAST_QUERY_SQL,
    _results_non_empty,
    build_in_memory_db,
    sql_query_agent_actions,
    sql_query_agent_goal,
    sql_query_agent_start,
)

from langgoap import (
    GoalInterpreter,
    GoapGraph,
    InterpretedGoal,
)
from langgoap.planner.astar import plan as astar_plan
from langgoap.state import PlanningState
from tests.conftest import FakeStructuredModel


class TestSqlQueryAgentPlanning:
    """A* sequencing and cheap-first selection."""

    def test_initial_plan_explores_schema_then_runs_fast_query(self) -> None:
        """A\\* prefers the cheap explore -> fast path (cost 2)."""
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        plan_obj = astar_plan(
            PlanningState.from_dict(sql_query_agent_start()),
            sql_query_agent_goal(),
            actions,
        )
        assert plan_obj is not None
        assert plan_obj.action_names == ["explore_schema", "fast_query_attempt"]
        assert plan_obj.total_cost == 2.0

    def test_without_schema_exploration_no_plan_is_possible(self) -> None:
        """Dropping explore_schema from the catalog breaks the chain."""
        conn = build_in_memory_db()
        actions = [
            a for a in sql_query_agent_actions(conn) if a.name != "explore_schema"
        ]
        plan_obj = astar_plan(
            PlanningState.from_dict(sql_query_agent_start()),
            sql_query_agent_goal(),
            actions,
        )
        assert plan_obj is None


class TestSqlQueryAgentEndToEnd:
    """End-to-end GoapGraph invocation recovers from the fast-query failure."""

    def test_graph_recovers_and_returns_correct_answer(self) -> None:
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        graph = GoapGraph(actions=actions)
        result = graph.invoke(
            goal=sql_query_agent_goal(),
            world_state=sql_query_agent_start(),
        )

        assert result["status"] == "goal_achieved"
        assert result["world_state"]["answer_ready"] is True
        assert result["world_state"]["sql"] == CAREFUL_QUERY_SQL
        assert result["world_state"]["results"] == list(EXPECTED_TOP_SPENDERS)

    def test_fast_query_is_blacklisted_and_careful_query_is_used(self) -> None:
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        graph = GoapGraph(actions=actions)
        result = graph.invoke(
            goal=sql_query_agent_goal(),
            world_state=sql_query_agent_start(),
        )

        assert result["replan_count"] >= 1
        assert "fast_query_attempt" in result.get("blacklisted_actions", [])

        successes = [h for h in result["execution_history"] if h.success]
        success_names = [h.action_name for h in successes]
        assert "explore_schema" in success_names
        assert "careful_query_attempt" in success_names

        failures = [h for h in result["execution_history"] if not h.success]
        failure_names = [h.action_name for h in failures]
        assert "fast_query_attempt" in failure_names
        # sqlite's error message mentions the misspelled column.
        fast_failure = next(
            h for h in failures if h.action_name == "fast_query_attempt"
        )
        assert "Frst_Name" in (fast_failure.error or "")

    def test_explore_schema_populates_table_list(self) -> None:
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        graph = GoapGraph(actions=actions)
        result = graph.invoke(
            goal=sql_query_agent_goal(),
            world_state=sql_query_agent_start(),
        )

        tables = result["world_state"].get("tables")
        assert tables == ["Customer", "Invoice"]

    async def test_ainvoke_end_to_end(self) -> None:
        """Async path reaches the goal using the same blacklist semantics."""
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        graph = GoapGraph(actions=actions)
        result = await graph.ainvoke(
            goal=sql_query_agent_goal(),
            world_state=sql_query_agent_start(),
        )

        assert result["status"] == "goal_achieved"
        assert "fast_query_attempt" in result.get("blacklisted_actions", [])
        assert result["world_state"]["results"] == list(EXPECTED_TOP_SPENDERS)


class TestSqlQueryAgentEffectValidator:
    """Direct coverage of the soundness validator."""

    def test_validator_rejects_empty_results(self) -> None:
        pre = {"schema_known": True, "answer_ready": False}
        post = {
            "schema_known": True,
            "answer_ready": True,
            "sql": CAREFUL_QUERY_SQL,
            "results": [],
        }
        assert _results_non_empty(pre, post) is False

    def test_validator_accepts_non_empty_results(self) -> None:
        pre = {"schema_known": True, "answer_ready": False}
        post = {
            "schema_known": True,
            "answer_ready": True,
            "sql": CAREFUL_QUERY_SQL,
            "results": [("Helena Holý", 49.62)],
        }
        assert _results_non_empty(pre, post) is True

    def test_validator_rejects_missing_results_key(self) -> None:
        pre = {"schema_known": True, "answer_ready": False}
        post = {"schema_known": True, "answer_ready": True}
        assert _results_non_empty(pre, post) is False

    def test_careful_query_validator_attached(self) -> None:
        conn = build_in_memory_db()
        actions = {a.name: a for a in sql_query_agent_actions(conn)}
        assert actions["careful_query_attempt"].effect_validator is not None
        # fast_query_attempt relies on execution exceptions, not a validator.
        assert actions["fast_query_attempt"].effect_validator is None

    def test_validator_catches_empty_database_live(self) -> None:
        """careful_query_attempt fires against an empty DB → validator False."""
        conn = build_in_memory_db(seed=False)
        actions = {a.name: a for a in sql_query_agent_actions(conn)}
        careful = actions["careful_query_attempt"]

        # Run the execute callable directly then feed its output to the
        # validator — mirrors what the executor does before flipping the
        # action to "action_failed".
        assert careful.execute is not None
        raw = careful.execute({"schema_known": True, "answer_ready": False})
        assert isinstance(raw, dict)
        assert raw["results"] == []
        post = {"schema_known": True, "answer_ready": False, **raw}
        assert careful.validate_effects({}, post) is False


class TestSqlQueryAgentFailureIsolation:
    """The broken ``fast_query_attempt`` isolates its error from the DB."""

    def test_connection_remains_usable_after_failed_draft(self) -> None:
        """sqlite raises on the bad query but the connection stays open."""
        conn = build_in_memory_db()
        try:
            conn.execute(FAST_QUERY_SQL).fetchall()
        except sqlite3.OperationalError as exc:
            assert "Frst_Name" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("fast query should have raised")

        # The correct query still works.
        rows = conn.execute(CAREFUL_QUERY_SQL).fetchall()
        assert len(rows) == 3


def _nl_interpreted_goal() -> InterpretedGoal:
    return InterpretedGoal(
        conditions={"answer_ready": True},
        constraints=[],
        objectives=[],
        reasoning="The user wants a concrete answer, so answer_ready must be True.",
    )


class TestSqlQueryAgentNaturalLanguage:
    """NL intake via FakeStructuredModel + GoalInterpreter."""

    def test_nl_request_produces_runnable_goal(self) -> None:
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        interpreter = GoalInterpreter(llm=llm, actions=actions)

        goal = interpreter.interpret(
            "Show me the three customers who have spent the most."
        )
        assert goal.conditions == {"answer_ready": True}

        result = GoapGraph(actions=actions).invoke(
            goal=goal,
            world_state=sql_query_agent_start(),
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["results"] == list(EXPECTED_TOP_SPENDERS)

    def test_invoke_nl_reaches_goal(self) -> None:
        conn = build_in_memory_db()
        actions = sql_query_agent_actions(conn)
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        result = GoapGraph(actions=actions).invoke_nl(
            "Show me the three customers who have spent the most.",
            llm=llm,
            world_state=sql_query_agent_start(),
        )
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["results"] == list(EXPECTED_TOP_SPENDERS)
