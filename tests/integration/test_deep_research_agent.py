r"""Integration test for the deep research agent tutorial (Tier 3).

Exercises LangGoap against the hermetic in-memory research corpus in
``examples/tutorials/tutorial_examples/data/deep_research_instance.py``.
The battery pins the five features the tutorial showcases:

- **Layer A on-ramp** — :func:`create_goap_agent` produces a compiled
  LangGraph from a list of ``@tool`` functions without any manual
  ``ActionSpec`` plumbing.
- **Replanning on rate-limit error** — the cheap ``search_broad_corpus``
  tool raises :class:`RateLimitError` against the default workspace.
  The executor blacklists the action and the planner's fallback routes
  through ``search_deep_corpus`` on the next pass.
- **Tracer observability** — a recording :class:`PlanningTracer` proves
  every plan/replan/goal-achieved event fires in the expected order.
- **Execution history in Store** — :class:`StoreExecutionHistory` backed
  by ``InMemoryStore`` records a success after a run and
  ``query_by_goal`` retrieves it for diagnostic use.
- **Natural language goal intake** — ``GoapGraph.invoke_nl`` driven by a
  :class:`FakeStructuredModel` produces the same successful report.

The corresponding tutorial file is
``examples/tutorials/tutorial_examples/deep_research_agent.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore
from tutorial_examples.data.deep_research_instance import (
    EXPECTED_BROAD_FINDING_IDS,
    EXPECTED_DEEP_FINDING_IDS,
    EXPECTED_TOPICS,
    MIN_REPORT_CITATIONS,
    RESEARCH_REQUEST,
)
from tutorial_examples.deep_research_agent import (
    InsufficientEvidenceError,
    RateLimitError,
    ResearchWorkspace,
    deep_research_costs,
    deep_research_effects,
    deep_research_goal,
    deep_research_preconditions,
    deep_research_start,
    deep_research_tools,
)

from langgoap import (
    ActionSpec,
    ExecutionRecord,
    GoapGraph,
    InterpretedGoal,
    StoreExecutionHistory,
    compute_goal_hash,
    create_goap_agent,
)
from langgoap.integrations import goapify_tool
from langgoap.testing import FakeStructuredModel

# ---------------------------------------------------------------------------
# Unit-level coverage of the ResearchWorkspace helpers
# ---------------------------------------------------------------------------


class TestResearchWorkspaceUnit:
    """Direct tests on the workspace helpers used by every research tool."""

    def test_save_request_persists_query(self) -> None:
        ws = ResearchWorkspace()
        ws.save_request("what is GOAP?")
        assert ws.research_request == "what is GOAP?"

    def test_decompose_without_request_raises(self) -> None:
        ws = ResearchWorkspace()
        with pytest.raises(RuntimeError, match="before save_request"):
            ws.decompose()

    def test_decompose_extracts_expected_topics(self) -> None:
        ws = ResearchWorkspace()
        ws.save_request(RESEARCH_REQUEST)
        topics = ws.decompose()
        assert tuple(topics) == EXPECTED_TOPICS
        assert tuple(ws.topics) == EXPECTED_TOPICS

    def test_search_broad_raises_when_rate_limited(self) -> None:
        ws = ResearchWorkspace()
        ws.save_request(RESEARCH_REQUEST)
        ws.decompose()
        with pytest.raises(RateLimitError, match="rate limit"):
            ws.search_broad()
        # Findings must stay empty — no partial state after a raise.
        assert ws.findings == []

    def test_search_broad_returns_one_per_topic_when_open(self) -> None:
        ws = ResearchWorkspace(rate_limit_active=False)
        ws.save_request(RESEARCH_REQUEST)
        ws.decompose()
        findings = ws.search_broad()
        assert tuple(f.id for f in findings) == EXPECTED_BROAD_FINDING_IDS
        assert tuple(f.id for f in ws.findings) == EXPECTED_BROAD_FINDING_IDS

    def test_search_deep_covers_all_matching_documents(self) -> None:
        ws = ResearchWorkspace(rate_limit_active=True)
        ws.save_request(RESEARCH_REQUEST)
        ws.decompose()
        findings = ws.search_deep()
        assert tuple(f.id for f in findings) == EXPECTED_DEEP_FINDING_IDS
        # search_deep never raises — rate_limit_active is irrelevant.

    def test_synthesize_requires_minimum_unique_citations(self) -> None:
        ws = ResearchWorkspace()
        ws.save_request(RESEARCH_REQUEST)
        ws.decompose()
        # Only feed a single document so unique URLs < MIN_REPORT_CITATIONS.
        ws.findings = [ws.corpus[0]]
        with pytest.raises(InsufficientEvidenceError) as excinfo:
            ws.synthesize()
        assert str(MIN_REPORT_CITATIONS) in str(excinfo.value)
        assert ws.report is None

    def test_synthesize_builds_report_with_citations(self) -> None:
        ws = ResearchWorkspace(rate_limit_active=False)
        ws.save_request(RESEARCH_REQUEST)
        ws.decompose()
        ws.search_broad()
        report = ws.synthesize()
        assert report["request"] == RESEARCH_REQUEST
        assert tuple(report["topics"]) == EXPECTED_TOPICS
        assert len(report["citations"]) == len(EXPECTED_BROAD_FINDING_IDS)
        urls = {c["url"] for c in report["citations"]}
        assert len(urls) >= MIN_REPORT_CITATIONS


# ---------------------------------------------------------------------------
# Layer A: create_goap_agent end-to-end
# ---------------------------------------------------------------------------


class TestDeepResearchLayerA:
    """End-to-end Layer A wiring via ``create_goap_agent``."""

    def test_happy_path_uses_broad_search_when_unrestricted(self) -> None:
        """With no rate limit the cheap broad path fires end-to-end."""
        workspace = ResearchWorkspace(rate_limit_active=False)
        agent = create_goap_agent(
            tools=deep_research_tools(workspace),
            goal=deep_research_goal(),
            preconditions=deep_research_preconditions(),
            effects=deep_research_effects(),
            costs=deep_research_costs(),
        )
        result = agent.invoke(
            {
                "goal": deep_research_goal(),
                "world_state": deep_research_start(),
            }
        )
        assert result["status"] == "goal_achieved"
        # No blacklist entries, no replans — the cheap path succeeded.
        assert result["replan_count"] == 0
        assert "search_broad_corpus" not in result.get("blacklisted_actions", [])

        # The workspace should hold a fully built report sourced from
        # the broad findings.
        assert workspace.report is not None
        assert tuple(workspace.topics) == EXPECTED_TOPICS
        assert tuple(f.id for f in workspace.findings) == EXPECTED_BROAD_FINDING_IDS
        assert len(workspace.report["citations"]) == len(EXPECTED_BROAD_FINDING_IDS)

    def test_rate_limited_run_falls_back_to_deep_search(self) -> None:
        """Rate limit forces the planner to replan through deep search."""
        workspace = ResearchWorkspace(rate_limit_active=True)
        agent = create_goap_agent(
            tools=deep_research_tools(workspace),
            goal=deep_research_goal(),
            preconditions=deep_research_preconditions(),
            effects=deep_research_effects(),
            costs=deep_research_costs(),
        )
        result = agent.invoke(
            {
                "goal": deep_research_goal(),
                "world_state": deep_research_start(),
            }
        )

        assert result["status"] == "goal_achieved"
        # Exactly one replan: broad fails once, then deep search carries.
        assert result["replan_count"] == 1
        assert "search_broad_corpus" in result.get("blacklisted_actions", [])

        history = result["execution_history"]
        failures = [h for h in history if not h.success]
        failure_names = [h.action_name for h in failures]
        assert "search_broad_corpus" in failure_names
        rate_failure = next(
            h for h in failures if h.action_name == "search_broad_corpus"
        )
        assert "rate limit" in (rate_failure.error or "")

        successes = [h for h in history if h.success]
        success_names = [h.action_name for h in successes]
        assert success_names == [
            "save_research_request",
            "decompose_topics",
            "search_deep_corpus",
            "synthesize_report",
        ]

        # The workspace's report must reflect the deep-search corpus.
        assert workspace.report is not None
        assert tuple(f.id for f in workspace.findings) == EXPECTED_DEEP_FINDING_IDS
        assert len(workspace.report["citations"]) == len(EXPECTED_DEEP_FINDING_IDS)

    async def test_ainvoke_recovers_from_rate_limit(self) -> None:
        """Async path traverses the same blacklist + fallback semantics."""
        workspace = ResearchWorkspace(rate_limit_active=True)
        agent = create_goap_agent(
            tools=deep_research_tools(workspace),
            goal=deep_research_goal(),
            preconditions=deep_research_preconditions(),
            effects=deep_research_effects(),
            costs=deep_research_costs(),
        )
        result = await agent.ainvoke(
            {
                "goal": deep_research_goal(),
                "world_state": deep_research_start(),
            }
        )
        assert result["status"] == "goal_achieved"
        assert "search_broad_corpus" in result.get("blacklisted_actions", [])
        assert workspace.report is not None
        assert len(workspace.report["citations"]) == len(EXPECTED_DEEP_FINDING_IDS)


# ---------------------------------------------------------------------------
# Tracing — recording tracer pins the observability hook order
# ---------------------------------------------------------------------------


class _RecordingTracer:
    """Tracer that appends ``(hook, payload)`` tuples for assertion.

    Every hook stores a tuple with the minimum information needed to
    make test assertions readable.  The recorder never reaches into
    the executor's internal return dict — it captures the ``action``
    argument of :meth:`on_action_start` and keys subsequent
    ``on_action_complete`` events off the most recent start, which
    makes the recorder resilient to executor internal changes (L1).
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self._current_action: str | None = None

    def on_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.events.append(("plan_start", strategy_name))

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.events.append(("plan_complete", tuple(plan.action_names)))

    def on_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.events.append(("plan_failed", reason))

    def on_action_start(self, action: Any, state: Any) -> None:
        self._current_action = action.name
        self.events.append(("action_start", action.name))

    def on_action_complete(self, result: Any) -> None:
        # Key the complete event off the most recent action_start.
        # The executor's return dict shape is an internal detail.
        self.events.append(("action_complete", self._current_action))
        self._current_action = None

    def on_replan(self, reason: str, new_plan: Any) -> None:
        plan_names = tuple(new_plan.action_names) if new_plan is not None else ()
        self.events.append(("replan", (reason, plan_names)))

    def on_goal_achieved(self, final_state: Any) -> None:
        self.events.append(("goal_achieved", None))

    # Async parity — sync helpers delegate to keep the recording single-sourced.
    async def aon_plan_start(self, goal: Any, state: Any, strategy_name: str) -> None:
        self.on_plan_start(goal, state, strategy_name)

    async def aon_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.on_plan_complete(plan, duration_ms)

    async def aon_plan_failed(self, reason: str, duration_ms: float) -> None:
        self.on_plan_failed(reason, duration_ms)

    async def aon_action_start(self, action: Any, state: Any) -> None:
        self.on_action_start(action, state)

    async def aon_action_complete(self, result: Any) -> None:
        self.on_action_complete(result)

    async def aon_replan(self, reason: str, new_plan: Any) -> None:
        self.on_replan(reason, new_plan)

    async def aon_goal_achieved(self, final_state: Any) -> None:
        self.on_goal_achieved(final_state)


def _build_actions(
    *, rate_limit_active: bool = False
) -> tuple[ResearchWorkspace, list[ActionSpec]]:
    """Build the ActionSpec list the same way create_goap_agent does.

    Used by the tracer/history tests that want a :class:`GoapGraph`
    directly rather than the compiled Layer A agent.  The
    ``rate_limit_active`` flag controls whether
    ``search_broad_corpus`` will raise on its first call — callers
    that want to exercise the replan path flip it to ``True`` instead
    of mutating the returned workspace after construction (M2).
    """
    workspace = ResearchWorkspace(rate_limit_active=rate_limit_active)
    pre = deep_research_preconditions()
    eff = deep_research_effects()
    costs = deep_research_costs()
    actions: list[ActionSpec] = [
        goapify_tool(
            tool,
            preconditions=pre.get(tool.name),
            effects=eff.get(tool.name),
            cost=costs.get(tool.name, 1.0),
        )
        for tool in deep_research_tools(workspace)
    ]
    return workspace, actions


class TestDeepResearchTracing:
    """``PlanningTracer`` is called at every documented checkpoint."""

    def test_tracer_fires_plan_action_and_goal_hooks(self) -> None:
        tracer = _RecordingTracer()
        workspace, actions = _build_actions(rate_limit_active=False)
        graph = GoapGraph(actions=actions, tracer=tracer)
        result = graph.invoke(
            goal=deep_research_goal(),
            world_state=deep_research_start(),
        )
        assert result["status"] == "goal_achieved"

        hooks = [hook for hook, _ in tracer.events]
        # Exactly one plan, no replan — happy path is deterministic.
        assert hooks[0] == "plan_start"
        assert hooks.count("plan_start") == 1
        assert hooks.count("plan_complete") == 1
        # Four actions fire in order — no >= weakening (M4).
        assert hooks.count("action_start") == 4
        assert hooks.count("action_complete") == 4
        assert "replan" not in hooks
        assert hooks[-1] == "goal_achieved"

        # The completed plan is the exact cheap-first chain.
        plan_completes = [
            payload for hook, payload in tracer.events if hook == "plan_complete"
        ]
        assert plan_completes == [
            (
                "save_research_request",
                "decompose_topics",
                "search_broad_corpus",
                "synthesize_report",
            )
        ]

        # action_start / action_complete names interleave in order.
        action_starts = [
            payload for hook, payload in tracer.events if hook == "action_start"
        ]
        action_completes = [
            payload for hook, payload in tracer.events if hook == "action_complete"
        ]
        expected_sequence = [
            "save_research_request",
            "decompose_topics",
            "search_broad_corpus",
            "synthesize_report",
        ]
        assert action_starts == expected_sequence
        assert action_completes == expected_sequence

    def test_tracer_records_replan_on_rate_limit(self) -> None:
        tracer = _RecordingTracer()
        workspace, actions = _build_actions(rate_limit_active=True)
        graph = GoapGraph(actions=actions, tracer=tracer)
        result = graph.invoke(
            goal=deep_research_goal(),
            world_state=deep_research_start(),
        )
        assert result["status"] == "goal_achieved"

        hooks = [hook for hook, _ in tracer.events]
        assert hooks[-1] == "goal_achieved"
        # Exactly one replan: search_broad_corpus raises, then search_deep.
        assert hooks.count("replan") == 1

        # The replan payload names the fallback plan.  ``new_plan``
        # should contain ``search_deep_corpus`` (and notably NOT
        # ``search_broad_corpus``, which was blacklisted).
        replan_payloads = [
            payload for hook, payload in tracer.events if hook == "replan"
        ]
        assert len(replan_payloads) == 1
        replan_reason, replan_plan = replan_payloads[0]
        assert replan_reason == "action_failed"
        assert "search_deep_corpus" in replan_plan
        assert "search_broad_corpus" not in replan_plan

        # search_broad_corpus started (and failed); search_deep_corpus
        # started and completed as part of the fallback plan.
        action_starts = [
            payload for hook, payload in tracer.events if hook == "action_start"
        ]
        assert "search_broad_corpus" in action_starts
        assert "search_deep_corpus" in action_starts
        # search_deep_corpus must come strictly after search_broad_corpus.
        assert action_starts.index("search_deep_corpus") > action_starts.index(
            "search_broad_corpus"
        )

        # The planner fires ``on_plan_complete`` once for the initial
        # optimistic plan, then ``on_replan`` (not a second
        # ``on_plan_complete``) for the fallback plan — that contract
        # is enforced in ``GoapPlanner.__call__``.
        plan_completes = [
            payload for hook, payload in tracer.events if hook == "plan_complete"
        ]
        assert len(plan_completes) == 1
        assert "search_broad_corpus" in plan_completes[0]
        # The replan payload already covered the deep-search fallback
        # assertions above.  A second plan_start DOES fire because
        # the planner node runs twice, but the completion hook is
        # routed through on_replan the second time.
        assert hooks.count("plan_start") == 2


# ---------------------------------------------------------------------------
# Execution history in a LangGraph Store
# ---------------------------------------------------------------------------


class TestDeepResearchHistory:
    """``StoreExecutionHistory`` persists run outcomes across invocations."""

    def test_store_records_success_and_allows_query_by_goal(self) -> None:
        store = InMemoryStore()
        history = StoreExecutionHistory(store)
        workspace, actions = _build_actions(rate_limit_active=False)
        graph = GoapGraph(actions=actions, history=history)
        result = graph.invoke(
            goal=deep_research_goal(),
            world_state=deep_research_start(),
        )
        assert result["status"] == "goal_achieved"

        # Query the history via the public goal-hash helper.  This is
        # the documented way to look up execution records for a goal
        # — tests must not reach into ``langgoap.graph.nodes`` for a
        # private function (audit H1).
        goal_hash = history.goal_hash_for(deep_research_goal())
        # Double-check the public module-level helper agrees.
        assert goal_hash == compute_goal_hash(deep_research_goal())

        records = history.query_by_goal(goal_hash)
        assert len(records) == 1
        record = records[0]
        assert isinstance(record, ExecutionRecord)
        assert record.outcome == "success"
        assert record.goal_conditions == {"report_written": True}
        assert "synthesize_report" in record.plan_actions
        assert record.replan_count == 0

    def test_store_accumulates_runs_across_multiple_invocations(self) -> None:
        """Two isolated runs both land in the store under the same hash."""
        store = InMemoryStore()
        history = StoreExecutionHistory(store)

        # Each run gets its own workspace/actions/graph so the second
        # run does not observe artifacts the first run wrote to the
        # workspace (audit M6).  The history/store is the only shared
        # state, which is exactly the accumulation property we want
        # to test.
        for _ in range(2):
            _workspace, actions = _build_actions(rate_limit_active=False)
            graph = GoapGraph(actions=actions, history=history)
            result = graph.invoke(
                goal=deep_research_goal(),
                world_state=deep_research_start(),
            )
            assert result["status"] == "goal_achieved"

        goal_hash = history.goal_hash_for(deep_research_goal())
        records = history.query_by_goal(goal_hash)
        assert len(records) == 2
        assert all(r.outcome == "success" for r in records)


# ---------------------------------------------------------------------------
# Natural-language goal intake
# ---------------------------------------------------------------------------


def _nl_interpreted_goal() -> InterpretedGoal:
    return InterpretedGoal(
        conditions={"report_written": True},
        constraints=[],
        objectives=[],
        reasoning=(
            "The operator wants a written research report, so "
            "report_written must flip to True."
        ),
    )


class TestDeepResearchNaturalLanguage:
    """NL intake path via ``GoapGraph.invoke_nl``."""

    def test_invoke_nl_reaches_goal_via_deep_search(self) -> None:
        llm = FakeStructuredModel(response=_nl_interpreted_goal())
        # Build the workspace and actions in rate-limited mode from
        # the start — no post-construction mutation of closed-over
        # state (audit M2).  The NL path exercises the full
        # interpret → replan → deep-search fallback chain.
        workspace, actions = _build_actions(rate_limit_active=True)
        graph = GoapGraph(actions=actions)
        result = graph.invoke_nl(
            "Research how AI agent frameworks combine RAG with transformers.",
            llm=llm,
            world_state=deep_research_start(),
        )
        assert result["status"] == "goal_achieved"
        assert "search_broad_corpus" in result.get("blacklisted_actions", [])
        assert workspace.report is not None
        assert len(workspace.report["citations"]) == len(EXPECTED_DEEP_FINDING_IDS)
