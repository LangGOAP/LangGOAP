"""The low-code on-ramps expose the full ``GoapGraph`` / ``ActionSpec`` surface
(issue #19) and the subgraph node propagates the parent's config and runs
async (issue #20).

``create_goap_agent``, ``GoapSubgraph`` and ``add_goap_subgraph`` used to
forward every keyword to ``GoapGraph.compile``, so a tracer, strategy, stuck
handler or termination policy raised ``TypeError`` and per-tool QoS / human
approval could not be declared at all.  ``add_goap_subgraph`` registered a
sync closure that called ``invoke`` without the parent's ``RunnableConfig``,
so a parent checkpointer never reached the child, ``interrupt()`` inside the
child could not resume, and async-only actions failed under ``ainvoke``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from langgoap import ActionSpec, GoalSpec
from langgoap.graph.state import rehydrate_state
from langgoap.integrations import (
    GoapSubgraph,
    add_goap_subgraph,
    create_goap_agent,
    goapify_tool,
)
from langgoap.planner.strategy import AStarStrategy
from langgoap.qos import ActionQos
from langgoap.stuck import (
    FunctionalStuckHandler,
    StuckHandlerResult,
    StuckHandlingResultCode,
)
from langgoap.termination import MaxActionsPolicy
from langgoap.tracing import NullTracer


class _RecordingTracer(NullTracer):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_plan_complete(self, plan: Any, duration_ms: float) -> None:
        self.events.append(("plan_complete", list(plan.action_names)))

    def on_action_complete(self, result: Any) -> None:
        self.events.append(("action_complete", None))


class _CountingStrategy:
    """A ``PlanningStrategy`` that counts calls and delegates to A*."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = AStarStrategy()

    def plan(
        self,
        start: Any,
        goal: Any,
        actions: Any,
        *,
        blacklisted_actions: list[str] | None = None,
        prior_plan: Any = None,
        current_step: int = 0,
    ) -> Any:
        self.calls += 1
        return self._inner.plan(
            start,
            goal,
            actions,
            blacklisted_actions=blacklisted_actions,
            prior_plan=prior_plan,
            current_step=current_step,
        )


@tool
def gather_data() -> str:
    """Gather raw data."""
    return "raw data"


@tool
def write_report() -> str:
    """Write the final report."""
    return "report.pdf"


_EFFECTS = {"gather_data": {"has_data": True}, "write_report": {"report_written": True}}
_PRECONDITIONS = {"write_report": {"has_data": True}}
_GOAL = GoalSpec(conditions={"report_written": True})


def _agent(**kwargs: Any) -> Any:
    return create_goap_agent(
        tools=[gather_data, write_report],
        goal=_GOAL,
        effects=_EFFECTS,
        preconditions=_PRECONDITIONS,
        **kwargs,
    )


def _by_name(agent: Any) -> dict[str, ActionSpec]:
    return {a.name: a for a in agent.goap_actions}


# ---------------------------------------------------------------------------
# goapify_tool exposes every declarative ActionSpec field
# ---------------------------------------------------------------------------


class TestGoapifyToolParity:
    def test_forwards_every_action_spec_field(self) -> None:
        qos = ActionQos(max_attempts=2)
        spec = goapify_tool(
            gather_data,
            effects={"has_data": True},
            qos=qos,
            require_human_approval=True,
            human_input_key="approval",
            can_rerun=False,
            read_only=True,
            utility=3.0,
            metadata={"team": "ops"},
        )
        assert spec.qos is qos
        assert spec.require_human_approval is True
        assert spec.human_input_key == "approval"
        assert spec.can_rerun is False
        assert spec.read_only is True
        assert spec.utility == 3.0
        # goapify_tool also records the tool description and args (issue #56).
        assert dict(spec.metadata or {})["team"] == "ops"
        assert "description" in dict(spec.metadata or {})

    def test_defaults_match_action_spec_defaults(self) -> None:
        spec = goapify_tool(gather_data, effects={"has_data": True})
        default = ActionSpec(name="x")
        assert spec.qos == default.qos
        assert spec.require_human_approval == default.require_human_approval
        assert spec.human_input_key == default.human_input_key
        assert spec.can_rerun == default.can_rerun
        assert spec.read_only == default.read_only
        assert spec.utility == default.utility
        # Metadata is never None for a wrapped tool: it carries the tool's
        # description and args (issue #56); nothing else is added.
        assert set(dict(spec.metadata or {})) == {"description", "args"}


# ---------------------------------------------------------------------------
# create_goap_agent: GoapGraph options
# ---------------------------------------------------------------------------


class TestCreateGoapAgentGraphOptions:
    def test_tracer_receives_plan_and_action_events(self) -> None:
        tracer = _RecordingTracer()
        agent = _agent(tracer=tracer)
        result = agent.invoke({"goal": _GOAL, "world_state": {}})
        assert result["status"] == "goal_achieved"
        assert ("plan_complete", ["gather_data", "write_report"]) in tracer.events
        assert sum(1 for kind, _ in tracer.events if kind == "action_complete") == 2

    def test_strategy_is_used_for_planning(self) -> None:
        strategy = _CountingStrategy()
        agent = _agent(strategy=strategy)
        result = agent.invoke({"goal": _GOAL, "world_state": {}})
        assert result["status"] == "goal_achieved"
        assert strategy.calls == 1

    def test_stuck_handler_rescues_an_unreachable_goal(self) -> None:
        seen: list[str] = []

        def relax(state: Any, reason: Any) -> StuckHandlerResult:
            seen.append("called")
            return StuckHandlerResult(
                code=StuckHandlingResultCode.REPLAN,
                message="relax to has_data",
                new_goal=GoalSpec(conditions={"has_data": True}),
            )

        agent = create_goap_agent(
            tools=[gather_data],
            goal=_GOAL,  # report_written is unreachable with one tool
            effects={"gather_data": {"has_data": True}},
            stuck_handlers=[FunctionalStuckHandler(name="relax", fn=relax)],
        )
        result = agent.invoke({"goal": _GOAL, "world_state": {}})
        assert seen == ["called"]
        assert result["status"] == "goal_achieved"
        assert result["world_state"]["has_data"] is True

    def test_termination_policy_is_enforced(self) -> None:
        agent = _agent(termination_policies=[MaxActionsPolicy(1)])
        result = agent.invoke({"goal": _GOAL, "world_state": {}})
        assert result["status"] == "terminated"
        assert len(result["execution_history"]) == 1

    def test_compile_options_still_route_to_compile(self) -> None:
        agent = _agent(checkpointer=MemorySaver(), interrupt_before=["executor"])
        config = {"configurable": {"thread_id": "t-compile-opts"}}
        paused = agent.invoke(
            {"goal": _GOAL.to_dict(), "world_state": {}}, config=config
        )
        assert paused["plan"] is not None
        assert agent.get_state(config).next == ("executor",)

    def test_unknown_option_raises_type_error_naming_it(self) -> None:
        with pytest.raises(TypeError, match="bogus"):
            _agent(bogus=1)

    def test_agent_exposes_its_actions_for_rehydration(self) -> None:
        agent = _agent()
        assert [a.name for a in agent.goap_actions] == ["gather_data", "write_report"]
        result = agent.invoke({"goal": agent.goap_goal, "world_state": {}})
        typed = rehydrate_state(result, agent.goap_actions)
        assert typed["plan"] is not None
        assert typed["plan"].actions[0] is agent.goap_actions[0]


# ---------------------------------------------------------------------------
# create_goap_agent: per-tool ActionSpec options
# ---------------------------------------------------------------------------


class TestCreateGoapAgentPerToolOptions:
    def test_per_tool_fields_reach_the_action_specs(self) -> None:
        qos = ActionQos(max_attempts=3)
        agent = _agent(
            qos={"gather_data": qos},
            require_human_approval={"write_report": True},
            human_input_key={"write_report": "sign_off"},
            can_rerun={"write_report": False},
            read_only={"gather_data": True},
            utility={"write_report": 2.0},
            metadata={"gather_data": {"team": "ops"}},
        )
        specs = _by_name(agent)
        assert specs["gather_data"].qos is qos
        assert specs["gather_data"].read_only is True
        assert dict(specs["gather_data"].metadata or {})["team"] == "ops"
        assert specs["write_report"].require_human_approval is True
        assert specs["write_report"].human_input_key == "sign_off"
        assert specs["write_report"].can_rerun is False
        assert specs["write_report"].utility == 2.0
        # Tools without an entry keep ActionSpec defaults.
        assert specs["write_report"].qos is None
        assert specs["gather_data"].require_human_approval is False

    def test_require_human_approval_pauses_and_resumes(self) -> None:
        agent = _agent(
            require_human_approval={"write_report": True},
            checkpointer=MemorySaver(),
        )
        config = {"configurable": {"thread_id": "t-hitl-prebuilt"}}
        paused = agent.invoke(
            {"goal": _GOAL.to_dict(), "world_state": {}}, config=config
        )
        assert "__interrupt__" in paused
        resumed = agent.invoke(Command(resume=True), config=config)
        assert resumed["status"] == "goal_achieved"


# ---------------------------------------------------------------------------
# GoapSubgraph / add_goap_subgraph: GoapGraph options
# ---------------------------------------------------------------------------


def _chain_actions(
    *, async_only: bool = False, approve_last: bool = False
) -> list[ActionSpec]:
    async def astep2(ws: dict[str, Any]) -> dict[str, Any]:
        return {"done": True}

    second: dict[str, Any] = (
        {"aexecute": astep2} if async_only else {"execute": lambda s: {"done": True}}
    )
    return [
        ActionSpec(name="step1", effects={"a": True}, execute=lambda s: {"a": True}),
        ActionSpec(
            name="step2",
            preconditions={"a": True},
            effects={"done": True},
            require_human_approval=approve_last,
            **second,
        ),
    ]


class ParentState(TypedDict, total=False):
    world_state: dict[str, Any]
    plan_result: dict[str, Any]


def _parent(
    actions: list[ActionSpec], *, checkpointer: Any = None, **options: Any
) -> Any:
    builder: StateGraph = StateGraph(ParentState)
    add_goap_subgraph(
        builder,
        name="goap",
        actions=actions,
        goal=GoalSpec(conditions={"done": True}),
        **options,
    )
    builder.add_edge(START, "goap")
    builder.add_edge("goap", END)
    return builder.compile(checkpointer=checkpointer)


class TestSubgraphGraphOptions:
    def test_goap_subgraph_accepts_tracer_and_strategy(self) -> None:
        tracer = _RecordingTracer()
        strategy = _CountingStrategy()
        sub = GoapSubgraph(
            actions=_chain_actions(),
            goal=GoalSpec(conditions={"done": True}),
            tracer=tracer,
            strategy=strategy,
        )
        result = sub.compile().invoke({"world_state": {}, "goal": sub.goal})
        assert result["status"] == "goal_achieved"
        assert strategy.calls == 1
        assert ("plan_complete", ["step1", "step2"]) in tracer.events

    def test_add_goap_subgraph_accepts_tracer(self) -> None:
        tracer = _RecordingTracer()
        parent = _parent(_chain_actions(), tracer=tracer)
        result = parent.invoke({"world_state": {}})
        assert result["plan_result"]["status"] == "goal_achieved"
        assert ("plan_complete", ["step1", "step2"]) in tracer.events

    def test_unknown_option_raises_type_error_naming_it(self) -> None:
        with pytest.raises(TypeError, match="bogus"):
            GoapSubgraph(actions=_chain_actions(), goal=GoalSpec(conditions={"done": True}), bogus=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# add_goap_subgraph: config propagation and async (issue #20)
# ---------------------------------------------------------------------------


class _ChainNames(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name:
            self.names.append(str(name))


class TestAddGoapSubgraphConfigAndAsync:
    def test_parent_checkpointer_lets_a_child_interrupt_resume(self) -> None:
        parent = _parent(_chain_actions(approve_last=True), checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t-parent-hitl"}}
        paused = parent.invoke({"world_state": {}}, config=config)
        assert "__interrupt__" in paused
        resumed = parent.invoke(Command(resume=True), config=config)
        assert resumed["plan_result"]["status"] == "goal_achieved"
        assert resumed["plan_result"]["world_state"]["done"] is True

    def test_parent_callbacks_reach_the_child_nodes(self) -> None:
        handler = _ChainNames()
        parent = _parent(_chain_actions())
        result = parent.invoke({"world_state": {}}, config={"callbacks": [handler]})
        assert result["plan_result"]["status"] == "goal_achieved"
        assert "planner" in handler.names and "executor" in handler.names

    @pytest.mark.asyncio
    async def test_async_only_actions_run_under_parent_ainvoke(self) -> None:
        parent = _parent(_chain_actions(async_only=True))
        result = await parent.ainvoke({"world_state": {}})
        assert result["plan_result"]["status"] == "goal_achieved"
        assert result["plan_result"]["world_state"]["done"] is True
