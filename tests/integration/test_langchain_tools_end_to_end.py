"""End-to-end integration test for the three-layer low-code on-ramp.

Exercises ``goapify_tool``, ``create_goap_agent``, and ``GoapSubgraph``
together against a realistic multi-step tool pipeline.  This is the
"does it all work together" smoke test for Epic 2.
"""

from __future__ import annotations

from langchain_core.tools import tool

from langgoap.actions import ActionSpec
from langgoap.goals import GoalSpec
from langgoap.graph.builder import GoapGraph
from langgoap.integrations import (
    GoapSubgraph,
    create_goap_agent,
    goapify_tool,
)


@tool
def fetch_user_profile() -> dict[str, str]:
    """Fetch the user's profile from the database."""
    return {"name": "Ada", "tier": "pro"}


@tool
def fetch_billing_info() -> dict[str, float]:
    """Fetch the user's billing info."""
    return {"balance": 42.50}


@tool
def generate_report() -> str:
    """Generate a combined report."""
    return "report body"


@tool
def send_email() -> str:
    """Send the report by email."""
    return "sent"


def _wired_actions() -> list[ActionSpec]:
    return [
        goapify_tool(
            fetch_user_profile,
            effects={"has_profile": True},
            resources={"cost_usd": 0.01},
        ),
        goapify_tool(
            fetch_billing_info,
            effects={"has_billing": True},
            resources={"cost_usd": 0.01},
        ),
        goapify_tool(
            generate_report,
            preconditions={"has_profile": True, "has_billing": True},
            effects={"report_ready": True},
            resources={"cost_usd": 0.05},
        ),
        goapify_tool(
            send_email,
            preconditions={"report_ready": True},
            effects={"email_sent": True},
            resources={"cost_usd": 0.02},
        ),
    ]


class TestLayerA_PrebuiltAgent:
    def test_prebuilt_agent_runs_full_pipeline(self) -> None:
        agent = create_goap_agent(
            tools=[
                fetch_user_profile,
                fetch_billing_info,
                generate_report,
                send_email,
            ],
            goal=GoalSpec(conditions={"email_sent": True}),
            effects={
                "fetch_user_profile": {"has_profile": True},
                "fetch_billing_info": {"has_billing": True},
                "generate_report": {"report_ready": True},
                "send_email": {"email_sent": True},
            },
            preconditions={
                "generate_report": {
                    "has_profile": True,
                    "has_billing": True,
                },
                "send_email": {"report_ready": True},
            },
            resources={
                "fetch_user_profile": {"cost_usd": 0.01},
                "fetch_billing_info": {"cost_usd": 0.01},
                "generate_report": {"cost_usd": 0.05},
                "send_email": {"cost_usd": 0.02},
            },
        )
        result = agent.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"email_sent": True}),
            }
        )
        assert result["status"] == "goal_achieved"
        names = result["plan"].action_names
        assert names[-1] == "send_email"
        assert names[-2] == "generate_report"
        assert "fetch_user_profile" in names[:2]
        assert "fetch_billing_info" in names[:2]


class TestLayerB_GoapifyTool:
    def test_goapified_actions_plug_into_goap_graph(self) -> None:
        actions = _wired_actions()
        graph = GoapGraph(actions=actions)
        result = graph.invoke(
            goal=GoalSpec(conditions={"email_sent": True}),
            world_state={},
        )
        assert result["status"] == "goal_achieved"


class TestLayerC_GoapSubgraph:
    def test_subgraph_compiles_and_runs(self) -> None:
        sub = GoapSubgraph(
            actions=_wired_actions(),
            goal=GoalSpec(conditions={"email_sent": True}),
        )
        compiled = sub.compile()
        result = compiled.invoke(
            {
                "world_state": {},
                "goal": GoalSpec(conditions={"email_sent": True}),
            }
        )
        assert result.get("status") == "goal_achieved"
