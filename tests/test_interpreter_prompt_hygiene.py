"""GoalInterpreter provider detection and prompt hygiene (issue #59).

Azure and other OpenAI-compatible models must get the ``function_calling``
structured-output path like ``ChatOpenAI``, and the prompt must carry only
planning-relevant world-state keys, clipped, never the whole world state.
"""

from __future__ import annotations

from typing import Any

from langgoap import ActionSpec
from langgoap.interpreter import GoalInterpreter, _default_structured_output_kwargs
from langgoap.testing import FakeStructuredModel


def _actions() -> list[ActionSpec]:
    return [
        ActionSpec(
            name="fetch", preconditions={"has_source": True}, effects={"has_data": True}
        ),
        ActionSpec(
            name="report",
            preconditions={"has_data": True},
            effects={"report_done": True},
        ),
    ]


class TestProviderDetection:
    def test_azure_gets_function_calling(self) -> None:
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_endpoint="https://example.openai.azure.com",
            api_key="test",
            api_version="2024-02-01",
            azure_deployment="gpt-4o-mini",
        )
        assert _default_structured_output_kwargs(llm) == {"method": "function_calling"}

    def test_chat_openai_still_gets_function_calling(self) -> None:
        from langchain_openai import ChatOpenAI

        assert _default_structured_output_kwargs(
            ChatOpenAI(model="gpt-4o-mini", api_key="test")
        ) == {"method": "function_calling"}

    def test_other_models_get_defaults(self) -> None:
        assert (
            _default_structured_output_kwargs(FakeStructuredModel(response=None)) == {}
        )


class TestPromptWorldState:
    def _interpreter(self, **kwargs: Any) -> GoalInterpreter:
        return GoalInterpreter(
            llm=FakeStructuredModel(response=None), actions=_actions(), **kwargs
        )

    def test_only_planning_keys_reach_the_prompt(self) -> None:
        messages = self._interpreter().build_messages(
            "make a report",
            {
                "has_source": True,
                "secret_token": "sk-live-123",
                "documents": ["x" * 200] * 20,
            },
        )
        system = messages[0].content
        assert "has_source" in system
        assert "sk-live-123" not in system
        assert "documents" not in system

    def test_world_state_is_clipped(self) -> None:
        huge = {f"has_source{i}": "v" * 100 for i in range(200)}
        huge["has_source"] = True
        interp = self._interpreter()
        system = interp.build_messages("make a report", huge)[0].content
        assert len(system) < 6000

    def test_world_state_can_be_excluded(self) -> None:
        system = (
            self._interpreter(include_world_state=False)
            .build_messages("make a report", {"has_source": True})[0]
            .content
        )
        # The action catalog still lists the precondition; the world-state
        # section itself must be empty.
        assert "## Current World State\n\n(not provided)" in system
