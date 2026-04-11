"""Public testing utilities for LangGoap.

Mirrors LangChain's ``langchain.testing`` pattern: small, dependency-free
helpers that downstream tests, notebooks, and tutorials can use without
reaching into LangGoap's internal test infrastructure.

Currently this module exposes:

- :class:`FakeStructuredModel` — a minimal :class:`BaseChatModel` subclass
  whose ``with_structured_output`` returns a pre-configured response.  Useful
  for exercising :class:`langgoap.GoalInterpreter` and any code path that
  consumes structured LLM output, with no API key dependency.

The intent is that user-facing tutorials import from here:

.. code-block:: python

    from langgoap.testing import FakeStructuredModel

instead of from ``tests.conftest`` (which is project-private and not on the
package path for installed users).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

__all__ = ["FakeStructuredModel"]


class FakeStructuredModel(BaseChatModel):
    """Minimal :class:`BaseChatModel` for testing structured output.

    Returns a pre-configured response from ``with_structured_output()``.  Both
    sync and async paths work via :class:`~langchain_core.runnables.RunnableLambda`,
    so the same instance can drive ``GoalInterpreter.interpret`` and
    ``GoalInterpreter.ainterpret`` interchangeably.

    Args:
        response: The object that ``with_structured_output()`` will return.
            Typically a Pydantic model instance — for LangGoap tutorials this
            is usually an :class:`langgoap.InterpretedGoal`.
        expected_schema: When provided, ``with_structured_output()`` asserts
            that the requested schema matches.  Use this in tests that want
            to verify the caller is asking for the right type.

    Example:
        >>> from langgoap import GoalInterpreter, InterpretedGoal
        >>> from langgoap.testing import FakeStructuredModel
        >>> llm = FakeStructuredModel(
        ...     response=InterpretedGoal(
        ...         conditions={"answer_ready": True},
        ...         constraints=[],
        ...         objectives=[],
        ...         reasoning="The user wants a concrete answer.",
        ...     )
        ... )
        >>> interpreter = GoalInterpreter(llm=llm, actions=[])
        >>> goal = interpreter.interpret("Give me an answer.")
        >>> goal.conditions
        {'answer_ready': True}
    """

    response: Any = None
    expected_schema: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        if self.expected_schema is not None and schema is not self.expected_schema:
            raise AssertionError(
                f"with_structured_output called with schema={schema!r}, "
                f"expected {self.expected_schema!r}"
            )
        resp = self.response
        return RunnableLambda(lambda x: resp)
