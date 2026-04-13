"""Three-valued condition logic for runtime GOAP precondition resolution.

Implements the Orient phase of the OODA loop: after sensors update the
world-state, condition resolvers fill in values that cannot be directly
observed — they must be *inferred* (e.g. by asking an LLM whether a
prerequisite is true given the current context).

Three values are possible for any condition key:

* :attr:`ConditionStatus.TRUE` — the condition is satisfied; the resolved
  value ``True`` is written into world_state so the A* planner can use it.
* :attr:`ConditionStatus.FALSE` — the condition is known to be unsatisfied;
  ``False`` is written into world_state so precondition checks fail cleanly.
* :attr:`ConditionStatus.UNKNOWN` — the resolver cannot determine the value;
  the key is left absent from world_state, which causes the A* planner to
  treat actions that require it as unreachable (safe degradation).

Usage::

    from langgoap.conditions import FunctionalConditionResolver, resolve_conditions

    resolver = FunctionalConditionResolver(
        "check_ready",
        lambda key, ws: ConditionStatus.TRUE if ws.get("loaded") else ConditionStatus.UNKNOWN,
    )
    updated_ws = resolve_conditions([resolver], ["ready"], world_state)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class ConditionStatus(str):
    """Three-valued truth value for runtime condition resolution.

    Subclasses ``str`` (not ``enum.Enum``) for JSON-serialisation
    compatibility and direct use as a dict key.  Use the module-level
    constants :data:`TRUE`, :data:`FALSE`, and :data:`UNKNOWN` instead of
    constructing instances directly.
    """

    TRUE: ConditionStatus
    FALSE: ConditionStatus
    UNKNOWN: ConditionStatus

    def __repr__(self) -> str:
        return f"ConditionStatus.{self.upper()}"

    def __bool__(self) -> bool:
        return self == ConditionStatus.TRUE


ConditionStatus.TRUE = ConditionStatus("true")
ConditionStatus.FALSE = ConditionStatus("false")
ConditionStatus.UNKNOWN = ConditionStatus("unknown")


def _to_status(value: Any) -> ConditionStatus:
    """Coerce a raw resolver return value to a :class:`ConditionStatus`."""
    if isinstance(value, ConditionStatus):
        return value
    if value is None:
        return ConditionStatus.UNKNOWN
    if isinstance(value, bool):
        return ConditionStatus.TRUE if value else ConditionStatus.FALSE
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("true", "yes", "1"):
            return ConditionStatus.TRUE
        if norm in ("false", "no", "0"):
            return ConditionStatus.FALSE
    return ConditionStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ConditionResolver(Protocol):
    """Synchronous condition resolver — duck-typed protocol."""

    @property
    def name(self) -> str:
        """Human-readable resolver name."""
        ...  # pragma: no cover

    def resolve(self, key: str, world_state: dict[str, Any]) -> ConditionStatus:
        """Return the truth value of *key* given the current *world_state*."""
        ...  # pragma: no cover


@runtime_checkable
class AsyncConditionResolver(Protocol):
    """Asynchronous condition resolver — duck-typed protocol."""

    @property
    def name(self) -> str:
        """Human-readable resolver name."""
        ...  # pragma: no cover

    async def aresolve(self, key: str, world_state: dict[str, Any]) -> ConditionStatus:
        """Async version of :meth:`ConditionResolver.resolve`."""
        ...  # pragma: no cover



# ---------------------------------------------------------------------------
# Concrete resolvers
# ---------------------------------------------------------------------------


class FunctionalConditionResolver:
    """Wrap a plain callable into a :class:`ConditionResolver`.

    The callable receives ``(key, world_state)`` and may return a
    :class:`ConditionStatus`, a plain ``bool``, or ``None`` (which maps
    to :attr:`ConditionStatus.UNKNOWN`).  Coroutine callables are also
    supported — they are awaitable via :meth:`aresolve`.

    Args:
        name: Human-readable resolver name.
        fn:   ``(key: str, world_state: dict) -> ConditionStatus | bool | None``
              A sync or async callable.

    Example::

        FunctionalConditionResolver(
            "stock_check",
            lambda key, ws: ConditionStatus.TRUE if ws.get("stock", 0) > 0
                            else ConditionStatus.FALSE,
        )
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self._name = name
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    def resolve(self, key: str, world_state: dict[str, Any]) -> ConditionStatus:
        result = self._fn(key, world_state)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, result)
                        return _to_status(future.result())
                return _to_status(loop.run_until_complete(result))
            except RuntimeError:
                return _to_status(asyncio.run(result))
        return _to_status(result)

    async def aresolve(
        self, key: str, world_state: dict[str, Any]
    ) -> ConditionStatus:
        result = self._fn(key, world_state)
        if asyncio.iscoroutine(result):
            return _to_status(await result)
        return _to_status(result)


class PromptConditionResolver:
    """Resolve a condition key by querying an LLM.

    Uses LangChain's ``BaseChatModel.with_structured_output`` to obtain a
    structured ``TRUE / FALSE / UNKNOWN`` answer from the model.  The
    model is given the condition key and the current world-state as context.

    Args:
        llm: Any LangChain ``BaseChatModel`` (ChatOpenAI, ChatAnthropic, …).
        prompt_template: Optional f-string template.  Available variables:
            ``{key}`` — the condition key being resolved;
            ``{world_state}`` — the current world state dict as a string.
            Defaults to a concise diagnostic prompt.
        timeout_s: Per-call timeout in seconds (default 30).

    Example::

        from langchain_openai import ChatOpenAI
        from langgoap.conditions import PromptConditionResolver

        resolver = PromptConditionResolver(ChatOpenAI(model="gpt-4o-mini"))
        status = resolver.resolve("has_permission", {"user_role": "admin"})
    """

    _DEFAULT_PROMPT = (
        "You are a planning assistant. Determine whether the following condition "
        "is TRUE, FALSE, or UNKNOWN given the current world state.\n\n"
        "Condition key: {key}\n"
        "World state: {world_state}\n\n"
        "Respond with exactly one of: true, false, unknown."
    )

    def __init__(
        self,
        llm: Any,
        *,
        prompt_template: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._llm = llm
        self._prompt = prompt_template or self._DEFAULT_PROMPT
        self._timeout_s = timeout_s
        self._chain: Any | None = None

    @property
    def name(self) -> str:
        return "PromptConditionResolver"

    def _get_chain(self) -> Any:
        if self._chain is None:
            from pydantic import BaseModel

            class _Answer(BaseModel):
                status: str
                reasoning: str = ""

            self._chain = self._llm.with_structured_output(_Answer)
        return self._chain

    def _build_prompt(self, key: str, world_state: dict[str, Any]) -> str:
        return self._prompt.format(key=key, world_state=str(world_state))

    def resolve(self, key: str, world_state: dict[str, Any]) -> ConditionStatus:
        """Synchronously query the LLM and return a :class:`ConditionStatus`."""
        try:
            from langchain_core.messages import HumanMessage

            chain = self._get_chain()
            response = chain.invoke(
                [HumanMessage(content=self._build_prompt(key, world_state))],
            )
            return _to_status(response.status)
        except Exception as exc:
            logger.warning(
                "PromptConditionResolver failed for key %r: %s — returning UNKNOWN",
                key,
                exc,
            )
            return ConditionStatus.UNKNOWN

    async def aresolve(self, key: str, world_state: dict[str, Any]) -> ConditionStatus:
        """Async LLM query."""
        try:
            from langchain_core.messages import HumanMessage

            chain = self._get_chain()
            response = await chain.ainvoke(
                [HumanMessage(content=self._build_prompt(key, world_state))],
            )
            return _to_status(response.status)
        except Exception as exc:
            logger.warning(
                "PromptConditionResolver.aresolve failed for key %r: %s — UNKNOWN",
                key,
                exc,
            )
            return ConditionStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def resolve_conditions(
    resolvers: list[ConditionResolver | AsyncConditionResolver],
    keys: list[str],
    world_state: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all resolvers against *keys* and return an updated world-state copy.

    Keys already present in *world_state* are skipped unless *overwrite* is
    ``True``.  :attr:`~ConditionStatus.TRUE` writes ``True``,
    :attr:`~ConditionStatus.FALSE` writes ``False``, and
    :attr:`~ConditionStatus.UNKNOWN` leaves the key absent.

    Args:
        resolvers: Registered condition resolvers.
        keys:      Condition keys to evaluate (typically from goal + action
                   preconditions).
        world_state: Current world state (not mutated; a copy is returned).
        overwrite:   Re-evaluate even when the key is already set (default
                     ``False``).

    Returns:
        Updated world-state dict with resolved TRUE/FALSE keys merged in.
    """
    updated = dict(world_state)
    for key in keys:
        if not overwrite and key in updated:
            continue
        resolved = False
        for resolver in resolvers:
            try:
                if hasattr(resolver, "resolve"):
                    status = resolver.resolve(key, updated)
                else:
                    logger.warning(
                        "Resolver %r is async-only; skipping key %r in sync path.",
                        resolver.name,
                        key,
                    )
                    continue
                if status == ConditionStatus.TRUE:
                    updated[key] = True
                    resolved = True
                    break
                if status == ConditionStatus.FALSE:
                    updated[key] = False
                    resolved = True
                    break
                # UNKNOWN: try next resolver
            except Exception as exc:
                logger.warning(
                    "Resolver %r raised for key %r: %s", resolver.name, key, exc
                )
        if not resolved and key not in updated:
            logger.debug(
                "All resolvers returned UNKNOWN for key %r — leaving absent", key
            )
    return updated


async def aresolve_conditions(
    resolvers: list[ConditionResolver | AsyncConditionResolver],
    keys: list[str],
    world_state: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Async variant of :func:`resolve_conditions`.

    Uses ``aresolve`` when available, falls back to ``resolve`` in a thread
    executor for sync-only resolvers.
    """
    updated = dict(world_state)
    loop = asyncio.get_running_loop()

    for key in keys:
        if not overwrite and key in updated:
            continue
        resolved = False
        for resolver in resolvers:
            try:
                if hasattr(resolver, "aresolve"):
                    status = await resolver.aresolve(key, updated)
                elif hasattr(resolver, "resolve"):
                    status = await loop.run_in_executor(
                        None, resolver.resolve, key, updated
                    )
                else:
                    continue
                if status == ConditionStatus.TRUE:
                    updated[key] = True
                    resolved = True
                    break
                if status == ConditionStatus.FALSE:
                    updated[key] = False
                    resolved = True
                    break
            except Exception as exc:
                logger.warning(
                    "Resolver %r raised for key %r: %s", resolver.name, key, exc
                )
        if not resolved and key not in updated:
            logger.debug(
                "All resolvers returned UNKNOWN for key %r — leaving absent", key
            )
    return updated


__all__ = [
    "AsyncConditionResolver",
    "ConditionResolver",
    "ConditionStatus",
    "FunctionalConditionResolver",
    "PromptConditionResolver",
    "aresolve_conditions",
    "resolve_conditions",
]
