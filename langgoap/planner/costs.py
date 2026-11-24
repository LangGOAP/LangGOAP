"""Dynamic costs during one planning call (issue #118)."""

from __future__ import annotations

from typing import Any

from langgoap.actions import ActionSpec
from langgoap.state import PlanningState

_MISSING: Any = object()


class CostMemo:
    """Asks a declared dynamic cost once per distinct input within one planning call.

    ``ActionSpec.cost`` is a function of the state it is given.  When the
    action declares ``cost_keys``, the input that matters is the value of
    those keys (a key the planning state does not hold counts as one
    distinct value, which also covers context keys, constant during a call)
    and the function is asked once per distinct value.  Without the
    declaration nothing is remembered and the function is asked at every
    evaluation, as it always was: remembering costs a lookup per
    evaluation, and a plain A* search never asks the same (action, state)
    twice, so the lookup only pays off when the user says what the cost
    depends on.  A static cost is returned as it is.  One memo serves one
    ``plan()`` call, anytime iterations and plan building included.
    """

    __slots__ = ("_by_action",)

    def __init__(self) -> None:
        # id(action) -> (declared keys in a fixed order, value tuple -> cost)
        self._by_action: dict[int, tuple[tuple[str, ...], dict[Any, Any]]] = {}

    def cost(
        self,
        action: ActionSpec,
        state: PlanningState,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """The action's cost in *state*; a declared cost at most once per distinct input.

        *context* is ``state.as_context()`` when the caller already built it.
        """
        cost = action.cost
        if not callable(cost):
            return cost
        keys = action.cost_keys
        if keys is None:
            return cost(state.as_context() if context is None else context)
        entry = self._by_action.get(id(action))
        if entry is None:
            entry = self._by_action[id(action)] = (tuple(sorted(keys)), {})
        order, seen = entry
        view = state._view()  # the state's cached dict view, one call for all keys
        if len(order) == 1:
            key: Any = view.get(order[0], _MISSING)
        else:
            key = tuple([view.get(k, _MISSING) for k in order])
        try:
            return seen[key]
        except KeyError:
            value = cost(state.as_context() if context is None else context)
            seen[key] = value
            return value
