"""Convenience builder for GOAP execution graphs.

GoapGraph assembles a LangGraph StateGraph with planner, executor,
and observer nodes wired together for the GOAP execution loop.
"""

from __future__ import annotations

import inspect
import random
import threading
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command

from langgoap.actions import ActionSpec, ensure_unique_action_names
from langgoap.conditions import AsyncConditionResolver, ConditionResolver
from langgoap.goals import GoalSpec, MultiGoal
from langgoap.graph.nodes import (
    GoapExecutor,
    GoapObserver,
    GoapPlanner,
    ParallelGoapExecutor,
)
from langgoap.graph.nodes._execution import (
    WAIT_INTERRUPT,
    _validate_effects_mode,
    _validate_history_state,
)
from langgoap.graph.state import (
    GoapResult,
    GoapStateChannels,
    goal_to_state,
    rehydrate_state,
    state_defaults,
)
from langgoap.guards import ActionGuard, AsyncActionGuard
from langgoap.history import StoreExecutionHistory
from langgoap.planner.transitions import TransitionModel
from langgoap.sensors import AsyncSensor, Sensor, as_sensor
from langgoap.tracing import PlanningTracer

if TYPE_CHECKING:
    from langgoap.planner.strategy import PlanningStrategy
    from langgoap.stuck import StuckHandler
    from langgoap.termination import TerminationPolicy


class GoapGraph:
    """Builder that produces a compiled LangGraph for GOAP execution.

    Supports two usage styles:

    **Explicit compile + invoke** (for power users who need the compiled graph)::

        graph = GoapGraph(actions=[action1, action2, ...])
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke({
            "goal": GoalSpec(conditions={"done": True}),
            "world_state": {"a": True},
        })

    **Convenience invoke** (single-shot, no persistence)::

        graph = GoapGraph(actions=[action1, action2, ...])
        result = graph.invoke(
            goal=GoalSpec(conditions={"done": True}),
            world_state={"a": True},
        )

    **Custom planning strategy**::

        from langgoap.planner.strategy import LazyDecompositionStrategy
        graph = GoapGraph(
            actions=[action1, action2, ...],
            strategy=LazyDecompositionStrategy(lookahead=2),
        )
        result = graph.invoke(goal=goal, world_state={})

    The graph structure is::

        START → planner → executor → observer ──→ END
                  ↑                     │
                  └─────────────────────┘

    Args:
        actions: The action library available to the planner and executor.
        strategy: Optional :class:`~langgoap.planner.strategy.PlanningStrategy`.
            Defaults to the A\\* strategy wired inside :class:`GoapPlanner`.
        tracer: Optional :class:`~langgoap.tracing.PlanningTracer` for
            planning/execution observability.
        history: Optional :class:`~langgoap.history.StoreExecutionHistory`
            for persisting execution traces.
        sensors: Optional list of :class:`~langgoap.sensors.Sensor` /
            :class:`~langgoap.sensors.AsyncSensor` run before every planning
            round, the first included.  A plain function
            ``(world_state) -> updates`` is accepted and named after itself.
        guards: Optional list of :class:`~langgoap.guards.ActionGuard` /
            :class:`~langgoap.guards.AsyncActionGuard` evaluated before each
            action executes.
        resolvers: Optional list of dynamic condition resolvers applied to
            precondition evaluation.
        record_expansions: Forwarded to :class:`GoapPlanner`; enables node
            expansion trace recording for post-hoc analysis.
        transition_model: Optional
            :class:`~langgoap.planner.transitions.TransitionModel` threaded
            into :class:`GoapExecutor`.  When provided, actions with no
            ``execute`` callable (or those returning ``None``) have their
            runtime effects drawn from ``transition_model.sample``.  Use
            the same model you passed to :class:`MCTSStrategy` so plan-time
            and runtime share one noise distribution.
        rng: Optional ``random.Random`` forwarded to the executor for
            reproducible sampling; retry back-off jitter never draws from it.
        executor: ``"sequential"`` (default) runs one plan step per
            executor node invocation with :class:`GoapExecutor`;
            ``"parallel"`` wires :class:`ParallelGoapExecutor`, which runs
            every independent action of the next wave in one invocation
            (actions requiring human approval always run alone).
        history_state: What each ``ActionResult`` in ``execution_history``
            keeps of the world state.  ``"full"`` (default) stores complete
            before/after snapshots; ``"delta"`` stores only the keys an action
            changed, with their prior values; ``"planning"`` stores only the
            keys preconditions, effects and the goal read.  Use a compact mode
            when the world state carries large non-planning data (documents,
            LLM output), so checkpoints stop growing with every step
            (issue #74).  Realised costs are recorded per action and do not
            depend on the snapshots.
        goal: Default goal for :meth:`start`, :meth:`invoke` and
            :meth:`ainvoke`: a
            :class:`~langgoap.goals.GoalSpec`, a
            :class:`~langgoap.goals.MultiGoal`, or a plain mapping of
            conditions.
        effects: Who says what an action did.  ``"declared"`` (default)
            believes a returned dict, which must then set every declared
            effect key, else the declared effects.  ``"merged"`` applies the
            declared effects and merges a returned dict on top, so an action
            returns only what it adds or what differs (issue #97).
            ``"observed"`` does the same, then asks the sensors and lets what
            they report for the action's keys win, so actions need not
            re-read and return their own facts.  Requires ``sensors``.
        always_observe: State keys read from the sensors after every action
            and watched by every wait, besides each action's effect keys and
            :attr:`ActionSpec.observe <langgoap.actions.ActionSpec>`.
        max_actions: Shorthand for
            ``termination_policies=[MaxActionsPolicy(max_actions)]``.
        checkpointer: Checkpointer for the thread API (:meth:`start`,
            :meth:`notify`, :meth:`resume`, :meth:`waiting`).  Defaults to an
            in-memory saver.  :meth:`compile` keeps taking its own.
        on_stop: ``(thread_id, state) -> None`` called whenever a run made
            through the thread API stops: at the goal, at a wait, at a
            question for a person, or on failure.

    **Threads** (issue #96).  With a ``goal`` the graph object runs threads
    itself; runs are serialised, so a call made while another run is in
    progress (an action that publishes an event) is delivered after it::

        graph = GoapGraph(actions, goal={"delivered": True}, sensors=[read_platform])
        graph.start({"order_id": "o-1"}, thread_id="o-1")
        graph.notify("o-1", event)          # wakes a wait; never answers a question
        graph.resume(True, thread_id="o-1") # answers a question; never ends a wait
        graph.waiting("o-1")                # the interrupt payload, or None

    ``astart``, ``anotify``, ``aresume`` and ``awaiting`` are the async forms.
    The calls may come from several OS threads; runs are still serialised.
    """

    def __init__(
        self,
        actions: list[ActionSpec],
        *,
        strategy: PlanningStrategy | None = None,
        tracer: PlanningTracer | None = None,
        history: StoreExecutionHistory | None = None,
        sensors: (
            Sequence[Sensor | AsyncSensor | Callable[[dict[str, Any]], dict[str, Any]]]
            | None
        ) = None,
        guards: list[ActionGuard | AsyncActionGuard] | None = None,
        resolvers: list[ConditionResolver | AsyncConditionResolver] | None = None,
        record_expansions: bool = False,
        transition_model: TransitionModel | None = None,
        rng: random.Random | None = None,
        stuck_handlers: "list[StuckHandler] | None" = None,
        max_stuck_iterations: int = 3,
        termination_policies: "list[TerminationPolicy] | None" = None,
        executor: Literal["sequential", "parallel"] = "sequential",
        history_state: Literal["full", "delta", "planning"] = "full",
        goal: GoalSpec | MultiGoal | Mapping[str, Any] | None = None,
        effects: Literal["declared", "merged", "observed"] = "declared",
        always_observe: Iterable[str] = (),
        max_actions: int | None = None,
        checkpointer: Checkpointer | None = None,
        on_stop: Callable[[str, GoapResult], Any] | None = None,
    ) -> None:
        ensure_unique_action_names(actions)
        self._history_state = _validate_history_state(history_state)
        self._effects = _validate_effects_mode(effects)
        if isinstance(always_observe, str):
            raise TypeError("always_observe must be an iterable of state keys")
        self._always_observe = tuple(always_observe)
        if self._effects == "observed" and not sensors:
            raise ValueError(
                'effects="observed" needs sensors: they are what reports an '
                "action's outcome."
            )
        waits = [a.name for a in actions if a.is_wait]
        if waits and not sensors:
            raise ValueError(
                f"Wait action(s) {waits!r} need sensors: a wait ends when a "
                "sensed value of one of its keys changes."
            )
        if max_actions is not None:
            from langgoap.termination import MaxActionsPolicy

            termination_policies = [
                *(termination_policies or []),
                MaxActionsPolicy(max_actions),
            ]
        self._goal = goal
        self._checkpointer = checkpointer
        self._on_stop = on_stop
        self._compiled: CompiledStateGraph | None = None
        self._mailbox: deque[tuple[str, str, Any, list[Any]]] = deque()
        self._running = False
        self._lock = threading.Lock()
        if executor not in ("sequential", "parallel"):
            raise ValueError(
                f"executor must be 'sequential' or 'parallel', got {executor!r}"
            )
        self._executor_kind = executor
        self.actions = actions
        self._strategy = strategy
        self._tracer = tracer
        self._history = history
        self._sensors: list[Sensor | AsyncSensor] | None = (
            [as_sensor(sensor) for sensor in sensors] if sensors else None
        )
        self._guards = guards
        self._resolvers = resolvers
        self._record_expansions = record_expansions
        self._transition_model = transition_model
        self._rng = rng
        self._stuck_handlers = stuck_handlers
        self._max_stuck_iterations = max_stuck_iterations
        self._termination_policies = termination_policies

    def compile(
        self,
        checkpointer: Checkpointer | None = None,
        store: Any = None,
        *,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> CompiledStateGraph:
        """Build and compile the GOAP StateGraph.

        Args:
            checkpointer: Optional LangGraph checkpointer for persistence.
                Any ``BaseCheckpointSaver`` works unchanged with its stock
                serializer: :class:`~langgoap.graph.state.GoapState` is
                JSON-native.  When a checkpointer is attached, build the
                input with :meth:`initial_state` (or pass ``goal.to_dict()``)
                because LangGraph serialises the raw input before the
                state reducers run.
            store: Optional LangGraph store for shared state.
            interrupt_before: Optional list of node names to interrupt
                before.  Forwarded to ``StateGraph.compile``.  Use
                ``["executor"]`` to gate each action execution behind a
                human-in-the-loop checkpoint boundary, then resume via
                ``compiled.invoke(None, config=...)``.
            interrupt_after: Optional list of node names to interrupt
                after.  Forwarded to ``StateGraph.compile``.

        Returns:
            A compiled StateGraph ready for invocation.
        """
        builder = StateGraph(GoapStateChannels)

        # All three nodes are wrapped with RunnableLambda so LangGraph
        # dispatches to the async variant under ``ainvoke`` and the sync
        # variant under ``invoke``.  Without this wiring, the planner
        # and observer would only ever see the sync ``__call__`` path
        # and their async tracer hooks would never fire.
        planner = GoapPlanner(
            self.actions,
            strategy=self._strategy,
            tracer=self._tracer,
            sensors=self._sensors,
            resolvers=self._resolvers,
            record_expansions=self._record_expansions,
            stuck_handlers=self._stuck_handlers,
            max_stuck_iterations=self._max_stuck_iterations,
        )
        executor_cls = (
            ParallelGoapExecutor if self._executor_kind == "parallel" else GoapExecutor
        )
        executor = executor_cls(
            tracer=self._tracer,
            guards=self._guards,
            transition_model=self._transition_model,
            rng=self._rng,
            actions=self.actions,
            history_state=self._history_state,
            sensors=self._sensors,
            effects=self._effects,
            always_observe=self._always_observe,
        )
        observer = GoapObserver(
            self.actions,
            tracer=self._tracer,
            history=self._history,
            termination_policies=self._termination_policies,
        )
        builder.add_node(
            "planner",
            RunnableLambda(func=planner.__call__, afunc=planner.acall),
        )
        builder.add_node(
            "executor",
            RunnableLambda(func=executor.__call__, afunc=executor.acall),
        )
        builder.add_node(
            "observer",
            RunnableLambda(func=observer.__call__, afunc=observer.acall),
        )

        # Wire edges
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "observer")
        # Observer uses Command(goto=...) for routing — no explicit edges needed

        compile_kwargs: dict[str, Any] = {
            "checkpointer": checkpointer,
            "store": store,
        }
        if interrupt_before is not None:
            compile_kwargs["interrupt_before"] = interrupt_before
        if interrupt_after is not None:
            compile_kwargs["interrupt_after"] = interrupt_after
        return builder.compile(**compile_kwargs)

    def initial_state(
        self,
        goal: GoalSpec | MultiGoal | dict[str, Any],
        world_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the JSON-native input for a compiled graph.

        Use this (rather than ``{"goal": GoalSpec(...)}``) whenever the
        compiled graph has a checkpointer attached: LangGraph serialises the
        raw input before the ``goal`` reducer can convert the object.

        Every bookkeeping channel (``blacklisted_actions``,
        ``action_failure_counts``, ``replan_count``, ...) is seeded from
        :data:`~langgoap.graph.state.STATE_DEFAULTS`, so the result of the
        run carries every key of the schema even when nothing was ever
        written to it.

        The world state may be partial.  Sensors run before every planning
        round, the first included, so it only needs the facts no sensor can
        derive (an order id, a flag the application owns); there is no need to
        call the sensors yourself before the first plan.

        Args:
            goal: A :class:`~langgoap.goals.GoalSpec`,
                :class:`~langgoap.goals.MultiGoal`, or an already
                serialised goal dict.
            world_state: Initial world state (defaults to an empty dict).
        """
        return {
            "goal": goal_to_state(goal),
            "world_state": dict(world_state or {}),
            **state_defaults(),
        }

    def rehydrate(self, state: Mapping[str, Any]) -> GoapResult:
        """Return *state* with typed ``goal``, ``plan`` and ``execution_history``.

        Plan actions are rebound by name to this graph's action list so
        their callables are restored.  Accepts the dict returned by
        ``compiled.invoke`` or ``compiled.get_state(config).values``.
        """
        return GoapResult(rehydrate_state(state, self.actions))

    # ------------------------------------------------------------------
    # Threads: start, wake and answer a run by thread id (issue #96)
    # ------------------------------------------------------------------

    @property
    def compiled(self) -> CompiledStateGraph:
        """The graph compiled once with this object's checkpointer.

        Used by :meth:`start`, :meth:`notify`, :meth:`resume` and
        :meth:`waiting`; an in-memory saver is used when no ``checkpointer``
        was given.
        """
        if self._compiled is None:
            self._compiled = self.compile(
                checkpointer=(
                    self._checkpointer
                    if self._checkpointer is not None
                    else InMemorySaver()
                )
            )
        return self._compiled

    @staticmethod
    def _thread_config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _pending_interrupt(snapshot: Any) -> dict[str, Any] | None:
        if not snapshot.next:
            return None
        for task in snapshot.tasks:
            for pending in getattr(task, "interrupts", ()):
                value = pending.value
                if isinstance(value, dict):
                    return dict(value)
                return {"type": "interrupt", "value": value}
        return None

    def waiting(self, thread_id: str) -> dict[str, Any] | None:
        """What the thread is paused on: its interrupt payload, or ``None``.

        A wait reports ``{"type": "goap_action_wait", "action": name, ...}``,
        an approval ``"goap_action_approval"``, a typed form
        ``"goap_action_form"``.  An ``interrupt()`` an action raised itself is
        returned as is when it is a dict, else under ``"value"``.  ``None``
        means the thread is unknown, finished, or not paused on an interrupt.
        """
        return self._pending_interrupt(
            self.compiled.get_state(self._thread_config(thread_id))
        )

    def waiting_action(self, thread_id: str) -> str | None:
        """The action name the thread is paused on if it is a wait, else ``None``.

        Convenience over :meth:`waiting` for the common case of deciding which
        event a waiting thread expects (issue #105).
        """
        pending = self.waiting(thread_id)
        if pending is None:
            return None
        if isinstance(pending, dict) and pending.get("type") == "goap_action_wait":
            return str(pending.get("action") or pending.get("waiting"))
        return None

    def _resolve_goal(
        self, goal: GoalSpec | MultiGoal | Mapping[str, Any] | None
    ) -> GoalSpec | MultiGoal:
        chosen = goal if goal is not None else self._goal
        if chosen is None:
            raise ValueError(
                "No goal: pass one to the call or to GoapGraph(..., goal=...)."
            )
        if isinstance(chosen, (GoalSpec, MultiGoal)):
            return chosen
        return GoalSpec(conditions=dict(chosen))

    async def awaiting(self, thread_id: str) -> dict[str, Any] | None:
        """Async :meth:`waiting`, for checkpointers that are async-only."""
        return self._pending_interrupt(
            await self.compiled.aget_state(self._thread_config(thread_id))
        )

    @staticmethod
    def _input_for(kind: str, payload: Any, pending: dict[str, Any] | None) -> Any:
        """The graph input for a queued call, or ``None`` when it has nothing to do.

        A ``"start"`` carries its input.  A ``"notify"`` resumes only a thread
        paused in a wait and a ``"resume"`` only one paused on a question, so
        an event never answers a question and an answer never ends a wait.
        """
        if kind == "start":
            return payload
        if pending is None:
            return None
        is_wait = pending.get("type") == WAIT_INTERRUPT
        if (kind == "notify") != is_wait:
            return None
        # LangGraph treats ``Command(resume=None)`` as "no resume value".
        return Command(resume=True if payload is None and is_wait else payload)

    def start(
        self,
        world_state: dict[str, Any] | None = None,
        *,
        thread_id: str,
        goal: GoalSpec | MultiGoal | Mapping[str, Any] | None = None,
    ) -> GoapResult | None:
        """Plan and run a thread until the goal, a wait, a question or a failure.

        Args:
            world_state: What no sensor can derive (see :meth:`initial_state`).
            thread_id: The thread to run; the checkpointer keeps its state.
            goal: Overrides the graph's default ``goal`` for this thread.

        Returns:
            The state where the run stopped, typed as for :meth:`invoke` (an
            ``"__interrupt__"`` entry is present when it paused), or ``None``
            when the call was made during another run and has been queued:
            ``on_stop`` receives the state then.
        """
        graph_input = self.initial_state(self._resolve_goal(goal), world_state)
        return self._post(thread_id, "start", graph_input)

    def notify(self, thread_id: str, event: Any = None) -> GoapResult | None:
        """Something happened: a thread paused in a wait looks again.

        The wait asks the sensors and ends only if one of its keys changed, so
        notifying too often is harmless.  A thread that is unknown, finished,
        or paused on a question is left alone and ``None`` is returned.
        ``event`` is the resume value; waits ignore it.
        """
        return self._post(thread_id, "notify", event)

    def resume(self, value: Any, *, thread_id: str) -> GoapResult | None:
        """Answer the question the thread is paused on (an approval, a form).

        A thread that is not paused on a question (it is in a wait, finished,
        or unknown) is left alone and ``None`` is returned.
        """
        return self._post(thread_id, "resume", value)

    def _enqueue(self, thread_id: str, kind: str, payload: Any) -> list[Any] | None:
        """Queue a call; return its result slot when this caller must drain."""
        slot: list[Any] = [None]
        with self._lock:
            self._mailbox.append((thread_id, kind, payload, slot))
            if self._running:
                return None  # the run in progress delivers it
            self._running = True
        return slot

    def _next_call(self) -> tuple[str, str, Any, list[Any]] | None:
        """The next queued call, or ``None`` after marking the mailbox idle."""
        with self._lock:
            if not self._mailbox:
                self._running = False
                return None
            return self._mailbox.popleft()

    def _post(self, thread_id: str, kind: str, payload: Any) -> GoapResult | None:
        """Run now, or queue behind the run in progress.

        Safe to call from several OS threads: one of them drains the mailbox
        and the others return ``None``.  If a run raises, the error reaches
        the draining caller and the calls still queued are delivered by the
        next call.
        """
        slot = self._enqueue(thread_id, kind, payload)
        if slot is None:
            return None
        try:
            while (call := self._next_call()) is not None:
                thread, call_kind, call_payload, result = call
                pending = None if call_kind == "start" else self.waiting(thread)
                graph_input = self._input_for(call_kind, call_payload, pending)
                if graph_input is None:
                    continue
                state = self.rehydrate(
                    self.compiled.invoke(graph_input, self._thread_config(thread))
                )
                result[0] = state
                if self._on_stop is not None:
                    self._on_stop(thread, state)
        except BaseException:
            with self._lock:
                self._running = False
            raise
        return slot[0]

    async def astart(
        self,
        world_state: dict[str, Any] | None = None,
        *,
        thread_id: str,
        goal: GoalSpec | MultiGoal | Mapping[str, Any] | None = None,
    ) -> GoapResult | None:
        """Async :meth:`start`."""
        graph_input = self.initial_state(self._resolve_goal(goal), world_state)
        return await self._apost(thread_id, "start", graph_input)

    async def anotify(self, thread_id: str, event: Any = None) -> GoapResult | None:
        """Async :meth:`notify`."""
        return await self._apost(thread_id, "notify", event)

    async def aresume(self, value: Any, *, thread_id: str) -> GoapResult | None:
        """Async :meth:`resume`."""
        return await self._apost(thread_id, "resume", value)

    async def _apost(
        self, thread_id: str, kind: str, payload: Any
    ) -> GoapResult | None:
        """Async :meth:`_post`; the mailbox is shared with the sync path."""
        slot = self._enqueue(thread_id, kind, payload)
        if slot is None:
            return None
        try:
            while (call := self._next_call()) is not None:
                thread, call_kind, call_payload, result = call
                pending = None if call_kind == "start" else await self.awaiting(thread)
                graph_input = self._input_for(call_kind, call_payload, pending)
                if graph_input is None:
                    continue
                state = self.rehydrate(
                    await self.compiled.ainvoke(
                        graph_input, self._thread_config(thread)
                    )
                )
                result[0] = state
                if self._on_stop is not None:
                    stopped = self._on_stop(thread, state)
                    if inspect.isawaitable(stopped):
                        await stopped
        except BaseException:
            with self._lock:
                self._running = False
            raise
        return slot[0]

    def invoke(
        self,
        goal: GoalSpec | MultiGoal | Mapping[str, Any] | None = None,
        world_state: dict[str, Any] | None = None,
        config: RunnableConfig | None = None,
    ) -> GoapResult:
        """Convenience method: compile and invoke the graph in one call.

        Suitable for single-shot executions that don't need persistence or
        time-travel.  For repeated invocations with the same compiled graph,
        use :meth:`compile` directly.

        Args:
            goal: The goal to achieve: a :class:`GoalSpec`, a
                :class:`MultiGoal` wrapping several sub-goals, or a plain
                mapping of conditions (``{"done": True}``).  Defaults to the
                graph's ``goal=``.
            world_state: Initial world state (defaults to an empty dict).
            config: Optional LangGraph run configuration
                (e.g. ``{"configurable": {"thread_id": "..."}}``)

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes as a :class:`~langgoap.graph.state.GoapResult`:
            the state dict, rehydrated so ``goal``, ``plan`` and
            ``execution_history`` are typed objects (see :meth:`rehydrate`),
            with ``succeeded``, ``actions`` and ``summary()``.
        """
        compiled = self.compile()
        result = compiled.invoke(
            self.initial_state(self._resolve_goal(goal), world_state), config=config
        )
        return self.rehydrate(result)

    async def ainvoke(
        self,
        goal: GoalSpec | MultiGoal | Mapping[str, Any] | None = None,
        world_state: dict[str, Any] | None = None,
        config: RunnableConfig | None = None,
    ) -> GoapResult:
        """Async convenience method: compile and invoke the graph.

        Identical to :meth:`invoke` but uses ``ainvoke`` on the compiled
        graph, enabling native async execution of action callables.

        Args:
            goal: The goal to achieve: a :class:`GoalSpec`, a
                :class:`MultiGoal` wrapping several sub-goals, or a plain
                mapping of conditions (``{"done": True}``).  Defaults to the
                graph's ``goal=``.
            world_state: Initial world state (defaults to an empty dict).
            config: Optional LangGraph run configuration.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes as a :class:`~langgoap.graph.state.GoapResult`:
            the state dict, rehydrated so ``goal``, ``plan`` and
            ``execution_history`` are typed objects (see :meth:`rehydrate`),
            with ``succeeded``, ``actions`` and ``summary()``.
        """
        compiled = self.compile()
        result = await compiled.ainvoke(
            self.initial_state(self._resolve_goal(goal), world_state), config=config
        )
        return self.rehydrate(result)

    def invoke_nl(
        self,
        request: str,
        world_state: dict[str, Any] | None = None,
        *,
        llm: BaseChatModel,
        config: RunnableConfig | None = None,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> GoapResult:
        """Interpret a natural language request and execute the GOAP loop.

        Convenience method that creates a :class:`~langgoap.interpreter.GoalInterpreter`,
        converts the request to a :class:`~langgoap.goals.GoalSpec`, and invokes
        the graph in one call.

        Args:
            request: Natural language description of the goal.
            world_state: Initial world state (defaults to an empty dict).
            llm: LangChain chat model for goal interpretation.
            config: Optional LangGraph run configuration.
            structured_output_kwargs: Forwarded to
                :class:`~langgoap.interpreter.GoalInterpreter` and then to
                ``llm.with_structured_output()``.  Use
                ``{"method": "function_calling"}`` for OpenAI when the
                conditions dict must stay open-ended.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        from langgoap.interpreter import (
            GoalInterpreter,
            _default_structured_output_kwargs,
        )

        if structured_output_kwargs is None:
            structured_output_kwargs = _default_structured_output_kwargs(llm)

        interpreter = GoalInterpreter(
            llm=llm,
            actions=self.actions,
            structured_output_kwargs=structured_output_kwargs,
        )
        goal = interpreter.interpret(request, world_state=world_state)
        return self.invoke(goal=goal, world_state=world_state, config=config)

    async def ainvoke_nl(
        self,
        request: str,
        world_state: dict[str, Any] | None = None,
        *,
        llm: BaseChatModel,
        config: RunnableConfig | None = None,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> GoapResult:
        """Async variant of :meth:`invoke_nl`.

        Args:
            request: Natural language description of the goal.
            world_state: Initial world state (defaults to an empty dict).
            llm: LangChain chat model for goal interpretation.
            config: Optional LangGraph run configuration.
            structured_output_kwargs: Forwarded to
                :class:`~langgoap.interpreter.GoalInterpreter` and then to
                ``llm.with_structured_output()``.  Use
                ``{"method": "function_calling"}`` for OpenAI when the
                conditions dict must stay open-ended.

        Returns:
            The final :class:`~langgoap.graph.state.GoapState` after the
            GOAP loop completes.
        """
        from langgoap.interpreter import (
            GoalInterpreter,
            _default_structured_output_kwargs,
        )

        if structured_output_kwargs is None:
            structured_output_kwargs = _default_structured_output_kwargs(llm)

        interpreter = GoalInterpreter(
            llm=llm,
            actions=self.actions,
            structured_output_kwargs=structured_output_kwargs,
        )
        goal = await interpreter.ainterpret(request, world_state=world_state)
        return await self.ainvoke(goal=goal, world_state=world_state, config=config)
