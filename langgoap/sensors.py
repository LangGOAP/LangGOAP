"""Sensor protocols for reading external state before planning.

Sensors formalize the Observe phase of the OODA loop: before the A*
planner runs, each registered sensor reads external state (APIs,
databases, file systems, etc.) and merges its findings into the
``world_state`` dict.  This decouples state acquisition from action
execution and mirrors the sensor concept found in GOApy and other
GOAP reference implementations.

Three flavours are provided:

* :class:`Sensor` — synchronous protocol (duck-typed).
* :class:`AsyncSensor` — asynchronous protocol (duck-typed).
* :class:`FunctionalSensor` — convenience wrapper that turns a plain
  ``Callable[[dict], dict]`` into a :class:`Sensor`.

Sensors are registered on :class:`~langgoap.graph.builder.GoapGraph`
and :class:`~langgoap.graph.nodes.GoapPlanner` via the ``sensors``
keyword argument.  They run **before** every planning round (including
replans) and never block the planner — exceptions are caught and
logged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Protocol, runtime_checkable

from langgoap._worker import run_in_worker

logger = logging.getLogger(__name__)


@runtime_checkable
class Sensor(Protocol):
    """Reads external state into the world state before planning."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        """Return key-value updates to merge into ``world_state``.

        An empty dict is a valid no-op return.
        """
        ...  # pragma: no cover


@runtime_checkable
class AsyncSensor(Protocol):
    """Async variant of :class:`Sensor`."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    async def asense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        """Async version of :meth:`Sensor.sense`."""
        ...  # pragma: no cover


class FunctionalSensor:
    """Convenience wrapper: construct a :class:`Sensor` from a plain function.

    Args:
        name: Human-readable sensor name.
        fn: A callable ``(world_state: dict) -> dict`` that returns
            state updates to merge.

    Example::

        sensor = FunctionalSensor("check_api", lambda ws: {"api_up": True})
    """

    def __init__(
        self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        self._name = name
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    def sense(self, world_state: dict[str, Any]) -> dict[str, Any]:
        return self._fn(world_state)


def as_sensor(
    sensor: Sensor | AsyncSensor | Callable[[dict[str, Any]], dict[str, Any]],
) -> Sensor | AsyncSensor:
    """Return *sensor* as a sensor object; a plain function is wrapped.

    A callable ``(world_state) -> updates`` becomes a
    :class:`FunctionalSensor` named after the function, so
    ``GoapGraph(sensors=[read_platform])`` needs no wrapper (issue #96).
    """
    if hasattr(sensor, "sense") or hasattr(sensor, "asense"):
        return sensor  # type: ignore[return-value]
    if asyncio.iscoroutinefunction(sensor):
        raise TypeError(
            "An async function cannot be passed as a sensor directly; implement "
            "AsyncSensor (a `name` and an `asense` coroutine) instead."
        )
    if callable(sensor):
        name = getattr(sensor, "__name__", None) or type(sensor).__name__
        return FunctionalSensor(name, sensor)
    raise TypeError(
        f"A sensor must implement Sensor / AsyncSensor or be a callable, got "
        f"{type(sensor).__name__}."
    )


async def _run_sensor_async(
    sensor: Sensor | AsyncSensor,
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Resolve and await the best callable for *sensor*.

    If the sensor implements ``asense``, call it directly.
    Otherwise fall back to ``sense`` run in the default executor.
    """
    if hasattr(sensor, "asense"):
        return await sensor.asense(world_state)
    return await run_in_worker(sensor.sense, world_state)


def _record_sensor_result(
    results: list[tuple[str, dict[str, Any]]],
    world_state: dict[str, Any],
    name: str,
    updates: dict[str, Any],
) -> None:
    """Append ``(name, updates)`` and merge into ``world_state`` in place.

    Shared between the sync and async loop drivers so the post-call
    bookkeeping cannot diverge.  Sequential execution (each sensor sees
    the cumulative state from prior sensors) is enforced here by the
    in-place ``world_state.update`` rather than by the loop drivers.
    """
    results.append((name, updates))
    world_state.update(updates)


def run_sensors_sync(
    sensors: list[Sensor | AsyncSensor],
    world_state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Run all sensors synchronously and return (name, updates) pairs.

    Exceptions from individual sensors are caught and logged.  A failing
    sensor never blocks the planner.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for sensor in sensors:
        try:
            sense = getattr(sensor, "sense", None)  # narrows for mypy 1.20 and 2.4
            if sense is None:
                # AsyncSensor-only — skip in sync path with a warning.
                logger.warning(
                    "Sensor %r is async-only; skipping in sync path. "
                    "Use GoapGraph.ainvoke() to run async sensors.",
                    sensor.name,
                )
                continue
            _record_sensor_result(results, world_state, sensor.name, sense(world_state))
        except Exception as exc:
            logger.warning("Sensor %r raised: %s", sensor.name, exc)
    return results


async def run_sensors_async(
    sensors: list[Sensor | AsyncSensor],
    world_state: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Run all sensors asynchronously and return (name, updates) pairs.

    Falls back to ``sense()`` in a thread executor for sensors that only
    implement the sync :class:`Sensor` protocol.  Exceptions from individual
    sensors are caught and logged so a single failing sensor never halts the
    rest.

    **Execution order**: sensors are awaited *sequentially* (one at a time).
    This is intentional \u2014 each sensor receives the cumulative ``world_state``
    that includes all updates from sensors that ran before it, enabling
    dependent sensor chains (e.g. sensor B reads the value written by sensor
    A).  If your sensors are fully independent and latency is critical, run
    them concurrently yourself with :func:`asyncio.gather` and merge the
    result dicts manually.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for sensor in sensors:
        try:
            _record_sensor_result(
                results,
                world_state,
                sensor.name,
                await _run_sensor_async(sensor, world_state),
            )
        except Exception as exc:
            logger.warning("Sensor %r raised: %s", sensor.name, exc)
    return results


__all__ = [
    "AsyncSensor",
    "FunctionalSensor",
    "Sensor",
    "as_sensor",
    "run_sensors_async",
    "run_sensors_sync",
]
