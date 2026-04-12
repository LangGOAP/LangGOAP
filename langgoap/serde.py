"""Custom serializer that handles LangGoap's ``MappingProxyType`` fields.

LangGoap's frozen dataclasses (:class:`~langgoap.actions.ActionSpec`,
:class:`~langgoap.goals.GoalSpec`, :class:`~langgoap.planner.types.Plan`,
etc.) wrap every dict field in ``types.MappingProxyType`` inside
``__post_init__``.  ``frozen=True`` on a dataclass only prevents the
*attribute* being rebound — it does not stop the wrapped dict from being
mutated in place — so ``MappingProxyType`` is how LangGoap enforces
deep immutability on the planning state.

LangGraph's stock ``JsonPlusSerializer`` has two encoding gaps for
LangGoap:

1. **``MappingProxyType`` has no msgpack encoding.**  The stock default
   handler raises ``TypeError: Type is not msgpack serializable: GoalSpec``
   the moment it recurses into a frozen dataclass that wraps a dict in a
   read-only view.

2. **``frozenset[tuple[...]]`` does not round-trip.**  ormsgpack does not
   distinguish tuples from lists on the wire — both decode as ``list``.
   :class:`~langgoap.state.PlanningState` stores its world state as
   ``frozenset[tuple[str, Any]]`` for hashability, and the stock encoder
   packs sets via ``(module, name, tuple(obj))``; after decode the inner
   tuples become lists and ``frozenset([[k, v], ...])`` fails with
   ``unhashable type: list``.  The stock ext hook silently swallows the
   exception and returns ``None``, which is then passed into
   ``PlanningState.__init__`` as ``conditions=None`` — corrupting state
   in a way that only surfaces when callers later try to iterate it.

This module ships two serializer subclasses:

- :class:`LangGoapSerializer` — for ``MemorySaver`` and ``PostgresSaver``
  (msgpack-only checkpointers).
- :class:`LangGoapRedisSerializer` — for ``RedisSaver`` /
  ``AsyncRedisSaver``.  The Redis checkpointer stores metadata as
  RedisJSON, requiring JSON-serializable bytes from ``dumps_typed``.
  This subclass preserves the JSON-first encoding from
  ``JsonPlusRedisSerializer`` and only overrides the msgpack fallback
  path to use LangGoap's custom encoder.

:func:`install_langgoap_serde` auto-detects which backend the
checkpointer uses and installs the correct subclass.
"""

from __future__ import annotations

import dataclasses
import pickle
from types import MappingProxyType
from typing import Any

import ormsgpack
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    JsonPlusSerializer,
    _msgpack_default as _stock_msgpack_default,
    _option as _stock_option,
)

__all__ = [
    "LangGoapSerializer",
    "LangGoapRedisSerializer",
    "install_langgoap_serde",
]


# ---------------------------------------------------------------------------
# Custom msgpack encoder
# ---------------------------------------------------------------------------

# ``_stock_option`` already includes ``OPT_PASSTHROUGH_DATACLASS`` (so the
# stock dataclass branch in ``_msgpack_default`` is reachable).  We add
# ``OPT_PASSTHROUGH_TUPLE`` so every tuple is routed through our default
# handler and wrapped as an Ext that reconstructs as a tuple on decode —
# critical for ``frozenset[tuple[...]]`` round-trip.
_LANGGOAP_OPTION = _stock_option | ormsgpack.OPT_PASSTHROUGH_TUPLE


def _pack_ext_payload(module: str, name: str, arg: Any) -> bytes:
    """Pack an Ext constructor payload ``(module, name, arg)``.

    The payload is serialized as a **list** rather than a tuple so that
    our own ``OPT_PASSTHROUGH_TUPLE`` option does not recursively wrap
    the metadata envelope in yet another tuple-Ext.  The stock ext hook
    decodes Ext payloads with ``tup[0], tup[1], tup[2]`` indexing, which
    works identically for lists and tuples.
    """
    return ormsgpack.packb(
        [module, name, arg],
        default=_langgoap_msgpack_default,
        option=_LANGGOAP_OPTION,
    )


def _langgoap_msgpack_default(obj: Any) -> Any:
    """Encoder hook that handles ``MappingProxyType``, dataclasses, tuples, and sets.

    See the module docstring for the encoding gaps this hook closes.
    Everything else (pydantic v1/v2, numpy, datetime, enums, ...) is
    delegated to the stock handler unchanged.
    """
    if isinstance(obj, MappingProxyType):
        return dict(obj)
    if isinstance(obj, tuple):
        # Preserve tuple identity across the wire.  ormsgpack natively
        # encodes tuples as msgpack arrays, which decode as lists — so
        # without this branch, ``frozenset[tuple[...]]`` would fail to
        # rehydrate (lists are unhashable).
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _pack_ext_payload("builtins", "tuple", list(obj)),
        )
    if isinstance(obj, (set, frozenset)):
        # Own the set/frozenset branch because the stock handler packs
        # ``tuple(obj)`` via the module-level ``_msgpack_enc`` (without
        # our tuple-passthrough option), which would lose tuple identity
        # for items inside the set.  Going through our own encoder here
        # keeps the tuple wrapping from the branch above in effect for
        # every set element.
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _pack_ext_payload(
                obj.__class__.__module__,
                obj.__class__.__name__,
                list(obj),
            ),
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Own the dataclass branch for the same reason we own the set
        # branch: the stock handler recurses through the module-level
        # ``_msgpack_enc`` which uses the stock default, bypassing our
        # hook and losing every LangGoap-specific encoding inside a
        # dataclass field.
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_KW_ARGS,
            _pack_ext_payload(
                obj.__class__.__module__,
                obj.__class__.__name__,
                {
                    field.name: getattr(obj, field.name)
                    for field in dataclasses.fields(obj)
                },
            ),
        )
    return _stock_msgpack_default(obj)


def _langgoap_msgpack_enc(obj: Any) -> bytes:
    return ormsgpack.packb(
        obj, default=_langgoap_msgpack_default, option=_LANGGOAP_OPTION
    )


# ---------------------------------------------------------------------------
# Serializer for MemorySaver / PostgresSaver (msgpack-only)
# ---------------------------------------------------------------------------


class LangGoapSerializer(JsonPlusSerializer):
    """``JsonPlusSerializer`` subclass that supports LangGoap state.

    Overrides ``dumps_typed`` to route msgpack encoding through
    :func:`_langgoap_msgpack_enc`.  The ``loads_typed`` path is unchanged
    because the stock ext hook already handles every Ext code we emit
    (``EXT_CONSTRUCTOR_SINGLE_ARG`` reconstructs tuples and sets via
    ``cls(arg)``; ``EXT_CONSTRUCTOR_KW_ARGS`` reconstructs dataclasses
    via ``cls(**kwargs)``).  LangGoap's dataclass ``__post_init__``
    hooks re-wrap decoded dicts into ``MappingProxyType`` automatically.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None:
            return "null", b""
        if isinstance(obj, bytes):
            return "bytes", obj
        if isinstance(obj, bytearray):
            return "bytearray", obj
        try:
            return "msgpack", _langgoap_msgpack_enc(obj)
        except ormsgpack.MsgpackEncodeError as exc:
            if self.pickle_fallback:
                return "pickle", pickle.dumps(obj)
            raise exc


# ---------------------------------------------------------------------------
# Serializer for RedisSaver / AsyncRedisSaver (JSON-first + msgpack fallback)
# ---------------------------------------------------------------------------

# Lazy import to avoid hard dependency on langgraph-checkpoint-redis.
_JsonPlusRedisSerializer: type | None = None


def _get_redis_serializer_base() -> type:
    """Import ``JsonPlusRedisSerializer`` on demand."""
    global _JsonPlusRedisSerializer
    if _JsonPlusRedisSerializer is None:
        from langgraph.checkpoint.redis.jsonplus_redis import (
            JsonPlusRedisSerializer,
        )

        _JsonPlusRedisSerializer = JsonPlusRedisSerializer
    return _JsonPlusRedisSerializer


def _make_langgoap_redis_serializer_cls() -> type:
    """Build the LangGoapRedisSerializer class at first use.

    We can't define it at module level because
    ``langgraph-checkpoint-redis`` is an optional dependency —
    referencing its base class at import time would raise ``ImportError``
    on installations without the extra.
    """
    base = _get_redis_serializer_base()

    class LangGoapRedisSerializer(base):  # type: ignore[valid-type,misc]
        """``JsonPlusRedisSerializer`` subclass that supports LangGoap state.

        The Redis checkpointer stores checkpoint data as RedisJSON, which
        requires every value to be JSON-serializable.  The stock
        ``JsonPlusRedisSerializer`` achieves this via a JSON-first
        encoding path (``orjson`` + ``_preprocess_interrupts``), falling
        back to msgpack only for structures containing ``bytes``.

        LangGoap's frozen dataclasses contain two types that the stock
        preprocessor doesn't handle:

        - ``MappingProxyType`` (deep-immutable dict wrapper)
        - ``frozenset`` (used for ``PlanningState.conditions``)

        This subclass adds those branches so the JSON encoding path
        succeeds for the full checkpoint, and overrides the msgpack
        fallback to use :func:`_langgoap_msgpack_enc`.
        """

        def _preprocess_interrupts(self, obj: Any) -> Any:
            import enum

            # Handle MappingProxyType by unwrapping to a plain dict
            # before recursive preprocessing.
            if isinstance(obj, MappingProxyType):
                return self._preprocess_interrupts(dict(obj))
            # Handle frozenset the same way the parent handles set —
            # convert to LC constructor format so it survives JSON
            # encoding and is reconstructed on load via
            # ``_reconstruct_from_constructor``.
            if isinstance(obj, frozenset):
                return {
                    "lc": 2,
                    "type": "constructor",
                    "id": ["builtins", "frozenset"],
                    "kwargs": {
                        "__set_items__": [
                            self._preprocess_interrupts(item)
                            for item in obj
                        ]
                    },
                }
            # Handle tuples explicitly: the parent converts them via
            # ``tuple(processed)`` which drops type info on decode
            # (JSON arrays → Python lists).  We wrap in an LC
            # constructor so that the round-trip through JSON
            # reconstructs a tuple (not a list) — critical for
            # ``frozenset[tuple[str, Any]]``.
            if isinstance(obj, tuple):
                return {
                    "lc": 2,
                    "type": "constructor",
                    "id": ["builtins", "tuple"],
                    "kwargs": {
                        "__tuple_items__": [
                            self._preprocess_interrupts(item)
                            for item in obj
                        ]
                    },
                }
            # Handle enums (e.g. ReplanStrategy, ObjectiveDirection).
            if isinstance(obj, enum.Enum):
                return {
                    "lc": 2,
                    "type": "constructor",
                    "id": [
                        obj.__class__.__module__,
                        obj.__class__.__name__,
                    ],
                    "kwargs": {"__enum_value__": obj.value},
                }
            # Override the dataclass branch.  The parent uses
            # ``dataclasses.asdict`` which calls ``copy.deepcopy`` on
            # field values — and Python 3.14 can't deep-copy
            # ``MappingProxyType``.  We iterate ``dataclasses.fields``
            # directly and recursively preprocess each field value.
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                processed_dict = {
                    f.name: self._preprocess_interrupts(
                        getattr(obj, f.name)
                    )
                    for f in dataclasses.fields(obj)
                }
                return self._encode_constructor_args(
                    type(obj), kwargs=processed_dict
                )
            return super()._preprocess_interrupts(obj)

        def _reconstruct_from_constructor(
            self, obj: dict[str, Any]
        ) -> Any:
            import importlib

            id_parts = obj.get("id", [])
            kwargs = obj.get("kwargs", {})

            # frozenset: same pattern as stock set handling.
            if id_parts == ["builtins", "frozenset"]:
                items = kwargs.get("__set_items__", [])
                return frozenset(
                    self._revive_if_needed(item) for item in items
                )

            # tuple: reconstruct from __tuple_items__.
            if id_parts == ["builtins", "tuple"]:
                items = kwargs.get("__tuple_items__", [])
                return tuple(
                    self._revive_if_needed(item) for item in items
                )

            # enum: reconstruct from __enum_value__.
            if "__enum_value__" in kwargs and len(id_parts) >= 2:
                module_path = ".".join(id_parts[:-1])
                class_name = id_parts[-1]
                cls = getattr(
                    importlib.import_module(module_path), class_name
                )
                return cls(kwargs["__enum_value__"])

            return super()._reconstruct_from_constructor(obj)

        def _default_handler(self, obj: Any) -> Any:
            # Catch any MappingProxyType that slips past preprocessing.
            if isinstance(obj, MappingProxyType):
                return dict(obj)
            return super()._default_handler(obj)

        def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
            if isinstance(obj, bytes):
                return "bytes", obj
            elif isinstance(obj, bytearray):
                return "bytearray", bytes(obj)
            elif obj is None:
                return "null", b""
            else:
                try:
                    # JSON-first path.  Works for metadata dicts, simple
                    # scalars, and LangGoap dataclasses (after
                    # _preprocess_interrupts converts MappingProxyType /
                    # frozenset to JSON-safe LC constructor format).
                    processed_obj = self._preprocess_interrupts(obj)
                    import orjson

                    json_bytes = orjson.dumps(
                        processed_obj, default=self._default_handler
                    )
                    return "json", json_bytes
                except (TypeError, Exception):
                    # Fallback to LangGoap's custom msgpack encoder for
                    # anything JSON can't handle.
                    try:
                        return "msgpack", _langgoap_msgpack_enc(obj)
                    except ormsgpack.MsgpackEncodeError as exc:
                        if self.pickle_fallback:
                            return "pickle", pickle.dumps(obj)
                        raise exc

    return LangGoapRedisSerializer


# Module-level cache for the dynamically-built class.
_LangGoapRedisSerializer: type | None = None


def _get_langgoap_redis_serializer_cls() -> type:
    global _LangGoapRedisSerializer
    if _LangGoapRedisSerializer is None:
        _LangGoapRedisSerializer = _make_langgoap_redis_serializer_cls()
    return _LangGoapRedisSerializer


# Re-export as a name for isinstance checks and the public API.
# At runtime this resolves lazily; for type-checking it's ``type``.
LangGoapRedisSerializer: type  # noqa: N816  (lazy class)


def __getattr__(name: str) -> Any:  # module-level __getattr__
    if name == "LangGoapRedisSerializer":
        return _get_langgoap_redis_serializer_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Auto-install helper
# ---------------------------------------------------------------------------


def _is_redis_serde(serde: Any) -> bool:
    """Check if *serde* is (a subclass of) ``JsonPlusRedisSerializer``."""
    try:
        base = _get_redis_serializer_base()
        return isinstance(serde, base)
    except ImportError:
        return False


def install_langgoap_serde(
    checkpointer: BaseCheckpointSaver,
) -> BaseCheckpointSaver:
    """Replace the checkpointer's serde with a LangGoap-aware variant.

    Auto-detects the backend:

    - **Redis** checkpointers (whose stock serde is
      ``JsonPlusRedisSerializer``) get a ``LangGoapRedisSerializer``
      that preserves the JSON-first encoding required by ``_dump_metadata``
      and falls back to LangGoap's custom msgpack encoder.
    - **All other** checkpointers (``MemorySaver``, ``PostgresSaver``, …)
      get a :class:`LangGoapSerializer` (pure msgpack).

    The helper is idempotent and preserves ``pickle_fallback`` /
    ``allowed_*_modules`` configuration from the original serde.
    """
    current = getattr(checkpointer, "serde", None)

    # Already installed — no-op.
    if isinstance(current, LangGoapSerializer):
        return checkpointer
    if _is_redis_serde(current):
        redis_cls = _get_langgoap_redis_serializer_cls()
        if isinstance(current, redis_cls):
            return checkpointer

    # Carry over every public configuration field we can see on the
    # existing serde so users who constructed ``JsonPlusSerializer(...)``
    # with custom allowlists or pickle_fallback keep their settings.
    init_kwargs: dict[str, Any] = {}
    if current is not None:
        if getattr(current, "pickle_fallback", False):
            init_kwargs["pickle_fallback"] = True
        allowed_json = getattr(current, "_allowed_json_modules", None)
        if allowed_json is not None:
            init_kwargs["allowed_json_modules"] = allowed_json
        allowed_msgpack = getattr(current, "_allowed_msgpack_modules", None)
        if allowed_msgpack is not None:
            init_kwargs["allowed_msgpack_modules"] = allowed_msgpack

    if _is_redis_serde(current):
        cls = _get_langgoap_redis_serializer_cls()
    else:
        cls = LangGoapSerializer

    checkpointer.serde = cls(**init_kwargs)
    return checkpointer
