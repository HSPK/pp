"""Telemetry primitives.

Python port of `packages/telemetry/src/index.ts` and `noop.ts`. The TypeScript
API is callback based (``startSpan(options, callback)``); the Python port keeps
that shape as an async method and additionally exposes it as an async context
manager, which is the idiomatic way to scope a span in Python.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

AttributeValue = str | int | float | bool | Sequence[str] | Sequence[int] | Sequence[float] | Sequence[bool]
SpanAttributes = dict[str, Any]

T = TypeVar("T")


@dataclass
class SpanOptions:
    name: str
    attributes: SpanAttributes = field(default_factory=dict)


@dataclass
class SpanError:
    name: str
    message: str


@dataclass
class SpanStatus:
    status: str
    """``"ok"`` or ``"error"``."""
    error: SpanError | None = None


@runtime_checkable
class TelemetryContext(Protocol):
    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Any | Awaitable[Any]],
    ) -> Any: ...


@runtime_checkable
class TelemetrySpan(TelemetryContext, Protocol):
    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...
