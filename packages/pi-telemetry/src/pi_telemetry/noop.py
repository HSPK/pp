"""No-op telemetry implementation.

Python port of `packages/telemetry/src/noop.ts`.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .types import SpanAttributes, SpanOptions, SpanStatus


class NoopTelemetrySpan:
    """A span that discards everything written to it.

    `__slots__` is the Python analogue of upstream's `Object.freeze`: the
    shared context is handed to every caller, so it must not be possible to
    hang per-caller state off it.
    """

    __slots__ = ()

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[NoopTelemetrySpan], Any | Awaitable[Any]],
    ) -> Any:
        result = callback(self)
        if inspect.isawaitable(result):
            return await result
        return result

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        return None

    def set_attributes(self, attributes: SpanAttributes) -> None:
        return None

    def set_status(self, status: SpanStatus) -> None:
        return None


NOOP_TELEMETRY_CONTEXT = NoopTelemetrySpan()
"""Shared telemetry context used when an application does not provide one."""
