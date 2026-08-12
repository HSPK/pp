"""In-memory telemetry recorder.

Python port of `packages/telemetry/src/memory.ts`. Recording is passive: a
failure while copying attributes never breaks the traced code.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .noop import NOOP_TELEMETRY_CONTEXT
from .types import SpanAttributes, SpanError, SpanOptions, SpanStatus


@dataclass(frozen=True)
class RecordedTelemetryEvent:
    name: str
    attributes: SpanAttributes


@dataclass(frozen=True)
class RecordedTelemetrySpan:
    id: int
    parent_id: int | None
    name: str
    attributes: SpanAttributes
    events: tuple[RecordedTelemetryEvent, ...]
    status: SpanStatus
    settled: bool
    end_sequence: int | None = None


@dataclass
class _MutableSpan:
    id: int
    parent_id: int | None
    name: str
    attributes: SpanAttributes
    events: list[RecordedTelemetryEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=lambda: SpanStatus("ok"))
    explicit_status: bool = False
    settled: bool = False
    end_sequence: int | None = None


@dataclass
class _State:
    spans: list[_MutableSpan] = field(default_factory=list)
    next_span_id: int = 1
    next_end_sequence: int = 1


def _copy_attribute_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _copy_attributes(attributes: SpanAttributes | None) -> SpanAttributes:
    if not attributes:
        return {}
    return {name: _copy_attribute_value(value) for name, value in attributes.items() if value is not None}


def _merge_attributes(current: SpanAttributes, attributes: SpanAttributes) -> SpanAttributes:
    merged = _copy_attributes(current)
    for name, value in attributes.items():
        if value is not None:
            merged[name] = _copy_attribute_value(value)
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.status == "ok":
        return SpanStatus("ok")
    if status.error is not None:
        return SpanStatus("error", SpanError(status.error.name, status.error.message))
    return SpanStatus("error")


def _automatic_error_status(error: BaseException | None) -> SpanStatus:
    try:
        if isinstance(error, BaseException):
            return SpanStatus("error", SpanError(type(error).__name__, str(error)))
    except Exception:
        # Error inspection is passive. Fall through to a status without details.
        pass
    return SpanStatus("error")


def _settle_span(state: _State, span: _MutableSpan, failed: bool, error: BaseException | None = None) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        span.status = _automatic_error_status(error)
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


class _InMemorySpan:
    def __init__(self, state: _State, recorded: _MutableSpan) -> None:
        self._state = state
        self._recorded = recorded

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[Any], Any | Awaitable[Any]],
    ) -> Any:
        return await _start_in_memory_span(self._state, self._recorded, options, callback)

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        if self._recorded.settled:
            return
        # Recording is passive. Ignore malformed telemetry payloads.
        with contextlib.suppress(Exception):
            self._recorded.events.append(RecordedTelemetryEvent(name, _copy_attributes(attributes)))

    def set_attributes(self, attributes: SpanAttributes) -> None:
        if self._recorded.settled:
            return
        with contextlib.suppress(Exception):
            self._recorded.attributes = _merge_attributes(self._recorded.attributes, attributes)

    def set_status(self, status: SpanStatus) -> None:
        if self._recorded.settled:
            return
        try:
            self._recorded.status = _copy_status(status)
            self._recorded.explicit_status = True
        except Exception:
            pass


async def _start_in_memory_span(
    state: _State,
    parent: _MutableSpan | None,
    options: SpanOptions,
    callback: Callable[[Any], Any | Awaitable[Any]],
) -> Any:
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        recorded = _MutableSpan(
            id=state.next_span_id,
            parent_id=parent.id if parent else None,
            name=options.name,
            attributes=_copy_attributes(options.attributes),
        )
        state.next_span_id += 1
        state.spans.append(recorded)
    except Exception:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    span = _InMemorySpan(state, recorded)
    try:
        result = callback(span)
        if inspect.isawaitable(result):
            result = await result
    except BaseException as error:
        _settle_span(state, recorded, True, error)
        raise
    _settle_span(state, recorded, False)
    return result


class InMemoryTelemetryContext:
    """Backend-neutral reference implementation that records spans in memory."""

    def __init__(self) -> None:
        self._state = _State()

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[Any], Any | Awaitable[Any]],
    ) -> Any:
        return await _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> list[RecordedTelemetrySpan]:
        """Return detached snapshots in span-start order."""
        return [
            RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_copy_attributes(span.attributes),
                events=tuple(
                    RecordedTelemetryEvent(event.name, _copy_attributes(event.attributes)) for event in span.events
                ),
                status=_copy_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            for span in self._state.spans
        ]
