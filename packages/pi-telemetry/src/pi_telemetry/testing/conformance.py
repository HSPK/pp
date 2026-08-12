"""Runner-independent conformance cases for the callback telemetry adapter contract.

Python port of `packages/telemetry/src/testing/conformance.ts`. Every adapter
that records spans should pass these, so a new backend can be checked against
the same contract the in-memory reference implementation satisfies.

Two upstream behaviours have no Python analogue and are noted at their case:
a rejection value of `undefined`, and admitting the callback *synchronously*
before the returned promise is awaited (`start_span` is `async def` here, so
nothing runs until it is awaited).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, NoReturn

from ..memory import RecordedTelemetryEvent, RecordedTelemetrySpan
from ..types import SpanError, SpanOptions, SpanStatus, TelemetrySpan
from .types import (
    TelemetryAdapterConformanceCase,
    TelemetryAdapterFixture,
    TelemetryAdapterFixtureFactory,
)


class UnreadableMapping(dict):
    """A mapping whose every read raises, like upstream's throwing `Proxy`."""

    def _raise(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("read")

    __getitem__ = _raise
    __iter__ = _raise
    __len__ = _raise
    __bool__ = _raise
    __contains__ = _raise
    get = _raise
    keys = _raise
    items = _raise
    values = _raise


class UnreadableSequence(list):
    """An attribute value that raises while it is being copied."""

    def __iter__(self) -> NoReturn:
        raise RuntimeError("read")

    def __len__(self) -> NoReturn:
        raise RuntimeError("read")


class UnreadableSpanOptions(SpanOptions):
    """`SpanOptions` whose fields raise when read."""

    def __getattribute__(self, name: str) -> NoReturn:
        raise RuntimeError("read")


class UnreadableSpanStatus(SpanStatus):
    """`SpanStatus` whose fields raise when read."""

    def __getattribute__(self, name: str) -> NoReturn:
        raise RuntimeError("read")


class UnreadableError(Exception):
    """An error whose message cannot be read, so status derivation must stay passive."""

    def __str__(self) -> NoReturn:
        raise RuntimeError("read")

    def __repr__(self) -> NoReturn:
        raise RuntimeError("read")


def unreadable_span_options() -> SpanOptions:
    return UnreadableSpanOptions.__new__(UnreadableSpanOptions)


def unreadable_span_status() -> SpanStatus:
    return UnreadableSpanStatus.__new__(UnreadableSpanStatus)


def _find_span(spans: Sequence[RecordedTelemetrySpan], name: str) -> RecordedTelemetrySpan:
    for span in spans:
        if span.name == name:
            return span
    raise AssertionError(f"Expected recorded span {name}")


async def _rejects_with_same_value(operation: Any, expected: BaseException, label: str) -> None:
    try:
        await operation
    except BaseException as error:
        if error is not expected:
            raise AssertionError(f"{label}: rejected with a different value") from None
        return
    raise AssertionError(f"{label}: expected the operation to reject")


async def _admits_once_and_preserves_the_result(fixture: TelemetryAdapterFixture) -> None:
    calls = 0
    expected = {"value": 42}

    def callback(_span: TelemetrySpan) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return expected

    # Upstream also asserts the callback is admitted *synchronously*, before the
    # returned promise is awaited. `start_span` is a coroutine function here, so
    # nothing can run until it is awaited; only the call count and the preserved
    # result are checkable.
    result = await fixture.context.start_span(SpanOptions(name="success"), callback)

    assert calls == 1
    assert result is expected
    spans = await fixture.get_spans()
    assert _find_span(spans, "success").status == SpanStatus("ok")
    assert _find_span(spans, "success").settled is True


async def _preserves_rejection_values(fixture: TelemetryAdapterFixture) -> None:
    sync_error = RuntimeError("sync")

    def raise_sync(_span: TelemetrySpan) -> NoReturn:
        raise sync_error

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="sync-error"), raise_sync),
        sync_error,
        "sync-error",
    )

    async_error = RuntimeError("async")

    async def raise_async(_span: TelemetrySpan) -> NoReturn:
        raise async_error

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="async-error"), raise_async),
        async_error,
        "async-error",
    )

    # Upstream additionally rejects with `undefined`. Python can only raise
    # exception instances, so that case has no analogue and is omitted rather
    # than replaced with a different value.

    unreadable_error = UnreadableError()

    def raise_unreadable(_span: TelemetrySpan) -> NoReturn:
        raise unreadable_error

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="unreadable-error"), raise_unreadable),
        unreadable_error,
        "unreadable-error",
    )

    async_unreadable_error = UnreadableError()

    async def raise_async_unreadable(_span: TelemetrySpan) -> NoReturn:
        raise async_unreadable_error

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="async-unreadable-error"), raise_async_unreadable),
        async_unreadable_error,
        "async-unreadable-error",
    )

    spans = await fixture.get_spans()
    for name in ("sync-error", "async-error", "unreadable-error", "async-unreadable-error"):
        assert _find_span(spans, name).status.status == "error"


async def _uses_last_explicit_status(fixture: TelemetryAdapterFixture) -> None:
    def set_two_statuses(span: TelemetrySpan) -> None:
        span.set_status(SpanStatus("error", SpanError("Expected", "first")))
        span.set_status(SpanStatus("ok"))

    await fixture.context.start_span(SpanOptions(name="last-status"), set_two_statuses)

    thrown = RuntimeError("after explicit status")

    def ok_then_throw(span: TelemetrySpan) -> NoReturn:
        span.set_status(SpanStatus("ok"))
        raise thrown

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="explicit-before-throw"), ok_then_throw),
        thrown,
        "explicit-before-throw",
    )

    rejected = RuntimeError("after async explicit status")

    async def error_then_reject(span: TelemetrySpan) -> NoReturn:
        span.set_status(SpanStatus("error", SpanError("Expected", "async failure")))
        raise rejected

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="explicit-before-rejection"), error_then_reject),
        rejected,
        "explicit-before-rejection",
    )

    def error_then_return(span: TelemetrySpan) -> dict[str, bool]:
        span.set_status(SpanStatus("error", SpanError("Expected", "returned failure")))
        return {"ok": False}

    await fixture.context.start_span(SpanOptions(name="expected-failure"), error_then_return)

    spans = await fixture.get_spans()
    assert _find_span(spans, "last-status").status == SpanStatus("ok")
    assert _find_span(spans, "explicit-before-throw").status == SpanStatus("ok")
    assert _find_span(spans, "explicit-before-rejection").status == SpanStatus(
        "error", SpanError("Expected", "async failure")
    )
    assert _find_span(spans, "expected-failure").status == SpanStatus(
        "error", SpanError("Expected", "returned failure")
    )


async def _merges_attributes_and_records_ordered_events(fixture: TelemetryAdapterFixture) -> None:
    def record(span: TelemetrySpan) -> None:
        span.set_attributes({"count": 1, "overwrite": "middle"})
        # `None` is this port's analogue of upstream's `undefined`: it must not
        # overwrite an already recorded value.
        span.set_attributes({"count": None, "overwrite": "end"})
        span.add_event("first", {"index": 1, "ignored": None})
        span.add_event("second", {"index": 2})

    await fixture.context.start_span(
        SpanOptions(name="recording", attributes={"start": "value", "overwrite": "start", "ignored": None}),
        record,
    )

    span = _find_span(await fixture.get_spans(), "recording")
    assert span.attributes == {"start": "value", "overwrite": "end", "count": 1}
    assert list(span.events) == [
        RecordedTelemetryEvent("first", {"index": 1}),
        RecordedTelemetryEvent("second", {"index": 2}),
    ]


async def _ignores_failed_attribute_calls_atomically(fixture: TelemetryAdapterFixture) -> None:
    def record(span: TelemetrySpan) -> None:
        span.set_attributes({"partial": "must not survive", "unreadable": UnreadableSequence(["value"])})

    await fixture.context.start_span(
        SpanOptions(name="atomic-attributes", attributes={"retained": "value"}),
        record,
    )

    assert _find_span(await fixture.get_spans(), "atomic-attributes").attributes == {"retained": "value"}


async def _makes_calls_after_settlement_inert(fixture: TelemetryAdapterFixture) -> None:
    settled_span: TelemetrySpan | None = None

    def capture(span: TelemetrySpan) -> None:
        nonlocal settled_span
        settled_span = span

    await fixture.context.start_span(SpanOptions(name="settled", attributes={"value": "initial"}), capture)
    captured_span = settled_span
    assert captured_span is not None

    captured_span.set_attributes({"value": "late"})
    captured_span.add_event("late", {"value": True})
    captured_span.set_status(SpanStatus("error"))

    child_admitted = False

    def child(_span: TelemetrySpan) -> int:
        nonlocal child_admitted
        child_admitted = True
        return 7

    child_result = await captured_span.start_span(SpanOptions(name="late-child"), child)
    # Upstream asserts `childAdmitted` before awaiting the child, pinning
    # synchronous admission; `start_span` is `async def` here, so admission can
    # only be observed after the await.
    assert child_admitted is True
    assert child_result == 7

    spans = await fixture.get_spans()
    assert len(spans) == 1
    assert spans[0].attributes == {"value": "initial"}
    assert list(spans[0].events) == []
    assert spans[0].status == SpanStatus("ok")


async def _records_nested_and_concurrent_children(fixture: TelemetryAdapterFixture) -> None:
    first_gate = asyncio.Event()

    async def parent_body(parent: TelemetrySpan) -> None:
        async def first_child(_span: TelemetrySpan) -> None:
            await first_gate.wait()

        first = asyncio.ensure_future(parent.start_span(SpanOptions(name="first-child"), first_child))
        # Let the first child start (and block) before the second one runs, so
        # the two are genuinely concurrent siblings.
        await asyncio.sleep(0)
        second = await parent.start_span(SpanOptions(name="second-child"), lambda _span: "done")
        assert second == "done"
        first_gate.set()
        await first

    await fixture.context.start_span(SpanOptions(name="parent"), parent_body)

    spans = await fixture.get_spans()
    parent = _find_span(spans, "parent")
    first = _find_span(spans, "first-child")
    second = _find_span(spans, "second-child")
    assert parent.parent_id is None
    assert first.parent_id == parent.id
    assert second.parent_id == parent.id
    assert second.end_sequence is not None
    assert first.end_sequence is not None
    assert parent.end_sequence is not None
    assert second.end_sequence < first.end_sequence
    assert first.end_sequence < parent.end_sequence


async def _suppresses_unreadable_payload_failures(fixture: TelemetryAdapterFixture) -> None:
    calls = 0

    def callback(_span: TelemetrySpan) -> int:
        nonlocal calls
        calls += 1
        return 9

    result = await fixture.context.start_span(unreadable_span_options(), callback)

    assert calls == 1
    assert result == 9
    assert list(await fixture.get_spans()) == []

    def record(span: TelemetrySpan) -> None:
        attributes = UnreadableMapping()
        span.set_attributes(attributes)
        span.add_event("unreadable-event", attributes)
        span.set_status(unreadable_span_status())

    await fixture.context.start_span(SpanOptions(name="unreadable-recording"), record)

    recorded = await fixture.get_spans()
    assert len(recorded) == 1
    assert recorded[0].attributes == {}
    assert list(recorded[0].events) == []
    assert recorded[0].status == SpanStatus("ok")


async def _ignores_failed_status_calls_atomically(fixture: TelemetryAdapterFixture) -> None:
    rejection = RuntimeError("rejected after unreadable status")

    async def unreadable_status_then_reject(span: TelemetrySpan) -> NoReturn:
        span.set_status(unreadable_span_status())
        raise rejection

    await _rejects_with_same_value(
        fixture.context.start_span(SpanOptions(name="unreadable-status"), unreadable_status_then_reject),
        rejection,
        "unreadable-status",
    )

    assert _find_span(await fixture.get_spans(), "unreadable-status").status.status == "error"


def _create_case(
    factory: TelemetryAdapterFixtureFactory,
    group: str,
    name: str,
    test: Any,
) -> TelemetryAdapterConformanceCase:
    async def run() -> None:
        fixture = await factory()
        try:
            await test(fixture)
        finally:
            await fixture.aclose()

    return TelemetryAdapterConformanceCase(group=group, name=name, run=run)


def create_telemetry_adapter_conformance(
    factory: TelemetryAdapterFixtureFactory,
) -> list[TelemetryAdapterConformanceCase]:
    """Creates runner-independent cases for the callback telemetry adapter contract."""
    return [
        _create_case(
            factory,
            "callback lifecycle",
            "admits once and preserves the result",
            _admits_once_and_preserves_the_result,
        ),
        _create_case(
            factory,
            "callback lifecycle",
            "preserves synchronous and asynchronous rejection values",
            _preserves_rejection_values,
        ),
        _create_case(
            factory,
            "status",
            "uses last explicit status without automatic overwrite",
            _uses_last_explicit_status,
        ),
        _create_case(
            factory,
            "recording",
            "merges attributes and records ordered events",
            _merges_attributes_and_records_ordered_events,
        ),
        _create_case(
            factory,
            "recording",
            "ignores failed attribute calls atomically",
            _ignores_failed_attribute_calls_atomically,
        ),
        _create_case(
            factory,
            "recording",
            "makes calls after settlement inert",
            _makes_calls_after_settlement_inert,
        ),
        _create_case(
            factory,
            "parentage",
            "records nested and concurrent child relationships",
            _records_nested_and_concurrent_children,
        ),
        _create_case(
            factory,
            "passivity",
            "suppresses unreadable telemetry payload failures",
            _suppresses_unreadable_payload_failures,
        ),
        _create_case(
            factory,
            "passivity",
            "ignores failed status calls atomically",
            _ignores_failed_status_calls_atomically,
        ),
    ]
