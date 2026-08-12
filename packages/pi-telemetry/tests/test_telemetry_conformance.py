"""Python port of `packages/telemetry/test/conformance.test.ts`.

Runs the shared adapter conformance suite against `InMemoryTelemetryContext`,
plus the bespoke case pinning that recorded snapshots are detached copies.

Named `test_telemetry_conformance` rather than `test_conformance` because
pytest's prepend import mode puts every package's `tests/` directory on
`sys.path`, so two `test_conformance.py` modules in different packages would
collide (`pi-server` already has one).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pi_telemetry import InMemoryTelemetryContext, RecordedTelemetrySpan, SpanOptions, TelemetrySpan
from pi_telemetry.testing import TelemetryAdapterFixture, create_telemetry_adapter_conformance
from pi_telemetry.testing.conformance import (
    UnreadableError,
    UnreadableMapping,
    UnreadableSequence,
    unreadable_span_options,
    unreadable_span_status,
)


class _InMemoryFixture:
    def __init__(self) -> None:
        self._context = InMemoryTelemetryContext()

    @property
    def context(self) -> InMemoryTelemetryContext:
        return self._context

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]:
        return self._context.get_spans()

    async def aclose(self) -> None:
        return None


async def _factory() -> TelemetryAdapterFixture:
    return _InMemoryFixture()


CONFORMANCE = create_telemetry_adapter_conformance(_factory)


@pytest.mark.parametrize(
    "case",
    CONFORMANCE,
    ids=[f"{case.group}: {case.name}" for case in CONFORMANCE],
)
async def test_in_memory_telemetry_context_conformance(case) -> None:
    await case.run()


async def test_the_unreadable_conformance_fixtures_really_do_raise_on_every_read() -> None:
    """Guards the passivity cases against passing vacuously."""
    with pytest.raises(RuntimeError):
        bool(UnreadableMapping())
    with pytest.raises(RuntimeError):
        list(UnreadableSequence(["value"]))
    with pytest.raises(RuntimeError):
        _ = unreadable_span_options().name
    with pytest.raises(RuntimeError):
        _ = unreadable_span_status().status
    with pytest.raises(RuntimeError):
        str(UnreadableError())


async def test_returns_detached_snapshots_without_exposing_mutable_recording_state() -> None:
    context = InMemoryTelemetryContext()
    open_settled: bool | None = None
    open_end_sequence: int | None = None

    def record(span: TelemetrySpan) -> None:
        nonlocal open_settled, open_end_sequence
        span.add_event("event", {"value": 1})
        open_span = context.get_spans()[0]
        open_settled = open_span.settled
        open_end_sequence = open_span.end_sequence

    await context.start_span(SpanOptions(name="snapshot", attributes={"tags": ["initial"]}), record)

    assert open_settled is False
    assert open_end_sequence is None
    first = context.get_spans()[0]
    assert first.settled is True
    assert first.end_sequence == 1
    first.attributes["tags"] = ["mutated"]
    first.events[0].attributes["value"] = 2

    second = context.get_spans()[0]
    assert second.attributes == {"tags": ["initial"]}
    assert [(event.name, event.attributes) for event in second.events] == [("event", {"value": 1})]
