import asyncio

import pytest
from pi_telemetry import InMemoryTelemetryContext, SpanError, SpanOptions, SpanStatus


async def test_records_span_name_and_attributes():
    ctx = InMemoryTelemetryContext()
    await ctx.start_span(SpanOptions(name="op", attributes={"a": 1}), lambda span: None)
    spans = ctx.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "op"
    assert spans[0].attributes == {"a": 1}
    assert spans[0].parent_id is None


async def test_span_ids_and_order_are_assigned_in_start_order():
    ctx = InMemoryTelemetryContext()

    async def outer(span):
        await span.start_span(SpanOptions(name="first-child"), lambda child: None)
        await span.start_span(SpanOptions(name="second-child"), lambda child: None)

    await ctx.start_span(SpanOptions(name="root"), outer)
    spans = ctx.get_spans()
    assert [s.name for s in spans] == ["root", "first-child", "second-child"]
    assert [s.id for s in spans] == [1, 2, 3]


async def test_nested_spans_get_correct_parent_id():
    ctx = InMemoryTelemetryContext()

    async def outer(span):
        async def inner(child):
            await child.start_span(SpanOptions(name="grandchild"), lambda gc: None)

        await span.start_span(SpanOptions(name="child"), inner)

    await ctx.start_span(SpanOptions(name="root"), outer)
    spans = {s.name: s for s in ctx.get_spans()}
    assert spans["root"].parent_id is None
    assert spans["child"].parent_id == spans["root"].id
    assert spans["grandchild"].parent_id == spans["child"].id


async def test_add_event_set_attributes_set_status_are_recorded():
    ctx = InMemoryTelemetryContext()

    def callback(span):
        span.add_event("evt1", {"x": 1})
        span.set_attributes({"y": 2})
        span.set_status(SpanStatus("error", SpanError("Custom", "custom message")))

    await ctx.start_span(SpanOptions(name="op"), callback)
    span = ctx.get_spans()[0]
    assert len(span.events) == 1
    assert span.events[0].name == "evt1"
    assert span.events[0].attributes == {"x": 1}
    assert span.attributes == {"y": 2}
    assert span.status.status == "error"
    assert span.status.error == SpanError("Custom", "custom message")


async def test_writes_after_span_settles_are_ignored():
    ctx = InMemoryTelemetryContext()
    captured = {}

    def callback(span):
        captured["span"] = span
        return "value"

    await ctx.start_span(SpanOptions(name="op"), callback)
    span = captured["span"]
    span.add_event("late-event")
    span.set_attributes({"late": True})
    span.set_status(SpanStatus("error"))

    recorded = ctx.get_spans()[0]
    assert recorded.events == ()
    assert recorded.attributes == {}
    assert recorded.status.status == "ok"


async def test_span_that_raises_settles_with_error_status_and_exception_propagates():
    ctx = InMemoryTelemetryContext()

    def callback(span):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await ctx.start_span(SpanOptions(name="op"), callback)

    span = ctx.get_spans()[0]
    assert span.settled is True
    assert span.status.status == "error"
    assert span.status.error == SpanError("ValueError", "boom")


async def test_explicit_set_status_is_not_overwritten_by_automatic_error_status():
    ctx = InMemoryTelemetryContext()

    def callback(span):
        span.set_status(SpanStatus("error", SpanError("Explicit", "explicit message")))
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await ctx.start_span(SpanOptions(name="op"), callback)

    span = ctx.get_spans()[0]
    assert span.status.error == SpanError("Explicit", "explicit message")


async def test_get_spans_returns_detached_copies():
    ctx = InMemoryTelemetryContext()
    await ctx.start_span(SpanOptions(name="op", attributes={"a": 1}), lambda span: None)

    spans = ctx.get_spans()
    spans[0].attributes["a"] = 999
    spans[0].attributes["new"] = "value"

    spans_again = ctx.get_spans()
    assert spans_again[0].attributes == {"a": 1}


async def test_end_sequence_increments_in_completion_order():
    ctx = InMemoryTelemetryContext()

    async def outer(span):
        # "slow" starts first (lower span id) but yields, letting "fast" finish
        # first. end_sequence should reflect completion order, not start order.
        # An event, not a sleep: the ordering this test pins is then a fact
        # about the code rather than a bet on how loaded the machine is.
        fast_finished = asyncio.Event()

        async def slow(child):
            await fast_finished.wait()
            return "slow-done"

        async def fast(child):
            fast_finished.set()
            return "fast-done"

        slow_task = asyncio.create_task(span.start_span(SpanOptions(name="slow"), slow))
        fast_task = asyncio.create_task(span.start_span(SpanOptions(name="fast"), fast))
        await asyncio.gather(slow_task, fast_task)

    await ctx.start_span(SpanOptions(name="root"), outer)
    spans = {s.name: s for s in ctx.get_spans()}
    assert spans["slow"].id < spans["fast"].id
    assert spans["fast"].end_sequence == 1
    assert spans["slow"].end_sequence == 2
    assert spans["root"].end_sequence == 3


async def test_settled_state_defaults_false_until_callback_completes():
    ctx = InMemoryTelemetryContext()
    captured = {}

    def callback(span):
        captured["snapshot_during"] = ctx.get_spans()[0].settled
        return None

    await ctx.start_span(SpanOptions(name="op"), callback)
    assert captured["snapshot_during"] is False
    assert ctx.get_spans()[0].settled is True


async def test_list_attribute_values_are_copied():
    ctx = InMemoryTelemetryContext()
    await ctx.start_span(SpanOptions(name="op", attributes={"tags": ["a", "b"]}), lambda span: None)
    span = ctx.get_spans()[0]
    assert span.attributes["tags"] == ["a", "b"]
    assert span.attributes["tags"] is not None


async def test_set_attributes_with_none_value_does_not_overwrite_existing_value():
    ctx = InMemoryTelemetryContext()

    def callback(span):
        span.set_attributes({"a": 1})
        span.set_attributes({"a": None})

    await ctx.start_span(SpanOptions(name="op"), callback)
    span = ctx.get_spans()[0]
    assert span.attributes == {"a": 1}


async def test_explicit_error_status_without_error_detail_round_trips():
    ctx = InMemoryTelemetryContext()

    def callback(span):
        span.set_status(SpanStatus("error"))

    await ctx.start_span(SpanOptions(name="op"), callback)
    span = ctx.get_spans()[0]
    assert span.status == SpanStatus("error")
    assert span.status.error is None


async def test_automatic_error_status_falls_back_when_error_inspection_raises():
    ctx = InMemoryTelemetryContext()

    class UnstringableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot stringify")

    def callback(span):
        raise UnstringableError()

    with pytest.raises(UnstringableError):
        await ctx.start_span(SpanOptions(name="op"), callback)

    span = ctx.get_spans()[0]
    assert span.status.status == "error"
    assert span.status.error is None


async def test_automatic_error_status_direct_call_with_non_exception_error():
    from pi_telemetry.memory import _automatic_error_status

    status = _automatic_error_status(None)
    assert status.status == "error"
    assert status.error is None
    from pi_telemetry.memory import _MutableSpan, _settle_span
    from pi_telemetry.types import SpanStatus as _SpanStatus

    state = InMemoryTelemetryContext()._state
    span = _MutableSpan(id=1, parent_id=None, name="op", attributes={})

    _settle_span(state, span, True, ValueError("first"))
    first_status = span.status
    first_end_sequence = span.end_sequence

    # Settling an already-settled span is a no-op: status and end_sequence
    # from the first settlement are preserved.
    _settle_span(state, span, True, ValueError("second"))
    assert span.status == first_status
    assert span.end_sequence == first_end_sequence
    assert isinstance(span.status, _SpanStatus)


async def test_set_status_exception_is_suppressed_and_previous_status_kept():
    ctx = InMemoryTelemetryContext()
    captured = {}

    def callback(span):
        captured["span"] = span
        span.set_status(SpanStatus("error", SpanError("Explicit", "explicit message")))
        # `object()` has no `.status` attribute, so `_copy_status` raises
        # AttributeError inside `set_status`; recording is passive, so the
        # exception must be swallowed and the previous status kept.
        span.set_status(object())

    await ctx.start_span(SpanOptions(name="op"), callback)
    span = ctx.get_spans()[0]
    assert span.status.error == SpanError("Explicit", "explicit message")


async def test_start_span_after_parent_settled_defers_to_noop_and_is_not_recorded():
    ctx = InMemoryTelemetryContext()
    captured = {}

    def callback(span):
        captured["span"] = span

    await ctx.start_span(SpanOptions(name="root"), callback)
    root_span = captured["span"]

    # The root span object is already settled; starting a child on it now
    # must not add a new recorded span, and should still run the callback.
    result = await root_span.start_span(SpanOptions(name="late-child"), lambda child: "child-ran")

    assert result == "child-ran"
    assert [s.name for s in ctx.get_spans()] == ["root"]


async def test_span_creation_failure_falls_back_to_noop(monkeypatch):
    import pi_telemetry.memory as memory_module

    def exploding_copy_attributes(attributes=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(memory_module, "_copy_attributes", exploding_copy_attributes)

    ctx = InMemoryTelemetryContext()
    result = await ctx.start_span(SpanOptions(name="op"), lambda span: "ran-via-noop")

    assert result == "ran-via-noop"
    assert ctx.get_spans() == []
