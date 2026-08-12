import pytest
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SpanOptions, SpanStatus


async def test_start_span_runs_sync_callback_and_returns_value():
    result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="op"), lambda span: 42)
    assert result == 42


async def test_start_span_runs_async_callback_and_returns_value():
    async def callback(span):
        return "done"

    result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="op"), callback)
    assert result == "done"


async def test_start_span_propagates_sync_exception():
    def callback(span):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="op"), callback)


async def test_start_span_propagates_async_exception():
    async def callback(span):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="op"), callback)


async def test_span_methods_are_no_ops():
    async def callback(span):
        span.add_event("evt", {"a": 1})
        span.set_attributes({"b": 2})
        span.set_status(SpanStatus("ok"))
        nested = await span.start_span(SpanOptions(name="child"), lambda child: "child-result")
        return nested

    result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="parent"), callback)
    assert result == "child-result"
