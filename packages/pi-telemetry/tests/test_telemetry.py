"""Python port of `packages/telemetry/test/telemetry.test.ts`.

Large parts of the upstream file are `expectTypeOf` / `@ts-expect-error`
assertions: they pin that an unknown span name, a missing required attribute,
a value outside a closed set, or a duplicate span name across schemas is
rejected *by the compiler*. Python has no type-level assertion to run and
nothing raises at runtime, because upstream states outright that "schema values
are used only for type inference; no runtime schema validation is performed"
and this port keeps that. Those assertions are individually noted below with
the reason instead of being silently dropped.

`unreadable()` upstream is a `Proxy` that throws on every read, proving the
no-op context never inspects payloads. The Python analogue is a class whose
`__getattribute__`/mapping reads raise.
"""

from __future__ import annotations

import json
from typing import Any, NoReturn

import pytest
from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    InMemoryTelemetryContext,
    SpanOptions,
    TelemetrySpan,
    create_typed_span_starter,
    define_telemetry_schema,
)


class _Unreadable(dict):
    """A mapping that raises on every read, like upstream's throwing `Proxy`."""

    def _boom(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("read")

    __getitem__ = _boom
    __iter__ = _boom
    __len__ = _boom
    get = _boom
    keys = _boom
    items = _boom
    values = _boom


class _UnreadableSpanOptions(SpanOptions):
    """`SpanOptions` whose fields raise when read."""

    def __getattribute__(self, name: str) -> NoReturn:
        raise RuntimeError("read")


def test_preserves_serializable_definitions_and_infers_exact_attributes():
    definition: dict[str, Any] = {
        "version": 1,
        "spans": {
            "operation": {
                "description": "Test operation",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "kind": {"type": "string", "required": True, "values": ["read", "write"], "description": "Kind"},
                },
                "endAttributes": {},
                "events": {
                    "result": {
                        "description": "Result",
                        "attributes": {
                            "outcome": {
                                "type": "string",
                                "required": True,
                                "values": ["ok", "error"],
                                "description": "Outcome",
                            },
                        },
                    },
                },
                "status": {"default": "ok", "errorWhen": "The operation fails"},
            },
        },
    }
    schema = define_telemetry_schema(definition)

    assert schema is definition
    json.dumps(schema)

    # `expectTypeOf<TelemetrySchemaSpanStartAttributes<...>>()` and the
    # `@ts-expect-error` block that follows it are compile-time only: they pin
    # that a required event attribute cannot be omitted, that closed-set values
    # are exact, that undeclared events are rejected, and that an empty
    # `endAttributes` schema rejects every attribute. Python has no type-level
    # assertion and this port performs no runtime validation (upstream does
    # not either), so only the schema data those types are derived from can be
    # checked here.
    operation = schema["spans"]["operation"]
    assert operation["startAttributes"]["kind"]["required"] is True
    assert operation["startAttributes"]["kind"]["values"] == ["read", "write"]
    assert operation["endAttributes"] == {}
    assert operation["events"]["result"]["attributes"]["outcome"]["values"] == ["ok", "error"]


@pytest.mark.asyncio
async def test_combines_schema_vocabularies_and_binds_child_starters_to_their_parent_spans():
    operation_schema = define_telemetry_schema(
        {
            "version": 1,
            "spans": {
                "operation": {
                    "description": "Operation",
                    "parents": {"kind": "root_or_external"},
                    "startAttributes": {
                        "kind": {
                            "type": "string",
                            "required": True,
                            "values": ["read", "write"],
                            "description": "Kind",
                        },
                    },
                    "endAttributes": {},
                    "status": {"default": "ok", "errorWhen": "The operation fails"},
                },
            },
        }
    )
    request_schema = define_telemetry_schema(
        {
            "version": 3,
            "spans": {
                "request": {
                    "description": "Request",
                    "parents": {"kind": "spans", "spans": ["operation"]},
                    "startAttributes": {
                        "provider": {"type": "string", "required": True, "description": "Provider"},
                    },
                    "endAttributes": {
                        "response": {"type": "string", "description": "Response kind"},
                    },
                    "status": {"default": "ok", "errorWhen": "The request fails"},
                },
            },
        }
    )
    telemetry_context = InMemoryTelemetryContext()
    start_span = create_typed_span_starter(telemetry_context, [operation_schema, request_schema])

    async def run_operation(_operation_span: TelemetrySpan, start_child_span: Any) -> Any:
        def run_request(request_span: TelemetrySpan, _start_grandchild_span: Any) -> int:
            request_span.set_attributes({"response": "cached"})
            return 42

        return await start_child_span("request", {"provider": "example"}, run_request)

    result = await start_span("operation", {"kind": "read"}, run_operation)

    assert result == 42

    spans = telemetry_context.get_spans()
    operation_span = next(span for span in spans if span.name == "operation")
    request_span = next(span for span in spans if span.name == "request")
    assert operation_span.parent_id is None
    assert request_span.parent_id == operation_span.id

    # `createTypedSpanStarter` never reads the schemas it is handed.
    create_typed_span_starter(telemetry_context, _Unreadable())

    sync_error = RuntimeError("sync")

    def raise_sync(_span: TelemetrySpan, _start_child: Any) -> NoReturn:
        raise sync_error

    with pytest.raises(RuntimeError) as caught_sync:
        await start_span("operation", {"kind": "write"}, raise_sync)
    assert caught_sync.value is sync_error

    async_error = RuntimeError("async")

    async def raise_async(_span: TelemetrySpan, _start_child: Any) -> NoReturn:
        raise async_error

    with pytest.raises(RuntimeError) as caught_async:
        await start_span("request", {"provider": "example"}, raise_async)
    assert caught_async.value is async_error

    # The remaining upstream assertions in this case are `@ts-expect-error`
    # compile-time checks (union-valued span names must be narrowed, unknown
    # attributes are rejected, unknown span names are rejected, attributes come
    # from the schema owning the span, duplicate span names across schemas are
    # rejected). None has a Python analogue: the starter accepts any name and
    # any attribute mapping at runtime, upstream included.


@pytest.mark.asyncio
async def test_noop_admits_callbacks_synchronously_and_reuses_one_inert_span():
    admitted = False
    first_span: TelemetrySpan | None = None

    async def run(span: TelemetrySpan) -> int:
        nonlocal admitted, first_span
        admitted = True
        first_span = span
        child = await span.start_span(SpanOptions(name="child"), lambda child_span: child_span)
        assert child is span
        return 42

    # Upstream asserts `admitted` is true *before* awaiting the returned
    # promise, pinning that the callback is admitted synchronously. That
    # distinction cannot exist here: `start_span` is `async def`, so its body
    # cannot run before the first `await`. (The guarantee it protects still
    # holds -- awaiting a coroutine that never suspends does not yield to the
    # event loop -- but there is no observation point to assert it from.)
    result = await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="first"), run)

    assert admitted is True
    assert result == 42
    # Upstream asserts `Object.isFrozen(firstSpan)`. The Python analogue of a
    # frozen object is one that rejects attribute writes, which `__slots__ = ()`
    # gives; the shared-inert-span half is asserted by identity.
    assert first_span is NOOP_TELEMETRY_CONTEXT
    with pytest.raises(AttributeError):
        first_span.injected = "state"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_noop_preserves_synchronous_and_asynchronous_rejection_values():
    sync_error = RuntimeError("sync")

    def raise_sync(_span: TelemetrySpan) -> NoReturn:
        raise sync_error

    with pytest.raises(RuntimeError) as caught_sync:
        await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="sync"), raise_sync)
    assert caught_sync.value is sync_error

    async_error = RuntimeError("async")

    async def raise_async(_span: TelemetrySpan) -> NoReturn:
        raise async_error

    with pytest.raises(RuntimeError) as caught_async:
        await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="async"), raise_async)
    assert caught_async.value is async_error


@pytest.mark.asyncio
async def test_noop_does_not_inspect_or_retain_telemetry_payloads():
    options = _UnreadableSpanOptions.__new__(_UnreadableSpanOptions)

    def run(span: TelemetrySpan) -> None:
        attributes = _Unreadable()
        status = _Unreadable()
        span.add_event("event", attributes)
        span.set_attributes(attributes)
        span.set_status(status)  # type: ignore[arg-type]

    await NOOP_TELEMETRY_CONTEXT.start_span(options, run)


def test_the_unreadable_fixtures_really_do_raise_on_every_read():
    options = _UnreadableSpanOptions.__new__(_UnreadableSpanOptions)
    with pytest.raises(RuntimeError):
        _ = options.name
    with pytest.raises(RuntimeError):
        _ = _Unreadable()["anything"]
    with pytest.raises(RuntimeError):
        list(_Unreadable())
