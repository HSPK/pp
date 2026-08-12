"""Telemetry schema definitions and the typed span starters built on them.

Python port of the schema half of `packages/telemetry/src/index.ts`.

TypeScript encodes a schema twice: once as data and once as a large set of
conditional types, so that `startSpan("operation", { kind: "read" }, ...)`
rejects an unknown span name, a missing required attribute, or a value outside
a closed set *at compile time*. Python has no equivalent, so this port keeps
the data and drops the type layer: schemas are plain dictionaries whose
TypeScript key spelling (`startAttributes`, `endAttributes`, `errorWhen`, ...)
is preserved, because they are serialized as-is into generated documentation.

Upstream states outright that "schema values are used only for type inference;
no runtime schema validation is performed", so `create_typed_span_starter`
ignores its `schemas` argument at runtime exactly as `createTypedSpanStarter`
does.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from .types import SpanAttributes, SpanOptions, TelemetryContext, TelemetrySpan

TelemetryAttributeType = Literal["string", "number", "boolean", "string[]", "number[]", "boolean[]"]

TelemetryAttributeDefinition = dict[str, Any]
"""`{"type": TelemetryAttributeType, "description": str, ...}`."""

TelemetryEventDefinition = dict[str, Any]
"""`{"description": str, "attributes": {name: attribute definition}}`."""

TelemetryParentDefinition = dict[str, Any]
"""`{"kind": "any" | "root_or_external" | "spans", "spans"?: [str]}`."""

TelemetrySpanDefinition = dict[str, Any]
"""One span: description, parents, start/end attributes, events, status."""

TelemetrySchemaDefinition = dict[str, Any]
"""One telemetry schema: `{"version": int, "spans": {name: span definition}}`."""

SpanStarter = Callable[
    [str, SpanAttributes, Callable[[TelemetrySpan, "SpanStarter"], Any]],
    Awaitable[Any],
]
"""`async (name, attributes, callback) -> result`, the port of `TypedSpanStarter`."""


def define_telemetry_schema(schema: TelemetrySchemaDefinition) -> TelemetrySchemaDefinition:
    """Identity helper for serializable telemetry schema data.

    Upstream exists purely to attach a `const` type parameter; it returns the
    very object it was given, and this port does the same so that callers can
    rely on identity.
    """
    return schema


def _bind_typed_span_starter(telemetry_context: TelemetryContext) -> SpanStarter:
    async def start_span(
        name: str,
        attributes: SpanAttributes,
        callback: Callable[[TelemetrySpan, SpanStarter], Any],
    ) -> Any:
        async def run(span: TelemetrySpan) -> Any:
            result = callback(span, _bind_typed_span_starter(span))
            if inspect.isawaitable(result):
                return await result
            return result

        return await telemetry_context.start_span(SpanOptions(name=name, attributes=attributes), run)

    return start_span


def create_typed_span_starter(
    telemetry_context: TelemetryContext,
    schemas: Sequence[TelemetrySchemaDefinition],
) -> SpanStarter:
    """Bind an explicit parent context to the combined span vocabulary of `schemas`.

    Schema values are used only for documentation and type inference; no runtime
    schema validation is performed, and `schemas` is never read here.
    """
    return _bind_typed_span_starter(telemetry_context)


__all__ = [
    "SpanStarter",
    "TelemetryAttributeDefinition",
    "TelemetryAttributeType",
    "TelemetryEventDefinition",
    "TelemetryParentDefinition",
    "TelemetrySchemaDefinition",
    "TelemetrySpanDefinition",
    "create_typed_span_starter",
    "define_telemetry_schema",
]
