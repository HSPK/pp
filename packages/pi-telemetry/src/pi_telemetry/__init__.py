"""Telemetry primitives (Python port of ``@earendil-works/pi-telemetry``)."""

from __future__ import annotations

from .memory import (
    InMemoryTelemetryContext,
    RecordedTelemetryEvent,
    RecordedTelemetrySpan,
)
from .noop import NOOP_TELEMETRY_CONTEXT, NoopTelemetrySpan
from .schema import (
    SpanStarter,
    TelemetryAttributeDefinition,
    TelemetryAttributeType,
    TelemetryEventDefinition,
    TelemetryParentDefinition,
    TelemetrySchemaDefinition,
    TelemetrySpanDefinition,
    create_typed_span_starter,
    define_telemetry_schema,
)
from .types import (
    AttributeValue,
    SpanAttributes,
    SpanError,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)

__all__ = [
    "NOOP_TELEMETRY_CONTEXT",
    "AttributeValue",
    "InMemoryTelemetryContext",
    "NoopTelemetrySpan",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "SpanAttributes",
    "SpanError",
    "SpanOptions",
    "SpanStarter",
    "SpanStatus",
    "TelemetryAttributeDefinition",
    "TelemetryAttributeType",
    "TelemetryContext",
    "TelemetryEventDefinition",
    "TelemetryParentDefinition",
    "TelemetrySchemaDefinition",
    "TelemetrySpan",
    "TelemetrySpanDefinition",
    "create_typed_span_starter",
    "define_telemetry_schema",
]
