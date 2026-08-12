# pi-telemetry

Vendor-neutral telemetry contracts and schema utilities for pi packages.

This package provides:

- an explicit, callback-based `TelemetryContext` / `TelemetrySpan` contract;
- a shared `NOOP_TELEMETRY_CONTEXT`;
- a reference `InMemoryTelemetryContext` implementation;
- serializable schema definitions;
- no exporter, global current-span state, or dependency on a telemetry backend.

Applications can use the in-memory reference or provide an adapter for OpenTelemetry, Sentry, logs, or another backend. Pi packages pass telemetry contexts explicitly and define their domain schemas separately.

## Table of Contents

- [Installation](#installation)
- [Telemetry Concepts](#telemetry-concepts)
- [Core Context API](#core-context-api)
- [Adapter Contract](#adapter-contract)
- [No-op Context](#no-op-context)
- [In-Memory Reference Adapter](#in-memory-reference-adapter)
- [Adapter Conformance](#adapter-conformance)
- [Typed Schemas](#typed-schemas)
  - [Start and Completion Attributes](#start-and-completion-attributes)
- [Schema Metadata](#schema-metadata)
- [Pi Package Integration](#pi-package-integration)
- [Security and Portability](#security-and-portability)
- [API Reference](#api-reference)
- [Development](#development)
- [License](#license)

## Installation

```bash
pip install pi-telemetry
```

## Telemetry Concepts

Telemetry describes what a program did while it was running. This package models that work using spans, attributes, events, statuses, and explicit context:

| Concept | Plain-language meaning |
|---|---|
| **Span** | A timed record of one operation, such as loading an account or making an AI request. It begins before the work and ends when the work finishes. |
| **Parent and child spans** | Operations can contain smaller operations. A request span might contain a cache lookup and a database query. Together they form a tree showing where time was spent. |
| **Attribute** | A named fact attached to a span, such as `provider: "openai"`, `cache.hit: true`, or `item_count: 12`. Attributes describe the operation and its result. |
| **Event** | A named occurrence at a point during a span, such as `retry.scheduled` or `cache.lookup`. Events have no duration and may carry their own attributes. |
| **Status** | The operation's outcome: `ok` or `error`. An error status may include an error name and message. |
| **Context** | A handle identifying where new work belongs in the span tree. Starting a span from a context makes it a child of that context. |

For example, loading an account could produce this telemetry:

```text
example.account.load                         span
├─ attributes: account.id=123, found=true   facts about the span
├─ event: example.cache.lookup              occurrence during the span
│  └─ attribute: cache.hit=false            fact about the event
└─ status: ok                               final outcome
```

A span is diagnostic data, not business state. Recording it must not change whether the account load runs, succeeds, fails, or is persisted. An adapter translates these generic concepts into the corresponding concepts used by OpenTelemetry, Sentry, logs, or another backend.

## Core Context API

A `TelemetryContext` starts a span around a callback. The callback receives a `TelemetrySpan`, which is also the explicit parent context for child spans. `start_span()` is async in Python.

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SpanOptions, TelemetryContext


async def read_account(account_id: str) -> dict[str, str] | None:
    return {"id": account_id}


async def load_account(
    account_id: str,
    telemetry_context: TelemetryContext = NOOP_TELEMETRY_CONTEXT,
):
    async def run(span):
        account = await read_account(account_id)
        span.set_attributes({"example.account.found": account is not None})
        return account

    return await telemetry_context.start_span(
        SpanOptions(
            name="example.account.load",
            attributes={"example.account.id": account_id},
        ),
        run,
    )
```

Pass the callback span to lower-level work to create explicit nesting:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SpanOptions


async def perform_work() -> str:
    return "done"


async def main() -> str:
    async def parent(parent_span):
        async def child(child_span):
            child_span.add_event("example.cache.lookup", {"example.cache.hit": True})
            return await perform_work()

        return await parent_span.start_span(SpanOptions(name="example.child"), child)

    return await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="example.parent"), parent)
```

There is no public `end()` method. `start_span()` owns settlement and keeps the span open until the callback's value or awaitable settles. For an expected failure represented by a normal return value, set the status explicitly:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SpanError, SpanOptions, SpanStatus


async def save() -> dict[str, str]:
    return {"ok": "false", "reason": "locked"}


async def main():
    async def run(span):
        result = await save()
        if result["ok"] != "true":
            span.set_status(SpanStatus("error", SpanError("SaveError", result["reason"])))
        return result

    return await NOOP_TELEMETRY_CONTEXT.start_span(SpanOptions(name="example.save"), run)
```

## Adapter Contract

An adapter implements `TelemetryContext` and bridges the generic API to its backend. It must:

- create a child span and invoke the callback exactly once after `start_span()` is awaited;
- preserve the callback's returned value and raised exception object;
- keep the native span open until a returned awaitable settles;
- treat normal completion as `ok` and raises as errors unless an explicit status was set;
- make repeated `set_status()` calls last-write-wins;
- merge `set_attributes()` calls, with later non-`None` values replacing earlier values and `None` ignored;
- make recording methods synchronous, passive, and non-throwing;
- ignore calls made after settlement;
- ignore a failed recording call atomically, suppress backend failures, and still execute the business callback exactly once.

Adapters may activate backend-native ambient context internally for automatic instrumentation, but pi code always propagates the parent through `TelemetryContext` arguments. Exporter buffering, flushing, sampling, backend IDs, and backend-specific context objects belong to the adapter. Use the [adapter conformance suite](#adapter-conformance) to check these observable semantics.

## No-op Context

Use `NOOP_TELEMETRY_CONTEXT` when telemetry is optional:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, SpanOptions


async def run_operation() -> str:
    return "ok"


async def main() -> str:
    return await NOOP_TELEMETRY_CONTEXT.start_span(
        SpanOptions(name="example.operation"),
        lambda span: run_operation(),
    )
```

The no-op context:

- invokes callbacks when the returned coroutine is awaited;
- preserves returned values and asynchronous exceptions;
- uses one shared inert span, including for nested spans;
- does not inspect or retain names, attributes, events, or statuses.

The TypeScript package freezes the shared no-op span. The Python port uses `__slots__` on `NoopTelemetrySpan` to prevent per-caller state from being attached.

## In-Memory Reference Adapter

`InMemoryTelemetryContext` is the backend-neutral reference implementation. It is useful for tests, local diagnostics, and applications that intentionally want process-local capture without an exporter:

```python
from pi_telemetry import InMemoryTelemetryContext, SpanOptions


async def main() -> None:
    telemetry = InMemoryTelemetryContext()

    async def run(span) -> None:
        span.add_event("example.started")
        span.set_attributes({"output_count": 3})

    await telemetry.start_span(
        SpanOptions(name="example.operation", attributes={"input": "demo"}),
        run,
    )

    print(telemetry.get_spans())
```

`get_spans()` returns detached snapshots in span-start order. Each `RecordedTelemetrySpan` contains a deterministic numeric ID, parent ID, merged attributes, ordered events, final status, settlement state, and deterministic end sequence. It records no timestamps.

The adapter is safe to use as an ordinary `TelemetryContext`, but storage is unbounded and process-local. Create a fresh instance to isolate tests or recording scopes, and do not capture sensitive attributes unless the caller's data policy allows them.

## Adapter Conformance

`pi_telemetry.testing` exports a runner-independent conformance suite modeled as grouped cases. A fixture supplies a fresh context and converts its backend's finished spans into normalized `RecordedTelemetrySpan` snapshots:

```python
import pytest

from pi_telemetry import InMemoryTelemetryContext
from pi_telemetry.testing import create_telemetry_adapter_conformance


class Fixture:
    def __init__(self) -> None:
        self.context = InMemoryTelemetryContext()

    async def get_spans(self):
        return self.context.get_spans()

    async def aclose(self) -> None:
        return None


async def make_fixture() -> Fixture:
    return Fixture()


conformance = create_telemetry_adapter_conformance(make_fixture)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", conformance, ids=lambda case: f"{case.group}: {case.name}")
async def test_telemetry_adapter(case) -> None:
    await case.run()
```

The suite checks single admission, result and exception identity, automatic and explicit status, attribute merging, event ordering, inert post-settlement calls, nested and concurrent parentage, and suppression of unreadable telemetry payload failures. Python has no analogue for JavaScript's `undefined` rejection value, and `async def start_span()` cannot admit the callback before the coroutine is awaited.

## Typed Schemas

The low-level span API intentionally accepts open names and attribute bags so adapters remain generic. Domain packages can define closed, serializable schemas. In TypeScript those schemas drive compile-time inference. In Python they are plain dictionaries; `define_telemetry_schema()` returns the same object and `create_typed_span_starter()` does not validate against the schema at runtime, matching upstream's runtime behavior.

```python
from pi_telemetry import (
    NOOP_TELEMETRY_CONTEXT,
    create_typed_span_starter,
    define_telemetry_schema,
)


EXAMPLE_TELEMETRY_SCHEMA = define_telemetry_schema(
    {
        "version": 1,
        "spans": {
            "example.read": {
                "description": "Read one resource",
                "parents": {"kind": "any"},
                "startAttributes": {
                    "example.resource": {
                        "type": "string",
                        "required": True,
                        "values": ["account", "project"],
                        "description": "Resource kind",
                    },
                },
                "endAttributes": {
                    "example.item_count": {
                        "type": "number",
                        "description": "Number of returned items",
                    },
                },
                "events": {
                    "example.cache": {
                        "description": "Cache lookup result",
                        "attributes": {
                            "example.cache.hit": {
                                "type": "boolean",
                                "required": True,
                                "description": "Whether the cache contained the resource",
                            },
                        },
                    },
                },
                "status": {
                    "default": "ok",
                    "errorWhen": "The read throws or returns an error result",
                },
            },
        },
    }
)

start_span = create_typed_span_starter(
    NOOP_TELEMETRY_CONTEXT,
    [EXAMPLE_TELEMETRY_SCHEMA],
)
```

The starter passes the span name and attributes to the parent context and passes a child starter into the callback:

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, create_typed_span_starter


async def read_accounts() -> list[str]:
    return ["a"]


async def read_projects() -> list[str]:
    return ["p"]


async def main():
    start_span = create_typed_span_starter(NOOP_TELEMETRY_CONTEXT, [])

    async def run(span, start_child_span):
        span.add_event("example.cache", {"example.cache.hit": True})
        accounts = await read_accounts()
        span.set_attributes({"example.item_count": len(accounts)})

        async def child(child_span, _start_grandchild):
            projects = await read_projects()
            child_span.set_attributes({"example.item_count": len(projects)})

        await start_child_span("example.read", {"example.resource": "project"}, child)
        return accounts

    return await start_span("example.read", {"example.resource": "account"}, run)
```

### Start and Completion Attributes

`startAttributes` and `endAttributes` describe when an attribute is normally known, not separate runtime storage:

| Schema field | How values are recorded | Requiredness |
|---|---|---|
| `startAttributes` | Passed in the starter's `attributes` argument when the span is created | Each definition explicitly sets `required: true` or `false` |
| `endAttributes` | Added later through the span's `set_attributes()` method | Always optional |

Both sets become ordinary attributes on the same backend span. There is no separate end-attribute payload or end callback.

```python
from pi_telemetry import NOOP_TELEMETRY_CONTEXT, create_typed_span_starter


async def read_accounts() -> list[str]:
    return ["a", "b"]


async def main():
    start_span = create_typed_span_starter(NOOP_TELEMETRY_CONTEXT, [])

    async def run(span, _start_child_span):
        accounts = await read_accounts()
        span.set_attributes({"example.item_count": len(accounts)})
        return accounts

    return await start_span("example.read", {"example.resource": "account"}, run)
```

“End” means completion enrichment: an end attribute may be set at any point while the callback is active, and it may be omitted when unavailable. Returning, resolving, raising, or cancelling controls settlement; `start_span()` performs the actual end operation. Adapter calls made after settlement are inert.

Schema dictionaries are not merged, inspected, or retained at runtime. Their TypeScript key spelling (`startAttributes`, `endAttributes`, `errorWhen`) is preserved because schemas are serializable documentation data.

## Schema Metadata

Supported attribute types are:

- `string`, `number`, and `boolean`;
- `string[]`, `number[]`, and `boolean[]`.

Attribute definitions support:

- `values`: a closed set for scalar values;
- `elementValues`: a closed set for array elements;
- `examples`: documentation examples;
- `sensitive`: marks data requiring special handling;
- `cardinality`: records expected `low` or `high` cardinality.

Start and event attributes declare `required`. End attributes do not; see [Start and Completion Attributes](#start-and-completion-attributes).

Parent metadata is descriptive schema data:

- `{"kind": "any"}`: root or any caller span;
- `{"kind": "root_or_external"}`: root or a caller-owned span outside the schema;
- `{"kind": "spans", "spans": [...]}`: only the listed schema spans.

Adapters do not need to understand schema objects. Instrumentation helpers and tests use them to keep emitted names and attributes consistent.

## Pi Package Integration

Package ownership is intentionally split:

- `pi_telemetry` owns the vendor-neutral contract, no-op and in-memory reference contexts, schema utilities, and adapter conformance suite;
- `pi_ai` accepts and propagates telemetry contexts in provider request options but owns no telemetry schema;
- the Python port does not currently expose the TypeScript `AGENT_TELEMETRY_SCHEMAS`, `AI_TELEMETRY_SCHEMA`, `HARNESS_TELEMETRY_SCHEMA`, `startAiSpan`, or `startHarnessSpan` exports from `pi_agent`.

The pi schemas use pi-owned `pi.ai.*`, `pi.harness.*`, and `pi.session.*` names when present. Adapters may translate them to backend conventions without changing the emitted pi vocabulary.

## Security and Portability

Telemetry is process-local diagnostics, not durable application state. Do not persist a `TelemetryContext`, `TelemetrySpan`, or backend-native trace object in records, messages, snapshots, or deferred handles.

Attribute values are intentionally limited to primitive scalars and arrays by schema convention. Domain instrumentation should avoid prompts, completions, tool arguments or output, file contents, provider payloads, headers, credentials, and free-form error details unless its schema and data policy explicitly allow them.

The package does not use `contextvars` or another ambient current-span mechanism. Backend adapters remain responsible for their own runtime compatibility.

## API Reference

### Core types and values

| Export | Purpose |
|---|---|
| `TelemetryContext` | Protocol for starting callback-managed child spans |
| `TelemetrySpan` | Protocol for recording attributes, events, and status; also acts as a child context |
| `SpanOptions` | Span name and optional start attributes |
| `SpanAttributes` / `AttributeValue` | Open adapter-level attribute bag and supported values |
| `SpanStatus` / `SpanError` | Explicit `ok` or `error` status |
| `NOOP_TELEMETRY_CONTEXT` | Shared passive context for disabled telemetry |
| `NoopTelemetrySpan` | Inert span implementation used by the no-op context |
| `InMemoryTelemetryContext` | Reference adapter with deterministic process-local recording |
| `RecordedTelemetrySpan` | Normalized captured span snapshot |
| `RecordedTelemetryEvent` | Normalized captured event snapshot |

### Schema definitions

| Export | Purpose |
|---|---|
| `define_telemetry_schema()` | Identity helper for serializable schema data |
| `create_typed_span_starter()` | Binds a parent context to one or more schema vocabularies |
| `SpanStarter` | Async starter callable over `(name, attributes, callback)` |
| `TelemetrySchemaDefinition` | Top-level schema shape |
| `TelemetrySpanDefinition` | Span metadata, parents, attributes, events, and status rule |
| `TelemetryAttributeType` | Supported scalar and array type names |
| `TelemetryAttributeDefinition` | Attribute type, allowed values, examples, and metadata |
| `TelemetryEventDefinition` | Event description and attribute definitions |
| `TelemetryParentDefinition` | Open, external-root, or finite schema-parent rule |

The TypeScript-only compile-time inference exports (`TypedSpanStarter`, `InferStartAttributes`, `ExactTelemetryAttributes`, and related conditional types) have no Python runtime equivalent.

### Testing subpath

| Export | Purpose |
|---|---|
| `create_telemetry_adapter_conformance()` | Creates runner-independent adapter conformance cases |
| `TelemetryAdapterFixture` | Fresh context and normalized snapshot reader for one case |
| `TelemetryAdapterFixtureFactory` | Creates isolated fixtures |
| `TelemetryAdapterConformanceCase` | Grouped case that test runners execute |

## Development

From the repository root:

```bash
uv sync --all-packages
uv run pytest packages/pi-telemetry
uv run ruff check packages/pi-telemetry
```

## License

MIT
