"""Types for the runner-independent telemetry adapter conformance suite.

Python port of `packages/telemetry/src/testing/types.ts`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..memory import RecordedTelemetrySpan
from ..types import TelemetryContext


class TelemetryAdapterFixture(Protocol):
    """A fresh adapter instance and normalized snapshot reader owned by one case.

    Upstream this is an `AsyncDisposable`; the Python analogue is an explicit
    `aclose()` the case runner always awaits in a `finally`.
    """

    @property
    def context(self) -> TelemetryContext: ...

    async def get_spans(self) -> Sequence[RecordedTelemetrySpan]: ...

    async def aclose(self) -> None: ...


TelemetryAdapterFixtureFactory = Callable[[], Awaitable[TelemetryAdapterFixture]]
"""Creates an isolated adapter fixture for one conformance case."""


@dataclass(frozen=True)
class TelemetryAdapterConformanceCase:
    """A runner-independent conformance case that can be registered with any test framework."""

    group: str
    name: str
    run: Callable[[], Awaitable[None]]
