"""Byte transport abstraction.

Python port of `packages/client/src/transport.ts`. `ByteTransport` is a duplex
byte stream abstraction so tests can substitute an in-memory transport, and
`unix.py` provides the real Unix-domain-socket implementation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol


class ByteTransport(Protocol):
    """One connected, authenticated duplex byte stream."""

    def send(self, chunk: bytes) -> Awaitable[None]:
        """Sends one byte chunk. Calls must be delivered in invocation order.

        Implementations must validate/reserve synchronously (before returning)
        so that concurrent, unawaited calls are ordered and bounds-checked the
        same way a JS `Promise`-returning function would be: its body runs up
        to the first `await` immediately, not once some later `await` resumes
        it. The returned awaitable then resolves once the chunk is flushed.
        """
        ...

    def close(self) -> None:
        """Closes the transport. Repeated calls must be harmless."""
        ...


class ByteTransportHandlers(Protocol):
    """Callbacks a transport uses to report inbound activity."""

    def on_data(self, chunk: bytes) -> None:
        """Delivers an arbitrary inbound byte chunk."""
        ...

    def on_close(self) -> None:
        """Reports an orderly terminal close."""
        ...

    def on_error(self, error: Exception) -> None:
        """Reports a terminal transport failure."""
        ...


ByteTransportFactory = Callable[[ByteTransportHandlers], ByteTransport | Awaitable[ByteTransport]]
"""Creates a fresh connected, authenticated transport. Exactly one terminal handler is expected."""
