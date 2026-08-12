"""RPC client (Python port of ``@earendil-works/pi-client``).

Ports `packages/client/src/index.ts` and the modules it re-exports.

`packages/client/src/promise.ts` has no Python module. It exists only to
polyfill `Promise.withResolvers()` for a TypeScript lib baseline older than
ES2024 -- its own comment says to delete it once that baseline moves.
`asyncio.Future` is the built-in Python equivalent (`set_result` /
`set_exception` on a future obtained from `loop.create_future()`), and this
port uses it directly throughout `client.py` and `session_handle.py`.
"""

from __future__ import annotations

from .client import PiClient
from .errors import (
    PiClientDisposedError,
    PiDisconnectedError,
    PiServerError,
    PiSessionDetachedError,
    PiSessionOwnershipError,
)
from .session_handle import SessionHandle, SessionHandleCallbacks, SessionLeaseMode
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import (
    ConnectionState,
    ConnectionStateChange,
    CreateSessionOptions,
    ListenerErrorHandler,
    PiClientOptions,
    Unsubscribe,
)
from .unix import create_unix_transport_factory

__all__ = [
    "ByteTransport",
    "ByteTransportFactory",
    "ByteTransportHandlers",
    "ConnectionState",
    "ConnectionStateChange",
    "CreateSessionOptions",
    "ListenerErrorHandler",
    "PiClient",
    "PiClientDisposedError",
    "PiClientOptions",
    "PiDisconnectedError",
    "PiServerError",
    "PiSessionDetachedError",
    "PiSessionOwnershipError",
    "SessionHandle",
    "SessionHandleCallbacks",
    "SessionLeaseMode",
    "Unsubscribe",
    "create_unix_transport_factory",
]
