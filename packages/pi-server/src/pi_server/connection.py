"""Byte-connection abstraction.

Python port of `packages/server/src/connection.ts`. `ByteConnection` is an
established, authorized ordered byte connection; `ConnectionState` tracks one
connection's handshake/dispatch lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pi_protocol import ClientMessageDecoder


class ByteConnection(Protocol):
    """An established, authorized ordered byte connection."""

    @property
    def closed(self) -> bool: ...

    def send(self, chunk: bytes) -> Awaitable[None]: ...

    def close(self, final_chunk: bytes | None = None) -> Awaitable[None] | None: ...


class ByteConnectionHandler(Protocol):
    def on_data(self, chunk: bytes) -> None: ...
    def on_close(self) -> None: ...
    def on_error(self, error: Exception) -> None: ...


ByteConnectionAcceptor = Callable[[ByteConnection], ByteConnectionHandler]

ConnectionStage = Literal["awaitingHello", "handshaking", "ready", "closing", "closed"]


@dataclass(eq=False)
class ConnectionState:
    """`eq=False` keeps identity-based equality/hash so instances work as `Set`
    members the same way TS's reference-identity `Set<ConnectionState>` does."""

    id: str
    connection: ByteConnection
    decoder: ClientMessageDecoder
    handshake_timeout_handle: asyncio.TimerHandle
    session_ids: set[str] = field(default_factory=set)
    stage: ConnectionStage = "awaitingHello"
    disconnected: bool = False
    handshake_complete: bool = False
    handshake: asyncio.Task[None] | None = None


def is_terminal_connection(state: ConnectionState) -> bool:
    return state.disconnected or state.stage in ("closing", "closed")
