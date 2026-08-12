"""Shared client option and state types.

Python port of `packages/client/src/types.ts`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .transport import ByteTransportFactory

ConnectionState = Literal["disconnected", "connecting", "connected"]


@dataclass
class ConnectionStateChange:
    state: ConnectionState
    error: Exception | None = None


Unsubscribe = Callable[[], None]
ListenerErrorHandler = Callable[[Exception], None]


@dataclass
class PiClientOptions:
    transport_factory: ByteTransportFactory
    max_frame_length: int | None = None
    """Reports subscriber failures without allowing them to corrupt client state."""
    on_listener_error: ListenerErrorHandler | None = None


@dataclass
class CreateSessionOptions:
    cwd: str | None = None
    name: str | None = None
    model: dict[str, Any] | None = None
    thinking_level: str | None = None
