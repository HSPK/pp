"""Options for the Unix-domain-socket `PiServerListener`.

Python port of `packages/server/src/transports/unix/types.ts`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class UnixListenerOptions:
    path: str
    mode: int | None = None
    """Socket filesystem permissions. Defaults to owner read/write only (0o600)."""
    max_pending_bytes: int | None = None
    """Maximum framed bytes queued per connection before a slow peer is disconnected."""
    graceful_close_timeout_ms: int | None = None
    max_frame_length: int | None = None
    """Used to derive and validate max_pending_bytes. Must match the server when customized."""
    on_error: Callable[[Exception], None] | None = None


@dataclass
class UnixServerOptions:
    """`UnixListenerOptions` plus the `PiServerOptions` fields other than `listeners`."""

    path: str
    mode: int | None = None
    max_pending_bytes: int | None = None
    graceful_close_timeout_ms: int | None = None
    max_frame_length: int | None = None
    handshake_timeout_ms: int | None = None
    server_id: str | None = None
    on_error: Callable[[Exception], None] | None = None
