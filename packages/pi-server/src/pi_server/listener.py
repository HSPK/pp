"""Listener composition.

Python port of `packages/server/src/listener.ts`.
"""

from __future__ import annotations

from typing import Protocol

from .connection import ByteConnectionAcceptor


class PiServerListener(Protocol):
    """Supplies established byte connections after any required transport authentication."""

    @property
    def address(self) -> str | None:
        """Human-readable bound address after startup, when the transport has one."""
        ...

    async def start(self, accept: ByteConnectionAcceptor) -> None:
        """Starts listening and passes authorized connections to accept."""
        ...

    async def close(self) -> None: ...
