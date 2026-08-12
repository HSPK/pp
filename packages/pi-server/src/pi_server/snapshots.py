"""Server snapshot revisions and broadcast serialization.

Python port of `packages/server/src/snapshots.ts`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from pi_ai.utils.tasks import spawn
from pi_protocol import PROTOCOL_VERSION

from .connection import ConnectionState
from .types import PiServerService


class ServerSnapshotPublisher:
    def __init__(
        self,
        server_id: str,
        service: PiServerService,
        connections: set[ConnectionState],
        is_closing: Callable[[], bool],
        list_sessions: Callable[[], Awaitable[list[dict[str, Any]]]],
        send_message: Callable[[ConnectionState, dict[str, Any]], Awaitable[bool]],
        report_error: Callable[[BaseException], None],
    ) -> None:
        self._server_id = server_id
        self._service = service
        self._connections = connections
        self._is_closing = is_closing
        self._list_sessions = list_sessions
        self._send_message = send_message
        self._report_error = report_error
        self._revision = 0
        # Chains broadcasts serially, mirroring the TS `broadcastQueue: Promise<void>`.
        self._broadcast_queue: Any = None

    @property
    def current_revision(self) -> int:
        return self._revision

    async def get(self, models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "serverId": self._server_id,
            "protocolVersion": PROTOCOL_VERSION,
            "revision": self._revision,
            "sessions": await self._list_sessions(),
            "models": models if models is not None else await self._service.list_models(),
        }

    def broadcast(self) -> Awaitable[None]:
        previous = self._broadcast_queue

        async def _run() -> None:
            if previous is not None:
                with contextlib.suppress(Exception):
                    await previous
            try:
                await self._perform_broadcast()
            except Exception as error:
                self._report_error(error)

        task = spawn(_run())
        self._broadcast_queue = task
        return task

    async def _perform_broadcast(self) -> None:
        ready_connections = [c for c in self._connections if c.stage == "ready" and not c.disconnected]
        if not ready_connections or self._is_closing():
            return
        self._revision += 1
        revision = self._revision
        models = await self._service.list_models()
        current = await self.get(models)
        snapshot = {**current, "revision": revision}
        envelope = {"type": "event", "event": {"type": "server_snapshot", "snapshot": snapshot}}
        for connection in ready_connections:
            await self._send_message(connection, envelope)
