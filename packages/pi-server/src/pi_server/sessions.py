"""Live session lifecycle management.

Python port of `packages/server/src/sessions.ts`. `LiveSessionManager` owns
the map of currently-live session runtimes, session-acquisition dedup while
opening, per-connection attach/detach bookkeeping, and idle disposal.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_ai.utils.tasks import spawn

from .connection import ByteConnection, ConnectionState
from .errors import PiServerError
from .types import (
    CreateSessionOptions,
    PiServerService,
    PiSessionRuntime,
    PiSessionRuntimeEvent,
    PromptInput,
    SteerInput,
)


def _to_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "id": snapshot["id"],
        "createdAt": snapshot["createdAt"],
        "cwd": snapshot["cwd"],
    }
    if "updatedAt" in snapshot:
        metadata["updatedAt"] = snapshot["updatedAt"]
    if "name" in snapshot:
        metadata["sessionName"] = snapshot["name"]
    return metadata


@dataclass
class _LiveSession:
    id: str
    runtime: PiSessionRuntime
    connections: set[ConnectionState] = field(default_factory=set)
    unsubscribe: Callable[[], None] = lambda: None
    operation_count: int = 0
    ready: bool = False
    terminal: bool = False
    disposing: Any = None  # asyncio.Task[None] | None


class LiveSessionManager:
    def __init__(
        self,
        service: PiServerService,
        is_closing: Callable[[], bool],
        send_message: Callable[[ConnectionState, dict[str, Any]], Awaitable[bool]],
        close_connection: Callable[[ByteConnection], Awaitable[None]],
        disconnect: Callable[[ConnectionState], Awaitable[None]],
        broadcast_server_snapshot: Callable[[], None],
        report_error: Callable[[BaseException], None],
    ) -> None:
        self._service = service
        self._is_closing = is_closing
        self._send_message = send_message
        self._close_connection = close_connection
        self._disconnect = disconnect
        self._broadcast_server_snapshot = broadcast_server_snapshot
        self._report_error = report_error
        self._live_sessions: dict[str, _LiveSession] = {}
        self._opening_sessions: dict[str, Any] = {}  # id -> asyncio.Task[_LiveSession]

    async def execute_command(self, connection: ConnectionState, command: dict[str, Any]) -> dict[str, Any]:
        kind = command["command"]
        if kind == "list":
            return {"command": "list", "sessions": await self.list_metadata()}
        if kind == "create":
            session_id = str(uuid.uuid4())
            options = CreateSessionOptions(
                id=session_id,
                cwd=command.get("cwd"),
                name=command.get("name"),
                model=command.get("model"),
                thinking_level=command.get("thinkingLevel"),
            )
            live = await self._acquire(session_id, lambda: self._service.create_session(options))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._broadcast_server_snapshot()
            return {"command": "create", "session": session}
        if kind == "attach":
            session_id = command["sessionId"]
            live = await self._acquire(session_id, lambda: self._service.open_session(session_id))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._broadcast_server_snapshot()
            return {"command": "attach", "session": session}
        if kind == "detach":
            session_id = command["sessionId"]
            live = self._live_sessions.get(session_id)
            if session_id in connection.session_ids:
                connection.session_ids.discard(session_id)
                if live is not None:
                    live.connections.discard(connection)
                    if live.connections and not live.terminal and live.disposing is None:
                        await self._broadcast_snapshot(live)
                    await self._maybe_dispose(live)
                self._broadcast_server_snapshot()
            return {"command": "detach", "sessionId": session_id}
        if kind == "prompt":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.prompt(PromptInput(text=command["text"]))
            )
            return {"command": "prompt", "session": session}
        if kind == "steer":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.steer(SteerInput(text=command["text"]))
            )
            return {"command": "steer", "session": session}
        if kind == "abort":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(connection, live, lambda: live.runtime.abort())
            return {"command": "abort", "session": session}
        if kind == "set_model":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(connection, live, lambda: live.runtime.set_model(command["model"]))
            return {"command": "set_model", "session": session}
        if kind == "set_thinking":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.set_thinking(command["thinkingLevel"])
            )
            return {"command": "set_thinking", "session": session}
        raise PiServerError("invalid_request", f"Unknown command: {kind}")

    async def disconnect(self, connection: ConnectionState) -> None:
        sessions = [self._live_sessions[i] for i in connection.session_ids if i in self._live_sessions]
        connection.session_ids.clear()
        for live in sessions:
            live.connections.discard(connection)
        for live in sessions:
            try:
                await self._maybe_dispose(live)
            except Exception as error:
                self._report_error(error)

    async def list_metadata(self) -> list[dict[str, Any]]:
        stored = await self._service.list_sessions()
        live_by_id: dict[str, dict[str, Any]] = {}
        for live in self._live_sessions.values():
            if live.disposing is not None:
                continue
            live_by_id[live.id] = await self._normalized_snapshot(live)
        metadata: list[dict[str, Any]] = []
        for item in stored:
            snapshot = live_by_id.pop(item["id"], None)
            metadata.append(item if snapshot is None else {**item, **_to_metadata(snapshot)})
        for snapshot in live_by_id.values():
            metadata.append(_to_metadata(snapshot))
        return metadata

    async def close(self) -> None:
        for task in list(self._opening_sessions.values()):
            try:
                await task
            except Exception as error:
                self._report_error(error)
        sessions = list(self._live_sessions.values())
        self._live_sessions.clear()

        async def _dispose(live: _LiveSession) -> None:
            if live.disposing is not None:
                await live.disposing
                return
            live.unsubscribe()
            await live.runtime.dispose()

        for live in sessions:
            await _dispose(live)

    async def _run_operation(
        self, connection: ConnectionState, live: _LiveSession, operation: Callable[[], Awaitable[None]]
    ) -> dict[str, Any]:
        live.operation_count += 1
        try:
            await operation()
            return self._for_connection(await self._broadcast_snapshot(live), connection)
        finally:
            live.operation_count -= 1
            self._schedule_maybe_dispose(live)

    async def _acquire(
        self, session_id: str, acquire_runtime: Callable[[], Awaitable[PiSessionRuntime]]
    ) -> _LiveSession:
        while True:
            existing = self._live_sessions.get(session_id)
            if existing is not None:
                if existing.terminal:
                    raise PiServerError("session_locked", f"Session runtime is terminating: {session_id}")
                if existing.disposing is not None:
                    await existing.disposing
                    continue
                return existing
            opening = self._opening_sessions.get(session_id)
            if opening is not None:
                return await opening
            pending = spawn(self._create(session_id, acquire_runtime))
            self._opening_sessions[session_id] = pending
            try:
                return await pending
            finally:
                if self._opening_sessions.get(session_id) is pending:
                    del self._opening_sessions[session_id]

    async def _create(
        self, session_id: str, acquire_runtime: Callable[[], Awaitable[PiSessionRuntime]]
    ) -> _LiveSession:
        runtime = await acquire_runtime()
        if self._is_closing():
            await runtime.dispose()
            raise RuntimeError("PiServer closed while acquiring a session runtime")
        live: _LiveSession | None = None
        try:
            snapshot = await _await_maybe(runtime.snapshot())
            if snapshot["id"] != session_id:
                raise PiServerError(
                    "invalid_request",
                    f"Service returned session {snapshot['id']} for server-assigned session {session_id}",
                )
            live = _LiveSession(id=session_id, runtime=runtime)
            live.unsubscribe = runtime.subscribe(lambda event, live=live: self._handle_runtime_event(live, event))
            self._live_sessions[session_id] = live
            live.ready = True
            return live
        except Exception:
            if live is not None:
                live.unsubscribe()
            try:
                await runtime.dispose()
            except Exception as dispose_error:
                self._report_error(dispose_error)
            raise

    def _handle_runtime_event(self, live: _LiveSession, event: PiSessionRuntimeEvent) -> None:
        if event.type == "error":

            async def _terminate() -> None:
                try:
                    await self._terminate(live, event.error)
                except Exception as error:
                    self._report_error(error)

            spawn(_terminate())
            return
        if event.type == "progress":
            envelope = {
                "type": "event",
                "event": {"type": "session_progress", "sessionId": live.id, "progress": event.progress},
            }
            for connection in live.connections:
                spawn(self._send_message(connection, envelope))
        else:

            async def _broadcast() -> None:
                try:
                    await self._broadcast_snapshot(live)
                except Exception as error:
                    self._report_error(error)

            spawn(_broadcast())
        self._schedule_maybe_dispose(live)

    async def _terminate(self, live: _LiveSession, error: PiServerError) -> None:
        if live.terminal:
            return
        live.terminal = True
        self._report_error(error)
        live.unsubscribe()
        connections = list(live.connections)
        for connection in connections:
            await self._close_connection(connection.connection)
        for connection in connections:
            await self._disconnect(connection)
        await self._maybe_dispose(live)

    async def _normalized_snapshot(self, live: _LiveSession) -> dict[str, Any]:
        snapshot = await _await_maybe(live.runtime.snapshot())
        if snapshot["id"] != live.id:
            raise PiServerError("invalid_request", f"Runtime session ID changed from {live.id} to {snapshot['id']}")
        return {**snapshot, "phase": live.runtime.get_phase(), "attached": bool(live.connections), "locked": True}

    def _for_connection(self, snapshot: dict[str, Any], connection: ConnectionState) -> dict[str, Any]:
        return {**snapshot, "attached": snapshot["id"] in connection.session_ids}

    async def _broadcast_snapshot(self, live: _LiveSession) -> dict[str, Any]:
        snapshot = await self._normalized_snapshot(live)
        envelope = {"type": "event", "event": {"type": "session_snapshot", "snapshot": snapshot}}
        for connection in live.connections:
            spawn(self._send_message(connection, envelope))
        return snapshot

    async def _attach(self, connection: ConnectionState, live: _LiveSession) -> None:
        if connection.disconnected or connection.stage != "ready" or connection.connection.closed:
            await self._maybe_dispose(live)
            raise PiServerError("invalid_request", "Connection closed while attaching to a session")
        connection.session_ids.add(live.id)
        live.connections.add(connection)

    def _require_attached(self, connection: ConnectionState, session_id: str) -> _LiveSession:
        if session_id not in connection.session_ids:
            raise PiServerError("invalid_request", f"Connection is not attached to session {session_id}")
        live = self._live_sessions.get(session_id)
        if live is None or live.terminal or live.disposing is not None:
            raise PiServerError("not_found", f"Session is not live: {session_id}")
        return live

    def _schedule_maybe_dispose(self, live: _LiveSession) -> None:
        async def _run() -> None:
            try:
                await self._maybe_dispose(live)
            except Exception as error:
                self._report_error(error)

        spawn(_run())

    async def _maybe_dispose(self, live: _LiveSession) -> None:
        if (
            self._is_closing()
            or not live.ready
            or live.disposing is not None
            or live.connections
            or live.operation_count > 0
            or (not live.terminal and live.runtime.get_phase() != "idle")
        ):
            if live.disposing is not None:
                await live.disposing
            return
        live.unsubscribe()

        async def _dispose() -> None:
            try:
                await live.runtime.dispose()
            finally:
                if self._live_sessions.get(live.id) is live:
                    del self._live_sessions[live.id]

        live.disposing = spawn(_dispose())
        await live.disposing
        if not self._is_closing():
            self._broadcast_server_snapshot()


async def _await_maybe(value: Any) -> Any:
    """Awaits `value` if it is awaitable (TS `MaybePromise<T>`), else returns it directly."""
    if hasattr(value, "__await__"):
        return await value
    return value
