"""The RPC server orchestrator.

Python port of `packages/server/src/server.ts`. `PiServer` accepts byte
connections from configured listeners, performs the hello handshake, dispatches
requests to `LiveSessionManager`, and fans out server/session snapshot events.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable
from typing import Any

from pi_ai.utils.tasks import spawn
from pi_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    ProtocolValidationError,
    encode_server_message,
    is_supported_protocol_version,
)

from .connection import ByteConnection, ByteConnectionHandler, ConnectionState, is_terminal_connection
from .errors import INTERNAL_SERVER_ERROR_MESSAGE, NOT_IMPLEMENTED_MESSAGE, InternalServerError, PiServerError
from .listener import PiServerListener
from .sessions import LiveSessionManager
from .snapshots import ServerSnapshotPublisher
from .types import PiServerOptions, PiServerService

_DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000
_MAX_UINT32 = 0xFFFF_FFFF
_MAX_TIMER_DELAY_MS = 2_147_483_647


class PiServer:
    def __init__(self, service: PiServerService, options: PiServerOptions) -> None:
        resolved = _resolve_options(options)
        self.id = options.server_id if options.server_id is not None else str(uuid.uuid4())
        self._listeners: list[PiServerListener] = options.listeners
        self._max_frame_length = resolved[0]
        self._handshake_timeout_ms = resolved[1]
        self._on_error = options.on_error
        self._connections: set[ConnectionState] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._start_task: asyncio.Task[PiServer] | None = None
        self._started = False

        self._sessions = LiveSessionManager(
            service=service,
            is_closing=lambda: self._closing,
            send_message=self._send_message,
            close_connection=self._close_connection,
            disconnect=self._disconnect,
            broadcast_server_snapshot=lambda: self._snapshots.broadcast(),
            report_error=self._report_error,
        )
        self._snapshots = ServerSnapshotPublisher(
            server_id=self.id,
            service=service,
            connections=self._connections,
            is_closing=lambda: self._closing,
            list_sessions=self._sessions.list_metadata,
            send_message=self._send_message,
            report_error=self._report_error,
        )

    @property
    def addresses(self) -> list[str]:
        return [listener.address for listener in self._listeners if listener.address is not None]

    def start(self) -> Awaitable[PiServer]:
        loop = asyncio.get_event_loop()
        if self._started:
            future: asyncio.Future[PiServer] = loop.create_future()
            future.set_exception(RuntimeError("PiServer is already started"))
            return future
        if self._start_task is not None:
            future = loop.create_future()
            future.set_exception(RuntimeError("PiServer is already starting"))
            return future
        if self._closing:
            future = loop.create_future()
            future.set_exception(RuntimeError("PiServer is closing or closed"))
            return future
        self._start_task = asyncio.ensure_future(self._start_internal())
        return self._start_task

    async def _start_internal(self) -> PiServer:
        started: list[PiServerListener] = []
        try:
            for listener in self._listeners:
                await listener.start(self.accept)
                started.append(listener)
            self._started = True
            return self
        except Exception:
            self._closing = True
            await asyncio.gather(*(listener.close() for listener in started), return_exceptions=True)
            await self._close_server_state()
            raise
        finally:
            self._start_task = None

    def accept(self, connection: ByteConnection) -> ByteConnectionHandler:
        if self._closing:
            spawn(self._close_connection(connection))
            return _NullHandler(self._report_error)

        state = ConnectionState(
            id=str(uuid.uuid4()),
            connection=connection,
            decoder=ClientMessageDecoder(self._max_frame_length),
            handshake_timeout_handle=None,  # type: ignore[arg-type]
        )

        def _on_handshake_timeout() -> None:
            spawn(self._fail_protocol(state, {"code": "invalid_request", "message": "Handshake timeout"}))

        loop = asyncio.get_event_loop()
        state.handshake_timeout_handle = loop.call_later(self._handshake_timeout_ms / 1000, _on_handshake_timeout)
        self._connections.add(state)
        return _ConnectionHandler(self, state)

    async def close(self) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._closing = True
        self._close_task = asyncio.ensure_future(self._close_internal())
        await self._close_task

    async def _close_internal(self) -> None:
        starting = self._start_task
        if starting is not None:
            with contextlib.suppress(Exception):
                await starting
        try:
            await asyncio.gather(*(listener.close() for listener in self._listeners))
        finally:
            await self._close_server_state()
            self._started = False

    def _receive(self, state: ConnectionState, chunk: bytes) -> None:
        if is_terminal_connection(state):
            return
        try:
            messages = state.decoder.push(chunk)
        except Exception as error:
            spawn(self._fail_protocol(state, self._to_protocol_error(error)))
            return
        for message in messages:
            if is_terminal_connection(state):
                return
            self._dispatch_message(state, message)

    def _dispatch_message(self, state: ConnectionState, message: dict[str, Any]) -> None:
        if state.stage == "awaitingHello":
            if message["type"] != "hello":
                spawn(
                    self._fail_protocol(
                        state, {"code": "invalid_request", "message": "The first client message must be hello"}
                    )
                )
                return
            state.stage = "handshaking"

            async def _run_handshake() -> None:
                try:
                    await self._finish_handshake(state, message)
                except Exception as error:
                    await self._fail_protocol(state, self._to_protocol_error(error))

            state.handshake = asyncio.ensure_future(_run_handshake())
            return

        if message["type"] == "hello":
            spawn(
                self._fail_protocol(
                    state, {"code": "invalid_request", "message": "hello may only be sent as the first message"}
                )
            )
            return

        if state.stage == "ready":
            spawn(self._handle_request(state, message))
            return
        if state.stage != "handshaking":
            return
        handshake = state.handshake
        if handshake is None:
            return

        async def _after_handshake() -> None:
            await handshake
            if state.stage == "ready" and not state.disconnected:
                await self._handle_request(state, message)

        spawn(_after_handshake())

    async def _finish_handshake(self, state: ConnectionState, hello: dict[str, Any]) -> None:
        if not is_supported_protocol_version(hello["version"]):
            await self._fail_protocol(
                state,
                {
                    "code": "version",
                    "message": f"Unsupported protocol version {hello['version']}; expected {PROTOCOL_VERSION}",
                },
            )
            return

        snapshot = await self._snapshots.get()
        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            return
        sent = await self._send_message(
            state, {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": state.id, "snapshot": snapshot}
        )
        if sent and not state.disconnected and state.stage == "handshaking":
            state.handshake_complete = True
            state.stage = "ready"
            state.handshake_timeout_handle.cancel()
            if snapshot["revision"] != self._snapshots.current_revision:
                current = await self._snapshots.get()
                await self._send_message(
                    state, {"type": "event", "event": {"type": "server_snapshot", "snapshot": current}}
                )

    async def _handle_request(self, state: ConnectionState, envelope: dict[str, Any]) -> None:
        try:
            result = await self._sessions.execute_command(state, envelope["request"])
            await self._send_message(state, {"type": "response", "id": envelope["id"], "ok": True, "result": result})
        except Exception as error:
            await self._send_message(
                state, {"type": "response", "id": envelope["id"], "ok": False, "error": self._to_protocol_error(error)}
            )

    def _transport_closed(self, connection: ConnectionState) -> None:
        if not connection.disconnected and connection.stage != "closing":
            try:
                connection.decoder.end()
            except Exception as error:
                self._report_error(error)
        spawn(self._disconnect(connection))

    async def _disconnect(self, connection: ConnectionState) -> None:
        if connection.disconnected:
            return
        handshake_complete = connection.handshake_complete
        connection.disconnected = True
        connection.stage = "closed"
        connection.handshake_timeout_handle.cancel()
        self._connections.discard(connection)
        await self._sessions.disconnect(connection)
        if not self._closing and handshake_complete:
            self._snapshots.broadcast()

    async def _send_message(self, connection: ConnectionState, message: dict[str, Any]) -> bool:
        if connection.disconnected or connection.connection.closed:
            return False
        try:
            frame = encode_server_message(message, max_frame_length=self._max_frame_length)
        except Exception as error:
            self._report_error(error)
            await self._close_connection(connection.connection)
            await self._disconnect(connection)
            return False
        try:
            await connection.connection.send(frame)
            return True
        except Exception as error:
            self._report_error(error)
            await self._close_connection(connection.connection)
            await self._disconnect(connection)
            return False

    async def _fail_protocol(self, connection: ConnectionState, error: dict[str, Any]) -> None:
        if connection.disconnected or connection.stage in ("closing", "closed"):
            return
        connection.stage = "closing"
        connection.handshake_timeout_handle.cancel()
        message = {"type": "hello_error", "error": error}
        final_frame: bytes | None = None
        try:
            final_frame = encode_server_message(message, max_frame_length=self._max_frame_length)
        except Exception as encode_error:
            self._report_error(encode_error)
        await self._close_connection(connection.connection, final_frame)
        await self._disconnect(connection)

    async def _close_server_state(self) -> None:
        connections = list(self._connections)
        for connection in connections:
            connection.stage = "closing"
            connection.handshake_timeout_handle.cancel()
        await asyncio.gather(*(self._close_connection(c.connection) for c in connections))
        await asyncio.gather(*(self._disconnect(c) for c in connections))
        await self._sessions.close()
        self._connections.clear()

    async def _close_connection(self, connection: ByteConnection, final_chunk: bytes | None = None) -> None:
        try:
            result = connection.close(final_chunk)
            if result is not None:
                await result
        except Exception as error:
            self._report_error(error)

    def _to_protocol_error(self, error: BaseException) -> dict[str, Any]:
        if isinstance(error, InternalServerError):
            self._report_error(error.__cause__ or error)
            return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}
        if isinstance(error, PiServerError):
            if error.code == "not_implemented":
                return {"code": "not_implemented", "message": NOT_IMPLEMENTED_MESSAGE}
            if error.details is None:
                return {"code": error.code, "message": error.message}
            return {"code": error.code, "message": error.message, "details": error.details}
        if isinstance(error, ProtocolValidationError):
            return {"code": "invalid_request", "message": str(error)}
        self._report_error(error)
        return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}

    def _report_error(self, error: BaseException) -> None:
        if self._on_error is None:
            return
        with contextlib.suppress(Exception):
            self._on_error(error if isinstance(error, Exception) else Exception(str(error)))


class _NullHandler:
    def __init__(self, report_error: Any) -> None:
        self._report_error = report_error

    def on_data(self, chunk: bytes) -> None:
        pass

    def on_close(self) -> None:
        pass

    def on_error(self, error: Exception) -> None:
        self._report_error(error)


class _ConnectionHandler:
    def __init__(self, server: PiServer, state: ConnectionState) -> None:
        self._server = server
        self._state = state

    def on_data(self, chunk: bytes) -> None:
        self._server._receive(self._state, chunk)

    def on_close(self) -> None:
        self._server._transport_closed(self._state)

    def on_error(self, error: Exception) -> None:
        self._server._report_error(error)
        connection = self._state

        async def _run() -> None:
            await self._server._close_connection(connection.connection)
            await self._server._disconnect(connection)

        spawn(_run())


def _resolve_options(options: PiServerOptions) -> tuple[int, int]:
    if not isinstance(options.listeners, list):
        raise TypeError("PiServer listeners must be an array")
    if options.server_id == "":
        raise TypeError("PiServer serverId must not be empty")
    max_frame_length = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
    if not isinstance(max_frame_length, int) or max_frame_length <= 0 or max_frame_length > _MAX_UINT32:
        raise TypeError(f"PiServer maxFrameLength must be an integer between 1 and {_MAX_UINT32}")
    handshake_timeout_ms = (
        options.handshake_timeout_ms if options.handshake_timeout_ms is not None else _DEFAULT_HANDSHAKE_TIMEOUT_MS
    )
    if (
        not isinstance(handshake_timeout_ms, int)
        or handshake_timeout_ms <= 0
        or handshake_timeout_ms > _MAX_TIMER_DELAY_MS
    ):
        raise TypeError(f"PiServer handshakeTimeoutMs must be an integer between 1 and {_MAX_TIMER_DELAY_MS}")
    return max_frame_length, handshake_timeout_ms
