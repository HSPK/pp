"""Connection lifecycle: handshake, framing, and transport failure handling.

Python port of `packages/client/src/connection.ts`. `Connection` owns exactly
one `ByteTransport` at a time and speaks the framed CBOR protocol over it,
tracking a small state machine (`disconnected` / `connecting` / `connected`)
that mirrors the TypeScript discriminated union via a single lifecycle
attribute instead of a tagged union type.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    ProtocolValidationError,
    ServerMessageDecoder,
    encode_client_message,
)

from .errors import PiDisconnectedError, PiServerError, to_disconnected_error, to_error
from .transport import ByteTransport, ByteTransportFactory
from .types import ConnectionState, ConnectionStateChange

MAX_UINT32 = 0xFFFF_FFFF


@dataclass
class _Lifecycle:
    state: ConnectionState
    id: int = 0
    decoder: ServerMessageDecoder | None = None
    transport: ByteTransport | None = None
    handshake: asyncio.Future[dict[str, Any]] | None = None


@dataclass
class _Handlers:
    on_data: Callable[[bytes], None]
    on_close: Callable[[], None]
    on_error: Callable[[Exception], None]


class Connection:
    def __init__(
        self,
        transport_factory: ByteTransportFactory,
        max_frame_length: int | None,
        on_handshake: Callable[[dict[str, Any]], None],
        on_message: Callable[[dict[str, Any]], None],
        on_state_change: Callable[[ConnectionStateChange], None],
    ) -> None:
        self._transport_factory = transport_factory
        self._on_handshake = on_handshake
        self._on_message = on_message
        self._on_state_change = on_state_change
        self._max_frame_length = DEFAULT_MAX_FRAME_LENGTH if max_frame_length is None else max_frame_length
        if (
            not isinstance(self._max_frame_length, int)
            or isinstance(self._max_frame_length, bool)
            or self._max_frame_length <= 0
            or self._max_frame_length > MAX_UINT32
        ):
            raise TypeError(f"PiClient maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        self._lifecycle = _Lifecycle(state="disconnected")
        self._sequence = 0
        self._background_tasks: set[asyncio.Task[Any]] = set()

    @property
    def state(self) -> ConnectionState:
        return self._lifecycle.state

    @property
    def max_frame_length(self) -> int:
        return self._max_frame_length

    def connect(self) -> asyncio.Future[dict[str, Any]]:
        if self._lifecycle.state != "disconnected":
            future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
            future.set_exception(PiDisconnectedError(f"PiClient is already {self._lifecycle.state}"))
            return future
        self._sequence += 1
        id_ = self._sequence
        handshake: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._lifecycle = _Lifecycle(
            state="connecting",
            id=id_,
            decoder=ServerMessageDecoder(self._max_frame_length),
            handshake=handshake,
        )
        self._on_state_change(ConnectionStateChange(state="connecting"))
        self._spawn(self._open_transport(id_))
        return handshake

    def disconnect(self, reason: str | Exception = "Client disconnected") -> None:
        if self._lifecycle.state == "disconnected":
            return
        error = PiDisconnectedError(reason) if isinstance(reason, str) else reason
        self._fail_and_close(error)

    def fail(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        self._lifecycle = _Lifecycle(state="disconnected")
        if lifecycle.handshake is not None and not lifecycle.handshake.done():
            lifecycle.handshake.set_exception(error)
        self._on_state_change(ConnectionStateChange(state="disconnected", error=error))

    def send(self, frame: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state != "connected" or lifecycle.transport is None:
            raise PiDisconnectedError()
        transport = lifecycle.transport

        async def _send() -> None:
            try:
                await transport.send(frame)
            except Exception as error:
                current = self._lifecycle
                if current.state != "disconnected" and current.transport is transport:
                    self._fail_and_close(to_disconnected_error(error))

        self._spawn(_send())

    async def _open_transport(self, id_: int) -> None:
        handlers = _Handlers(
            on_data=lambda chunk: self._handle_data(id_, chunk),
            on_close=lambda: self._handle_close() if self._is_current(id_) else None,
            on_error=lambda error: (
                self._fail_and_close(to_disconnected_error(error)) if self._is_current(id_) else None
            ),
        )
        try:
            result = self._transport_factory(handlers)
            transport = await result if isinstance(result, Awaitable) else result
        except Exception as error:
            if self._is_current(id_):
                self.fail(to_disconnected_error(error))
            return
        lifecycle = self._lifecycle
        if lifecycle.state != "connecting" or lifecycle.id != id_:
            transport.close()
            return
        lifecycle.transport = transport
        try:
            frame = encode_client_message(
                {"type": "hello", "version": PROTOCOL_VERSION}, max_frame_length=self._max_frame_length
            )
            await transport.send(frame)
        except Exception as error:
            if self._is_current(id_):
                self._fail_and_close(to_disconnected_error(error))

    def _handle_data(self, id_: int, chunk: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected" or lifecycle.id != id_:
            return
        if lifecycle.state == "connecting" and lifecycle.transport is None:
            self._fail_and_close(ProtocolValidationError("Received server data before the client hello was sent"))
            return
        assert lifecycle.decoder is not None
        try:
            messages = lifecycle.decoder.push(chunk)
        except Exception as error:
            self._fail_and_close(to_error(error))
            return
        for message in messages:
            if self._lifecycle.state == "disconnected":
                return
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "connecting":
            if message["type"] == "hello_error":
                self._fail_and_close(PiServerError(message["error"]))
                return
            if message["type"] != "hello":
                self._fail_and_close(ProtocolValidationError("Expected server hello as first message"))
                return
            if lifecycle.transport is None:
                self._fail_and_close(ProtocolValidationError("Received server hello before the client hello was sent"))
                return
            connected = _Lifecycle(
                state="connected",
                id=lifecycle.id,
                decoder=lifecycle.decoder,
                transport=lifecycle.transport,
                handshake=lifecycle.handshake,
            )
            self._lifecycle = connected
            try:
                self._on_handshake(message["snapshot"])
            except Exception as error:
                if self._lifecycle is connected:
                    self._fail_and_close(to_error(error))
                return
            if self._lifecycle is not connected:
                return
            self._on_state_change(ConnectionStateChange(state="connected"))
            if self._lifecycle is not connected:
                return
            self._lifecycle = _Lifecycle(
                state="connected", id=connected.id, decoder=connected.decoder, transport=connected.transport
            )
            handshake = lifecycle.handshake
            if handshake is not None and not handshake.done():
                handshake.set_result(message["snapshot"])
            return
        if lifecycle.state != "connected":
            return
        if message["type"] in ("hello", "hello_error"):
            self._fail_and_close(ProtocolValidationError("Unexpected handshake message"))
            return
        self._on_message(message)

    def _handle_close(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        error: Exception = PiDisconnectedError("Byte transport closed")
        try:
            if lifecycle.decoder is not None:
                lifecycle.decoder.end()
        except Exception as decoder_error:
            error = to_error(decoder_error)
        self.fail(error)

    def _fail_and_close(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        transport = None if lifecycle.state == "disconnected" else lifecycle.transport
        self.fail(error)
        if transport is not None:
            transport.close()

    def _is_current(self, id_: int) -> bool:
        return self._lifecycle.state != "disconnected" and self._lifecycle.id == id_

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
