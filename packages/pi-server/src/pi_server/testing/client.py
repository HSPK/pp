"""Wire-level protocol test client.

Python port of `packages/server/src/testing/client.ts`. `ProtocolTestClient`
drives the raw client-side wire protocol (hello/request/response/event
messages) directly, independent of `pi_client`, so `pi_server` tests can
assert on exact wire behaviour. `connect_unix_test_client` connects one over a
real Unix-domain socket.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pi_protocol import PROTOCOL_VERSION, ServerMessageDecoder, encode_client_message

from .service import Deferred


class WireChannel(Protocol):
    def send(self, chunk: bytes) -> Awaitable[None]: ...
    def send_fragmented(self, chunk: bytes, split_at: int) -> Awaitable[None]: ...
    def close(self) -> Awaitable[None]: ...


class _Waiter:
    __slots__ = ("future", "predicate")

    def __init__(self, predicate: Callable[[dict[str, Any]], bool], future: asyncio.Future[dict[str, Any]]) -> None:
        self.predicate = predicate
        self.future = future


class ProtocolTestClient:
    def __init__(self, channel: WireChannel) -> None:
        self.messages: list[dict[str, Any]] = []
        self._channel = channel
        self._decoder = ServerMessageDecoder()
        self._waiters: set[_Waiter] = set()
        self._closed_deferred = Deferred()
        self._request_sequence = 0
        self._closed_value = False
        self._read_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed_value

    async def hello(self, version: int = PROTOCOL_VERSION) -> dict[str, Any]:
        response = self.next(lambda message: message["type"] in ("hello", "hello_error"))
        await self.send_message({"type": "hello", "version": version})
        return await response

    async def request(self, command: dict[str, Any], id: str | None = None) -> dict[str, Any]:
        if id is None:
            self._request_sequence += 1
            id = f"request-{self._request_sequence}"
        response = self.next(lambda message: message["type"] == "response" and message["id"] == id)
        await self.send_message({"type": "request", "id": id, "request": command})
        return await response

    async def send_message(self, message: dict[str, Any]) -> None:
        await self._channel.send(encode_client_message(message))

    async def send_bytes(self, chunk: bytes) -> None:
        await self._channel.send(chunk)

    async def send_fragmented_message(self, message: dict[str, Any], split_at: int) -> None:
        await self._channel.send_fragmented(encode_client_message(message), split_at)

    def next(self, predicate: Callable[[dict[str, Any]], bool]) -> Awaitable[dict[str, Any]]:
        return self.next_from(0, predicate)

    def next_from(self, index: int, predicate: Callable[[dict[str, Any]], bool]) -> Awaitable[dict[str, Any]]:
        for message in self.messages[index:]:
            if predicate(message):
                loop = asyncio.get_event_loop()
                future: asyncio.Future[dict[str, Any]] = loop.create_future()
                future.set_result(message)
                return future
        if self._closed_value:
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            future.set_exception(RuntimeError("Wire client is closed"))
            return future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._waiters.add(_Waiter(predicate, future))
        return future

    async def wait_for_close(self) -> None:
        if self._closed_value:
            return
        await self._closed_deferred.future

    async def close(self) -> None:
        await self._channel.close()

    def receive(self, chunk: bytes) -> None:
        try:
            messages = self._decoder.push(chunk)
        except Exception as error:
            self.fail(error if isinstance(error, Exception) else Exception(str(error)))
            return
        for message in messages:
            self.messages.append(message)
            for waiter in list(self._waiters):
                if not waiter.predicate(message):
                    continue
                self._waiters.discard(waiter)
                if not waiter.future.done():
                    waiter.future.set_result(message)

    def mark_closed(self) -> None:
        if self._closed_value:
            return
        self._closed_value = True
        self._closed_deferred.resolve(None)
        self.fail(RuntimeError("Wire connection closed"))

    def fail(self, error: Exception) -> None:
        for waiter in list(self._waiters):
            self._waiters.discard(waiter)
            if not waiter.future.done():
                waiter.future.set_exception(error)


async def connect_unix_test_client(path: str) -> ProtocolTestClient:
    reader, writer = await asyncio.open_unix_connection(path)
    client_ref: list[ProtocolTestClient] = []

    async def _send(chunk: bytes) -> None:
        writer.write(chunk)
        await writer.drain()

    async def _send_fragmented(chunk: bytes, split_at: int) -> None:
        await _send(chunk[:split_at])
        await _send(chunk[split_at:])

    async def _close() -> None:
        if writer.is_closing():
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    class _Channel:
        send = staticmethod(_send)
        send_fragmented = staticmethod(_send_fragmented)
        close = staticmethod(_close)

    client = ProtocolTestClient(_Channel())
    client_ref.append(client)

    async def _read_loop() -> None:
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    client.mark_closed()
                    return
                client.receive(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            client.fail(error if isinstance(error, Exception) else Exception(str(error)))
            client.mark_closed()

    client._read_task = asyncio.ensure_future(_read_loop())
    return client
