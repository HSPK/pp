"""Shared test fixtures for pi_client tests.

Python port of `packages/client/test/support.ts`. `MemoryByteServer` is the
in-memory transport double the ported tests use in place of a real socket.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pi_client import PiClient, PiClientOptions
from pi_protocol import PROTOCOL_VERSION, ClientMessageDecoder, encode_server_message


class MemoryByteServer:
    def __init__(self) -> None:
        self._handlers: Any = None
        self._decoder = ClientMessageDecoder()
        self._message_listeners: set[Callable[[dict[str, Any]], None]] = set()
        self.sent_by_client: list[bytes] = []
        self.client_close_count = 0

    def connect(self, handlers: Any) -> Any:
        self._handlers = handlers
        closed = {"value": False}

        class _Transport:
            async def send(_self, chunk: bytes) -> None:
                if closed["value"]:
                    raise RuntimeError("Transport is closed")
                self.sent_by_client.append(bytes(chunk))
                for message in self._decoder.push(chunk):
                    for listener in list(self._message_listeners):
                        listener(message)

            def close(_self) -> None:
                if closed["value"]:
                    return
                closed["value"] = True
                self.client_close_count += 1

        return _Transport()

    def on_message(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._message_listeners.add(listener)
        return lambda: self._message_listeners.discard(listener)

    def send(self, message: dict[str, Any], split_at: int | None = None) -> None:
        frame = encode_server_message(message)
        if split_at is None:
            self.send_raw(frame)
            return
        self.send_raw(frame[:split_at])
        self.send_raw(frame[split_at:])

    def send_together(self, messages: list[dict[str, Any]]) -> None:
        chunk = b"".join(encode_server_message(message) for message in messages)
        self.send_raw(chunk)

    def send_raw(self, chunk: bytes) -> None:
        if self._handlers is not None:
            self._handlers.on_data(chunk)

    def close(self) -> None:
        if self._handlers is not None:
            self._handlers.on_close()

    def error(self, error: Exception) -> None:
        if self._handlers is not None:
            self._handlers.on_error(error)


base_server_snapshot: dict[str, Any] = {
    "serverId": "server-1",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 1,
    "sessions": [],
    "models": [],
}


def session_snapshot(id_: str, **overrides: Any) -> dict[str, Any]:
    snapshot = {
        "id": id_,
        "cwd": "/workspace",
        "createdAt": 1,
        "updatedAt": 1,
        "phase": "idle",
        "model": {"provider": "faux", "id": "model"},
        "thinkingLevel": "off",
        "attached": True,
        "locked": True,
        "revision": 1,
        "transcript": [],
        "queuedSteer": [],
        "queuedSteerCount": 0,
    }
    snapshot.update(overrides)
    return snapshot


def create_client(server: MemoryByteServer) -> PiClient:
    return PiClient(PiClientOptions(transport_factory=server.connect))


async def connect_client(server: MemoryByteServer) -> PiClient:
    client = create_client(server)

    def on_hello(message: dict[str, Any]) -> None:
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_hello)
    await client.connect()
    return client


def collect_requests(server: MemoryByteServer) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    def on_message(message: dict[str, Any]) -> None:
        if message["type"] == "request":
            requests.append(message)

    server.on_message(on_message)
    return requests


async def attach_session(client: PiClient, server: MemoryByteServer, snapshot: dict[str, Any]) -> Any:
    requests = collect_requests(server)
    attaching = asyncio.ensure_future(client.attach_session(snapshot["id"]))
    request = None
    for _ in range(1000):
        request = next((candidate for candidate in requests if candidate["request"]["command"] == "attach"), None)
        if request is not None:
            break
        await asyncio.sleep(0)
    if request is None:
        raise RuntimeError("Missing attach request")
    server.send(
        {"type": "response", "id": request["id"], "ok": True, "result": {"command": "attach", "session": snapshot}}
    )
    return await attaching
