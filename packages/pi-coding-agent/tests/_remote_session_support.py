"""Shared in-memory server fixture for `RemoteSession` tests.

Mirrors `pi_client`'s own `tests/support.py` (itself a port of
`packages/client/test/support.ts`), duplicated here (rather than imported
cross-package) because test directories aren't installed packages in this
workspace. Adds `open_remote_session`, matching
`packages/coding-agent/test/client/support.ts`'s helper of the same name.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pi_client import PiClient, PiClientOptions
from pi_coding_agent.client.remote_session import RemoteSession, RemoteSessionOptions
from pi_protocol import PROTOCOL_VERSION, ClientMessageDecoder, encode_server_message


class MemoryByteServer:
    def __init__(self) -> None:
        self._handlers: Any = None
        self._decoder = ClientMessageDecoder()
        self._message_listeners: set[Callable[[dict[str, Any]], None]] = set()

    def connect(self, handlers: Any) -> Any:
        self._handlers = handlers
        closed = {"value": False}

        class _Transport:
            async def send(_self, chunk: bytes) -> None:
                if closed["value"]:
                    raise RuntimeError("Transport is closed")
                for message in self._decoder.push(chunk):
                    for listener in list(self._message_listeners):
                        listener(message)

            def close(_self) -> None:
                closed["value"] = True

        return _Transport()

    def on_message(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._message_listeners.add(listener)
        return lambda: self._message_listeners.discard(listener)

    def send(self, message: dict[str, Any]) -> None:
        if self._handlers is not None:
            self._handlers.on_data(encode_server_message(message))

    def close(self) -> None:
        if self._handlers is not None:
            self._handlers.on_close()


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


async def connect_client(server: MemoryByteServer) -> PiClient:
    client = PiClient(PiClientOptions(transport_factory=server.connect))

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


async def _find_request(requests: list[dict[str, Any]], command: str) -> dict[str, Any]:
    for _ in range(2000):
        found = next((candidate for candidate in requests if candidate["request"]["command"] == command), None)
        if found is not None:
            return found
        await asyncio.sleep(0)
    raise AssertionError(f"Missing {command} request")


async def next_request(server: MemoryByteServer, command: str) -> dict[str, Any]:
    """Waits for the next request matching `command`, ordered by arrival (like `nextRequest` in the TS suite)."""
    found: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    def on_message(message: dict[str, Any]) -> None:
        if found.done():
            return
        if message["type"] == "request" and message["request"]["command"] == command:
            found.set_result(message)

    unsubscribe = server.on_message(on_message)
    try:
        return await asyncio.wait_for(found, timeout=2)
    finally:
        unsubscribe()


async def open_remote_session(
    client: PiClient,
    server: MemoryByteServer,
    snapshot: dict[str, Any],
    options: RemoteSessionOptions | None = None,
) -> RemoteSession:
    requests = collect_requests(server)
    opening = asyncio.ensure_future(RemoteSession.open(client, snapshot["id"], options))
    request = await _find_request(requests, "attach")
    server.send(
        {"type": "response", "id": request["id"], "ok": True, "result": {"command": "attach", "session": snapshot}}
    )
    return await opening
