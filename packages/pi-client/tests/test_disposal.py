"""Client disposal tests.

Python port of `packages/client/test/disposal.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_client import PiClient, PiClientDisposedError, PiClientOptions
from pi_protocol import PROTOCOL_VERSION
from support import MemoryByteServer, attach_session, base_server_snapshot, connect_client, session_snapshot


@pytest.mark.asyncio
async def test_connects_through_its_ownership_factory():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] != "hello":
            return
        server.send(
            {
                "type": "hello",
                "version": PROTOCOL_VERSION,
                "connectionId": "connection-1",
                "snapshot": base_server_snapshot,
            }
        )

    server.on_message(on_message)

    client = await asyncio.wait_for(PiClient.open(PiClientOptions(transport_factory=server.connect)), timeout=5)

    assert client.connected is True
    await client.dispose()


@pytest.mark.asyncio
async def test_disconnects_invalidates_child_handles_and_rejects_pending_requests():
    server = MemoryByteServer()
    client = await connect_client(server)
    handle = await attach_session(client, server, session_snapshot("session-1"))
    pending = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.01)

    first_disposal = client.dispose()
    second_disposal = client.dispose()

    assert second_disposal is first_disposal
    assert client.disposed is True
    assert client.connected is False
    assert handle.attached is False
    with pytest.raises(PiClientDisposedError):
        await asyncio.wait_for(pending, timeout=5)
    with pytest.raises(PiClientDisposedError):
        await asyncio.wait_for(handle.prompt("after disposal"), timeout=5)
    await first_disposal


@pytest.mark.asyncio
async def test_supports_explicit_async_disposal():
    server = MemoryByteServer()
    client = await connect_client(server)

    async with client:
        pass

    assert client.disposed is True
    assert client.connection_state == "disconnected"
