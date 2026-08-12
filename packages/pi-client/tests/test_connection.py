"""Connection lifecycle and handshake tests.

Python port of `packages/client/test/connection.test.ts`.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_client import PiClient, PiClientOptions, PiDisconnectedError, PiServerError
from pi_protocol import PROTOCOL_VERSION, ProtocolValidationError, encode_cbor, encode_frame, encode_server_message
from support import (
    MemoryByteServer,
    attach_session,
    base_server_snapshot,
    connect_client,
    create_client,
    session_snapshot,
)


@pytest.mark.asyncio
async def test_sends_framed_hello_before_accepting_fragmented_server_hello():
    server = MemoryByteServer()
    received = []

    def on_message(message):
        received.append(message)
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                },
                3,
            )

    server.on_message(on_message)
    client = create_client(server)

    result = await asyncio.wait_for(client.connect(), timeout=5)
    assert result == base_server_snapshot
    assert received[0] == {"type": "hello", "version": PROTOCOL_VERSION}
    assert client.connection_state == "connected"


@pytest.mark.asyncio
async def test_rejects_server_data_before_client_hello_is_sent():
    close_count = 0
    send_count = 0

    def factory(handlers):
        nonlocal send_count, close_count
        handlers.on_data(
            encode_server_message(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                }
            )
        )

        class _Transport:
            async def send(self, chunk):
                nonlocal send_count
                send_count += 1

            def close(self):
                nonlocal close_count
                close_count += 1

        return _Transport()

    client = PiClient(PiClientOptions(transport_factory=factory))

    with pytest.raises(ProtocolValidationError, match="Received server data before the client hello was sent"):
        await asyncio.wait_for(client.connect(), timeout=5)
    assert client.connection_state == "disconnected"
    assert send_count == 0
    assert close_count == 1


@pytest.mark.asyncio
async def test_isolates_subscriber_failures_from_handshake():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_message)
    client = create_client(server)

    def failing_listener(_snapshot):
        raise RuntimeError("consumer failure")

    client.subscribe(failing_listener)

    result = await asyncio.wait_for(client.connect(), timeout=5)
    assert result == base_server_snapshot
    assert client.connection_state == "connected"


@pytest.mark.asyncio
async def test_reports_subscriber_failures_without_changing_connection_state():
    server = MemoryByteServer()
    listener_errors = []

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_message)
    client = PiClient(
        PiClientOptions(transport_factory=server.connect, on_listener_error=lambda error: listener_errors.append(error))
    )

    def failing_listener(_snapshot):
        raise RuntimeError("consumer failure")

    client.subscribe(failing_listener)

    result = await asyncio.wait_for(client.connect(), timeout=5)
    assert result == base_server_snapshot
    assert len(listener_errors) == 1
    assert str(listener_errors[0]) == "consumer failure"
    assert client.connection_state == "connected"


@pytest.mark.asyncio
async def test_does_not_restore_connection_after_listener_disconnects_during_handshake():
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
    client = create_client(server)
    client.subscribe(lambda _snapshot: client.disconnect())

    with pytest.raises(PiDisconnectedError):
        await asyncio.wait_for(client.connect(), timeout=5)
    assert client.connection_state == "disconnected"
    assert server.client_close_count == 1


@pytest.mark.asyncio
async def test_does_not_restore_stale_connection_when_listener_reconnects_during_handshake():
    """The reconnect a snapshot listener starts must win; the aborted first connection must not revive it."""
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = {"count": 0}
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] != "hello":
                return
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": f"connection-{connection['count']}",
                    "snapshot": {**base_server_snapshot, "revision": connection["count"]},
                }
            )

        server.on_message(on_message)

    def factory(handlers):
        server = first if connection["count"] == 0 else second
        connection["count"] += 1
        return server.connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=factory))
    state = {"reconnect": None, "requested": False}

    def listener(_snapshot):
        if state["requested"]:
            return
        state["requested"] = True
        client.disconnect()
        state["reconnect"] = asyncio.ensure_future(client.reconnect())

    client.subscribe(listener)

    with pytest.raises(PiDisconnectedError):
        await asyncio.wait_for(client.connect(), timeout=5)
    assert state["reconnect"] is not None

    result = await asyncio.wait_for(state["reconnect"], timeout=5)
    assert result["revision"] == 2
    assert client.connection_state == "connected"
    assert first.client_close_count == 1


@pytest.mark.asyncio
async def test_rejects_typed_handshake_version_error():
    server = MemoryByteServer()

    def on_message(_message):
        server.send({"type": "hello_error", "error": {"code": "version", "message": "Unsupported protocol version"}})

    server.on_message(on_message)
    client = create_client(server)

    with pytest.raises(PiServerError) as exc_info:
        await asyncio.wait_for(client.connect(), timeout=5)
    assert exc_info.value.code == "version"
    assert str(exc_info.value) == "Unsupported protocol version"
    assert client.connection_state == "disconnected"
    assert server.client_close_count == 1


@pytest.mark.asyncio
async def test_rejects_pending_requests_on_close_and_reconnects():
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = {"count": 0}
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] == "hello":
                server.send(
                    {
                        "type": "hello",
                        "version": PROTOCOL_VERSION,
                        "connectionId": f"connection-{connection['count']}",
                        "snapshot": {**base_server_snapshot, "revision": connection["count"]},
                    }
                )

        server.on_message(on_message)

    def factory(handlers):
        server = first if connection["count"] == 0 else second
        connection["count"] += 1
        return server.connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=factory))
    states = []
    client.on_connection_state_change(lambda change: states.append(change.state))
    await asyncio.wait_for(client.connect(), timeout=5)
    pending = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.01)
    first.close()

    with pytest.raises(PiDisconnectedError):
        await asyncio.wait_for(pending, timeout=5)
    assert client.connection_state == "disconnected"

    result = await asyncio.wait_for(client.reconnect(), timeout=5)
    assert result["revision"] == 2
    assert client.connection_state == "connected"
    assert states == ["connecting", "connected", "disconnected", "connecting", "connected"]


@pytest.mark.asyncio
async def test_supports_synchronous_reconnect_from_a_disconnection_listener():
    """A connection-state listener may start a reconnect the moment it sees `disconnected`."""
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = {"count": 0}
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] != "hello":
                return
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": f"connection-{connection['count']}",
                    "snapshot": {**base_server_snapshot, "revision": connection["count"]},
                }
            )

        server.on_message(on_message)

    def factory(handlers):
        server = first if connection["count"] == 0 else second
        connection["count"] += 1
        return server.connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=factory))
    await asyncio.wait_for(client.connect(), timeout=5)
    state = {"reconnect": None}

    def on_change(change):
        if change.state == "disconnected" and state["reconnect"] is None:
            state["reconnect"] = asyncio.ensure_future(client.reconnect())

    client.on_connection_state_change(on_change)

    first.close()
    assert state["reconnect"] is not None

    result = await asyncio.wait_for(state["reconnect"], timeout=5)
    assert result["revision"] == 2
    assert client.connection_state == "connected"


@pytest.mark.asyncio
async def test_rejects_pending_requests_on_transport_errors():
    server = MemoryByteServer()
    client = await connect_client(server)
    pending = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.01)
    server.error(Exception("read failed"))

    with pytest.raises(PiDisconnectedError, match="read failed"):
        await asyncio.wait_for(pending, timeout=5)
    assert client.connection_state == "disconnected"


@pytest.mark.asyncio
async def test_enforces_configured_frame_limit():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_message)
    client = PiClient(PiClientOptions(transport_factory=server.connect, max_frame_length=512))
    await asyncio.wait_for(client.connect(), timeout=5)
    handle = await attach_session(client, server, session_snapshot("session-1"))
    sent_before = len(server.sent_by_client)

    with pytest.raises(ProtocolValidationError):
        await asyncio.wait_for(handle.prompt("x" * 1000), timeout=5)
    assert len(server.sent_by_client) == sent_before

    server.send_raw(bytes([0, 0, 2, 1]))
    assert client.connection_state == "disconnected"


@pytest.mark.asyncio
async def test_disconnects_on_invalid_protocol_data():
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send_raw(encode_frame(encode_cbor({"type": "event", "event": {"type": "session_removed", "sessionId": 1}})))
    await asyncio.sleep(0.01)
    assert client.connection_state == "disconnected"


@pytest.mark.asyncio
async def test_reports_truncated_framing_when_transport_closes():
    server = MemoryByteServer()
    client = await connect_client(server)
    pending = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.01)
    server.send_raw(bytes([0, 0, 0, 2, 1]))
    server.close()

    with pytest.raises(ProtocolValidationError, match=r"(?i)truncated"):
        await asyncio.wait_for(pending, timeout=5)
    assert client.connection_state == "disconnected"


def test_rejects_frame_limits_outside_uint32_range():
    server = MemoryByteServer()
    with pytest.raises(TypeError, match=r"max_frame_length|maxFrameLength"):
        PiClient(PiClientOptions(transport_factory=server.connect, max_frame_length=0x1_0000_0000))
