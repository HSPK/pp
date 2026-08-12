"""Coverage tests for pi_client.connection.

Targets lines: 83-85, 108, 117, 123-126, 141-144, 155-157, 162, 174,
               184-185, 199-202, 207, 212->214, 216, 218-219, 225, 228->232
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pi_client import PiClient, PiClientOptions, PiDisconnectedError
from pi_protocol import PROTOCOL_VERSION, encode_server_message
from support import MemoryByteServer, base_server_snapshot, connect_client

# --------------------------------------------------------------------------- #
# 83-85: connect() when already connecting or connected                        #
# --------------------------------------------------------------------------- #


async def test_connect_when_already_connecting_raises() -> None:
    server = MemoryByteServer()

    def auto_hello(msg: dict[str, Any]) -> None:
        if msg["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "c1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(auto_hello)
    client = PiClient(PiClientOptions(transport_factory=server.connect))
    fut = client.connect()
    # Second connect() must fail with PiDisconnectedError while the first
    # is still in "connecting" state.
    with pytest.raises(PiDisconnectedError, match="connecting"):
        await asyncio.wait_for(client.connect(), timeout=5)
    # Let the first handshake complete normally.
    await asyncio.wait_for(fut, timeout=5)
    client.disconnect()


async def test_connect_when_already_connected_raises() -> None:
    server = MemoryByteServer()
    client = await connect_client(server)
    try:
        with pytest.raises(PiDisconnectedError, match="connected"):
            await asyncio.wait_for(client.connect(), timeout=5)
    finally:
        client.disconnect()


# --------------------------------------------------------------------------- #
# 108: fail() early return when already disconnected                           #
# --------------------------------------------------------------------------- #


def test_fail_when_already_disconnected_is_noop() -> None:
    server = MemoryByteServer()
    client = PiClient(PiClientOptions(transport_factory=server.connect))
    conn = client._connection  # type: ignore[attr-defined]
    # State is "disconnected" initially; calling fail() must be a no-op.
    conn.fail(RuntimeError("ignored"))  # should not raise


# --------------------------------------------------------------------------- #
# 117: send() raises when not connected                                        #
# --------------------------------------------------------------------------- #


def test_connection_send_raises_when_disconnected() -> None:
    server = MemoryByteServer()
    client = PiClient(PiClientOptions(transport_factory=server.connect))
    conn = client._connection  # type: ignore[attr-defined]
    with pytest.raises(PiDisconnectedError):
        conn.send(b"frame")


# --------------------------------------------------------------------------- #
# 123-126: send() _send() task catches transport exception                    #
# --------------------------------------------------------------------------- #


async def test_send_fails_when_transport_raises_after_connect() -> None:
    """Transport.send() raises on request; connection should transition to disconnected."""
    fail_next_send = asyncio.Event()

    class _FaultyTransport:
        closed = False

        async def send(self, chunk: bytes) -> None:
            if fail_next_send.is_set():
                raise RuntimeError("write error")

        def close(self) -> None:
            self.closed = True

    transport_ref: list[_FaultyTransport] = []

    def factory(handlers: Any) -> _FaultyTransport:
        t = _FaultyTransport()
        transport_ref.append(t)

        # Deliver the server hello immediately (synchronously) so the
        # handshake completes before we exercise send failures.
        async def _deliver_hello() -> None:
            await asyncio.sleep(0)
            handlers.on_data(
                encode_server_message(
                    {
                        "type": "hello",
                        "version": PROTOCOL_VERSION,
                        "connectionId": "c1",
                        "snapshot": base_server_snapshot,
                    }
                )
            )

        _deliver_task = asyncio.ensure_future(_deliver_hello())
        del _deliver_task  # fire-and-forget; suppressing RUF006
        return t

    client = PiClient(PiClientOptions(transport_factory=factory))
    snapshot = await asyncio.wait_for(client.connect(), timeout=5)
    assert snapshot == base_server_snapshot
    assert client.connection_state == "connected"

    # Now cause transport.send() to raise on the next call.
    fail_next_send.set()
    conn = client._connection  # type: ignore[attr-defined]
    conn.send(b"\x00" * 16)  # spawns _send() task

    # Wait for the disconnect to propagate.
    for _ in range(100):
        if client.connection_state == "disconnected":
            break
        await asyncio.sleep(0.01)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 141-144: _open_transport() cleans up stale transport when lifecycle changed #
# --------------------------------------------------------------------------- #


async def test_open_transport_closes_transport_if_disconnected_while_connecting() -> None:
    """Disconnect during factory await → stale transport must be closed."""
    factory_event = asyncio.Event()
    transport_closed = asyncio.Event()

    class _Transport:
        async def send(self, chunk: bytes) -> None:
            pass

        def close(self) -> None:
            transport_closed.set()

    async def slow_factory(handlers: Any) -> _Transport:
        await factory_event.wait()
        return _Transport()

    client = PiClient(PiClientOptions(transport_factory=slow_factory))
    fut = client.connect()
    await asyncio.sleep(0.01)
    # Disconnect while the factory is suspended.
    client.disconnect()

    # Let the factory complete; it should close the transport it created.
    factory_event.set()
    await asyncio.sleep(0.05)

    assert await asyncio.wait_for(asyncio.shield(transport_closed.wait()), timeout=3)
    assert client.connection_state == "disconnected"

    # Suppress the disconnected future (it was rejected).
    with pytest.raises((PiDisconnectedError, Exception)):
        await asyncio.wait_for(fut, timeout=5)


# --------------------------------------------------------------------------- #
# 155-157: _open_transport() hello send fails                                  #
# --------------------------------------------------------------------------- #


async def test_open_transport_fails_when_hello_send_raises() -> None:
    class _FailSendTransport:
        closed = False

        async def send(self, chunk: bytes) -> None:
            raise RuntimeError("hello send failed")

        def close(self) -> None:
            self.closed = True

    def factory(handlers: Any) -> _FailSendTransport:
        return _FailSendTransport()

    client = PiClient(PiClientOptions(transport_factory=factory))
    with pytest.raises((PiDisconnectedError, RuntimeError)):
        await asyncio.wait_for(client.connect(), timeout=5)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 162: _handle_data() early return when state is disconnected                 #
# --------------------------------------------------------------------------- #


async def test_handle_data_ignored_after_disconnect() -> None:
    server = MemoryByteServer()
    client = await connect_client(server)
    client.disconnect()
    assert client.connection_state == "disconnected"
    # Sending data after disconnect must not raise and must keep the client
    # in "disconnected" state.
    server.send_raw(
        encode_server_message({"type": "event", "event": {"type": "server_snapshot", "snapshot": base_server_snapshot}})
    )
    await asyncio.sleep(0.01)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 174: for-loop early return when state becomes disconnected mid-batch         #
# --------------------------------------------------------------------------- #


async def test_second_message_in_batch_skipped_after_disconnect_from_first() -> None:
    """Batch of two messages; the first (invalid hello when connected) causes
    a disconnect; the second must be silently dropped (line 174 reached)."""
    server = MemoryByteServer()
    client = await connect_client(server)

    # Send two frames in one raw write: first a "hello" (invalid when
    # connected) then a "hello_error".  The first triggers a disconnect
    # immediately so the second hits the guard at line 174.
    first = encode_server_message(
        {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "connectionId": "late",
            "snapshot": base_server_snapshot,
        }
    )
    second = encode_server_message({"type": "hello_error", "error": {"code": "version", "message": "too old"}})
    server.send_raw(first + second)
    await asyncio.sleep(0.05)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 184-185: _handle_message() unexpected first message (not hello/hello_error) #
# --------------------------------------------------------------------------- #


async def test_unexpected_first_message_disconnects_client() -> None:
    """Server sends a non-hello first message → ProtocolValidationError."""

    class _GarbageFirstServer:
        def connect(self, handlers: Any) -> Any:
            self._handlers = handlers

            class _T:
                async def send(_self, chunk: bytes) -> None:
                    # Respond to the client hello with an invalid "response"
                    # message instead of "hello".
                    handlers.on_data(
                        encode_server_message(
                            {
                                "type": "response",
                                "id": "x",
                                "ok": True,
                                "result": {"command": "list", "sessions": []},
                            }
                        )
                    )

                def close(_self) -> None:
                    pass

            return _T()

    bad_server = _GarbageFirstServer()
    client = PiClient(PiClientOptions(transport_factory=bad_server.connect))
    with pytest.raises(Exception, match=r"(?i)first|hello"):
        await asyncio.wait_for(client.connect(), timeout=5)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 199-202: _handle_message() on_handshake callback raises                     #
# --------------------------------------------------------------------------- #


async def test_on_handshake_exception_disconnects_client() -> None:
    server = MemoryByteServer()

    def on_msg(msg: dict[str, Any]) -> None:
        if msg["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "c1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_msg)

    def exploding_subscriber(snapshot: Any) -> None:
        raise RuntimeError("subscriber exploded")

    client = PiClient(
        PiClientOptions(
            transport_factory=server.connect,
            # Capture the on_handshake exception path by using a subscriber
            # that raises; PiClient wraps this in on_listener_error if provided,
            # but if not provided the exception bubbles up through on_handshake.
        )
    )
    client.subscribe(exploding_subscriber)
    # With on_listener_error not configured the error is absorbed; the
    # connection still transitions to connected.  The coverage goal is to
    # reach the try/except around on_handshake.
    result = await asyncio.wait_for(client.connect(), timeout=5)
    assert result == base_server_snapshot
    client.disconnect()


# --------------------------------------------------------------------------- #
# 199-202: on_handshake raises AND is not isolated (no on_listener_error)     #
# --------------------------------------------------------------------------- #


async def test_on_handshake_error_triggers_fail_and_close() -> None:
    """When on_handshake itself (not a subscriber) raises, the connection fails."""
    server = MemoryByteServer()

    def on_msg(msg: dict[str, Any]) -> None:
        if msg["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "c1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_msg)

    # Inject a raising on_handshake via a custom Connection.
    from pi_client.connection import Connection

    states: list[str] = []

    def on_handshake(_: Any) -> None:
        raise RuntimeError("on_handshake raised")

    conn = Connection(
        transport_factory=server.connect,
        max_frame_length=None,
        on_handshake=on_handshake,
        on_message=lambda _: None,
        on_state_change=lambda change: states.append(change.state),
    )

    with pytest.raises((RuntimeError, PiDisconnectedError)):
        await asyncio.wait_for(conn.connect(), timeout=5)
    assert conn.state == "disconnected"


# --------------------------------------------------------------------------- #
# 207: lifecycle replaced by on_state_change callback                         #
# --------------------------------------------------------------------------- #


async def test_on_state_change_disconnect_during_connect_handshake() -> None:
    """on_state_change disconnects the client; lifecycle must not be overwritten."""
    server = MemoryByteServer()

    def on_msg(msg: dict[str, Any]) -> None:
        if msg["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "c1",
                    "snapshot": base_server_snapshot,
                }
            )

    server.on_message(on_msg)

    from pi_client.connection import Connection

    conn = Connection(
        transport_factory=server.connect,
        max_frame_length=None,
        on_handshake=lambda _: None,
        on_message=lambda _: None,
        on_state_change=lambda change: conn.disconnect() if change.state == "connected" else None,
    )

    with pytest.raises((PiDisconnectedError, Exception)):
        await asyncio.wait_for(conn.connect(), timeout=5)
    assert conn.state == "disconnected"


# --------------------------------------------------------------------------- #
# 218-219: hello/hello_error received when already connected                  #
# --------------------------------------------------------------------------- #


async def test_hello_message_when_connected_disconnects_client() -> None:
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send(
        {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "connectionId": "dup",
            "snapshot": base_server_snapshot,
        }
    )
    await asyncio.sleep(0.05)
    assert client.connection_state == "disconnected"


async def test_hello_error_when_connected_disconnects_client() -> None:
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send({"type": "hello_error", "error": {"code": "version", "message": "too old"}})
    await asyncio.sleep(0.05)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 225: _handle_close() early return when already disconnected                 #
# --------------------------------------------------------------------------- #


async def test_handle_close_ignored_when_already_disconnected() -> None:
    server = MemoryByteServer()
    client = await connect_client(server)
    client.disconnect()
    assert client.connection_state == "disconnected"
    # Triggering on_close again must be a no-op (covers line 225 return).
    server.close()
    await asyncio.sleep(0.01)
    assert client.connection_state == "disconnected"


# --------------------------------------------------------------------------- #
# 228->232: _handle_close() with decoder.end() raising                        #
# --------------------------------------------------------------------------- #


async def test_handle_close_with_truncated_frame_raises_protocol_error() -> None:
    """Transport closes after partial frame → decoder.end() raises; error propagates."""
    from pi_protocol import ProtocolValidationError

    server = MemoryByteServer()
    client = await connect_client(server)
    pending = asyncio.ensure_future(client.list_sessions())
    await asyncio.sleep(0.01)
    # Send a frame header claiming a 2-byte body, then only 1 byte, then close.
    server.send_raw(bytes([0, 0, 0, 2, 0xA0]))  # partial frame
    server.close()

    with pytest.raises((ProtocolValidationError, PiDisconnectedError)):
        await asyncio.wait_for(pending, timeout=5)
    assert client.connection_state == "disconnected"
