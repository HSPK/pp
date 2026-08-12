"""Coverage tests for pi_server.server.

Targets lines: 78-80, 86-88, 110-111, 139-140, 149, 157, 174-175,
               191-202, 244-245, 273-277, 281, 296-297, 308-309, 320,
               335, 338, 341, 344, 373, 376
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import ServerHarness, wait, wait_until
from pi_protocol import PROTOCOL_VERSION
from pi_server.errors import InternalServerError, NotImplementedProtocolError, PiServerError
from pi_server.server import PiServer
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server
from pi_server.types import PiServerOptions


def _service() -> TestServerService:
    return TestServerService()


# --------------------------------------------------------------------------- #
# 78-80: start() when already started                                         #
# --------------------------------------------------------------------------- #


async def test_start_when_already_started_raises(harness: ServerHarness) -> None:
    started = await harness.start_server()
    with pytest.raises(RuntimeError, match="already started"):
        await wait(started.server.start())


# --------------------------------------------------------------------------- #
# 86-88: start() when closing/closed                                          #
# --------------------------------------------------------------------------- #


async def test_start_when_closing_raises(harness: ServerHarness) -> None:
    started = await harness.start_server()
    # Close the server; then try to start it again.
    await wait(started.server.close())
    with pytest.raises(RuntimeError, match="closing or closed"):
        await wait(started.server.start())


# --------------------------------------------------------------------------- #
# 110-111: accept() when server is closing returns NullHandler                #
# --------------------------------------------------------------------------- #


async def test_accept_when_closing_returns_null_handler(harness: ServerHarness) -> None:
    started = await harness.start_server()

    class _MockConnection:
        closed = False
        close_count = 0

        def send(self, chunk: bytes) -> asyncio.Future[None]:
            f: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            f.set_result(None)
            return f

        def close(self, final_chunk: bytes | None = None) -> None:
            self.closed = True
            self.close_count += 1

    started.server._closing = True
    conn = _MockConnection()
    handler = started.server.accept(conn)  # type: ignore[arg-type]
    # NullHandler: on_data and on_close are no-ops.
    handler.on_data(b"x")
    handler.on_close()
    # Close must have been called on the mock connection.
    await wait_until(lambda: conn.close_count > 0, "the null handler closed the connection")
    assert conn.close_count > 0


# --------------------------------------------------------------------------- #
# 139-140: _start_internal() failure rolls back started listeners             #
# --------------------------------------------------------------------------- #


async def test_start_failure_rolls_back_started_listeners(socket_dir: Any) -> None:
    class _GoodListener:
        address = "good"
        close_count = 0

        async def start(self, accept: Any) -> None:
            pass

        async def close(self) -> None:
            self.close_count += 1
            self.address = None

    class _FailListener:
        address = "bad"

        async def start(self, accept: Any) -> None:
            raise RuntimeError("listener start failed")

        async def close(self) -> None:
            self.address = None

    good = _GoodListener()
    bad = _FailListener()
    server = PiServer(_service(), PiServerOptions(listeners=[good, bad]))
    with pytest.raises(RuntimeError, match="listener start failed"):
        await wait(server.start())
    assert good.close_count == 1


# --------------------------------------------------------------------------- #
# 149: _close_internal() awaits an in-progress start                         #
# --------------------------------------------------------------------------- #


async def test_close_waits_for_in_progress_start(socket_dir: Any) -> None:
    path = str(socket_dir / "ci.sock")
    server = create_unix_server(_service(), UnixServerOptions(path=path))
    start_future = asyncio.ensure_future(server.start())
    # Close while start is still running — must not hang.
    await wait(server.close())
    # Start may succeed or fail; we just need it to complete.
    import contextlib

    with contextlib.suppress(Exception):
        await wait(start_future)


# --------------------------------------------------------------------------- #
# 174-175: _dispatch_message() hello after first message                     #
# --------------------------------------------------------------------------- #


async def test_hello_after_first_message_fails_connection(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    # Complete the handshake.
    hello = await wait(client.hello())
    assert hello["type"] == "hello"
    # Send a second hello — server should reject.
    await wait(client.send_message({"type": "hello", "version": PROTOCOL_VERSION}))
    await wait_until(lambda: client.closed, "the server closed the connection after a duplicate hello")
    assert client.closed


# --------------------------------------------------------------------------- #
# 191-202: _finish_handshake() version mismatch                              #
# --------------------------------------------------------------------------- #


async def test_unsupported_protocol_version_rejected(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    response = await wait(client.hello(version=9999))
    assert response["type"] == "hello_error"
    assert response["error"]["code"] == "version"


# --------------------------------------------------------------------------- #
# 191-202: _finish_handshake() closing during handshake                      #
# --------------------------------------------------------------------------- #


async def test_finish_handshake_aborted_when_server_closes(socket_dir: Any) -> None:
    """Server closes while the handshake is in flight (state != 'handshaking')."""
    path = str(socket_dir / "fh.sock")
    delay = TestServerService()
    # Inject a list-sessions delay so the handshake snapshot.get() stalls.
    d = delay.delay_next_list()
    server = create_unix_server(delay, UnixServerOptions(path=path))
    await wait(server.start())
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        from pi_server.testing.client import ProtocolTestClient

        class _Channel:
            async def send(self, chunk: bytes) -> None:
                writer.write(chunk)
                await writer.drain()

            async def send_fragmented(self, chunk: bytes, split_at: int) -> None:
                await self.send(chunk[:split_at])
                await self.send(chunk[split_at:])

            async def close(self) -> None:
                writer.close()

        client = ProtocolTestClient(_Channel())

        async def _read() -> None:
            try:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        client.mark_closed()
                        return
                    client.receive(chunk)
            except Exception as e:
                client.fail(e)
                client.mark_closed()

        client._read_task = asyncio.ensure_future(_read())
        try:
            client.next(lambda m: m["type"] in ("hello", "hello_error"))
            await wait(client.send_message({"type": "hello", "version": PROTOCOL_VERSION}))
            # Release the delay once the handshake has actually reached the
            # stalled `list_sessions`; `entered` is the deterministic seam a
            # fixed sleep was only approximating.
            await wait(d.entered.future)
            d.release.resolve(None)
            await wait(server.close())
            # The handshake may or may not have completed; either outcome is fine.
        finally:
            writer.close()
            client._read_task.cancel()
    finally:
        await wait(server.close())


# --------------------------------------------------------------------------- #
# 244-245: _handle_request() propagates service exception as error response  #
# --------------------------------------------------------------------------- #


async def test_handle_request_returns_error_response_on_exception(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    # "attach" with an unknown session should return a not_found error.
    response = await wait(client.request({"command": "attach", "sessionId": "no-such-session"}))
    assert response["ok"] is False
    assert response["error"]["code"] == "not_found"


# --------------------------------------------------------------------------- #
# 273-277: _transport_closed() when decoder.end() raises                     #
# --------------------------------------------------------------------------- #


async def test_transport_closed_with_truncated_frame(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    # Send a partial frame then close the connection.
    await wait(client.send_bytes(bytes([0, 0, 0, 4, 0xA0])))  # header says 4 bytes, only 1 follows
    await wait(client.close())
    await wait_until(lambda: client.closed, "the truncated frame closed the connection")
    assert client.closed


# --------------------------------------------------------------------------- #
# 281: _disconnect() when already disconnected is a no-op                    #
# --------------------------------------------------------------------------- #


async def test_disconnect_idempotent(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    await wait(client.close())
    await wait_until(
        lambda: all(cs.disconnected for cs in started.server._connections),
        "the server observed the client disconnect",
    )
    # Calling disconnect again via the server must not raise.
    for cs in list(started.server._connections):
        await wait(started.server._disconnect(cs))


# --------------------------------------------------------------------------- #
# 296-297: _send_message() when encode_server_message raises                 #
# --------------------------------------------------------------------------- #


async def test_send_message_handles_encode_error(harness: ServerHarness) -> None:
    """Inject an encode failure: server must report the error and close the connection."""
    started = await harness.start_server()
    errors: list[Exception] = []
    started.server._on_error = lambda e: errors.append(e)

    client = await harness.connect(started.server)
    await wait(client.hello())

    from unittest.mock import patch

    call_count = [0]
    real_encode = __import__("pi_protocol").encode_server_message

    def patched_encode(msg: Any, **kw: Any) -> bytes:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("encode failed")
        return real_encode(msg, **kw)

    import contextlib

    with patch("pi_server.server.encode_server_message", side_effect=patched_encode), contextlib.suppress(Exception):
        await wait(client.request({"command": "list"}))

    await wait_until(
        lambda: len(errors) >= 1 or client.closed,
        "the encode failure was reported or closed the connection",
    )
    assert len(errors) >= 1 or client.closed


# --------------------------------------------------------------------------- #
# 308-309: _send_message() when connection.send() raises                     #
# --------------------------------------------------------------------------- #


async def test_send_message_handles_send_error(harness: ServerHarness) -> None:
    started = await harness.start_server()
    errors: list[Exception] = []
    started.server._on_error = lambda e: errors.append(e)

    client = await harness.connect(started.server)
    await wait(client.hello())

    # Find the ConnectionState and replace its connection with one whose send() raises.
    states = list(started.server._connections)
    assert len(states) == 1
    cs = states[0]

    original_connection = cs.connection

    class _FailSend:
        closed = original_connection.closed

        def send(self, chunk: bytes) -> asyncio.Future[None]:
            raise RuntimeError("send failed")

        def close(self, final_chunk: bytes | None = None) -> None:
            original_connection.close(final_chunk)

    cs.connection = _FailSend()  # type: ignore[assignment]
    # Sending any message should trigger the error path.
    await wait(
        started.server._send_message(
            cs, {"type": "event", "event": {"type": "server_snapshot", "snapshot": {"test": 1}}}
        )
    )
    await wait_until(lambda: cs.disconnected, "the failing send disconnected the connection")
    assert cs.disconnected


# --------------------------------------------------------------------------- #
# 320: _fail_protocol() when already disconnected is a no-op                 #
# --------------------------------------------------------------------------- #


async def test_fail_protocol_when_disconnected_is_noop(harness: ServerHarness) -> None:
    started = await harness.start_server()
    client = await harness.connect(started.server)
    await wait(client.hello())
    await wait(client.close())
    await wait_until(
        lambda: all(cs.disconnected for cs in started.server._connections),
        "the server observed the client disconnect",
    )
    # All connections should be gone; calling _fail_protocol on a fake
    # disconnected state must be a no-op.
    import uuid

    from pi_protocol import ClientMessageDecoder
    from pi_server.connection import ConnectionState

    loop = asyncio.get_event_loop()
    fake_handle = loop.call_later(60, lambda: None)
    fake_handle.cancel()

    class _ClosedConn:
        closed = True

        def send(self, chunk: bytes) -> asyncio.Future[None]:
            f: asyncio.Future[None] = loop.create_future()
            f.set_result(None)
            return f

        def close(self, final_chunk: bytes | None = None) -> None:
            pass

    fake_state = ConnectionState(
        id=str(uuid.uuid4()),
        connection=_ClosedConn(),  # type: ignore[arg-type]
        decoder=ClientMessageDecoder(),
        handshake_timeout_handle=loop.call_later(60, lambda: None),
    )
    fake_state.disconnected = True
    await wait(started.server._fail_protocol(fake_state, {"code": "invalid_request", "message": "test"}))


# --------------------------------------------------------------------------- #
# 335, 338, 341, 344: _to_protocol_error() branches                          #
# --------------------------------------------------------------------------- #


def test_to_protocol_error_internal_server_error() -> None:
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    errors: list[Exception] = []
    server._on_error = lambda e: errors.append(e)
    cause = ValueError("root cause")
    err = InternalServerError(cause)
    err.__cause__ = cause
    result = server._to_protocol_error(err)
    assert result["code"] == "internal_error"
    assert len(errors) == 1


def test_to_protocol_error_not_implemented() -> None:
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    err = NotImplementedProtocolError()
    result = server._to_protocol_error(err)
    assert result["code"] == "not_implemented"


def test_to_protocol_error_pi_server_error_with_details() -> None:
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    err = PiServerError("not_found", "not found", details={"extra": "info"})
    result = server._to_protocol_error(err)
    assert result["code"] == "not_found"
    assert result["details"] == {"extra": "info"}


def test_to_protocol_error_pi_server_error_no_details() -> None:
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    err = PiServerError("not_found", "not found")
    result = server._to_protocol_error(err)
    assert result["code"] == "not_found"
    assert "details" not in result


def test_to_protocol_error_generic_exception() -> None:
    errors: list[Exception] = []
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    server._on_error = lambda e: errors.append(e)
    result = server._to_protocol_error(RuntimeError("unexpected"))
    assert result["code"] == "internal_error"
    assert len(errors) == 1


# --------------------------------------------------------------------------- #
# 373: _report_error() when on_error is None                                  #
# --------------------------------------------------------------------------- #


def test_report_error_when_on_error_is_none() -> None:
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    assert server._on_error is None
    server._report_error(RuntimeError("ignored"))  # must not raise


# --------------------------------------------------------------------------- #
# 376: _NullHandler.on_error() reports via report_error                      #
# --------------------------------------------------------------------------- #


async def test_null_handler_on_error_reports() -> None:
    errors: list[Exception] = []
    server = PiServer(_service(), PiServerOptions(listeners=[]))
    server._on_error = lambda e: errors.append(e)
    server._closing = True  # so accept() returns NullHandler

    class _MockConn:
        closed = False

        def send(self, chunk: bytes) -> asyncio.Future[None]:
            f: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            f.set_result(None)
            return f

        def close(self, final_chunk: bytes | None = None) -> None:
            self.closed = True

    handler = server.accept(_MockConn())  # type: ignore[arg-type]
    handler.on_error(RuntimeError("null handler error"))
    assert len(errors) == 1
    assert "null handler error" in str(errors[0])
    await wait(server.close())


# --------------------------------------------------------------------------- #
# _resolve_options: empty server_id                                           #
# --------------------------------------------------------------------------- #


def test_empty_server_id_raises() -> None:
    with pytest.raises(TypeError, match="serverId"):
        PiServer(_service(), PiServerOptions(listeners=[], server_id=""))


# --------------------------------------------------------------------------- #
# close() when already closing awaits same task                               #
# --------------------------------------------------------------------------- #


async def test_close_twice_awaits_same_close_task(harness: ServerHarness) -> None:
    started = await harness.start_server()
    c1 = asyncio.ensure_future(started.server.close())
    c2 = asyncio.ensure_future(started.server.close())
    await wait(asyncio.gather(c1, c2))
    assert started.server.addresses == []
