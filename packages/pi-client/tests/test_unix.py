"""Unix-domain socket transport tests.

Python port of `packages/client/test/unix.test.ts`, built on
`asyncio.start_unix_server` instead of `node:net`.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from pi_client import PiClient, PiClientOptions
from pi_client.unix import create_unix_transport_factory
from pi_protocol import PROTOCOL_VERSION, ClientMessageDecoder, encode_server_message

server_snapshot = {
    "serverId": "unix-server",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 4,
    "sessions": [],
    "models": [],
}


def test_rejects_invalid_unix_transport_options():
    with pytest.raises(TypeError, match="must not be empty"):
        create_unix_transport_factory("")
    with pytest.raises(TypeError, match="too long"):
        create_unix_transport_factory("/tmp/" + "x" * 512)
    with pytest.raises(TypeError, match="positive"):
        create_unix_transport_factory("/tmp/pi.sock", max_pending_bytes=0)


@pytest.fixture
def socket_dir():
    """A short directory for Unix sockets.

    A Unix socket path is capped at 107 bytes. pytest's `tmp_path` embeds the
    test name, and under `pytest-xdist` a worker id too, which together
    overflow that limit and fail these tests for a reason unrelated to what
    they check.
    """
    root = tempfile.mkdtemp(prefix="pi-c-")
    try:
        yield Path(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
async def test_exchanges_fragmented_framed_messages_over_a_real_unix_socket(socket_dir):
    socket_path = str(socket_dir / "pi.sock")

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        decoder = ClientMessageDecoder()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    return
                for message in decoder.push(chunk):
                    if message["type"] == "hello":
                        hello = encode_server_message(
                            {
                                "type": "hello",
                                "version": PROTOCOL_VERSION,
                                "connectionId": "unix-connection",
                                "snapshot": server_snapshot,
                            }
                        )
                        for byte in hello:
                            writer.write(bytes([byte]))
                            await writer.drain()
                    else:
                        response = encode_server_message(
                            {
                                "type": "response",
                                "id": message["id"],
                                "ok": True,
                                "result": {"command": "list", "sessions": []},
                            }
                        )
                        split = len(response) // 2
                        writer.write(response[:split])
                        await writer.drain()
                        writer.write(response[split:])
                        await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    async with server:
        client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(socket_path)))
        try:
            snapshot = await asyncio.wait_for(client.connect(), timeout=5)
            assert snapshot == server_snapshot
            results = await asyncio.wait_for(asyncio.gather(client.list_sessions(), client.list_sessions()), timeout=5)
            assert results == [[], []]
        finally:
            client.disconnect()


@pytest.mark.asyncio
async def test_bounds_pending_writes_preserves_order_and_reports_remote_end_once(socket_dir):
    socket_path = str(socket_dir / "pi.sock")
    # Matches the TS test's 2 MiB chunks: large enough that the kernel socket
    # buffer cannot absorb a single write synchronously, so `drain()` actually
    # suspends and both sends stay genuinely pending until `resume_event` fires.
    first = bytes([1]) * (2 * 1024 * 1024)
    second = bytes([2]) * (2 * 1024 * 1024)
    expected_length = len(first) + len(second)
    state = {"received_length": 0, "invalid_order": False}
    received = asyncio.get_event_loop().create_future()
    server_ready = asyncio.get_event_loop().create_future()
    resume_event = asyncio.Event()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not server_ready.done():
            server_ready.set_result(None)
        await resume_event.wait()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    return
                for index, byte in enumerate(chunk):
                    expected = 1 if state["received_length"] + index < len(first) else 2
                    if byte != expected:
                        state["invalid_order"] = True
                state["received_length"] += len(chunk)
                if state["received_length"] >= expected_length:
                    writer.write(bytes([9]))
                    await writer.drain()
                    writer.write_eof()
                    if not received.done():
                        received.set_result(None)
                    return
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    async with server:
        inbound: list[int] = []
        errors: list[Exception] = []
        closed = asyncio.get_event_loop().create_future()

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                inbound.extend(chunk)

            def on_close(self) -> None:
                if not closed.done():
                    closed.set_result(None)

            def on_error(self, error: Exception) -> None:
                errors.append(error)

        transport = await create_unix_transport_factory(socket_path, max_pending_bytes=expected_length)(_Handlers())

        try:
            await asyncio.wait_for(server_ready, timeout=5)
            first_write = transport.send(first)
            second_write = transport.send(second)
            await asyncio.sleep(0.01)
            with pytest.raises(Exception, match="pending byte limit"):
                await asyncio.wait_for(transport.send(bytes([3])), timeout=5)
            resume_event.set()
            await asyncio.wait_for(asyncio.gather(first_write, second_write, received, closed), timeout=5)
            assert state["received_length"] == expected_length
            assert state["invalid_order"] is False
            assert inbound == [9]
            assert errors == []
        finally:
            transport.close()


@pytest.mark.asyncio
async def test_rejects_a_truncated_final_frame_from_a_real_unix_socket(socket_dir):
    socket_path = str(socket_dir / "pi.sock")

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        decoder = ClientMessageDecoder()
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    return
                for message in decoder.push(chunk):
                    if message["type"] == "hello":
                        writer.write(
                            encode_server_message(
                                {
                                    "type": "hello",
                                    "version": PROTOCOL_VERSION,
                                    "connectionId": "unix-truncated",
                                    "snapshot": server_snapshot,
                                }
                            )
                        )
                        await writer.drain()
                    else:
                        writer.write(bytes([0, 0, 0, 2, 1]))
                        await writer.drain()
                        writer.write_eof()
                        return
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    async with server:
        client = PiClient(PiClientOptions(transport_factory=create_unix_transport_factory(socket_path)))
        try:
            await asyncio.wait_for(client.connect(), timeout=5)
            with pytest.raises(Exception, match=r"ProtocolValidationError|frame|truncat"):
                await asyncio.wait_for(client.list_sessions(), timeout=5)
            assert client.connection_state == "disconnected"
        finally:
            client.disconnect()


@pytest.mark.asyncio
async def test_rejects_connection_errors(socket_dir):
    missing_path = str(socket_dir / "missing.sock")
    errors: list[Exception] = []

    class _Handlers:
        def on_data(self, chunk: bytes) -> None:
            pass

        def on_close(self) -> None:
            pass

        def on_error(self, error: Exception) -> None:
            errors.append(error)

    with pytest.raises(FileNotFoundError):
        await asyncio.wait_for(create_unix_transport_factory(missing_path)(_Handlers()), timeout=5)
    assert not os.path.exists(missing_path)
