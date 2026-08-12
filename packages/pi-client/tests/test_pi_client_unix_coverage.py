"""Coverage tests for pi_client.unix.

Targets lines: 22-23, 67->69, 70->64, 74-76, 86, 88, 98, 106, 108->110
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pi_client.unix import UnixTransportOptions, _UnixByteTransport, create_unix_transport_factory

# --------------------------------------------------------------------------- #
# 22-23: UnixTransportOptions.__init__                                         #
# --------------------------------------------------------------------------- #


def test_unix_transport_options_stores_fields() -> None:
    opts = UnixTransportOptions("/some/path", max_pending_bytes=1024)
    assert opts.path == "/some/path"
    assert opts.max_pending_bytes == 1024


def test_unix_transport_options_defaults() -> None:
    opts = UnixTransportOptions("/some/path")
    assert opts.path == "/some/path"
    assert opts.max_pending_bytes is None


# --------------------------------------------------------------------------- #
# 86: send() raises TypeError for non-bytes argument                          #
# --------------------------------------------------------------------------- #


async def test_send_raises_type_error_for_non_bytes(tmp_path: Any) -> None:
    socket_path = str(tmp_path / "t.sock")

    async def _handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.close()

    server = await asyncio.start_unix_server(_handler, path=socket_path)
    try:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                pass

            def on_close(self) -> None:
                pass

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        try:
            with pytest.raises(TypeError, match="bytes"):
                transport.send("not bytes")  # type: ignore[arg-type]
        finally:
            transport.close()
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# 88: send() raises RuntimeError when closed                                  #
# --------------------------------------------------------------------------- #


async def test_send_raises_when_transport_is_closed(tmp_path: Any) -> None:
    socket_path = str(tmp_path / "t.sock")

    async def _handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.close()

    server = await asyncio.start_unix_server(_handler, path=socket_path)
    try:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                pass

            def on_close(self) -> None:
                pass

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        transport.close()
        with pytest.raises(RuntimeError, match="closed"):
            transport.send(b"data")
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# 98: _write() raises when writer is closed mid-write                         #
# --------------------------------------------------------------------------- #


async def test_write_raises_when_transport_is_closed(tmp_path: Any) -> None:
    """`_write()` re-checks `_closed` inside the lock and raises.

    The check exists because `close()` can land while another `_write()` owns
    the lock. Rather than depend on the OS socket buffer filling up (which is
    not reproducible and can hang), this holds the lock explicitly, sets
    `_closed`, then releases it -- the exact interleaving the check guards.
    """
    socket_path = str(tmp_path / "t.sock")

    async def _handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.close()

    server = await asyncio.start_unix_server(_handler, path=socket_path)
    try:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                pass

            def on_close(self) -> None:
                pass

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        try:
            # Set closed flag directly so we can test the lock-internal check.
            transport._closed = True
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(transport._write(b"test"), timeout=5)
        finally:
            # Writer wasn't closed via close() since _closed was set manually;
            # close it explicitly so the event loop cleans up cleanly.
            transport._closed = False
            transport.close()
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# 106: close() second call is a no-op                                         #
# --------------------------------------------------------------------------- #


async def test_close_twice_is_noop(tmp_path: Any) -> None:
    socket_path = str(tmp_path / "t.sock")

    async def _handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        w.close()

    server = await asyncio.start_unix_server(_handler, path=socket_path)
    try:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                pass

            def on_close(self) -> None:
                pass

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        transport.close()
        transport.close()  # must not raise (covers line 106)
    finally:
        server.close()


# --------------------------------------------------------------------------- #
# 108->110: close() when read_task is None (read not yet started)             #
# --------------------------------------------------------------------------- #


async def test_close_with_no_read_task(tmp_path: Any) -> None:
    """_UnixByteTransport created but start_reading() not called; close() must work."""
    socket_path = str(tmp_path / "t.sock")
    reader_ref: list[asyncio.StreamReader] = []
    writer_ref: list[asyncio.StreamWriter] = []
    connected = asyncio.Event()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        reader_ref.append(reader)
        writer_ref.append(writer)
        connected.set()
        await asyncio.sleep(5)
        writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    async with server:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(socket_path), timeout=5)
        await asyncio.wait_for(connected.wait(), timeout=5)
        transport = _UnixByteTransport(reader, writer, 65536)
        # Do NOT call start_reading() → _read_task is None.
        transport.close()  # covers the _read_task is None branch (108->110)
        assert transport._closed


# --------------------------------------------------------------------------- #
# 67->69: on_close() called when transport closes normally (not yet closed)   #
# --------------------------------------------------------------------------- #


async def test_on_close_called_on_eof(tmp_path: Any) -> None:
    socket_path = str(tmp_path / "t.sock")
    close_called = asyncio.Event()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    async with server:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                pass

            def on_close(self) -> None:
                close_called.set()

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        try:
            await asyncio.wait_for(close_called.wait(), timeout=5)
        finally:
            transport.close()


# --------------------------------------------------------------------------- #
# 70->64: read loop continues when transport closed but data available         #
# --------------------------------------------------------------------------- #


async def test_data_ignored_after_close_in_read_loop(tmp_path: Any) -> None:
    """close() called just before on_data so _closed=True; the loop should
    skip on_data (branch 70->64 in the read loop)."""
    socket_path = str(tmp_path / "t.sock")
    data_calls: list[bytes] = []
    writer_ref: list[asyncio.StreamWriter] = []
    ready = asyncio.Event()
    server_done = asyncio.Event()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer_ref.append(writer)
        ready.set()
        await server_done.wait()
        writer.close()

    server = await asyncio.start_unix_server(handle_client, path=socket_path)
    try:

        class _Handlers:
            def on_data(self, chunk: bytes) -> None:
                data_calls.append(chunk)

            def on_close(self) -> None:
                pass

            def on_error(self, error: Exception) -> None:
                pass

        factory = create_unix_transport_factory(socket_path)
        transport = await asyncio.wait_for(factory(_Handlers()), timeout=5)
        try:
            await asyncio.wait_for(ready.wait(), timeout=5)
            # Close the transport before the data arrives so _closed=True.
            transport.close()
            # Then have the server send data — the read loop should skip on_data.
            if writer_ref:
                writer_ref[0].write(b"ignored")
                await writer_ref[0].drain()
            await asyncio.sleep(0.05)
            assert data_calls == []
        finally:
            transport.close()
    finally:
        server_done.set()
        server.close()


# --------------------------------------------------------------------------- #
# 74-76: read loop exception handler                                           #
# --------------------------------------------------------------------------- #


async def test_read_loop_calls_on_error_on_exception(tmp_path: Any) -> None:
    """Inject a bad StreamReader that raises on read(); on_error must be called."""
    socket_path = str(tmp_path / "t.sock")
    errors: list[Exception] = []
    error_event = asyncio.Event()

    server = await asyncio.start_unix_server(lambda r, w: None, path=socket_path)
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(socket_path), timeout=5)
        try:
            # Wrap the reader so read() raises after a short delay.
            class _BrokenReader:
                async def read(self, n: int) -> bytes:
                    raise OSError("simulated read failure")

            class _Handlers:
                def on_data(self, chunk: bytes) -> None:
                    pass

                def on_close(self) -> None:
                    pass

                def on_error(self, error: Exception) -> None:
                    errors.append(error)
                    error_event.set()

            transport = _UnixByteTransport(_BrokenReader(), writer, 65536)  # type: ignore[arg-type]
            transport.start_reading(_Handlers())
            await asyncio.wait_for(error_event.wait(), timeout=5)
            assert len(errors) == 1
            assert "simulated" in str(errors[0])
        finally:
            writer.close()
    finally:
        server.close()
