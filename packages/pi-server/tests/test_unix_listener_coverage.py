"""Coverage tests for pi_server.transports.unix.listener.

Targets lines: 44, 88, 90, 98-99, 103-106, 111-112, 119-120, 151-152,
               167->172, 209, 211, 213, 222, 236, 247-248, 256-258,
               271-276, 280-287, 303, 306
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server
from pi_server.transports.unix.listener import (
    UnixByteConnection,
    UnixListenerOptions,
    _is_socket_live,
    _remove_stale_socket,
    _set_socket_mode,
    create_unix_listener,
    validate_unix_socket_path,
)


def _service() -> TestServerService:
    return TestServerService()


# --------------------------------------------------------------------------- #
# 44: validate_unix_socket_path with empty string                             #
# --------------------------------------------------------------------------- #


def test_validate_unix_socket_path_empty_raises() -> None:
    with pytest.raises(TypeError, match="must not be empty"):
        validate_unix_socket_path("", "MyPath")


def test_validate_unix_socket_path_happy() -> None:
    # Should not raise.
    validate_unix_socket_path("/tmp/pi.sock")


# --------------------------------------------------------------------------- #
# 88, 90: start() when already started or closing                             #
# --------------------------------------------------------------------------- #


async def test_start_when_already_started_raises(socket_dir: Any) -> None:
    path = str(socket_dir / "a.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    finally:
        await asyncio.wait_for(listener.close(), timeout=5)


async def test_start_when_closing_raises(socket_dir: Any) -> None:
    path = str(socket_dir / "b.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    close_task = asyncio.ensure_future(listener.close())
    listener._closing = True
    # Attempt to start a NEW listener on same path to exercise the "closing" guard.
    listener2 = create_unix_listener(UnixListenerOptions(path=str(socket_dir / "c.sock")))
    listener2._closing = True
    with pytest.raises(RuntimeError, match="closing"):
        await asyncio.wait_for(listener2.start(lambda c: None), timeout=5)
    await asyncio.wait_for(close_task, timeout=5)


# --------------------------------------------------------------------------- #
# 98-99: start() when asyncio.start_unix_server raises (socket path error)   #
# --------------------------------------------------------------------------- #


async def test_start_raises_when_bind_fails() -> None:
    # Use a path inside a non-existent nested directory that cannot be created
    # because the parent is a file, not a directory. The directory is a short
    # private temp dir rather than `tmp_path`: this test is about bind failing,
    # and a long `tmp_path` (longer still under `pytest-xdist`) would trip the
    # 107-byte socket-path check first and pass for the wrong reason.
    root = tempfile.mkdtemp(prefix="pi-b-")
    file_path = str(Path(root) / "notadir")
    with open(file_path, "w") as f:
        f.write("x")
    bad_path = str(Path(root) / "notadir" / "pi.sock")
    listener = create_unix_listener(UnixListenerOptions(path=bad_path))
    try:
        with pytest.raises(OSError):
            await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 103-106: start() when chmod fails                                           #
# --------------------------------------------------------------------------- #


async def test_start_cleanup_when_chmod_fails(socket_dir: Any) -> None:
    path = str(socket_dir / "mode.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))

    def raise_on_chmod(p: str, mode: int) -> None:
        raise OSError("chmod not allowed")

    with (
        patch("pi_server.transports.unix.listener.os.chmod", side_effect=raise_on_chmod),
        pytest.raises(OSError, match="chmod"),
    ):
        await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    # After failure, the socket file should be cleaned up and server set to None.
    assert listener._server is None
    assert not os.path.exists(path)


# --------------------------------------------------------------------------- #
# 111-112: _accept_socket() when closing                                      #
# --------------------------------------------------------------------------- #


async def test_accept_socket_closes_connection_when_listener_is_closing(socket_dir: Any) -> None:
    path = str(socket_dir / "closing.sock")
    server = create_unix_server(_service(), UnixServerOptions(path=path))
    await asyncio.wait_for(server.start(), timeout=5)
    try:
        # Find the UnixListener and mark it as closing.
        listener = server._listeners[0]  # type: ignore[attr-defined]
        listener._closing = True
        # Connect a raw client; the listener should close it immediately.
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        try:
            # The connection should be closed by the server side.
            data = await asyncio.wait_for(reader.read(1024), timeout=3)
            assert data == b""  # EOF — listener closed the socket
        finally:
            writer.close()
    finally:
        await asyncio.wait_for(server.close(), timeout=5)


# --------------------------------------------------------------------------- #
# 119-120: _accept_socket() when accept is None                               #
# --------------------------------------------------------------------------- #


async def test_accept_socket_closes_connection_when_accept_is_none(socket_dir: Any) -> None:
    path = str(socket_dir / "noacc.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    # Start with a real accept so the socket binds, then clear it.
    await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    try:
        listener._accept = None  # Simulate accept being unset.
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        try:
            data = await asyncio.wait_for(reader.read(1024), timeout=3)
            assert data == b""  # EOF — listener closed the socket
        finally:
            writer.close()
    finally:
        await asyncio.wait_for(listener.close(), timeout=5)


# --------------------------------------------------------------------------- #
# 167->172: _close_server_and_cleanup() when server is None (covers else)    #
# --------------------------------------------------------------------------- #


async def test_close_server_and_cleanup_when_server_is_none(socket_dir: Any) -> None:
    path = str(socket_dir / "none.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    # Do not start; call _close_server_and_cleanup directly with server=None.
    await asyncio.wait_for(listener._close_server_and_cleanup(), timeout=5)
    # Must not raise even when _server is None.


# --------------------------------------------------------------------------- #
# 209, 211, 213: _do_close() paths with final_chunk and already-closing writer #
# --------------------------------------------------------------------------- #


async def test_do_close_with_final_chunk(socket_dir: Any) -> None:
    path = str(socket_dir / "fc.sock")
    received: list[bytes] = []
    server_ready = asyncio.Event()
    writer_ref: list[asyncio.StreamWriter] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer_ref.append(writer)
        server_ready.set()
        data = await reader.read(65536)
        if data:
            received.append(data)
        writer.close()

    srv = await asyncio.start_unix_server(handle, path=path)
    async with srv:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        await asyncio.wait_for(server_ready.wait(), timeout=5)
        conn = UnixByteConnection(reader, writer, 1000, 65536)
        # close() with a final_chunk exercises line 213.
        await asyncio.wait_for(conn.close(final_chunk=b"\xff\xfe"), timeout=5)
        assert conn.closed


async def test_do_close_when_writer_already_closing(socket_dir: Any) -> None:
    path = str(socket_dir / "wc.sock")

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await asyncio.sleep(0.1)
        writer.close()

    srv = await asyncio.start_unix_server(handle, path=path)
    async with srv:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        conn = UnixByteConnection(reader, writer, 1000, 65536)
        # Call close() twice to exercise the already-closing branch (line 211).
        f1 = conn.close()
        f2 = conn.close()  # returns the same future
        await asyncio.wait_for(f1, timeout=5)
        await asyncio.wait_for(f2, timeout=5)
        assert conn.closed


# --------------------------------------------------------------------------- #
# 222: mark_closed() when already closed                                      #
# --------------------------------------------------------------------------- #


async def test_mark_closed_idempotent(socket_dir: Any) -> None:
    path = str(socket_dir / "mc.sock")

    srv = await asyncio.start_unix_server(lambda r, w: None, path=path)
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), timeout=5)
        conn = UnixByteConnection(reader, writer, 1000, 65536)
        conn.mark_closed()
        assert conn.closed
        conn.mark_closed()  # second call must be a no-op (covers line 222)
        assert conn.closed
        writer.close()
    finally:
        srv.close()


# --------------------------------------------------------------------------- #
# 236: _remove_stale_socket() for a non-socket file                           #
# --------------------------------------------------------------------------- #


async def test_remove_stale_socket_refuses_non_socket_file(socket_dir: Any) -> None:
    regular_file = str(socket_dir / "regular.txt")
    with open(regular_file, "w") as f:
        f.write("not a socket")
    with pytest.raises(RuntimeError, match="Refusing"):
        await asyncio.wait_for(_remove_stale_socket(regular_file), timeout=5)


# --------------------------------------------------------------------------- #
# 247-248: _is_socket_live() returns True for a live socket                  #
# --------------------------------------------------------------------------- #


async def test_is_socket_live_returns_true_for_live_socket(socket_dir: Any) -> None:
    path = str(socket_dir / "live.sock")
    srv = await asyncio.start_unix_server(lambda r, w: None, path=path)
    try:
        result = await asyncio.wait_for(_is_socket_live(path), timeout=5)
        assert result is True
    finally:
        srv.close()


async def test_is_socket_live_returns_false_for_dead_socket(socket_dir: Any) -> None:
    path = str(socket_dir / "dead.sock")
    # Create a socket file whose path does not exist.
    result = await asyncio.wait_for(_is_socket_live(path), timeout=5)
    assert result is False


# --------------------------------------------------------------------------- #
# 256-258: _set_socket_mode()                                                 #
# --------------------------------------------------------------------------- #


def test_set_socket_mode_is_noop_on_not_implemented(socket_dir: Any) -> None:
    path = str(socket_dir / "mode.sock")
    path_str = str(path)
    with open(path_str, "w") as f:
        f.write("")
    # Patch os.chmod to raise NotImplementedError (Windows-like behaviour).
    with patch("pi_server.transports.unix.listener.os.chmod", side_effect=NotImplementedError):
        # Must not raise.
        _set_socket_mode(path_str, 0o600)


# --------------------------------------------------------------------------- #
# 271-276: _resolve_unix_listener_options() mode validation                  #
# --------------------------------------------------------------------------- #


def test_invalid_mode_raises() -> None:
    with pytest.raises(TypeError, match="mode"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", mode=0o1000))


def test_negative_mode_raises() -> None:
    with pytest.raises(TypeError, match="mode"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", mode=-1))


def test_string_mode_raises() -> None:
    with pytest.raises(TypeError, match="mode"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", mode="rw"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 280-287: max_frame_length and max_pending_bytes validation                  #
# --------------------------------------------------------------------------- #


def test_invalid_max_frame_length_raises() -> None:
    with pytest.raises(TypeError, match="maxFrameLength"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", max_frame_length=0))


def test_max_frame_length_too_large_raises() -> None:
    with pytest.raises(TypeError, match="maxFrameLength"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", max_frame_length=0x1_0000_0001))


def test_max_pending_bytes_too_small_raises() -> None:
    with pytest.raises(TypeError, match="maxPendingBytes"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", max_frame_length=256, max_pending_bytes=10))


# --------------------------------------------------------------------------- #
# 303: graceful_close_timeout_ms validation                                   #
# --------------------------------------------------------------------------- #


def test_graceful_close_timeout_zero_raises() -> None:
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", graceful_close_timeout_ms=0))


def test_graceful_close_timeout_too_large_raises() -> None:
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", graceful_close_timeout_ms=2_147_483_648))


def test_graceful_close_timeout_negative_raises() -> None:
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", graceful_close_timeout_ms=-1))


# --------------------------------------------------------------------------- #
# 306: on_error field stored (option coverage)                                #
# --------------------------------------------------------------------------- #


def test_on_error_option_accepted() -> None:
    errors: list[Exception] = []
    listener = create_unix_listener(UnixListenerOptions(path="/tmp/x.sock", on_error=lambda e: errors.append(e)))
    assert listener._options.on_error is not None


# --------------------------------------------------------------------------- #
# Stale socket cleanup: live socket blocks rebind                             #
# --------------------------------------------------------------------------- #


async def test_start_refuses_to_rebind_live_socket(socket_dir: Any) -> None:
    path = str(socket_dir / "live.sock")
    srv = await asyncio.start_unix_server(lambda r, w: None, path=path)
    try:
        listener = create_unix_listener(UnixListenerOptions(path=path))
        with pytest.raises(RuntimeError, match="already running"):
            await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    finally:
        srv.close()


# --------------------------------------------------------------------------- #
# Stale dead socket file is removed and rebound                               #
# --------------------------------------------------------------------------- #


async def test_start_removes_stale_dead_socket(socket_dir: Any) -> None:
    path = str(socket_dir / "stale.sock")
    # Create a dead socket (not listening).
    import socket as _socket

    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.bind(path)
    s.close()
    assert os.path.exists(path)
    assert stat.S_ISSOCK(os.stat(path).st_mode)

    # UnixListener should remove the stale socket and start listening.
    listener = create_unix_listener(UnixListenerOptions(path=path))
    await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    try:
        assert listener.address == path
    finally:
        await asyncio.wait_for(listener.close(), timeout=5)


# --------------------------------------------------------------------------- #
# close() when already closing delegates to existing close_task              #
# --------------------------------------------------------------------------- #


async def test_close_twice_awaits_same_task(socket_dir: Any) -> None:
    path = str(socket_dir / "ct.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    await asyncio.wait_for(listener.start(lambda c: None), timeout=5)
    c1 = asyncio.ensure_future(listener.close())
    c2 = asyncio.ensure_future(listener.close())
    await asyncio.wait_for(asyncio.gather(c1, c2), timeout=5)
    assert not os.path.exists(path)
