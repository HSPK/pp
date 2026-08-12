"""Unix-domain-socket `PiServerListener`.

Python port of `packages/server/src/transports/unix/listener.ts`, built on
`asyncio.start_unix_server` / `asyncio.open_unix_connection`.

Like the TypeScript original, the listener binds to a private per-process
"owned bind path" (`.p-<hash of path>` next to the requested path) and
hard-links that onto the requested path, so the publicly visible socket file is
created atomically and the listener always knows which inode is its own. It
tracks the bound socket's `(dev, ino)` identity so shutdown never removes a
socket file some *other* process put at the path in the meantime.

Binding a socket the listener creates itself is also what keeps asyncio out of
the way: `asyncio.start_unix_server(path=...)` unconditionally `os.remove()`s
any socket file already at the path before binding (and, from 3.13 on, unlinks
whatever sits there at close time), which would defeat both the stale-socket
check and the identity-guarded cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import os
import socket as socket_module
import stat
import sys
import uuid
from collections.abc import Callable

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH

from ...listener import PiServerListener
from .types import UnixListenerOptions

_DEFAULT_SOCKET_MODE = 0o600
_DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS = 5_000
_MAX_UINT32 = 0xFFFF_FFFF
_MAX_TIMER_DELAY_MS = 2_147_483_647
_SOCKET_PROBE_TIMEOUT_S = 1.0
_MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform.startswith("linux") else 103


def validate_unix_socket_path(path: str, description: str = "Unix socket path") -> None:
    if not path:
        raise TypeError(f"{description} must not be empty")
    if len(path.encode("utf-8")) > _MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"{description} is too long; maximum is {_MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")


class _Resolved:
    __slots__ = ("graceful_close_timeout_ms", "max_pending_bytes", "mode", "on_error", "path")

    def __init__(
        self,
        path: str,
        mode: int,
        graceful_close_timeout_ms: int,
        max_pending_bytes: int,
        on_error: Callable[[Exception], None] | None,
    ) -> None:
        self.path = path
        self.mode = mode
        self.graceful_close_timeout_ms = graceful_close_timeout_ms
        self.max_pending_bytes = max_pending_bytes
        self.on_error = on_error


class UnixListener:
    """`PiServerListener` over a Unix-domain socket."""

    def __init__(self, options: UnixListenerOptions) -> None:
        self._options = _resolve_unix_listener_options(options)
        self._path = self._options.path
        self._mode = self._options.mode
        self._connections: set[UnixByteConnection] = set()
        self._server: asyncio.AbstractServer | None = None
        self._bound_path: str | None = None
        self._bound_identity: tuple[int, int] | None = None
        self._owned_bind_path: str | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._accept = None
        self._pump_tasks: set[asyncio.Task[None]] = set()

    @property
    def address(self) -> str | None:
        return self._bound_path

    async def start(self, accept) -> None:
        if self._server is not None:
            raise RuntimeError("Unix listener is already started")
        if self._closing:
            raise RuntimeError("Unix listener is closing or closed")
        self._accept = accept

        owned_bind_path = _owned_bind_path(self._path)
        validate_unix_socket_path(owned_bind_path, "PiServer private Unix bind path")
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, mode=0o700, exist_ok=True)
        await _remove_stale_socket(self._path)
        await _remove_stale_socket(owned_bind_path)
        self._owned_bind_path = owned_bind_path

        bind_socket = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        try:
            bind_socket.bind(owned_bind_path)
            bind_socket.setblocking(False)
        except BaseException:
            bind_socket.close()
            self._owned_bind_path = None
            raise
        try:
            server = await asyncio.start_unix_server(self._accept_socket, sock=bind_socket)
        except BaseException:
            bind_socket.close()
            _remove_path(owned_bind_path)
            self._owned_bind_path = None
            raise
        self._server = server
        try:
            identity = _socket_identity(owned_bind_path)
            if identity is None:
                raise RuntimeError(f"Unix listener path is not a socket after binding: {owned_bind_path}")
            self._bound_identity = identity
            os.link(owned_bind_path, self._path)
            _set_socket_mode(self._path, self._mode)
        except BaseException:
            await self._close_server_and_cleanup()
            self._server = None
            raise
        self._bound_path = self._path

    def _accept_socket(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing:
            writer.close()
            return
        connection = UnixByteConnection(
            reader, writer, self._options.graceful_close_timeout_ms, self._options.max_pending_bytes
        )
        self._connections.add(connection)
        accept = self._accept
        if accept is None:
            writer.close()
            return
        handler = accept(connection)
        task = asyncio.ensure_future(self._pump(connection, reader, writer, handler))
        self._pump_tasks.add(task)
        task.add_done_callback(self._pump_tasks.discard)

    async def _pump(
        self,
        connection: UnixByteConnection,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler,
    ) -> None:
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                handler.on_data(chunk)
        except Exception as error:
            handler.on_error(error)
        finally:
            connection.mark_closed()
            if not writer.is_closing():
                with contextlib.suppress(Exception):
                    writer.close()
            self._connections.discard(connection)
            handler.on_close()

    async def close(self) -> None:
        if self._close_task is not None:
            await self._close_task
            return
        self._closing = True
        self._close_task = asyncio.ensure_future(self._close_internal())
        await self._close_task

    async def _close_internal(self) -> None:
        self._bound_path = None
        await asyncio.gather(*(connection.close() for connection in list(self._connections)))
        await self._close_server_and_cleanup()
        self._connections.clear()
        self._server = None

    async def _close_server_and_cleanup(self) -> None:
        server = self._server
        try:
            if server is not None:
                server.close()
                with contextlib.suppress(Exception):
                    await server.wait_closed()
        finally:
            owned = self._owned_bind_path
            self._owned_bind_path = None
            try:
                self._cleanup_owned_socket()
            finally:
                if owned is not None:
                    _remove_path(owned)

    def _cleanup_owned_socket(self) -> None:
        """Removes the socket file only while it is still the one this listener bound.

        A restart (or anything else) may have replaced the inode at this path
        after the bind; removing that replacement would destroy another
        process's socket, so the recorded `(dev, ino)` identity is rechecked,
        and the removal itself goes through a rename so the check cannot be
        raced between the `lstat` and the `unlink`.
        """
        identity = self._bound_identity
        self._bound_identity = None
        if identity is None:
            return
        if _socket_identity(self._path) != identity:
            return

        preserved = os.path.join(os.path.dirname(self._path) or ".", f".c-{uuid.uuid4().hex[:6]}")
        try:
            os.rename(self._path, preserved)
        except FileNotFoundError:
            return
        moved = _socket_identity(preserved)
        if moved == identity:
            _remove_path(preserved)
            return
        if not os.path.lexists(self._path):
            os.rename(preserved, self._path)
        raise RuntimeError(f"Unix listener path changed during cleanup; preserved replacement at {preserved}")


class UnixByteConnection:
    """`ByteConnection` over one accepted Unix-domain socket connection.

    Exported for transport-level test verification, mirroring the TS export.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        graceful_close_timeout_ms: int,
        max_pending_bytes: int,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._graceful_close_timeout_ms = graceful_close_timeout_ms
        self._max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self._closed_value = False
        self._closing = False
        self._write_lock = asyncio.Lock()
        self._close_future: asyncio.Future[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed_value

    def send(self, chunk: bytes):
        # Bounds-check and byte-count reservation happen synchronously, before
        # any `await`, so concurrent unawaited `send()` calls are ordered and
        # bounded correctly (see `pi_client/unix.py` for the same pattern and
        # rationale).
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("Unix connection chunks must be bytes")
        if self._closed_value or self._closing:
            raise RuntimeError("Unix connection is closed")
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            raise RuntimeError("Unix connection exceeded its pending byte limit")
        self._pending_bytes += len(chunk)
        data = bytes(chunk)
        return asyncio.ensure_future(self._write(data))

    async def _write(self, chunk: bytes) -> None:
        try:
            async with self._write_lock:
                if self._closed_value or self._writer.is_closing():
                    raise RuntimeError("Unix connection is closed")
                self._writer.write(chunk)
                await self._writer.drain()
        finally:
            self._pending_bytes -= len(chunk)

    def close(self, final_chunk: bytes | None = None):
        loop = asyncio.get_event_loop()
        if self._closed_value or self._writer.is_closing():
            self.mark_closed()
            future: asyncio.Future[None] = loop.create_future()
            future.set_result(None)
            return future
        if self._close_future is not None:
            return self._close_future
        self._closing = True
        self._close_future = asyncio.ensure_future(self._do_close(final_chunk))
        return self._close_future

    async def _do_close(self, final_chunk: bytes | None) -> None:
        timeout = self._graceful_close_timeout_ms / 1000
        try:
            async with asyncio.timeout(timeout):
                async with self._write_lock:
                    pass
        except TimeoutError:
            pass
        if not self._writer.is_closing():
            try:
                if final_chunk:
                    self._writer.write(final_chunk)
                self._writer.close()
                async with asyncio.timeout(timeout):
                    await self._writer.wait_closed()
            except Exception:
                with contextlib.suppress(Exception):
                    self._writer.close()
        self.mark_closed()

    def mark_closed(self) -> None:
        if self._closed_value:
            return
        self._closed_value = True
        self._closing = True


async def _remove_stale_socket(path: str) -> None:
    try:
        original = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(original.st_mode):
        raise RuntimeError(f"Refusing to remove non-socket Unix listener path: {path}")
    if await _is_socket_live(path):
        raise RuntimeError(f"Unix listener is already running: {path}")

    identity = (original.st_dev, original.st_ino)
    preserved = os.path.join(os.path.dirname(path) or ".", f".s-{uuid.uuid4().hex[:6]}")
    try:
        os.rename(path, preserved)
    except FileNotFoundError:
        return
    if _socket_identity(preserved) != identity:
        if not os.path.lexists(path):
            os.rename(preserved, path)
        raise RuntimeError(f"Unix listener path changed while checking for a stale socket: {path}")
    _remove_path(preserved)


def _owned_bind_path(path: str) -> str:
    suffix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    return os.path.join(os.path.dirname(path) or ".", f".p-{suffix}")


def _remove_path(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _socket_identity(path: str) -> tuple[int, int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISSOCK(info.st_mode):
        return None
    return (info.st_dev, info.st_ino)


async def _is_socket_live(path: str) -> bool:
    try:
        _reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(path), _SOCKET_PROBE_TIMEOUT_S)
    except TimeoutError:
        # A socket that accepts the connection but never completes it is still
        # owned by a live process; TypeScript's probe resolves `true` here.
        return True
    except (ConnectionRefusedError, FileNotFoundError, BrokenPipeError, ConnectionResetError):
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


def _set_socket_mode(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except NotImplementedError:
        return
    except OSError as error:
        if error.errno not in (errno.ENOSYS, errno.ENOTSUP):
            raise


def create_unix_listener(options: UnixListenerOptions) -> PiServerListener:
    return UnixListener(options)


def _resolve_unix_listener_options(options: UnixListenerOptions) -> _Resolved:
    validate_unix_socket_path(options.path, "PiServer Unix socket path")
    mode = options.mode if options.mode is not None else _DEFAULT_SOCKET_MODE
    if not isinstance(mode, int) or mode < 0 or mode > 0o777:
        raise TypeError("PiServer Unix socket mode must be an integer between 0 and 0o777")
    max_frame_length = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
    if not isinstance(max_frame_length, int) or max_frame_length <= 0 or max_frame_length > _MAX_UINT32:
        raise TypeError(f"PiServer maxFrameLength must be an integer between 1 and {_MAX_UINT32}")
    max_pending_bytes = options.max_pending_bytes if options.max_pending_bytes is not None else max_frame_length * 4
    if not isinstance(max_pending_bytes, int) or max_pending_bytes < max_frame_length + 4:
        raise TypeError("PiServer maxPendingBytes must be an integer at least maxFrameLength + 4")
    graceful_close_timeout_ms = (
        options.graceful_close_timeout_ms
        if options.graceful_close_timeout_ms is not None
        else _DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS
    )
    if (
        not isinstance(graceful_close_timeout_ms, int)
        or graceful_close_timeout_ms <= 0
        or graceful_close_timeout_ms > _MAX_TIMER_DELAY_MS
    ):
        raise TypeError(f"PiServer gracefulCloseTimeoutMs must be an integer between 1 and {_MAX_TIMER_DELAY_MS}")
    return _Resolved(
        path=options.path,
        mode=mode,
        graceful_close_timeout_ms=graceful_close_timeout_ms,
        max_pending_bytes=max_pending_bytes,
        on_error=options.on_error,
    )
