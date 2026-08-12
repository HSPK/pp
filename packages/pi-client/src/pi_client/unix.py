"""Unix-domain-socket transport for `PiClient`.

Python port of `packages/client/src/unix.ts`, built on
`asyncio.open_unix_connection` instead of `node:net`.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from pi_protocol import DEFAULT_MAX_FRAME_LENGTH

from .transport import ByteTransport, ByteTransportHandlers

MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform.startswith("linux") else 103


class UnixTransportOptions:
    def __init__(self, path: str, max_pending_bytes: int | None = None) -> None:
        self.path = path
        self.max_pending_bytes = max_pending_bytes


def create_unix_transport_factory(path: str, max_pending_bytes: int | None = None):
    """Creates fresh Unix-domain socket transports for `PiClient` connection attempts."""
    if len(path) == 0:
        raise TypeError("Unix transport path must not be empty")
    if len(path.encode("utf-8")) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"Unix transport path is too long; maximum is {MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")
    resolved_max_pending_bytes = DEFAULT_MAX_FRAME_LENGTH * 4 if max_pending_bytes is None else max_pending_bytes
    if not isinstance(resolved_max_pending_bytes, int) or resolved_max_pending_bytes <= 0:
        raise TypeError("Unix transport maxPendingBytes must be a positive integer")

    async def factory(handlers: ByteTransportHandlers) -> ByteTransport:
        return await _connect_unix_socket(path, resolved_max_pending_bytes, handlers)

    return factory


async def _connect_unix_socket(path: str, max_pending_bytes: int, handlers: ByteTransportHandlers) -> ByteTransport:
    reader, writer = await asyncio.open_unix_connection(path)
    transport = _UnixByteTransport(reader, writer, max_pending_bytes)
    transport.start_reading(handlers)
    return transport


class _UnixByteTransport:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, max_pending_bytes: int) -> None:
        self._reader = reader
        self._writer = writer
        self._max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._read_task: asyncio.Task[None] | None = None

    def start_reading(self, handlers: ByteTransportHandlers) -> None:
        self._read_task = asyncio.ensure_future(self._read_loop(handlers))

    async def _read_loop(self, handlers: ByteTransportHandlers) -> None:
        try:
            while True:
                chunk = await self._reader.read(65536)
                if not chunk:
                    if not self._closed:
                        handlers.on_close()
                    return
                if not self._closed:
                    handlers.on_data(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._closed:
                handlers.on_error(error if isinstance(error, Exception) else Exception(str(error)))

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        # The bounds check and byte reservation happen synchronously (before any
        # `await`), mirroring the JS `Promise`-returning `send()` whose body runs
        # immediately up to its first `await`. Python coroutines are lazy, so if
        # this logic lived inside an `async def` body it would only run once the
        # returned coroutine/task is scheduled, letting concurrent unawaited
        # callers race past the pending-byte limit.
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("Unix transport chunks must be bytes")
        if self._closed:
            raise RuntimeError("Unix transport is closed")
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            raise RuntimeError("Unix transport exceeded its pending byte limit")
        self._pending_bytes += len(chunk)
        return asyncio.ensure_future(self._write(bytes(chunk)))

    async def _write(self, chunk: bytes) -> None:
        try:
            async with self._write_lock:
                if self._closed:
                    raise RuntimeError("Unix transport is closed")
                self._writer.write(chunk)
                await self._writer.drain()
        finally:
            self._pending_bytes -= len(chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._read_task is not None:
            self._read_task.cancel()
        with contextlib.suppress(Exception):
            self._writer.close()
