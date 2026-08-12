"""Port of `packages/server/test/unix-connection.test.ts`.

The TS test drives `UnixByteConnection` against a hand-rolled `net.Socket`
stub with a controllable write callback. Python's `asyncio.StreamWriter` has
no equivalent seam to intercept, so this port exercises the same guarantee
(the final chunk passed to `close()` is only written after in-flight writes
complete, and the connection eventually closes) against a real Unix-domain
socket with a deliberately slow reader to create genuine backpressure,
following the "size is load-bearing" lesson from `pi_client`'s Unix tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

from conftest import wait, wait_until
from pi_protocol import ServerMessageDecoder, encode_server_message
from pi_server.transports.unix import UnixListenerOptions, create_unix_listener

_LARGE_CHUNK = bytes(range(256)) * (2 * 1024 * 1024 // 256)  # 2 MiB, exceeds the kernel socket buffer


async def test_queues_a_final_protocol_error_behind_pending_output_before_closing(socket_dir: Any) -> None:
    path = str(socket_dir / "server.sock")
    listener = create_unix_listener(UnixListenerOptions(path=path))
    connection_ref: list[Any] = []

    class _Handler:
        def on_data(self, chunk: bytes) -> None:
            pass

        def on_close(self) -> None:
            pass

        def on_error(self, error: Exception) -> None:
            pass

    def accept(connection: Any) -> Any:
        connection_ref.append(connection)
        return _Handler()

    await wait(listener.start(accept))
    try:
        reader, writer = await wait(asyncio.open_unix_connection(path))
        try:
            # Wait for the accept-side handler to run rather than sleeping a
            # fixed 50 ms: under `-n auto` the grace period can expire before
            # the listener has accepted, and `connection_ref[0]` would then
            # raise IndexError for a reason unrelated to this test.
            await wait_until(lambda: bool(connection_ref), "the listener accepted the connection")
            connection = connection_ref[0]

            pending_write = connection.send(_LARGE_CHUNK)
            final_message = {
                "type": "hello_error",
                "error": {"code": "invalid_request", "message": "Protocol violation"},
            }
            final_frame = encode_server_message(final_message)
            closing = connection.close(final_frame)

            # The reader hasn't drained anything yet, so the connection
            # shouldn't have finished closing yet.
            assert connection.closed is False

            received = bytearray()

            async def _drain_until_eof() -> None:
                while True:
                    chunk = await reader.read(1 << 20)
                    if not chunk:
                        return
                    received.extend(chunk)

            await wait(_drain_until_eof(), timeout=15.0)
            await wait(pending_write)
            await wait(closing)
            assert connection.closed is True

            assert bytes(received[: len(_LARGE_CHUNK)]) == _LARGE_CHUNK
            trailing = bytes(received[len(_LARGE_CHUNK) :])
            messages = ServerMessageDecoder().push(trailing)
            assert messages == [final_message]
        finally:
            writer.close()
    finally:
        await wait(listener.close())
