"""Port of `packages/server/test/server.test.ts`."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import pytest
from conftest import wait
from pi_protocol import ServerMessageDecoder
from pi_server.server import PiServer
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server
from pi_server.types import PiServerOptions


def _service() -> TestServerService:
    return TestServerService()


def test_requires_explicit_listeners() -> None:
    with pytest.raises(TypeError, match="listeners"):
        PiServer(_service(), PiServerOptions(listeners=None))  # type: ignore[arg-type]


def test_rejects_unix_socket_paths_that_cannot_fit_in_sockaddr_un() -> None:
    with pytest.raises(TypeError, match="too long"):
        create_unix_server(_service(), UnixServerOptions(path="/tmp/" + "x" * 512))


async def test_rejects_an_overlong_derived_private_unix_bind_path() -> None:
    max_length = 107 if sys.platform.startswith("linux") else 103
    suffix_length = len(b"/tmp//s")
    path = f"/tmp/{'x' * (max_length - suffix_length)}/s"
    server = create_unix_server(_service(), UnixServerOptions(path=path))

    with pytest.raises(TypeError, match=r"private Unix bind path.*too long"):
        await wait(server.start())


async def test_rejects_concurrent_start_calls_without_leaking_the_unix_listener(socket_dir: Any) -> None:
    path = str(socket_dir / "server.sock")
    server = create_unix_server(_service(), UnixServerOptions(path=path))
    starting = server.start()
    with pytest.raises(RuntimeError, match="starting"):
        await wait(server.start())
    await wait(starting)
    await wait(server.close())
    assert server.addresses == []
    # TypeScript asserts `lstat(path)` rejects with ENOENT, which `lexists`
    # mirrors and `exists` would not (a dangling symlink still lstats).
    assert not os.path.lexists(path)


async def test_handshake_timeout_cleanup_does_not_wait_for_a_blocked_output_queue() -> None:
    class BlockedConnection:
        def __init__(self) -> None:
            self.closed = False
            self.final_chunk: bytes | None = None

        def send(self, chunk: bytes) -> asyncio.Future[None]:
            future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            return future  # Never resolves: simulates a blocked output queue.

        def close(self, final_chunk: bytes | None = None) -> None:
            self.final_chunk = final_chunk
            self.closed = True

    core = PiServer(_service(), PiServerOptions(listeners=[], max_frame_length=1024, handshake_timeout_ms=10))
    connection = BlockedConnection()
    core.accept(connection)  # type: ignore[arg-type]

    async def _wait_closed() -> None:
        while not connection.closed:
            await asyncio.sleep(0.01)

    await wait(_wait_closed())
    assert isinstance(connection.final_chunk, (bytes, bytearray))
    messages = ServerMessageDecoder().push(connection.final_chunk)
    assert len(messages) == 1
    assert messages[0]["type"] == "hello_error"
    assert messages[0]["error"]["code"] == "invalid_request"
    await wait(core.close())


def test_rejects_timeout_values_above_max_timer_delay(socket_dir: Any) -> None:
    # A per-test directory rather than a fixed path: nothing is ever bound here
    # (the constructor rejects the options first), but a shared name across
    # xdist workers is the kind of thing that bites later.
    path = str(socket_dir / "timeout.sock")
    with pytest.raises(TypeError, match="handshakeTimeoutMs"):
        create_unix_server(_service(), UnixServerOptions(path=path, handshake_timeout_ms=2_147_483_648))
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        create_unix_server(_service(), UnixServerOptions(path=path, graceful_close_timeout_ms=2_147_483_648))


async def test_rejects_pending_byte_limits_smaller_than_one_maximum_frame(socket_dir: Any) -> None:
    path = str(socket_dir / "server.sock")
    with pytest.raises(TypeError, match="maxPendingBytes"):
        create_unix_server(_service(), UnixServerOptions(path=path, max_frame_length=128, max_pending_bytes=131))
