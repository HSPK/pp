"""Shared test harness for `pi_server` tests.

Python port of the `startServer`/`connect`/`attach` helpers duplicated across
`packages/server/test/{server,sessions,conformance}.test.ts`.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pi_server.server import PiServer
from pi_server.testing.client import ProtocolTestClient, connect_unix_test_client
from pi_server.testing.service import TestServerService
from pi_server.transports.unix import UnixServerOptions, create_unix_server

DEFAULT_TIMEOUT = 5.0


async def wait(awaitable: Awaitable[Any], timeout: float = DEFAULT_TIMEOUT) -> Any:
    """Awaits `awaitable` with a short bound so a bug never hangs the suite."""
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def wait_until(
    predicate: Callable[[], bool],
    description: str,
    timeout: float = DEFAULT_TIMEOUT,
    interval: float = 0.005,
) -> None:
    """Polls until `predicate` holds, then returns; raises if it never does.

    Preferred over ``await asyncio.sleep(0.05)`` before an assertion. A fixed
    sleep is both slower than it needs to be and wrong under load: the suite now
    runs with ``-n auto``, so a 50 ms grace period that was ample on an idle
    machine can expire before the server has done anything, and the test fails
    for a reason unrelated to what it is checking.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"Timed out after {timeout}s waiting until {description}")
        await asyncio.sleep(interval)


@dataclass
class StartedServer:
    server: PiServer
    service: TestServerService


class ServerHarness:
    """Tracks servers/clients created by a test so they can all be torn down together."""

    def __init__(self, tmp_path: Any) -> None:
        self._tmp_path = tmp_path
        self._socket_root: Path | None = None
        self._servers: list[PiServer] = []
        self._clients: list[ProtocolTestClient] = []
        self._counter = 0

    async def start_server(self, service: TestServerService | None = None, **overrides: Any) -> StartedServer:
        resolved_service = service if service is not None else TestServerService()
        self._counter += 1
        path = str(self._socket_dir() / f"s{self._counter}.sock")
        server = create_unix_server(resolved_service, UnixServerOptions(path=path, **overrides))
        self._servers.append(server)
        await wait(server.start())
        return StartedServer(server=server, service=resolved_service)

    def _socket_dir(self) -> Path:
        """A short-enough directory for a Unix socket.

        A Unix socket path is capped at 107 bytes. pytest's `tmp_path` already
        embeds the test name, and under `pytest-xdist` it also embeds a worker
        id, which together overflow that limit and fail the test for a reason
        that has nothing to do with what it is checking. So sockets live in a
        short private temp directory instead, torn down with the fixture.
        """
        if self._socket_root is None:
            self._socket_root = Path(tempfile.mkdtemp(prefix="pi-t-"))
        return self._socket_root

    async def connect(self, server: PiServer) -> ProtocolTestClient:
        client = await wait(connect_unix_test_client(server.addresses[0]))
        self._clients.append(client)
        return client

    async def aclose(self) -> None:
        await asyncio.gather(*(wait(c.close()) for c in self._clients), return_exceptions=True)
        self._clients.clear()
        await asyncio.gather(*(wait(s.close()) for s in self._servers), return_exceptions=True)
        self._servers.clear()
        if self._socket_root is not None:
            shutil.rmtree(self._socket_root, ignore_errors=True)
            self._socket_root = None


async def attach(client: ProtocolTestClient, session_id: str) -> dict[str, Any]:
    response = await wait(client.request({"command": "attach", "sessionId": session_id}))
    if not response["ok"] or response["result"]["command"] != "attach":
        raise AssertionError(f"Attach failed: {response}")
    return response["result"]["session"]


@pytest.fixture
async def harness(tmp_path: Any):
    instance = ServerHarness(tmp_path)
    try:
        yield instance
    finally:
        await wait(instance.aclose(), timeout=10.0)


@pytest.fixture
def socket_dir():
    """A short-enough directory for Unix sockets.

    Same reason as `ServerHarness._socket_dir`: a Unix socket path is capped at
    107 bytes, and `tmp_path` embeds the test name plus (under `pytest-xdist`)
    a worker id, which can overflow that limit and fail a test for a reason
    that has nothing to do with what it is checking.
    """
    directory = Path(tempfile.mkdtemp(prefix="pi-s-"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
